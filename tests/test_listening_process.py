"""Real disposable Windows jobs: a worker's exit/cancellation also reaps its child."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subkit.proc import CancelledError, KillableRun, hidden


# This script runs only in the disposable root. It never assigns the GUI or the
# unittest process to a job, and its only descendant is the sleeping test Python.
WORKER = r'''
import ctypes
from ctypes import wintypes as w
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, sys.argv[1])
from subkit import listening_process
from subkit.proc import hidden

listening_process.guard_worker()
job = listening_process._job
listening_process.guard_worker()
assert listening_process._job == job

# Independent names and declarations from JOBOBJECT_BASIC_LIMIT_INFORMATION and
# JOBOBJECT_EXTENDED_LIMIT_INFORMATION verify the Windows ABI through a real query.
class Basic(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong),
                ('PerJobUserTimeLimit', ctypes.c_longlong), ('LimitFlags', w.DWORD),
                ('MinimumWorkingSetSize', ctypes.c_size_t), ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', w.DWORD), ('Affinity', ctypes.c_size_t),
                ('PriorityClass', w.DWORD), ('SchedulingClass', w.DWORD)]
class Extended(ctypes.Structure):
    _fields_ = [('BasicLimitInformation', Basic), ('IoInfo', ctypes.c_ulonglong * 6),
                ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]
class Members(ctypes.Structure):
    _fields_ = [('NumberOfAssignedProcesses', w.DWORD), ('NumberOfProcessIdsInList', w.DWORD),
                ('ProcessIdList', ctypes.c_size_t * 2)]

kernel = ctypes.WinDLL('kernel32', use_last_error=True)
kernel.QueryInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.POINTER(w.DWORD)]
kernel.QueryInformationJobObject.restype = w.BOOL
kernel.GetHandleInformation.argtypes = [w.HANDLE, ctypes.POINTER(w.DWORD)]
kernel.GetHandleInformation.restype = w.BOOL
limits, flags = Extended(), w.DWORD()
if not kernel.QueryInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits), None):
    raise ctypes.WinError(ctypes.get_last_error())
if not kernel.GetHandleInformation(job, ctypes.byref(flags)):
    raise ctypes.WinError(ctypes.get_last_error())
child = subprocess.Popen([sys.executable, '-B', '-c', 'import time; time.sleep(60)'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **hidden())
members = Members()
if not kernel.QueryInformationJobObject(job, 3, ctypes.byref(members), ctypes.sizeof(members), None):
    raise ctypes.WinError(ctypes.get_last_error())
data = dict(root_pid=__import__('os').getpid(), child_pid=child.pid,
            pointer_bytes=ctypes.sizeof(ctypes.c_void_p), basic_bytes=ctypes.sizeof(Basic),
            extended_bytes=ctypes.sizeof(Extended), kill_on_close=bool(limits.BasicLimitInformation.LimitFlags & 0x2000),
            inheritable=bool(flags.value & 1), member_pids=list(members.ProcessIdList)[:members.NumberOfProcessIdsInList])
ready, gate = Path(sys.argv[2]), Path(sys.argv[3])
scratch = ready.with_suffix('.tmp')
scratch.write_text(json.dumps(data), encoding='utf-8')
scratch.replace(ready)
deadline = time.monotonic() + 20
while not gate.exists():
    if time.monotonic() >= deadline:
        raise TimeoutError('test parent did not finish its handshake')
    time.sleep(.02)
# Returning normally closes the worker's non-inheritable job handle. The child
# remains asleep, so its termination must come from JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.
'''


@unittest.skipUnless(os.name == "nt", "Windows Job Object lifecycle requires Windows")
class ListeningProcessJob(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, args, result in (
                ("OpenProcess", [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
                ("GetProcessId", [wintypes.HANDLE], wintypes.DWORD),
                ("WaitForSingleObject", [wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
                ("TerminateProcess", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
                ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL)):
            function = getattr(cls.kernel, name)
            function.argtypes, function.restype = args, result

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="duosub-listening-job-test-")
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.script = self.directory / "worker.py"
        self.script.write_text(WORKER, encoding="utf-8")
        self.ready = self.directory / "ready.json"
        self.gate = self.directory / "exit.gate"
        self.handles = []
        self.addCleanup(self.close_handles)

    def close_handles(self):
        for handle in self.handles:
            self.kernel.CloseHandle(handle)

    def command(self):
        return [sys.executable, "-B", "-X", "utf8", str(self.script), str(ROOT / "src"),
                str(self.ready), str(self.gate)]

    def open_owned_process(self, pid):
        self.assertNotEqual(pid, os.getpid(), "The unittest/GUI parent must never enter the disposable job")
        # PROCESS_TERMINATE is reserved for cleaning up this known test process
        # if an assertion exposes a broken guard; a retained handle prevents PID reuse.
        handle = self.kernel.OpenProcess(0x00100000 | 0x00001000 | 0x0001, False, pid)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self.handles.append(handle)
        self.assertEqual(self.kernel.GetProcessId(handle), pid)
        self.assertEqual(self.kernel.WaitForSingleObject(handle, 0), 258)
        return handle

    def read_ready(self, root_poll):
        deadline = time.monotonic() + 8
        while not self.ready.is_file():
            self.assertIsNone(root_poll(), "Disposable worker exited before the job handshake")
            if time.monotonic() >= deadline:
                self.fail("Disposable worker did not report its job within eight seconds")
            time.sleep(.02)
        data = json.loads(self.ready.read_text(encoding="utf-8"))
        self.assertNotEqual(data["root_pid"], data["child_pid"])
        return data

    def assert_guard_info(self, data):
        self.assertEqual(data["pointer_bytes"], 8, "This acceptance probe targets the installed 64-bit Python")
        self.assertEqual((data["basic_bytes"], data["extended_bytes"]), (64, 144))
        self.assertTrue(data["kill_on_close"])
        self.assertFalse(data["inheritable"])
        self.assertEqual(set(data["member_pids"]), {data["root_pid"], data["child_pid"]})

    def reap_known_child(self, handle):
        if handle is not None and self.kernel.WaitForSingleObject(handle, 0) == 258:
            self.kernel.TerminateProcess(handle, 91)
            self.kernel.WaitForSingleObject(handle, 3000)

    def test_normal_worker_exit_closes_job_and_kills_its_sleeping_child(self):
        process = subprocess.Popen(self.command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, **hidden())
        child_handle = None
        try:
            data = self.read_ready(process.poll)
            self.assertEqual(data["root_pid"], process.pid)
            root_handle = self.open_owned_process(process.pid)
            child_handle = self.open_owned_process(data["child_pid"])
            self.assert_guard_info(data)
            self.gate.write_text("normal exit", encoding="utf-8")
            stdout, stderr = process.communicate(timeout=8)
            self.assertEqual(process.returncode, 0, (stdout, stderr))
            self.assertEqual(self.kernel.WaitForSingleObject(root_handle, 3000), 0)
            self.assertEqual(self.kernel.WaitForSingleObject(child_handle, 3000), 0,
                             "Worker exited but its sleeping child survived the job handle")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            self.reap_known_child(child_handle)
            if process.stdout:
                process.stdout.close()
            if process.stderr:
                process.stderr.close()

    def test_killable_cancel_terminates_worker_and_its_child_before_completion(self):
        runner = KillableRun()
        outcome = {}
        def execute():
            try:
                outcome["result"] = runner.run(self.command(), check=True, timeout=30)
            except BaseException as exc:
                outcome["error"] = exc
        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
        child_handle = None
        try:
            # The parent process must remain alive until explicit cancellation.
            data = self.read_ready(lambda: None if thread.is_alive() else "ended")
            root_handle = self.open_owned_process(data["root_pid"])
            child_handle = self.open_owned_process(data["child_pid"])
            self.assert_guard_info(data)
            runner.cancel()
            thread.join(timeout=8)
            self.assertFalse(thread.is_alive(), "Cancelled worker pipeline did not finish")
            self.assertIsInstance(outcome.get("error"), CancelledError)
            self.assertNotIn("result", outcome)
            self.assertEqual(self.kernel.WaitForSingleObject(root_handle, 3000), 0)
            self.assertEqual(self.kernel.WaitForSingleObject(child_handle, 3000), 0,
                             "Cancellation returned with a live inference helper")
        finally:
            runner.cancel()
            thread.join(timeout=5)
            self.reap_known_child(child_handle)


if __name__ == "__main__":
    unittest.main()

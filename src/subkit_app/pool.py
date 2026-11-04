"""Start background jobs on the thread pool without PySide keeping every finished job alive.

PySide6 (checked on 6.11.1) makes the pool the owner of each runnable passed to ``QThreadPool.start``.  Qt deletes
an auto-deleting runnable after it ran, which releases it; a runnable that does not auto-delete is never deleted,
so the pool keeps the Python object - and everything it references, such as a whole preview document or the audio
samples - until the program exits.  DuoSub's jobs do not auto-delete (callers keep them to cancel, wait and drop
late signals), so they run inside a small auto-deleting carrier instead: Qt deletes the carrier after the run, and
the job is freed as soon as its caller lets go of it.
"""

from __future__ import annotations

import threading
from typing import Optional

from PySide6.QtCore import QRunnable, QThreadPool


class _Carrier(QRunnable):
    def __init__(self, job: QRunnable) -> None:
        super().__init__()
        self.job: Optional[QRunnable] = job
        self.lock = threading.Lock()
        self.started = False

    def run(self) -> None:  # worker thread
        with self.lock:
            self.started = True
            job, self.job = self.job, None
        if job is not None:
            job.run()


def start(job: QRunnable, priority: int = 0, pool: Optional[QThreadPool] = None) -> None:
    carrier = _Carrier(job)
    job._pool_carrier = carrier
    (pool or QThreadPool.globalInstance()).start(carrier, priority)


def try_take(job: QRunnable, pool: Optional[QThreadPool] = None) -> bool:
    """Remove a job that has not started yet from the queue (``QThreadPool.tryTake``); False once it runs or ran."""
    carrier = getattr(job, "_pool_carrier", None)
    if carrier is None:
        return False
    # Holding the lock keeps a dequeued carrier from running, and so from being deleted, during tryTake.
    with carrier.lock:
        if carrier.started:
            return False
        taken = (pool or QThreadPool.globalInstance()).tryTake(carrier)
        if taken:
            carrier.job = None
    if taken:
        job._pool_carrier = None              # the carrier is ours again and is freed with this reference
    return taken

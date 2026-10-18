"""Versioned audio/model cache with atomic publication of complete files."""
from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

import numpy as np

from .models import DEMUCS_ID, KANA_ID, KANA_REVISION, MMS_ID
from .song import ModelEmission

KALIGN_VERSION = "kalign-v1"


def rounded_clip(clip):
    """Nearest 10 ms, ties to even, used by both decoding and cache identity."""
    return tuple(int(round(t / 10)) * 10 for t in clip)


def cache_key(video, audio_index, clip):
    """The video's size and modified time (not its path: a moved video keeps its cache), track and clip."""
    st = Path(video).stat()
    identity = ["video", st.st_size, st.st_mtime_ns, audio_index, rounded_clip(clip) if clip is not None else "dialogue",
                DEMUCS_ID, MMS_ID, KANA_ID, KANA_REVISION, KALIGN_VERSION]
    return hashlib.sha1(json.dumps(identity, ensure_ascii=False).encode("utf-8")).hexdigest()


def legacy_cache_key(video, audio_index, clip, *, recorded=None):
    """The pre-0.21 path-based key; ``recorded`` is where the video was when the entry was written."""
    st = Path(video).stat()
    identity = [str(Path(recorded or video).resolve()), st.st_size, st.st_mtime_ns, audio_index,
                rounded_clip(clip) if clip is not None else "dialogue",
                DEMUCS_ID, MMS_ID, KANA_ID, KANA_REVISION, KALIGN_VERSION]
    return hashlib.sha1(json.dumps(identity, ensure_ascii=False).encode("utf-8")).hexdigest()


def cache_directory(root, video, audio_index, clip, *, recorded=None):
    """``root/<key>``; an entry under the older path-based key (this path, or ``recorded``) is renamed."""
    from ..audio import adopt_legacy

    root = Path(root)
    directory = root / cache_key(video, audio_index, clip)
    if not directory.exists():
        for where in (None, recorded) if recorded is not None else (None,):
            if adopt_legacy(root / legacy_cache_key(video, audio_index, clip, recorded=where), directory):
                break
    return directory


def atomic_write(path, writer):
    """Stage on the destination volume, then replace; incomplete files aren't cache entries.

    Staging is outside kalign/ so even a hard kill leaves only scratch, never a
    partly published cache file. Ordinary failures remove their own staging file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Cache writes need the same volume for os.replace (run TEMP may be on C:).
    staging = next((p.parent for p in path.parents if p.name == "kalign"), path.parent)
    staging = staging / ".kalign-staging"
    staging.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        if os.name == "nt":
            # Delete-on-close also covers a parent killing the worker mid-write.
            # O_TEMPORARY grants delete sharing, so rename is legal while open.
            import ctypes
            from ctypes import wintypes
            import msvcrt
            tmp = staging / (uuid.uuid4().hex + ".tmp")
            fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_BINARY | os.O_TEMPORARY)
            with os.fdopen(fd, "w+b") as stream:
                writer(stream)
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(tmp, path)
                # After atomic publication, retain the complete file on close.
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                set_info = kernel.SetFileInformationByHandle
                set_info.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
                set_info.restype = wintypes.BOOL
                delete = wintypes.DWORD(8)  # FILE_DISPOSITION_ON_CLOSE without DELETE
                if not set_info(msvcrt.get_osfhandle(stream.fileno()), 21, ctypes.byref(delete), ctypes.sizeof(delete)):
                    raise ctypes.WinError(ctypes.get_last_error())
            return
        with tempfile.NamedTemporaryFile(dir=staging, suffix=".tmp", delete=False) as stream:
            tmp = Path(stream.name)
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, lambda f: f.write(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")))


def write_array(path, value):
    atomic_write(path, lambda f: np.save(f, value, allow_pickle=False))


def write_emission(directory, emission, *, dtype=None):
    directory = Path(directory)
    meta = {f.name: getattr(emission, f.name) for f in fields(emission) if f.name != "log_probs"}
    lp = emission.log_probs if dtype is None else emission.log_probs.astype(dtype)
    write_array(directory / (emission.name + ".npy"), lp)
    # Metadata is the completion marker: missing metadata causes recomputation.
    write_json(directory / (emission.name + ".json"), meta)


def load_emission(directory, name):
    directory = Path(directory)
    try:
        meta = json.loads((directory / (name + ".json")).read_text(encoding="utf-8"))
        lp = np.load(directory / (name + ".npy"), allow_pickle=False)
        return ModelEmission(log_probs=lp, **meta)
    except (OSError, ValueError, TypeError, KeyError):
        return None

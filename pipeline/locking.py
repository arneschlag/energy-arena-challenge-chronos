"""Cross-container file locks for shared data and the single GPU.

All paths are supplied through environment variables and live on bind-mounted
host directories.  The implementation intentionally uses only the standard
library so it also works in the small data/summary container.
"""
from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path


LOCK_DIR = Path(os.environ.get("LOCK_DIR", "/locks"))
DATA_LOCK = Path(os.environ.get("DATA_LOCK_FILE", LOCK_DIR / "data.lock"))
GPU_LOCK = Path(os.environ.get("GPU_LOCK_FILE", LOCK_DIR / "gpu.lock"))


@contextmanager
def file_lock(path: str | os.PathLike, *, shared: bool = False):
    """Hold an advisory lock for the duration of the context."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def data_read_lock():
    return file_lock(DATA_LOCK, shared=True)


def data_write_lock():
    return file_lock(DATA_LOCK, shared=False)


def gpu_lock():
    return file_lock(GPU_LOCK, shared=False)

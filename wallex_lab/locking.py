"""Single-writer OS lock, released by the kernel even after a process crash."""
from contextlib import contextmanager
import errno
import math
import os
import time


def atomic_replace(source, destination, timeout=1.):
    """Windows readers/antivirus can briefly deny replacement of a closed JSON file."""
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
        raise ValueError('Replacement timeout must be finite nonnegative seconds')
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(.02, remaining))


@contextmanager
def exclusive(path, timeout=0):
    """Fail fast by default; optionally serialize transient contention for bounded seconds."""
    if isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or timeout<0:
        raise ValueError('Lock timeout must be finite nonnegative seconds')
    with open(path, 'a+b') as handle:
        if not handle.seek(0, os.SEEK_END):
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
        else:
            import fcntl
        deadline=time.monotonic()+timeout
        while True:
            try:
                if os.name=='nt':msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (PermissionError,BlockingIOError) as error:
                remaining=deadline-time.monotonic()
                if error.errno not in (errno.EACCES,errno.EAGAIN,errno.EWOULDBLOCK) or remaining<=0:raise
                time.sleep(min(.01,remaining))
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)

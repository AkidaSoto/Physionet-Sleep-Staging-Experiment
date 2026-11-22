"""Measure time for the development compute budget across laptop sleep."""

import ctypes
import sys
from time import monotonic


if sys.platform == "win32":
    _query_awake_time = ctypes.windll.kernel32.QueryUnbiasedInterruptTime
    _query_awake_time.argtypes = (ctypes.POINTER(ctypes.c_uint64),)
    _query_awake_time.restype = ctypes.c_int

    def active_clock() -> float:
        """Return monotonic seconds excluding Windows sleep and hibernation."""
        ticks = ctypes.c_uint64()
        if not _query_awake_time(ctypes.byref(ticks)):
            raise ctypes.WinError()
        return ticks.value / 10_000_000
else:
    active_clock = monotonic

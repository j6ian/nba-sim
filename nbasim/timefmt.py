"""Date formatting that works on Windows as well as Mac/Linux.

Windows' strftime rejects the glibc/BSD "%-d" and "%-I" (no leading zero)
codes with "ValueError: Invalid format string", so expand them by hand.
"""
from __future__ import annotations


def fmt(when, pattern: str) -> str:
    """strftime with portable support for %-d (day) and %-I (12-hour)."""
    pattern = pattern.replace("%-d", str(when.day))
    pattern = pattern.replace("%-I", str(int(when.strftime("%I"))))
    return when.strftime(pattern)

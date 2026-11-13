"""Keep the standard library's ``platform`` module from querying WMI on Windows.

Since Python 3.12, ``platform.system()``, ``win32_ver()``, ``machine()`` and
``processor()`` ask WMI first, through the private ``platform._wmi_query``,
and fall back to ``sys.getwindowsversion()``, ``cmd /c ver`` and environment
variables when that raises ``OSError``. Under CPU load a WMI query in a newly
spawned process can close one of that process's handles after it returns
(TFS 65866): parse-pool workers that ran ``platform.system()`` while
importing ChunkHound later failed with ``[WinError 6] The handle is invalid``
releasing their call-queue semaphore, or released the wrong semaphore and
stalled the pool. Raising ``OSError`` instead sends ``platform`` down its own
fallback, which reports the same OS for the uses ChunkHound has.

This module imports only the standard library: it runs from the package
``__init__``, so in every ChunkHound process (spawned workers re-import the
package) before anything calls ``platform``, including build environments
that have none of ChunkHound's dependencies.
"""

from __future__ import annotations

import logging
import sys


def _wmi_query_disabled(*_args: object, **_kwargs: object) -> object:
    """Stand-in for ``platform._wmi_query`` that sends callers to the fallback."""
    raise OSError("ChunkHound does not query WMI from platform (TFS 65866)")


def disable_platform_wmi_queries() -> None:
    """Make ``platform`` use its non-WMI fallback in this process.

    Does nothing off Windows, and logs at debug level and does nothing when
    this Python's ``platform`` has no ``_wmi_query`` (before 3.12, or a
    release that renamed it).
    """
    if sys.platform != "win32":
        return
    import platform

    if not hasattr(platform, "_wmi_query"):
        logging.getLogger(__name__).debug(
            "platform has no _wmi_query; WMI queries are not disabled"
        )
        return
    platform._wmi_query = _wmi_query_disabled  # type: ignore[attr-defined]

"""Signal a kernel inotify queue overflow to the realtime service.

When the kernel inotify queue overflows, the kernel drops events and queues one
IN_Q_OVERFLOW record. watchdog 7.1.0 emits no public event for that record,
and 7.1.1 emits an EventsLostEvent that ChunkHound does not map. The FD reader
calls ``on_overflow`` after watchdog handles the record on either version.

The hook relies on ``inotify_c.InotifyFD.handle_event``,
``InotifyObserver._inotify_fd``, and ``InotifyObserver._emitter_kwargs``.
Detection reports "active" for a hooked reader or a recognized lazy observer
whose first schedule() will install the hook. On Linux, an unrecognized reader
layout or missing reader API yields "unavailable". A future watchdog with the
same lazy layout but a different assignment path could bypass the hook.
CI exercises the locked watchdog version, so the 7.1.0 constructor path and
rejection of an unfamiliar layout are outside its regular coverage.

watchdog is imported inside the function so importing this module never loads
it (see tests/unit/test_watchdog_import_isolation.py).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

if TYPE_CHECKING:
    from .startup import RealtimeObserver

# Whether a kernel inotify overflow reaches the service: "active" (hook
# installed), "unavailable" (Linux, but no hook could be installed) or
# "not_applicable" (not Linux).
InotifyOverflowDetection = Literal["active", "unavailable", "not_applicable"]


def create_inotify_observer(
    on_overflow: Callable[[], None],
) -> RealtimeObserver | None:
    """Return an inotify observer that reports overflows, or None.

    None means no hook can be installed (not Linux, inotify unusable, or
    watchdog < 7.1, which drops the overflow record); the caller then uses
    watchdog's default observer.
    """
    if not sys.platform.startswith("linux"):
        return None
    from watchdog.utils import UnsupportedLibcError

    # The same guard watchdog's own observer selection uses on Linux.
    try:
        from watchdog.observers import inotify_c
        from watchdog.observers.inotify import InotifyObserver
    except (ImportError, UnsupportedLibcError):
        return None
    reader = getattr(inotify_c, "InotifyFD", None)
    if not isinstance(reader, type) or not callable(
        getattr(reader, "handle_event", None)
    ):
        return None
    inotify_fd_cls = reader

    class _OverflowSignalingInotifyFD(inotify_fd_cls):  # type: ignore[misc,valid-type]
        def handle_event(self, event: Any) -> None:
            try:
                super().handle_event(event)
            finally:
                # After watchdog's synchronous watch rebuild, so the resync
                # debounce starts only once every watch is back.
                if event.is_queue_overflow:
                    try:
                        on_overflow()
                    except Exception as error:
                        logger.warning(
                            f"inotify overflow could not be reported: {error}"
                        )

    class _OverflowObserver(InotifyObserver):
        def __setattr__(self, name: str, value: Any) -> None:
            # watchdog assigns the reader during construction in 7.1.0 and
            # during the first schedule() in 7.1.1, before installing a watch.
            if name == "_inotify_fd" and isinstance(value, inotify_fd_cls):
                value.__class__ = _OverflowSignalingInotifyFD
            super().__setattr__(name, value)

    observer = _OverflowObserver()
    fd = getattr(observer, "_inotify_fd", ...)
    # In 7.1.1, the empty reader slot and emitter kwargs dict identify the
    # observer that creates the reader and passes it to emitters in schedule().
    lazy = fd is None and isinstance(getattr(observer, "_emitter_kwargs", None), dict)
    if not lazy and not isinstance(fd, _OverflowSignalingInotifyFD):
        observer.stop()
        return None
    return observer

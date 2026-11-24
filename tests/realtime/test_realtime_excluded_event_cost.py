"""An excluded event crosses to the event loop once (TFS 65868).

The watchdog handler runs on watchdog's thread and forwards its records to the
service's event loop; the two threads share one interpreter lock, so each
crossing is paid by both. After an npm install hundreds of thousands of
excluded events arrive at once, and an edit made then waits behind all of them.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chunkhound.core.config.config import Config
from chunkhound.services.realtime.service import RealtimeIndexingService

_BURST = 200


class _CountingLoop:
    """The running loop, counting the calls made to it from another thread."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self.crossings = 0

    def call_soon_threadsafe(self, *args: Any) -> asyncio.Handle:
        self.crossings += 1
        return self._loop.call_soon_threadsafe(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loop, name)


@pytest.mark.asyncio
async def test_excluded_burst_costs_one_loop_crossing_per_event(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "repo").resolve()
    edited = root / "src" / "app.py"
    edited.parent.mkdir(parents=True)
    edited.write_text("x = 1\n")
    config = Config(
        database={"path": str(tmp_path / "db.duckdb"), "provider": "duckdb"},
        indexing={"realtime_backend": "watchdog"},
        target_dir=root,
    )
    # The database is the external boundary: note how many crossings had been
    # made each time it is asked whether a file was indexed.
    crossings_at_lookup: list[int] = []

    def get_file_by_path(_path: str) -> None:
        crossings_at_lookup.append(counting.crossings)

    service = RealtimeIndexingService(
        SimpleNamespace(provider=SimpleNamespace(get_file_by_path=get_file_by_path)),
        config,
    )
    loop = asyncio.get_running_loop()
    counting = _CountingLoop(loop)

    # The handler exactly as the service builds it for its watchdog observer.
    bootstrapped = service._bootstrap_fs_monitor(
        root,
        counting,  # type: ignore[arg-type]
        threading.Event(),
    )
    assert bootstrapped is not None
    handler = bootstrapped.event_handler
    # Only the events below reach the handler.
    service._stop_bootstrap_observer(bootstrapped.observer)
    counting.crossings = 0

    for i in range(_BURST):
        handler.on_any_event(
            SimpleNamespace(
                event_type="created",
                is_directory=False,
                src_path=str(root / "node_modules" / "pkg" / f"f{i}.js"),
            )
        )
    handler.on_any_event(
        SimpleNamespace(event_type="modified", is_directory=False, src_path=str(edited))
    )
    drained = loop.create_future()
    loop.call_soon(drained.set_result, None)
    await drained

    # One record per excluded event; the edit records its source, then queues.
    assert counting.crossings == _BURST + 2
    status = service.health_snapshot()
    assert status["pipeline"]["filtered_event_count"] == _BURST
    assert status["event_queue"]["accepted"] == 1

    # An excluded delete asks the database, which can wait; its source record
    # has crossed before that, so a status read counts it (TFS 65773).
    crossings_before_delete = counting.crossings
    handler.on_any_event(
        SimpleNamespace(
            event_type="deleted",
            is_directory=False,
            src_path=str(root / "node_modules" / "pkg" / "gone.js"),
        )
    )
    assert crossings_at_lookup == [crossings_before_delete + 1]

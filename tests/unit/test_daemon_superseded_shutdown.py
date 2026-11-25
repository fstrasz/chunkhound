"""A daemon superseded by another live daemon must end, and leave its lock alone.

A daemon whose lock names another live daemon is no longer discoverable, so a
daemon in that state with no clients would otherwise idle forever. When it
does shut down, it must not delete the live daemon's lock or registry entry,
or the next proxy starts yet another daemon.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chunkhound.core.config.config import Config
from chunkhound.daemon import server as daemon_server
from chunkhound.daemon.discovery import DaemonDiscovery
from chunkhound.daemon.server import ChunkHoundDaemon

_SHUTDOWN_BOUND_SECONDS = 10.0


async def _supersede_and_wait_for_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[DaemonDiscovery, int, bool]:
    """Run a daemon, hand its lock to another live pid, and wait for exit.

    Returns the discovery, the other pid, and whether the daemon exited on its
    own within the bound.
    """
    monkeypatch.setenv("CHUNKHOUND_DAEMON_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("CHUNKHOUND_DAEMON_REGISTRY_DIR", str(tmp_path / "registry"))
    monkeypatch.setattr(daemon_server, "_PUBLICATION_CHECK_INTERVAL", 0.05)

    project_dir = tmp_path / "project"
    db_path = project_dir / ".chunkhound" / "test.db"
    db_path.parent.mkdir(parents=True)
    config = Config(
        target_dir=project_dir,
        database={"path": str(db_path), "provider": "duckdb"},
        indexing={},
    )
    socket_path = "tcp:127.0.0.1:1"
    daemon = ChunkHoundDaemon(
        config,
        args=MagicMock(path=project_dir),
        socket_path=socket_path,
        project_dir=project_dir,
    )

    async def fake_initialize() -> None:
        daemon._start_startup_phase("initialize")
        daemon._initialized = True
        daemon._scan_target_path = daemon._normalize_requested_target_path(project_dir)
        daemon._startup_publish_complete.clear()
        daemon._deferred_start_task = asyncio.create_task(asyncio.sleep(0))
        daemon._startup_failure_message = None

    async def fake_barrier() -> None:
        daemon._start_startup_phase("startup_barrier")

    daemon.initialize = fake_initialize  # type: ignore[assignment]
    daemon.await_startup_barrier = fake_barrier  # type: ignore[assignment]

    mock_server = AsyncMock()
    mock_server.__aenter__ = AsyncMock(return_value=None)
    mock_server.__aexit__ = AsyncMock(return_value=None)

    discovery = daemon._discovery
    registry_path = discovery.get_registry_entry_path()
    # The test process's parent stands in for the live daemon that took over.
    other_pid = os.getppid()

    with patch(
        "chunkhound.daemon.server.ipc.create_server", new_callable=AsyncMock
    ) as create_server:
        create_server.return_value = (mock_server, socket_path)
        run_task = asyncio.create_task(daemon.run())
        exited = False
        try:
            for _ in range(250):
                lock: dict[str, Any] | None = discovery.read_lock()
                if (
                    lock is not None
                    and lock.get("pid") == os.getpid()
                    and registry_path.exists()
                ):
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("daemon never published its lock")

            discovery.write_lock(other_pid, "tcp:127.0.0.1:2")
            discovery.write_registry_entry(other_pid, "tcp:127.0.0.1:2")

            try:
                await asyncio.wait_for(
                    asyncio.shield(run_task), timeout=_SHUTDOWN_BOUND_SECONDS
                )
                exited = True
            except asyncio.TimeoutError:
                pass
        finally:
            daemon._shutdown_event.set()
            await asyncio.wait_for(run_task, timeout=15)

    return discovery, other_pid, exited


@pytest.mark.asyncio
async def test_superseded_daemon_without_clients_shuts_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, exited = await _supersede_and_wait_for_exit(tmp_path, monkeypatch)

    assert exited, (
        f"superseded daemon with no clients was still running after "
        f"{_SHUTDOWN_BOUND_SECONDS}s"
    )


@pytest.mark.asyncio
async def test_superseded_daemon_shutdown_keeps_the_live_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    discovery, other_pid, _ = await _supersede_and_wait_for_exit(tmp_path, monkeypatch)

    lock = discovery.read_lock()
    entry = discovery.read_registry_entry()
    assert lock is not None and lock["pid"] == other_pid, lock
    assert entry is not None and entry["pid"] == other_pid, entry

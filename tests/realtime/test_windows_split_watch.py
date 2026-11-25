"""Windows watchdog contracts at the dispatch and DuckDB boundaries."""

import asyncio
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from watchdog.events import FileMovedEvent
from watchdog.observers.api import BaseObserver
from watchdog.version import VERSION_INFO

from chunkhound.core.config.config import Config
from chunkhound.database_factory import create_services
from chunkhound.mcp_server.base import MCPServerBase
from chunkhound.services.realtime.events import SimpleEventHandler
from chunkhound.services.realtime.service import RealtimeIndexingService

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        sys.platform != "win32" or VERSION_INFO != (7, 1, 0),
        reason="Windows watchdog 7.1.0 delivery contract",
    ),
]


class _Server(MCPServerBase):
    """Exercise the production resync without opening a transport."""

    def _register_tools(self) -> None:
        pass

    async def run(self) -> None:
        pass


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator:
    root = tmp_path / "project"
    (root / "src" / "config").mkdir(parents=True)
    (root / "tools").mkdir()
    external = tmp_path / "chunkhound.json"
    external.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("CHUNKHOUND_GLOBAL_CONFIG_FILE", str(external))
    db = tmp_path / "index.duckdb"
    config = Config(
        target_dir=root,
        database={"path": str(db), "provider": "duckdb"},
        embeddings_disabled=True,
        indexing={"realtime_backend": "watchdog", "workspace_gitignore_nonrepo": True},
    )
    services = create_services(db, config)
    services.provider.connect()
    server = _Server(config)
    server.services = services
    server._scan_target_path = root
    service = RealtimeIndexingService(
        services, config, resync_callback=server._request_realtime_resync
    )
    server.realtime_indexing = service
    try:
        yield root, external, service, services.provider
    finally:
        services.provider.disconnect()


async def wait_until(
    condition: Callable[[], bool], failure: str = "watcher did not converge"
) -> None:
    deadline = asyncio.get_running_loop().time() + 25
    while not condition():
        assert asyncio.get_running_loop().time() < deadline, failure
        await asyncio.sleep(0.05)


def indexed_text(provider, path: Path, marker: str) -> bool:
    row = provider.get_file_by_path(str(path))
    return row is not None and any(
        marker in chunk["code"] for chunk in provider.get_chunks_by_file_id(row["id"])
    )


async def test_install_recreation_keeps_dependency_descendants_out_of_dispatch(
    project, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, service, provider = project
    deps = root / "node_modules"
    deps.mkdir()
    direct = root / "root.ts"
    source = root / "src/config/hostDirs.ts"
    direct.write_text("export const rootValue = 0;\n")
    source.write_text("export const hostDirs = 0;\n")
    delivered: list[Path] = []
    dispatch = SimpleEventHandler.dispatch

    def record(handler, event):
        if event.src_path:
            delivered.append(Path(event.src_path))
        dispatch(handler, event)

    monkeypatch.setattr(SimpleEventHandler, "dispatch", record)
    try:
        await service.start(root)
        for cycle in range(2):
            pkg = deps / "pkg"
            pkg.mkdir()
            generated = pkg / "index.ts"
            generated.write_text(f"export const generated = {cycle};\n")
            generated.unlink()
            pkg.rmdir()
            deps.rmdir()
            deps.mkdir()
        direct.write_text("export const rootValue = 'rootAfterInstall';\n")
        source.write_text("export const hostDirs = 'sourceAfterInstall';\n")
        await wait_until(
            lambda: (
                deps in delivered
                and indexed_text(provider, direct, "rootAfterInstall")
                and indexed_text(provider, source, "sourceAfterInstall")
            )
        )
        assert not [p for p in delivered if deps in p.parents], (
            "dependency descendant reached dispatch"
        )
        assert service.health_snapshot()["watched_directories_count"] == 3
        assert not provider.snapshot_file_paths_under_directory("node_modules")
    finally:
        await service.stop()


async def test_policy_change_reconciles_rows_and_concurrent_edits(project) -> None:
    root, external, service, provider = project
    tool = root / "tools/task.py"
    tool.write_text("def seeded_tool(): return 1\n")
    await service.services.indexing_coordinator.process_file(tool, skip_embeddings=True)
    assert indexed_text(provider, tool, "seeded_tool")
    ignore_file = root / "src/.gitignore"
    ignore_file.write_text("# initial rules\n")
    for policy_file in (external, ignore_file):
        during = f"during_{policy_file.suffix[1:]}_handoff"
        after = f"after_{policy_file.suffix[1:]}_handoff"
        wrote_during = False

        def edit_during_handoff(status):
            nonlocal wrote_during
            if (
                status.get("watchdog_fallback_reason")
                and not status["monitoring_ready"]
                and not wrote_during
            ):
                wrote_during = True
                tool.write_text(f"def {during}(): return 2\n")

        service._status_callback = edit_during_handoff
        try:
            await service.start(root)
            previous_scans = service.health_snapshot()["resync"]["performed_count"]
            if policy_file == external:
                replacement = external.with_suffix(".new")
                replacement.write_text(
                    json.dumps({"indexing": {"exclude": ["tools/**"]}})
                )
                replacement.replace(external)
            else:
                policy_file.write_text("generated/\n")
            await wait_until(
                lambda: (
                    service.health_snapshot()["resync"]["performed_count"]
                    > previous_scans
                ),
                f"{policy_file.name} change did not trigger a handoff resync",
            )
            status = service.health_snapshot()
            assert (
                status["effective_backend"] == "watchdog"
                and status["watched_directories_count"] == 1
            )
            assert "policy" in status["watchdog_fallback_reason"] and wrote_during
            assert indexed_text(provider, tool, during), (
                "handoff lost the edit to the seeded file"
            )
            tool.write_text(f"def {after}(): return 3\n")
            await wait_until(lambda: indexed_text(provider, tool, after))
            rows = provider.execute_query(
                "SELECT id FROM files WHERE path = ?", ["tools/task.py"]
            )
            assert len(rows) == 1
        finally:
            await service.stop()
        external.write_text("{}")


async def test_moves_and_failed_attach_reconcile_before_reporting_coverage(
    project, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _, service, provider = project
    old = root / "src/source.py"
    moved = root / "tools/source.py"
    direct = root / "root.py"
    direct.write_text("def original_root(): return 1\n")
    old.write_text("def moving_source(): return 1\n")
    await service.services.indexing_coordinator.process_file(old, skip_embeddings=True)
    assert indexed_text(provider, old, "moving_source")
    try:
        await service.start(root)
        old.rename(moved)
        # The Windows NEW record may have no OLD on this emitter.
        service.event_handler.dispatch(FileMovedEvent("", str(moved)))
        await wait_until(
            lambda: service.health_snapshot()["resync"]["performed_count"] >= 1
        )
        assert provider.get_file_by_path(str(old)) is None and indexed_text(
            provider, moved, "moving_source"
        )
        temp = root / "root.py.tmp"
        temp.write_text("def replaced_root(): return 2\n")
        temp.replace(direct)

        def root_replacement_indexed() -> bool:
            status = service.health_snapshot()
            assert (
                status.get("watchdog_fallback_reason") is None
                and status["watched_directories_count"] == 3
            ), "atomic replacement of a root file lost split coverage"
            return indexed_text(provider, direct, "replaced_root")

        await wait_until(root_replacement_indexed)
        (root / "tools").rename(root / "renamed")
        renamed = root / "renamed/source.py"
        await wait_until(
            lambda: (
                service.health_snapshot()["watched_directories_count"] == 1
                and service.monitoring_ready.is_set()
                and indexed_text(provider, renamed, "moving_source")
            )
        )
        assert provider.get_file_by_path(str(moved)) is None
        assert service.health_snapshot()["effective_backend"] == "watchdog"
        await service.stop()

        schedule = BaseObserver.schedule

        def fail_child(observer, handler, path, **kwargs):
            if Path(path) == root / "renamed":
                raise OSError("child attach denied")
            return schedule(observer, handler, path, **kwargs)

        monkeypatch.setattr(BaseObserver, "schedule", fail_child)
        snapshots = []
        service._status_callback = snapshots.append
        previous_count = service.health_snapshot()["resync"]["performed_count"]
        await service.start(root)
        ready = next(s for s in snapshots if s["monitoring_ready"])
        assert ready["resync"]["performed_count"] > previous_count, (
            "ready before failed-attach reconciliation"
        )
        assert (
            ready["watched_directories_count"] == 1
            and "child attach denied" in ready["watchdog_fallback_reason"]
        )
        assert ready["effective_backend"] == "watchdog"
    finally:
        await service.stop()

"""Windows watchdog coverage for ordinary top-level directories."""

from __future__ import annotations

import asyncio
import hashlib
import os
import stat
from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any

from chunkhound.core.config.config import Config
from chunkhound.utils.ignore_engine import compile_watch_exclusions

from .events import SimpleEventHandler, watchdog_event_filter

if TYPE_CHECKING:
    from .startup import RealtimeObserver


class WindowsSplitWatch:
    """Own top-level coverage until a transition requires a recursive root."""

    def __init__(
        self,
        root: Path,
        config: Config,
        provider: Any,
        handler: SimpleEventHandler,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.root = root
        self.handler = handler
        self.provider = provider
        self.loop = loop
        self.events: asyncio.Queue[Any] = asyncio.Queue()
        self.covered: set[Path] = set()
        self.omitted: set[Path] = set()
        self.config = config
        self.config_excludes: Callable[[str], bool] | None = None
        pinned_global = os.getenv("CHUNKHOUND_GLOBAL_CONFIG_FILE")
        self.config_paths = {
            *(p for p in (config.global_config_file, config.config_file) if p),
            root / ".chunkhound.json",
        }
        if pinned_global:
            self.config_paths.add(Path(pinned_global))
        else:
            for candidate in config.get_global_config_candidates():
                self.config_paths.add(candidate)
                if candidate == config.global_config_file:
                    break
        self.policy_names = {p.name.lower() for p in self.config_paths} | {".gitignore"}
        self.external_paths = tuple(
            p for p in self.config_paths if not p.is_relative_to(root)
        )
        self.revision = self.external_revision()

    def external_revision(self) -> tuple[bytes | int | None, ...]:
        revision: list[bytes | int | None] = []
        for path in self.external_paths:
            try:
                revision.append(hashlib.sha256(path.read_bytes()).digest())
            except FileNotFoundError:
                revision.append(None)
            except OSError as error:
                revision.append(error.errno)
        return tuple(revision)

    def can_omit(self, path: Path) -> bool:
        if self.config_excludes is None or not self.config_excludes(path.name):
            return False
        policy = self.handler._path_filter
        if not policy.excludes_directory(path) or policy.is_degraded:
            return False
        try:
            # The entry is an immediate child. Preserve its DB spelling and
            # query anew even when an earlier occupant of this name was empty.
            rows = self.provider.snapshot_file_paths_under_directory(path.name)
            return isinstance(rows, frozenset) and not rows
        except Exception:
            return False

    def start(self, observer: RealtimeObserver, abort: Event) -> None:
        filters = watchdog_event_filter()
        observer.schedule(
            self.handler, str(self.root), recursive=False, event_filter=filters
        )
        observer.start()
        self.config_excludes = compile_watch_exclusions(
            self.config.indexing.get_effective_config_excludes()
        )
        for path in self.root.iterdir():
            if abort.is_set():
                return
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            directory = bool(info.st_file_attributes & stat.FILE_ATTRIBUTE_DIRECTORY)
            if not directory:
                continue
            if self.can_omit(path):
                self.omitted.add(path)
                continue
            if info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise OSError(f"top-level directory link: {path}")
            try:
                observer.schedule(
                    self.handler, str(path), recursive=True, event_filter=filters
                )
            except Exception as error:
                raise OSError(f"child watch failed for {path}: {error}") from error
            self.covered.add(path)

    def intercept(self, event: Any) -> bool:
        """Run on the dispatcher. Only boundary events leave its normal fast path."""
        src = event.src_path
        dest = getattr(event, "dest_path", "")
        for spelling in (src, dest):
            if spelling and self.is_policy_file(spelling):
                self.loop.call_soon_threadsafe(
                    self.events.put_nowait, ("policy", spelling)
                )
                break
        if event.event_type in {"created", "deleted", "moved"}:
            if any(p and Path(p).parent == self.root for p in (src, dest)):
                self.loop.call_soon_threadsafe(
                    self.events.put_nowait, ("boundary", event)
                )
                return True
        return False

    def is_policy_file(self, spelling: str) -> bool:
        name = os.path.basename(spelling).lower()
        return name in self.policy_names and (
            name == ".gitignore" or Path(spelling) in self.config_paths
        )

    def use_recursive_root(self, observer: RealtimeObserver) -> None:
        observer.unschedule_all()
        self.handler.intercept = None
        observer.schedule(
            self.handler,
            str(self.root),
            recursive=True,
            event_filter=watchdog_event_filter(),
        )

    async def wait_for_transition(self, stop: asyncio.Event) -> str | None:
        """Consume direct-entry and policy changes until full coverage is needed."""
        next_check = self.loop.time() + 1.0
        while not stop.is_set():
            delay = next_check - self.loop.time()
            if delay <= 0:
                if await asyncio.to_thread(self.external_revision) != self.revision:
                    return "policy changed: external config"
                next_check = self.loop.time() + 1.0
                continue
            try:
                kind, value = await asyncio.wait_for(self.events.get(), delay)
            except asyncio.TimeoutError:
                continue
            if kind == "policy":
                return f"policy changed: {value}"
            try:
                reason = await asyncio.to_thread(self.boundary_change, value)
            except OSError as error:
                return f"watchdog coverage check failed: {error}"
            if reason:
                return reason
        return None

    def boundary_change(self, event: Any) -> str | None:
        """Classify off the dispatcher and event loop. Return a coverage loss reason."""
        path = Path(event.src_path or event.dest_path)
        if event.event_type == "moved":
            return f"top-level rename: {path}"
        if path in self.covered:
            return f"top-level directory {event.event_type}: {path}"
        if event.event_type == "created":
            try:
                info = path.lstat()
            except FileNotFoundError:
                info = None
            if info and info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                return f"top-level link: {path}"
            if event.is_directory or (info and stat.S_ISDIR(info.st_mode)):
                if not self.can_omit(path):
                    return f"top-level directory created: {path}"
                self.omitted.add(path)
            elif info is not None:
                self.omitted.discard(path)
        if path in self.omitted:
            if event.event_type == "deleted":
                self.omitted.discard(path)
            self.handler._record_rejected_event(event.event_type, path)
        else:
            self.handler.on_any_event(event)
        return None

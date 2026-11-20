"""A live but stalled parse worker must not prevent recovery of the scan."""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from concurrent.futures.process import BrokenProcessPool
from multiprocessing.connection import _ConnectionBase
from multiprocessing.synchronize import Semaphore
from pathlib import Path

import psutil
import pytest

import chunkhound.pipeline_bridge as bridge


def _block_worker(ready: _ConnectionBase, release: Semaphore) -> None:
    ready.send(os.getpid())
    ready.close()
    release.acquire()


def test_a_stalled_batch_reports_a_transient_pool_failure_and_replaces_its_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline" / "main.py"
    ctx = multiprocessing.get_context("spawn")
    reader, writer = ctx.Pipe(duplex=False)
    release = ctx.Semaphore(0)
    bridge._shut_down_parse_pool_at_exit()
    with (
        reader,
        writer,
        ProcessPoolExecutor(
            max_workers=1,
            mp_context=ctx,
            initializer=_block_worker,
            initargs=(writer, release),
        ) as stalled,
        ProcessPoolExecutor(max_workers=1, mp_context=ctx) as replacement,
        ThreadPoolExecutor(max_workers=1) as caller,
    ):
        pools = iter((stalled, replacement))
        monkeypatch.setattr(bridge, "_new_parse_pool_locked", lambda _: next(pools))
        notifications: list[BrokenProcessPool] = []
        try:
            stalled.submit(os.getpid)
            assert reader.poll(30), "stalled worker did not reach its initializer"
            pid = reader.recv()
            assert psutil.pid_exists(pid)
            with monkeypatch.context() as short_bound:
                short_bound.setattr(bridge, "_PARSE_POOL_STALL_SECONDS", 0.5)
                call = caller.submit(
                    bridge.parse_batch_callback,
                    [str(fixture)],
                    on_pool_broken=notifications.append,
                )
                try:
                    with pytest.raises(bridge._ParsePoolStalled) as failure:
                        call.result(timeout=15)
                except FutureTimeoutError:
                    pytest.fail(
                        "parse batch still blocked after 15 s with a live worker"
                    )
            assert notifications == [failure.value]
            assert not psutil.pid_exists(pid)
            ((language, chunks, error, _),) = bridge.parse_batch_callback(
                [str(fixture)]
            )
            assert (language, error) == ("python", None)
            assert chunks
            assert replacement.submit(os.getpid).result(timeout=15) != pid
        finally:
            release.release()
            bridge._shut_down_parse_pool_at_exit()

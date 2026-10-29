"""The Python reconnect after the Rust pipeline must survive the Rust side's DuckDB still holding the
file for a moment (Windows sharing violation), and must not mask any other error."""

import asyncio

import pytest

from chunkhound.services.rust_pipeline_runner import _reconnect_after_rust

SHARING = (
    'IO Error: Cannot open file "C:\\x\\chunks.db": The process cannot access the file '
    "because it is being used by another process."
)


class FakeDb:
    def __init__(self, errors: list[Exception]) -> None:
        self.errors = list(errors)
        self.calls = 0

    def connect(self) -> None:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    async def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


def test_retries_a_sharing_violation_until_the_file_is_released() -> None:
    db = FakeDb([OSError(SHARING), OSError(SHARING), OSError(SHARING)])
    clock = FakeClock()
    asyncio.run(_reconnect_after_rust(db, sleep=clock.sleep, clock=clock.now))
    assert db.calls == 4
    assert len(clock.slept) == 3


def test_connects_first_time_without_sleeping() -> None:
    db = FakeDb([])
    clock = FakeClock()
    asyncio.run(_reconnect_after_rust(db, sleep=clock.sleep, clock=clock.now))
    assert db.calls == 1
    assert clock.slept == []


def test_other_errors_are_raised_immediately() -> None:
    db = FakeDb([RuntimeError("Catalog Error: something else")])
    clock = FakeClock()
    with pytest.raises(RuntimeError, match="something else"):
        asyncio.run(_reconnect_after_rust(db, sleep=clock.sleep, clock=clock.now))
    assert db.calls == 1


def test_gives_up_with_the_original_error_after_the_timeout() -> None:
    db = FakeDb([OSError(SHARING)] * 1000)
    clock = FakeClock()
    with pytest.raises(OSError, match="being used by another process"):
        asyncio.run(_reconnect_after_rust(db, sleep=clock.sleep, clock=clock.now, timeout_s=1.0))
    assert 10 <= db.calls <= 12  # 1.0 s at 0.1 s polls

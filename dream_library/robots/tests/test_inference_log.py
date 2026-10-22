"""Behaviour tests for the inference log the loop writes (`robots.inference_log`, `inference_log_path`).

Each test runs a loop over the fakes into a JSON Lines file under `tmp_path` and reads the
records back: one per get_action outcome, one per caught error, one per stall burst, and the
run boundaries. `robots.control` drives the writes; the module's own helpers are pinned last.
"""

import asyncio
import json
import pathlib
import re
import threading
from collections.abc import Callable, Iterator
from typing import Any

import conftest
import dream
import pytest
import test_control

from robots import errors, inference_log

## ISO 8601 UTC with milliseconds, what every record's `t` must match.
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}\+00:00$")

## The keys every `inference` record carries, in order.
INFERENCE_KEYS = [
    "kind",
    "t",
    "run",
    "prompt",
    "session_id",
    "chunk",
    "rtt_ms",
    "serve_infer_ms",
    "gateway_ms",
    "network_ms",
    "up_mbps",
    "tcp_cwnd_kb",
    "tcp_retrans",
    "tcp_rtt_ms",
    "tcp_app_limited",
    "request_bytes",
    "response_bytes",
    "rows",
    "mode",
    "executed_since_obs",
    "due",
    "skipped",
    "discarded",
    "motor_temps_c",
]

## The keys every `error` record carries, in order.
ERROR_KEYS = ["kind", "t", "run", "error_type", "code", "reason", "detail", "retry_after_s", "action"]


@pytest.fixture
def build() -> Iterator[Callable[..., conftest.LoopHarness]]:
    """A `make_loop` factory on a YieldingClock, as `test_control.build`; capture threads stopped on teardown."""
    built: list[conftest.LoopHarness] = []

    def factory(**kwargs: Any) -> conftest.LoopHarness:
        kwargs.setdefault("clock", test_control.YieldingClock())
        harness = conftest.make_loop(**kwargs)
        built.append(harness)
        return harness

    yield factory
    for harness in built:
        for reader in harness.readers:
            if reader.is_alive():
                reader.stop()


def read_records(path: pathlib.Path) -> list[dict[str, Any]]:
    """Every line of the file as a dict, in order."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def kinds(records: list[dict[str, Any]]) -> list[str]:
    """The `kind` of every record, in order."""
    return [record["kind"] for record in records]


def test_a_run_writes_run_start_one_inference_per_chunk_and_run_stop(
    build: Callable[..., conftest.LoopHarness], tmp_path: pathlib.Path
) -> None:
    """Two chunks over four ticks: run_start, two inference records with the alignment numbers the chunk line
    logs (executed / due / skipped), run_stop with the counts; every record stamped with an ISO time and the run."""
    path = tmp_path / "inference.jsonl"
    harness = build(action_horizon=4, refill_at_remaining=2, inference_log_path=str(path), run_number=7)

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            await harness.loop.step(0.1)
            # Two rows left before this tick: the prefetch starts here and is let land before the next.
            await harness.loop.step(0.2)
            await test_control.wait_for_refill_task()
            await harness.loop.step(0.3)

    asyncio.run(scenario())

    records = read_records(path)
    assert kinds(records) == ["run_start", "inference", "inference", "run_stop"]
    for record in records:
        assert ISO_RE.match(record["t"]), record["t"]
        assert record["run"] == 7
    start = records[0]
    assert start["prompt"] == conftest.TASK
    assert start["control_hz"] == 30.0 and start["action_horizon"] == 4 and start["refill"] == 2
    first, second = records[1], records[2]
    assert list(first) == INFERENCE_KEYS
    # Chunk 1 answered the tick it was requested on (the sync wait): no rows executed, none due, none skipped.
    assert first["chunk"] == 1 and first["rows"] == 4 and first["rtt_ms"] == 0.0
    assert (first["executed_since_obs"], first["due"], first["skipped"], first["discarded"]) == (0, 0, 0, None)
    assert first["prompt"] == conftest.TASK and first["session_id"] is None
    assert all(first[key] is None for key in inference_log.TIMING_KEYS)
    # Chunk 2 was prefetched at 0.2 s and collected at 0.3 s: one row executed, three due by wall time.
    assert second["chunk"] == 2 and second["rtt_ms"] == 100.0
    assert (second["executed_since_obs"], second["due"], second["skipped"]) == (1, 3, 1)
    stop = records[-1]
    assert stop["reason"] == "stopped" and stop["chunks"] == 2 and stop["stalls"] == 0 and stop["clamps"] == 0


def test_a_retryable_refusal_writes_an_error_record_with_the_envelope_and_a_stall_record(
    build: Callable[..., conftest.LoopHarness], tmp_path: pathlib.Path
) -> None:
    """A cold-pool refusal is an `error` record carrying DA-2000, the reason, the hint and action `retry`; the
    empty tick that follows it, before the first chunk, is one `stall` record for the burst."""
    path = tmp_path / "inference.jsonl"
    client = test_control.WarmingClient(horizon=4, refused={0}, retry_after_s=2.0)
    harness = build(action_horizon=4, client=client, inference_log_path=str(path))

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            await harness.loop.step(1.0)
            await harness.loop.step(3.0)

    asyncio.run(scenario())

    records = read_records(path)
    assert kinds(records) == ["run_start", "error", "stall", "inference", "run_stop"]
    error = records[1]
    assert list(error) == ERROR_KEYS
    assert error["error_type"] == "DreamServerError" and error["code"] == "DA-2000"
    assert error["reason"] == "pool_warming" and error["retry_after_s"] == 2.0 and error["action"] == "retry"
    assert "NO_CAPACITY" in error["detail"]
    assert records[2]["chunk"] == 0 and records[2]["max_stall_ticks"] == conftest.LOOP_DEFAULTS["max_stall_ticks"]
    assert records[3]["chunk"] == 1


def test_a_session_expiry_is_an_error_record_with_action_reset(
    build: Callable[..., conftest.LoopHarness], tmp_path: pathlib.Path
) -> None:
    """DreamSessionExpired drops the buffer and refills; the record says `reset` and carries DA-0001."""
    path = tmp_path / "inference.jsonl"
    harness = build(action_horizon=4, refill_at_remaining=2, inference_log_path=str(path))

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)
            await harness.loop.step(0.1)
            harness.client.raise_once = test_control.SESSION_EXPIRED
            await harness.loop.step(0.2)
            await test_control.wait_for_refill_task()
            await harness.loop.step(0.3)

    asyncio.run(scenario())

    records = read_records(path)
    assert kinds(records) == ["run_start", "inference", "error", "inference", "run_stop"]
    assert records[2]["error_type"] == "DreamSessionExpired"
    assert records[2]["code"] == "DA-0001" and records[2]["action"] == "reset"


def test_an_unrecoverable_error_is_recorded_with_action_stop_and_names_the_run_stop_reason(
    build: Callable[..., conftest.LoopHarness], tmp_path: pathlib.Path
) -> None:
    """DreamAuthError stops the loop: one `error` record with action `stop` from __aexit__, then `run_stop`
    whose reason is the exception class."""
    path = tmp_path / "inference.jsonl"
    client = conftest.FakeClient(horizon=4, raise_once=test_control.UNRECOVERABLE_ERRORS[0])
    harness = build(action_horizon=4, client=client, inference_log_path=str(path))

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)

    with pytest.raises(dream.DreamAuthError):
        asyncio.run(scenario())

    records = read_records(path)
    assert kinds(records) == ["run_start", "error", "run_stop"]
    assert records[1]["error_type"] == "DreamAuthError" and records[1]["code"] == "DA-1000"
    assert records[1]["action"] == "stop"
    assert records[2]["reason"] == "DreamAuthError" and records[2]["chunks"] == 0


def test_a_stall_burst_after_the_first_chunk_is_one_record_and_run_stop_counts_every_stall_tick(
    build: Callable[..., conftest.LoopHarness], tmp_path: pathlib.Path
) -> None:
    """Three stall ticks on one silent refill are one `stall` record (like the WARNING, once per burst); the
    `run_stop` `stalls` field and `ControlLoop.stall_count` count all three."""
    path = tmp_path / "inference.jsonl"
    gate = threading.Event()
    client = test_control.AnswerOnceClient(horizon=1, gate=gate)
    harness = build(
        action_horizon=1,
        action_timeout_s=test_control.STALL_TIMEOUT_S,
        max_stall_ticks=3,
        client=client,
        inference_log_path=str(path),
    )

    async def scenario() -> None:
        try:
            async with harness.loop:
                for now_s in (0.0, 1.0, 2.0, 3.0):
                    await harness.loop.step(now_s)
                assert harness.loop.stall_count == 3
                assert harness.loop.chunk_count == 1
        finally:
            gate.set()
            await test_control.wait_for_refill_task()

    asyncio.run(scenario())

    records = read_records(path)
    assert kinds(records) == ["run_start", "inference", "stall", "run_stop"]
    assert records[2]["chunk"] == 1
    assert records[3]["stalls"] == 3


def test_records_are_appended_across_runs_and_flushed_per_record(
    build: Callable[..., conftest.LoopHarness], tmp_path: pathlib.Path
) -> None:
    """A second run into the same file follows the first (append mode), and a record is on disk as soon as it
    is written, not at exit: the file is read while the first loop is still open."""
    path = tmp_path / "inference.jsonl"
    first = build(action_horizon=4, inference_log_path=str(path), run_number=1)
    second = build(action_horizon=4, inference_log_path=str(path), run_number=2)
    seen_while_open: list[str] = []

    async def scenario() -> None:
        async with first.loop:
            await first.loop.step(0.0)
            seen_while_open.extend(kinds(read_records(path)))
        async with second.loop:
            await second.loop.step(0.0)

    asyncio.run(scenario())

    assert seen_while_open == ["run_start", "inference"]
    records = read_records(path)
    assert [(r["run"], r["kind"]) for r in records] == [
        (1, "run_start"),
        (1, "inference"),
        (1, "run_stop"),
        (2, "run_start"),
        (2, "inference"),
        (2, "run_stop"),
    ]


def test_the_session_id_and_last_timing_are_read_from_the_client_when_it_has_them(
    build: Callable[..., conftest.LoopHarness], tmp_path: pathlib.Path
) -> None:
    """The dream client's `session_id` and `last_timing` both reach the record, and `ControlLoop.timing` carries
    the split too."""
    path = tmp_path / "inference.jsonl"
    harness = build(action_horizon=4, inference_log_path=str(path))
    harness.client.session_id = "s-42"
    harness.client.last_timing = {
        "rtt_ms": 1234.5,
        "serve_infer_ms": 900.0,
        "gateway_ms": 50.0,
        "network_ms": 284.5,
        "request_bytes": 120000,
        "response_bytes": 5000,
        "up_mbps": 2.1,
    }

    async def scenario() -> None:
        async with harness.loop:
            await harness.loop.step(0.0)

    asyncio.run(scenario())

    record = read_records(path)[1]
    assert record["session_id"] == "s-42"
    assert (record["serve_infer_ms"], record["gateway_ms"], record["network_ms"]) == (900.0, 50.0, 284.5)
    assert (record["request_bytes"], record["response_bytes"]) == (120000, 5000)
    assert harness.loop.timing.serve_infer_ms == 900.0 and harness.loop.timing.response_bytes == 5000
    assert record["up_mbps"] == 2.1 and harness.loop.timing.up_mbps == 2.1, "the upload rate reaches the live view"
    assert harness.loop.session_id == "s-42"
    assert read_records(path)[-1]["session_id"] == "s-42"


def test_without_a_path_nothing_is_written_and_the_log_is_a_no_op(tmp_path: pathlib.Path) -> None:
    """`InferenceLog(None, ...)` opens nothing and write() is free; a loop without `inference_log_path` leaves
    `tmp_path` empty."""
    log = inference_log.InferenceLog(None, run_number=1)
    log.open()
    assert not log.is_open
    log.write("inference", chunk=1)
    log.close()
    assert list(tmp_path.iterdir()) == []


def test_a_write_failure_disables_the_log_and_never_raises(tmp_path: pathlib.Path) -> None:
    """The log is diagnostics: a failing write is logged once and the log closes itself, the loop goes on."""
    path = tmp_path / "inference.jsonl"
    log = inference_log.InferenceLog(str(path), run_number=3)
    log.open()
    log.write("run_start", prompt="x")
    log._file.close()
    log._file = _FailingFile()
    log.write("inference", chunk=1)
    assert not log.is_open
    log.write("inference", chunk=2)
    assert [r["kind"] for r in read_records(path)] == ["run_start"]


class _FailingFile:
    """A file whose write raises OSError, standing in for a full disk."""

    def write(self, data: str) -> int:
        raise OSError("disk full")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_error_fields_for_a_plain_exception_have_no_envelope() -> None:
    """A ControlLoopError from the adapter's shape check has no code, reason or hint; the type and message stay."""
    fields = inference_log.error_fields(errors.ControlLoopError("chunk has 3 rows, expected 4"), "retry")
    assert fields == {
        "error_type": "ControlLoopError",
        "code": None,
        "reason": None,
        "detail": "chunk has 3 rows, expected 4",
        "retry_after_s": None,
        "action": "retry",
    }


def test_client_timing_tolerates_a_client_without_it() -> None:
    """A client with no `last_timing`, or one that is not a dict (none yet), yields nulls."""
    assert inference_log.client_timing(object()) == dict.fromkeys(inference_log.TIMING_KEYS)

    class Client:
        last_timing = "not a dict"

    assert inference_log.client_timing(Client()) == dict.fromkeys(inference_log.TIMING_KEYS)

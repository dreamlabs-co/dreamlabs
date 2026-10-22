"""Pacing tests: the execution horizon, the row rate and the sync/async switch the client decides for its loop.

The contract tests run the client against `FakeDreamAPI` with its `latency_s` set, so the mode is chosen by a real
round trip and the cut is seen on the dict `get_action` hands back; the unit tests drive `dream.pacing.Pacer`
directly with chosen round trips.
"""

from __future__ import annotations

from typing import Any

import pytest
from conftest import FakeDreamAPI

import dream
from dream import pacing
from dream.errors import DreamConfigError

## Rows per chunk the fake serves here: the Dream-1 base model's 48.
ROWS = 48


@pytest.fixture
def long_config(config: dict[str, Any], server: FakeDreamAPI) -> dict[str, Any]:
    """The shared config and fake with `[ROWS, 7]` chunks."""
    server.horizon = ROWS
    for key in ("q_r_chunk", "q_l_chunk"):
        config["embodiment"]["action"][key]["shape"] = [ROWS, 7]
    return config


def test_defaults_are_a_30_row_execution_horizon_at_20_hz_and_sync_above_700_ms(config: dict[str, Any]) -> None:
    """The station's numbers are the library's defaults: nothing in the config, and the client paces at them."""
    with dream._open_policy(config) as client:
        assert client.execution_horizon == 30
        assert client.action_hz == 20.0
        assert client.mode == "async", "async before any round trip"
    assert (pacing.EXECUTION_HORIZON, pacing.ACTION_HZ, pacing.SYNC_ABOVE_MS) == (30, 20.0, 700.0)


def test_pacing_keys_are_consumed_locally_and_never_sent(
    long_config: dict[str, Any], server: FakeDreamAPI, observation: dict[str, Any]
) -> None:
    """The three pacing keys configure the client and do not reach DreamAPI's request body."""
    long_config.update({"execution_horizon": 24, "action_hz": 30.0, "sync_above_ms": 500})
    with dream._open_policy(long_config) as client:
        assert (client.execution_horizon, client.action_hz) == (24, 30.0)
        client.get_action(observation)
    sent = server.requests[0].body
    assert not {"execution_horizon", "action_hz", "sync_above_ms"} & set(sent)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("execution_horizon", 0),
        ("execution_horizon", 2.5),
        ("action_hz", 0),
        ("action_hz", True),
        ("sync_above_ms", -1),
    ],
)
def test_a_bad_pacing_key_is_a_config_error_naming_it(config: dict[str, Any], key: str, value: Any) -> None:
    """A pacing value of the wrong type or range fails at init, naming the key."""
    config[key] = value
    with pytest.raises(DreamConfigError, match=key):
        dream._open_policy(config)


def test_a_fast_reply_is_async_and_carries_every_row(long_config: dict[str, Any], observation: dict[str, Any]) -> None:
    """Async: the round trip is within the threshold, so the whole chunk comes back; the rows past the execution
    horizon are the runway the loop plays only when the next chunk is late."""
    with dream._open_policy(long_config) as client:
        action = client.get_action(observation)
        assert client.mode == "async"
    assert len(action["q_r_chunk"]) == ROWS and len(action["q_l_chunk"]) == ROWS


def test_a_slow_reply_switches_to_sync_and_keeps_every_row(
    long_config: dict[str, Any], server: FakeDreamAPI, observation: dict[str, Any]
) -> None:
    """Sync: a median round trip past `sync_above_ms` switches the mode (the first reply is the whole median), and
    that reply still carries the whole action horizon (48 rows), not `execution_horizon`; the loop runs all of it
    before asking again. Once most recent replies are fast again the mode is back to async."""
    long_config["sync_above_ms"] = 50
    with dream._open_policy(long_config) as client:
        server.latency_s = 0.1
        slow = client.get_action(observation)
        assert client.mode == "sync"
        assert len(slow["q_r_chunk"]) == ROWS and len(slow["q_l_chunk"]) == ROWS
        server.latency_s = 0.0
        client.get_action(observation)
        fast = client.get_action(observation)
        assert client.mode == "async"
        assert len(fast["q_r_chunk"]) == ROWS


def test_sync_runs_the_whole_chunk_before_asking_again(
    long_config: dict[str, Any], server: FakeDreamAPI, observation: dict[str, Any]
) -> None:
    """Sync is not due at the execution horizon or anywhere short of the chunk's end: only once no row is left."""
    long_config["sync_above_ms"] = 50
    with dream._open_policy(long_config) as client:
        server.latency_s = 0.1
        client.get_action(observation)
        assert client.mode == "sync"
        assert not client.refill_due(executed=30, remaining=ROWS - 30), "the execution horizon is async's"
        assert not client.refill_due(executed=ROWS - 1, remaining=1), "sync waits for the last row"
        assert client.refill_due(executed=ROWS, remaining=0)


def test_a_sync_reply_is_the_same_rows_an_async_reply_would_be(
    long_config: dict[str, Any], server: FakeDreamAPI, observation: dict[str, Any]
) -> None:
    """The mode changes when the loop asks, never what comes back: a slow reply's rows equal a fast reply's."""
    long_config["sync_above_ms"] = 50
    with dream._open_policy(long_config) as client:
        fast = client.get_action(observation)
    with dream._open_policy(long_config) as client:
        server.latency_s = 0.1
        slow = client.get_action(observation)
        assert client.mode == "sync"
    assert slow["q_r_chunk"] == fast["q_r_chunk"] and slow["q_l_chunk"] == fast["q_l_chunk"]


# dream.pacing.Pacer


def test_async_requests_so_the_reply_lands_as_the_execution_horizon_ends() -> None:
    """Async at 20 Hz with a 500 ms round trip: 10 rows of lead, so the request goes out after 20 executed rows."""
    pacer = pacing.Pacer()
    pacer.record(0.5)
    assert pacer.mode == pacing.ASYNC
    assert not pacer.refill_due(executed=19, remaining=29)
    assert pacer.refill_due(executed=20, remaining=28)


def test_async_lead_is_the_shortest_recent_round_trip() -> None:
    """The lead is sized from the shortest of the last RTT_WINDOW round trips, the one no queue at the serve added
    to: a reply that waited behind another robot's inference does not send the next requests early."""
    pacer = pacing.Pacer()
    for rtt_s in (0.6, 0.6, 0.4, 0.6):
        pacer.record(rtt_s)
    # 0.4 s at 20 Hz is 8 rows of lead: due after 22 executed, not after 18 as the slow replies alone would say.
    assert not pacer.refill_due(executed=21, remaining=27)
    assert pacer.refill_due(executed=22, remaining=26)
    for _ in range(pacing.RTT_WINDOW):
        pacer.record(0.6)
    assert pacer.refill_due(executed=18, remaining=30), "the fast reply has left the window: 12 rows of lead"


def test_a_reply_that_waited_moves_the_next_request_later() -> None:
    """On a shared serve: a reply 400 ms late (it queued behind the other robot) is installed later, and the next
    request still goes out the same number of executed rows after the install as before, so this robot's whole
    cycle moves later by the wait, away from the other robot's next inference, instead of earlier onto it."""
    pacer = pacing.Pacer()
    pacer.record(0.5)
    pacer.record(0.5)
    due_before = next(executed for executed in range(30) if pacer.refill_due(executed, 48 - executed))
    pacer.record(0.9)
    assert pacer.mode == pacing.ASYNC
    due_after = next(executed for executed in range(30) if pacer.refill_due(executed, 48 - executed))
    assert due_after == due_before == 20


def test_async_requests_early_when_the_rows_left_cannot_cover_the_round_trip() -> None:
    """A short chunk (a sync cut on the way back to async) is refilled once its rows left are worth less than the
    round trip plus the margin, so the buffer never runs dry before the reply."""
    pacer = pacing.Pacer()
    pacer.record(0.5)
    # 0.5 + 0.25 s at 20 Hz is 15 rows.
    assert not pacer.refill_due(executed=0, remaining=16)
    assert pacer.refill_due(executed=0, remaining=15)


def test_the_switch_is_the_threshold_on_the_median_round_trip() -> None:
    """A median over 700 ms is sync, 700 ms or less is async, decided again on every reply."""
    pacer = pacing.Pacer()
    pacer.record(0.901)
    assert pacer.mode == pacing.SYNC
    pacer.record(0.6)
    pacer.record(0.7)
    assert pacer.mode == pacing.ASYNC, "median of 0.6, 0.7, 0.701 is 0.7"


def test_one_slow_reply_does_not_switch_to_sync_but_most_do() -> None:
    """A single reply that waited behind another robot does not hold the arms through the next inference; the third
    slow reply of five does, as a link that really got slower should."""
    pacer = pacing.Pacer()
    for rtt_s in (0.5, 0.5, 0.5, 1.2):
        pacer.record(rtt_s)
    assert pacer.mode == pacing.ASYNC
    pacer.record(1.2)
    assert pacer.mode == pacing.ASYNC, "two slow of five"
    pacer.record(1.2)
    assert pacer.mode == pacing.SYNC, "three slow of the last five"


def test_any_empty_buffer_is_due_in_either_mode() -> None:
    """Nothing left is always due, in either mode."""
    pacer = pacing.Pacer()
    assert pacer.refill_due(executed=0, remaining=0)
    pacer.record(1.0)
    assert pacer.refill_due(executed=0, remaining=0)


def test_at_the_threshold_an_async_reply_still_fits_a_48_row_chunk() -> None:
    """The arithmetic behind 700 ms: the async reply is installed past the rows run during the round trip, and the
    execution horizon must fit after them: 14 + 30 = 44 rows of 48."""
    lead_rows = round(pacing.SYNC_ABOVE_MS / 1e3 * pacing.ACTION_HZ)
    assert lead_rows + pacing.EXECUTION_HORIZON <= ROWS

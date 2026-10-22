"""Tests for the slow-network warning (`dream.slow_network`).

The watch is pure (a clock is passed in), so its rules are pinned on synthetic `last_timing` dicts: one warning when
the median of the last five round trips reaches the slow line, quiet for `QUIET_S` after it while they stay slow, at
once again after a recovery, and words that blame the network or the model. Then the client, against the in-process
fake gateway with the slow line lowered so its quick round trips count: one warning on the caller's own line plus a
`dream` log record, never the key, never an exception, and silenced by the standard filter.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from typing import Any

import pytest

import dream
from dream import slow_network
from dream.client import PolicyClient

## A key the client sends with every request; it must appear in no warning and no log line.
KEY = "dl_live_test"


def timing(rtt_ms: float, network_ms: float | None = 1100.0, gateway_ms: float | None = 350.0, **over: Any) -> dict:
    """A `last_timing` dict whose round trip went mostly to the network, unless told otherwise."""
    serve_infer_ms = over.pop("serve_infer_ms", 300.0 if gateway_ms is not None else None)
    return {
        "rtt_ms": rtt_ms,
        "network_ms": network_ms,
        "gateway_ms": gateway_ms,
        "serve_infer_ms": serve_infer_ms,
        "up_mbps": over.pop("up_mbps", 2.1),
        **over,
    }


def feed(watch: slow_network.Watch, rtt_ms: float, count: int, start_s: float, **parts: Any) -> list[str]:
    """Records `count` round trips of `rtt_ms`, one a second from `start_s`; returns the warnings worded."""
    worded = [watch.record(timing(rtt_ms, **parts), start_s + i) for i in range(count)]
    return [text for text in worded if text is not None]


@pytest.fixture
def slow_client(config: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> Iterator[PolicyClient]:
    """A client against the fake gateway, with a key, whose every round trip counts as slow (the lines lowered)."""
    monkeypatch.setattr(slow_network, "SLOW_MS", 0.0)
    monkeypatch.setattr(slow_network, "RECOVERED_MS", 0.0)
    with dream._open_policy(config, api_key=KEY) as client:
        yield client


def test_one_warning_once_the_median_of_five_is_slow() -> None:
    """Four slow round trips are too few to judge; the fifth warns; the sixth, still slow, does not warn again."""
    watch = slow_network.Watch()
    assert feed(watch, 1450, 4, 0.0) == []
    assert len(feed(watch, 1450, 1, 4.0)) == 1
    assert feed(watch, 1450, 20, 5.0) == [], "slow on: quiet"


def test_one_slow_round_trip_among_quick_ones_never_warns() -> None:
    """A single 9-second reply moves the median of five not at all."""
    watch = slow_network.Watch()
    assert feed(watch, 300, 10, 0.0) == []
    assert feed(watch, 9000, 1, 10.0) == []
    assert feed(watch, 300, 10, 11.0) == []
    assert feed(watch, 9000, 2, 21.0) == [], "two slow of five is still a quick median"


def test_quiet_for_the_cooldown_then_a_reminder_while_it_stays_slow() -> None:
    """Slow without a break: one warning, then nothing for `QUIET_S`, then one more."""
    watch = slow_network.Watch()
    first = feed(watch, 1450, 5, 0.0)
    quiet = feed(watch, 1450, int(slow_network.QUIET_S) - 1, 5.0)
    again = [watch.record(timing(1450), 4.0 + slow_network.QUIET_S)]
    assert (len(first), quiet, again[0] is not None) == (1, [], True)


def test_a_recovery_rearms_it_and_the_next_slowdown_warns_at_once() -> None:
    """Under `RECOVERED_MS` the watch re-arms; a median between that and `SLOW_MS` does not re-arm it."""
    watch = slow_network.Watch()
    assert len(feed(watch, 1450, 5, 0.0)) == 1
    assert feed(watch, 900, 5, 5.0) == [], "900 ms is not slow"
    assert feed(watch, 1450, 5, 10.0) == [], "nor a recovery: still inside the quiet time"
    assert slow_network.RECOVERED_MS < 900 < slow_network.SLOW_MS
    assert feed(watch, 400, 5, 15.0) == []
    assert watch.is_armed
    assert len(feed(watch, 1450, 5, 20.0)) == 1, "recovered, then slow again: warns within the quiet time"


def test_the_words_blame_the_network_with_its_upload_rate_or_the_model() -> None:
    """The network when it is the largest part, with the upload rate when known; the model otherwise; no cause
    without a network figure."""
    network = slow_network.message([timing(1450)] * 5)
    assert network == (
        "Network is slow: get_action round trips are about 1450 ms (network 1100 ms, upload 2.1 Mb/s); "
        "the robot's motion may be less smooth. See client.last_timing."
    )
    assert "upload" not in slow_network.message([timing(1450, up_mbps=None)] * 5)
    model = slow_network.message([timing(1450, network_ms=150, gateway_ms=1300, serve_infer_ms=1200)] * 5)
    assert model == (
        "The model is responding slowly: get_action round trips are about 1450 ms "
        "(the model and our servers 1300 ms, network 150 ms); the robot's motion may be less smooth. "
        "See client.last_timing."
    )
    servers = slow_network.message([timing(1450, network_ms=400, gateway_ms=1050, serve_infer_ms=300)] * 5)
    assert servers.startswith("The model is responding slowly"), "our servers' part is larger than the network's"
    unknown = slow_network.message([timing(1450, network_ms=None, gateway_ms=None)] * 5)
    assert unknown.startswith("get_action round trips are slow: about 1450 ms;")
    for text in (network, model, unknown):
        assert "DreamAPI" not in text and "gateway" not in text and "serve " not in text


def test_the_client_warns_once_on_the_callers_line_and_logs_it(
    slow_client: PolicyClient, observation: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """Ten slow calls: one `DreamSlowNetworkWarning` attributed to this file, the same line on the `dream` logger,
    every action returned, and the key in neither."""
    caplog.set_level(logging.WARNING, logger="dream")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actions = [slow_client.get_action(observation) for _ in range(10)]
    slow = [w for w in caught if issubclass(w.category, dream.DreamSlowNetworkWarning)]
    assert len(slow) == 1 and all("q_r_chunk" in action for action in actions)
    assert slow[0].filename == __file__, "the warning names the caller's line, not the library's"
    text = str(slow[0].message)
    assert "get_action round trips" in text and KEY not in text
    logged = [
        record for record in caplog.records if record.name.startswith("dream") and "round trips" in record.message
    ]
    assert [record.getMessage() for record in logged] == [text]
    assert KEY not in caplog.text


def test_the_warning_never_raises_even_as_an_error(slow_client: PolicyClient, observation: dict[str, Any]) -> None:
    """`-W error` for the category must not cost the caller an action: get_action returns, and nothing is raised."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", dream.DreamSlowNetworkWarning)
        actions = [slow_client.get_action(observation) for _ in range(6)]
    assert all("q_r_chunk" in action for action in actions)


def test_the_standard_filter_silences_it(slow_client: PolicyClient, observation: dict[str, Any]) -> None:
    """`warnings.filterwarnings("ignore", category=dream.DreamSlowNetworkWarning)` is the documented off switch."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=dream.DreamSlowNetworkWarning)
        for _ in range(6):
            slow_client.get_action(observation)
    assert not [w for w in caught if issubclass(w.category, dream.DreamSlowNetworkWarning)]


def test_quick_calls_never_warn(config: dict[str, Any], observation: dict[str, Any]) -> None:
    """At the shipped lines the fake gateway's quick round trips warn about nothing."""
    with warnings.catch_warnings(record=True) as caught, dream._open_policy(config) as client:
        warnings.simplefilter("always")
        for _ in range(8):
            client.get_action(observation)
    assert not [w for w in caught if issubclass(w.category, dream.DreamSlowNetworkWarning)]


def test_it_is_a_public_user_warning() -> None:
    """A `UserWarning`, in `dream.__all__`, and not a `DreamError`: nothing that catches errors catches it."""
    assert issubclass(dream.DreamSlowNetworkWarning, UserWarning)
    assert "DreamSlowNetworkWarning" in dream.__all__
    assert not issubclass(dream.DreamSlowNetworkWarning, dream.DreamError)

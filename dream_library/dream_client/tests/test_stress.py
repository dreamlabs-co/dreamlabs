"""Stress tests S-1..S-7.

Short variants run by default and finish in seconds; the full variants (`DREAM_STRESS=full`) are
the long lengths, capped by the `DREAM_*_S` knobs so a run fits a CI budget.
"""

from __future__ import annotations

import base64
import os
import resource
import threading
import time
from typing import Any

import numpy as np
import pytest
from conftest import CHECKPOINT, FakeDreamAPI, make_observation

import dream
from dream.errors import DreamSessionExpired, DreamTimeout

## Full-length variants run only when asked for (nightly).
full_only = pytest.mark.skipif(os.environ.get("DREAM_STRESS") != "full", reason="nightly: DREAM_STRESS=full")

## Relative RSS growth tolerated over a long run (S-1: flat within 5 %).
RSS_TOLERANCE = 0.05


def _fds() -> int:
    return len(os.listdir("/proc/self/fd"))


def _rss_kb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _sequential(server: FakeDreamAPI, config: dict[str, Any], calls: int) -> None:
    """S-1 body: `calls` round trips on one client; FDs flat, RSS flat within tolerance after warm-up."""
    observation = make_observation()
    warmup = min(200, calls // 10)
    server.log_requests = False
    with dream._open_policy(config) as client:
        for _ in range(warmup):
            client.get_action(observation)
        fds, rss = _fds(), _rss_kb()
        latencies = []
        for _ in range(calls - warmup):
            started = time.perf_counter()
            client.get_action(observation)
            latencies.append(time.perf_counter() - started)
    assert _fds() <= fds
    assert _rss_kb() <= rss * (1 + RSS_TOLERANCE)
    latencies.sort()
    assert latencies[int(len(latencies) * 0.99)] < 0.1
    assert server.request_count == calls


def test_s01_sequential_calls_short(server: FakeDreamAPI, config: dict[str, Any]) -> None:
    """S-1 (short): 2 000 sequential calls, no FD growth, RSS flat, p99 stable."""
    _sequential(server, config, 2_000)


@full_only
def test_s01_sequential_calls_full(server: FakeDreamAPI, config: dict[str, Any]) -> None:
    """S-1 (full): 100 000 sequential calls."""
    _sequential(server, config, int(os.environ.get("DREAM_S1_CALLS", "100000")))


def _init_close_loop(config: dict[str, Any], cycles: int) -> None:
    fds = _fds()
    for _ in range(cycles):
        client = dream._open_policy(config)
        client.close()
    assert _fds() <= fds


def test_s03_init_close_loop_short(config: dict[str, Any]) -> None:
    """S-3 (short): 300 init/close cycles leave the FD count where it was."""
    _init_close_loop(config, 300)


@full_only
def test_s03_init_close_loop_full(config: dict[str, Any]) -> None:
    """S-3 (full): 10 000 init/close cycles."""
    _init_close_loop(config, 10_000)


def test_s04_large_observations_scale_linearly(server: FakeDreamAPI, config: dict[str, Any]) -> None:
    """S-4: a 4-camera 1080p JPEG window observation encodes in time linear in its size."""
    del config["embodiment"]
    frame_1080p = base64.b64encode(np.random.default_rng(0).bytes(300_000)).decode("ascii")
    frame_small = base64.b64encode(np.random.default_rng(1).bytes(30_000)).decode("ascii")

    def observation(frame: str, cameras: int) -> dict[str, Any]:
        return {**make_observation(frame), **{f"image_window.cam{i}": [frame] for i in range(cameras)}}

    with dream._open_policy(config) as client:
        for _ in range(5):
            client.get_action(observation(frame_small, 1))
        started = time.perf_counter()
        for _ in range(20):
            client.get_action(observation(frame_small, 1))
        small_s = (time.perf_counter() - started) / 20
        started = time.perf_counter()
        for _ in range(20):
            client.get_action(observation(frame_1080p, 4))
        big_s = (time.perf_counter() - started) / 20
    ratio_bytes = (4 * 300_000) / 30_000
    assert big_s / small_s < ratio_bytes * 2


def _timeout_storm(server: FakeDreamAPI, config: dict[str, Any], duration_s: float) -> None:
    """S-5 body: the gateway holds every other reply; every failure is DreamTimeout, never a hang."""
    observation = make_observation()
    fds = _fds()
    deadline = time.monotonic() + duration_s
    timeouts = successes = 0
    with dream._open_policy({**config, "timeout_ms": 150}) as client:
        while time.monotonic() < deadline:
            server.hold_next = 1 if (timeouts + successes) % 2 == 0 else 0
            started = time.monotonic()
            try:
                client.get_action(observation)
                successes += 1
            except DreamTimeout:
                timeouts += 1
            assert time.monotonic() - started < 2.0, "a call outlived its timeout by far: wedged"
    assert timeouts > 0 and successes > 0
    assert _fds() <= fds + 2


def test_s05_timeout_storm_short(server: FakeDreamAPI, config: dict[str, Any]) -> None:
    """S-5 (short): 3 s of a gateway dropping half its replies."""
    _timeout_storm(server, config, 3.0)


@full_only
def test_s05_timeout_storm_full(server: FakeDreamAPI, config: dict[str, Any]) -> None:
    """S-5 (full): the storm for `DREAM_STORM_S` seconds (default 300)."""
    _timeout_storm(server, config, float(os.environ.get("DREAM_STORM_S", "300")))


def _flap(server: FakeDreamAPI, config: dict[str, Any], duration_s: float, period_s: float) -> None:
    """S-6 body: the gateway forgets its sessions every `period_s`; exactly one report per flap, then steady."""
    observation = make_observation()
    reports = flaps = 0
    with dream._open_policy(config) as client:
        client.get_action(observation)
        deadline = time.monotonic() + duration_s
        next_flap = time.monotonic() + period_s
        while time.monotonic() < deadline:
            if time.monotonic() >= next_flap:
                server.restart()
                flaps += 1
                next_flap += period_s
            try:
                client.get_action(observation)
            except DreamSessionExpired:
                reports += 1
            time.sleep(0.005)
    assert reports == flaps
    assert flaps >= 3


def test_s06_gateway_flap_short(server: FakeDreamAPI, config: dict[str, Any]) -> None:
    """S-6 (short): a restart every 0.5 s for 3 s; one DreamSessionExpired per restart and no other error."""
    _flap(server, config, 3.0, 0.5)


@full_only
def test_s06_gateway_flap_full(server: FakeDreamAPI, config: dict[str, Any]) -> None:
    """S-6 (full): a restart every 10 s for `DREAM_FLAP_S` seconds (default 1800)."""
    _flap(server, config, float(os.environ.get("DREAM_FLAP_S", "1800")), 10.0)


def test_s07_slow_loris_no_spurious_timeouts(server: FakeDreamAPI, config: dict[str, Any]) -> None:
    """S-7: replies arriving just under `timeout_ms` never time out."""
    config["timeout_ms"] = 400
    server.latency_s = 0.3
    observation = make_observation()
    with dream._open_policy(config) as client:
        for _ in range(10):
            assert client.get_action(observation)["q_r_chunk"]
    assert len(server.requests) == 10


def test_s02_many_clients_no_cross_talk(server: FakeDreamAPI, config: dict[str, Any]) -> None:
    """S-2 (short): 8 threads with one client each; every reply belongs to its own session."""
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            with dream._open_policy({**config, "checkpoint": f"{CHECKPOINT}-{index}"}) as client:
                observation = make_observation()
                observation["prompt"] = f"worker-{index}"
                for _ in range(50):
                    client.get_action(observation)
                mine = [r for r in server.requests if r.body["checkpoint"] == f"{CHECKPOINT}-{index}"]
                sessions = {r.body["session_id"] for r in mine if r.body["session_id"] is not None}
                assert len(sessions) == 1, f"worker {index} saw sessions {sessions}"
        except BaseException as exc:  # noqa: B036  # thread boundary: every outcome is recorded
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30.0)
        assert not thread.is_alive()
    assert errors == []
    assert len(server.requests) == 400

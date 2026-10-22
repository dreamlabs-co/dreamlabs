"""Behaviour tests for `robots.station_api`: the routes, the status codes and the JSON bodies over a real socket.

The controller runs on an event loop in a background thread, the way `station.serve` runs
it, and the API binds an ephemeral port on 127.0.0.1; requests go through `urllib`, so what is
pinned is the wire: method, path, body and reply.
"""

import asyncio
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

import conftest
import numpy as np
import pytest
import test_station

from robots import station, station_api

## Longest a test waits for the run task to reach a state.
WAIT_S = 5.0


class BackgroundLoop:
    """An asyncio loop on a daemon thread, the controller's home while the API is exercised from the test thread."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, name="station-loop", daemon=True)
        self._thread.start()

    def run(self, coroutine: Any) -> Any:
        """Runs a coroutine on the loop and returns its result."""
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(WAIT_S)

    def close(self) -> None:
        """Stops the loop and joins the thread."""
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(WAIT_S)
        self.loop.close()


@pytest.fixture
def served(build: Callable[..., test_station.Station]) -> Iterator[tuple[test_station.Station, str]]:
    """A station over fakes, its controller on a background loop, the API on an ephemeral port; yields the base URL."""
    parts = build()
    background = BackgroundLoop()
    api = station_api.StationApi(parts.controller, background.loop, host="127.0.0.1", port=0)
    api.start()
    try:
        yield parts, f"http://127.0.0.1:{api.port}"
    finally:
        background.run(parts.controller.close())
        api.stop()
        background.close()


## `test_station.build` under this module's name, so `served` can use it.
build = test_station.build


def request(
    base: str, method: str, path: str, body: Any = None, raw: bytes | None = None
) -> tuple[int, dict[str, Any]]:
    """Sends one request and returns `(status, json body)`, for error replies too."""
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=WAIT_S) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def wait_until(predicate: Callable[[], bool], what: str) -> None:
    """Polls from the test thread until `predicate` holds."""
    deadline_s = time.monotonic() + WAIT_S
    while not predicate():
        if time.monotonic() > deadline_s:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.005)


def test_status_start_prompt_stop_and_home_over_http(served: tuple[test_station.Station, str]) -> None:
    """The happy path end to end: idle status, start with a prompt, a running status with counts, a prompt change,
    a stop that returns once the arms hold, and a home that returns once the arms are at zeros."""
    parts, base = served

    status, body = request(base, "GET", "/status")
    assert status == 200 and body["state"] == station.IDLE and body["run"] == 0

    status, body = request(base, "POST", "/start", {"prompt": "wipe the table"})
    assert status == 200 and body["state"] == station.STARTING and body["prompt"] == "wipe the table"
    wait_until(lambda: parts.controller.loop is not None and parts.controller.loop.chunk_count >= 2, "chunks")
    status, body = request(base, "GET", "/status")
    assert status == 200 and body["state"] == station.RUNNING and body["chunks"] >= 2
    assert body["timing"]["count"] >= 2 and body["run"] == 1

    status, body = request(base, "POST", "/prompt", {"prompt": "stack the blocks"})
    assert status == 200 and body["prompt"] == "stack the blocks"
    wait_until(lambda: parts.clients[0].resets == 1, "the episode reset")

    status, body = request(base, "POST", "/stop")
    assert status == 200 and body["state"] == station.IDLE
    assert parts.clients[0].closed

    status, body = request(base, "POST", "/home")
    assert status == 200 and body["state"] == station.IDLE
    for robot in parts.robots.values():
        np.testing.assert_allclose(robot.commands[-1], [0.0] * 6 + [1.0])


def test_refused_transitions_are_409_with_an_error_body(served: tuple[test_station.Station, str]) -> None:
    """A second start and a home while running are refused with 409 and `{"error": ...}`; the run goes on."""
    parts, base = served
    assert request(base, "POST", "/start")[0] == 200
    wait_until(lambda: parts.controller.state == station.RUNNING, "running")

    status, body = request(base, "POST", "/start")
    assert status == 409 and body == {"error": "start refused: station is running"}
    status, body = request(base, "POST", "/home")
    assert status == 409 and body["error"].startswith("home refused: station is running")
    assert parts.controller.state == station.RUNNING and parts.controller.run_number == 1

    assert request(base, "POST", "/stop")[1]["state"] == station.IDLE


def test_bad_bodies_are_400_and_unknown_routes_404(served: tuple[test_station.Station, str]) -> None:
    """Invalid JSON, a non-object body, a missing or blank prompt are 400; other paths and GET on a POST route 404."""
    parts, base = served

    status, body = request(base, "POST", "/prompt", raw=b"{not json")
    assert status == 400 and "not valid JSON" in body["error"]
    status, body = request(base, "POST", "/prompt", raw=b"[1, 2]")
    assert status == 400 and "JSON object" in body["error"]
    status, body = request(base, "POST", "/prompt", {})
    assert status == 400 and "'prompt' key" in body["error"]
    status, body = request(base, "POST", "/prompt", {"prompt": "   "})
    assert status == 400 and "non-empty string" in body["error"]
    status, body = request(base, "POST", "/start", {"prompt": ""})
    assert status == 400 and "non-empty string" in body["error"]
    assert parts.controller.state == station.IDLE and parts.controller.prompt == conftest.TASK

    status, body = request(base, "GET", "/nope")
    assert status == 404 and "no such route: GET /nope" in body["error"]
    status, body = request(base, "POST", "/nope")
    assert status == 404 and "no such route: POST /nope" in body["error"]
    status, body = request(base, "GET", "/start")
    assert status == 404


def test_an_empty_post_body_is_fine_and_a_query_string_is_ignored(served: tuple[test_station.Station, str]) -> None:
    """`curl -X POST /start` sends no body: that starts with the current prompt; `/status?x=1` is `/status`."""
    parts, base = served

    status, body = request(base, "POST", "/start", raw=b"")
    assert status == 200 and body["prompt"] == conftest.TASK
    wait_until(lambda: parts.controller.state == station.RUNNING, "running")
    status, body = request(base, "GET", "/status?verbose=1")
    assert status == 200 and body["state"] == station.RUNNING
    assert request(base, "POST", "/stop")[0] == 200


def test_a_controller_fault_is_500_with_the_exception_named(
    served: tuple[test_station.Station, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anything but a refusal or a bad request is reported, never left as a dropped connection."""
    parts, base = served

    async def broken() -> dict[str, Any]:
        raise RuntimeError("controller exploded")

    monkeypatch.setattr(parts.controller, "home", broken)

    status, body = request(base, "POST", "/home")
    assert status == 500 and body == {"error": "RuntimeError: controller exploded"}


def test_the_server_uses_the_default_port_constant_and_stop_is_idempotent(
    build: Callable[..., test_station.Station],
) -> None:
    """`StationApi()` defaults to `station.DEFAULT_API_PORT`; stop() before start() and twice after are no-ops."""
    parts = build()
    api = station_api.StationApi(parts.controller, asyncio.new_event_loop())
    assert api.port == station.DEFAULT_API_PORT == 8091
    api.stop()
    api.stop()

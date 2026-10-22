"""The session release: reset_episode() and close() end the held session on the gateway with `/v1/end_session`,
best effort, so a stop or a new episode frees the seat now instead of on idle eviction."""

from __future__ import annotations

import time
from typing import Any

from conftest import FakeDreamAPI, make_observation

import dream

API_KEY = "sk-live-SECRET-TOKEN-0123456789"


def test_reset_episode_releases_the_session_and_the_next_request_starts_a_new_one(
    server: FakeDreamAPI, config: dict[str, Any]
) -> None:
    client = dream._open_policy(config, api_key=API_KEY)
    client.get_action(make_observation())
    first = server.requests[-1].body["session_id"]
    minted = server.sessions.copy()
    assert first is None and len(minted) == 1
    client.reset_episode()
    expected = {"checkpoint": config["checkpoint"], "session_id": next(iter(minted))}
    assert [request.body for request in server.ended] == [expected]
    assert server.ended[0].headers["authorization"] == f"Bearer {API_KEY}"
    assert server.sessions == set(), "the fake freed the session, as the gateway does"
    client.get_action(make_observation())
    assert server.requests[-1].body["session_id"] is None, "the retry carries no id: the gateway mints a new one"
    assert len(server.sessions) == 1 and server.sessions != minted
    client.close()


def test_close_releases_the_session(server: FakeDreamAPI, config: dict[str, Any]) -> None:
    client = dream._open_policy(config)
    client.get_action(make_observation())
    (session_id,) = server.sessions
    client.close()
    assert [request.body["session_id"] for request in server.ended] == [session_id]
    assert "authorization" not in server.ended[0].headers, "no key was configured, so none is sent"
    client.close()
    assert len(server.ended) == 1, "close is idempotent: the second sends nothing"


def test_reset_and_close_before_any_request_send_nothing(server: FakeDreamAPI, config: dict[str, Any]) -> None:
    client = dream._open_policy(config)
    client.reset_episode()
    client.reset_episode()
    client.close()
    assert server.ended == [] and server.other_paths == []


def test_a_gateway_that_is_gone_does_not_fail_the_release(server: FakeDreamAPI, config: dict[str, Any]) -> None:
    """The release is best effort and bounded: a stop never fails, and never hangs, on a gateway that went away."""
    client = dream._open_policy(config)
    client.get_action(make_observation())
    server.close()
    started = time.perf_counter()
    client.reset_episode()
    client.close()
    assert time.perf_counter() - started < 5.0

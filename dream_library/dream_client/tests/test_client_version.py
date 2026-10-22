"""The library names its build on every request, and a gateway's refusal of it as outdated says what to run.

Every request to the gateway carries `Dream-Client-Version: <dream.__version__>` and nothing else in that header; a
gateway that refuses the version answers `DA-4000 client_outdated` (`dream.transport.CLIENT_VERSION_HEADER` says
when), which the client raises as #DreamBadRequest ending with the install of the current library from PyPI, the
same line it prints for a missing dependency (`dream.datasets._install_hint()`).
"""

from __future__ import annotations

from typing import Any

import pytest
from conftest import FakeDreamAPI, make_observation

import dream
from dream.errors import DreamBadRequest, DreamConfigError, DreamServerError
from dream.transport import CLIENT_VERSION_HEADER, Transport

API_KEY = "sk-live-SECRET-TOKEN-0123456789"

## A refusal's detail, in the words of a gateway that compares builds; the client repeats whatever detail
## the gateway sends.
DETAIL = "this dream library (version 3.0.0+gfedcba987654) is not the one this server runs (commit 0123456789ab)"


def test_every_request_carries_the_library_version_and_nothing_else_in_it(
    server: FakeDreamAPI, config: dict[str, Any]
) -> None:
    """Each `get_action` and the session release name the build, `dream.__version__`; the key stays in its own
    header."""
    client = dream._open_policy(config, api_key=API_KEY)
    client.get_action(make_observation())
    client.get_action(make_observation())
    client.close()

    sent = [request.headers for request in (*server.requests, *server.ended)]
    assert len(sent) == 3
    assert [headers[CLIENT_VERSION_HEADER.lower()] for headers in sent] == [dream.__version__] * 3
    assert all(API_KEY not in headers[CLIENT_VERSION_HEADER.lower()] for headers in sent)
    assert all(headers["authorization"] == f"Bearer {API_KEY}" for headers in sent)


def test_the_channel_adds_the_version_to_whatever_headers_it_is_given(server: FakeDreamAPI) -> None:
    """The header is the transport's, so every route the client calls carries it without being asked."""
    transport = Transport(server.host, server.port, timeout_ms=2000)
    try:
        transport.request({"checkpoint": "c", "session_id": None, "reset": False, "observation": {}}, {})
    finally:
        transport.close()
    assert server.requests[0].headers["dream-client-version"] == dream.__version__


def test_each_client_names_itself_on_a_user_agent_it_keeps_for_every_request(
    server: FakeDreamAPI, config: dict[str, Any]
) -> None:
    """A robot's first request carries no session, so the gateway tells the robots of one key apart, when a pool is
    full, by their `User-Agent`: `dream-client/<version> <8 hex>`, the hex drawn once per client. One client sends the
    same value on every request, its session release included; another client sends another."""
    first = dream._open_policy(config, api_key=API_KEY)
    first.get_action(make_observation())
    first.get_action(make_observation())
    first.close()
    second = dream._open_policy(config, api_key=API_KEY)
    second.get_action(make_observation())
    second.close()

    asked = [request.headers["user-agent"] for request in server.requests]
    ended = [request.headers["user-agent"] for request in server.ended]
    assert (len(asked), len(ended)) == (3, 2)
    first_agents, second_agents = {asked[0], asked[1], ended[0]}, {asked[2], ended[1]}
    assert len(first_agents) == 1 and len(second_agents) == 1, "one value per client, kept for all its requests"
    assert first_agents != second_agents, "another client, another value"
    for agent in first_agents | second_agents:
        product, tag = agent.split(" ")
        assert product == f"dream-client/{dream.__version__}"
        assert len(tag) == 8 and all(c in "0123456789abcdef" for c in tag)


@pytest.mark.parametrize("platform_url", ["https://platform.dreamlabs.ai", "https://staging.platform.example.com"])
def test_an_outdated_library_is_a_bad_request_that_ends_with_the_install_command(
    server: FakeDreamAPI, config: dict[str, Any], monkeypatch: pytest.MonkeyPatch, platform_url: str
) -> None:
    """`DA-4000 client_outdated` is a #DreamBadRequest, never retryable, carrying the gateway's detail and then the
    install of the current library from PyPI, the same whichever platform the credentials name."""
    monkeypatch.setenv("DREAM_API_URL", platform_url)
    server.fail_next("DA-4000", "client_outdated", detail=DETAIL)
    with dream._open_policy(config) as client, pytest.raises(DreamConfigError) as raised:
        client.get_action(make_observation())

    error = raised.value
    assert isinstance(error, DreamBadRequest) and isinstance(error, DreamServerError)
    assert (error.code, error.reason, error.retryable) == ("DA-4000", "client_outdated", False)
    assert str(error) == (
        f"DreamAPI refused the inference request: DA-4000 BAD_REQUEST (client_outdated): {DETAIL}; install the "
        "library: pip install --upgrade dreamlabs"
    )

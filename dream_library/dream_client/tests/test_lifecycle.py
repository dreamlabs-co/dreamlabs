"""Lifecycle and surface tests: C-18..C-26, S-9 and S-10.

Everything here is about what a user can see or touch on the client: its repr, `dir()`,
`help()`, the context manager, the wording of every error message, and how it behaves when
dropped without `close()` or shared across threads. Every test names its requirement id.
"""

from __future__ import annotations

import logging
import pydoc
import re
import subprocess
import sys
import threading
from typing import Any

import numpy as np
import pytest
from conftest import CHECKPOINT, EMBODIMENT, REMOVED_MEMBERS, FakeDreamAPI, make_observation

import dream
from dream import errors
from dream.client import PolicyClient

## A `_`-prefixed identifier at a word start: what a user-facing string must never contain.
PRIVATE_NAME_RE = re.compile(r"\b_[a-z]")

## The six names `dir(client)` shows.
PUBLIC_SURFACE = [
    "get_action",
    "reset_episode",
    "close",
    "last_timing",
    "mode",
    "refill_due",
    "execution_horizon",
    "action_hz",
    "__enter__",
    "__exit__",
]

## A credential that is easy to grep for in anything the client emits.
API_KEY = "sk-live-SECRET-TOKEN-0123456789"

## Longest a thread may take in S-10 before that is reported as a hang.
THREAD_JOIN_TIMEOUT_S = 5.0


def test_c24_repr_names_endpoint_only(server: FakeDreamAPI, config: dict[str, Any]) -> None:
    """C-24: `repr(client)` is `<dream.Client host=... port=...>` and leaks no token."""
    with dream._open_policy(config, api_key=API_KEY) as client:
        assert repr(client) == f"<dream.Client host={server.host} port={server.port}>"
        assert API_KEY not in repr(client)
        assert PRIVATE_NAME_RE.search(repr(client)) is None


def test_c18_api_key_never_in_repr_or_logs(
    server: FakeDreamAPI, config: dict[str, Any], observation: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """C-18: the api key appears in no repr, no log record and no error message."""
    caplog.set_level(logging.DEBUG)
    with dream._open_policy(config, api_key=API_KEY) as client:
        client.get_action(observation)
        server.fail_next("DA-1000", "unknown_key", detail="bad token")
        with pytest.raises(errors.DreamAuthError) as info:
            client.get_action(observation)
        assert API_KEY not in str(info.value)
    assert server.requests[0].headers["authorization"] == f"Bearer {API_KEY}"
    for record in caplog.records:
        assert API_KEY not in record.getMessage()


def test_c19_context_manager(
    fd_guard: None, server: FakeDreamAPI, config: dict[str, Any], observation: dict[str, Any]
) -> None:
    """C-19: `__enter__` returns the client, `__exit__` closes it, and close is idempotent."""
    with dream._open_policy(config) as client:
        assert isinstance(client, PolicyClient)
        client.get_action(observation)
    with pytest.raises(errors.DreamError, match="closed"):
        client.get_action(observation)
    client.close()
    client.close()
    assert len(server.requests) == 1


def test_c19_exit_propagates_the_exception(config: dict[str, Any]) -> None:
    """C-19: `__exit__` closes on the way out of an exception and does not swallow it."""
    with pytest.raises(KeyError):
        with dream._open_policy(config) as client:
            raise KeyError("from the body")
    with pytest.raises(errors.DreamError, match="closed"):
        client.get_action({})


def test_c10_c20_removed_members_are_gone(config: dict[str, Any]) -> None:
    """C-10 C-20: `kill_server` and the rest of the GR00T lifecycle and policy-hook surface."""
    with dream._open_policy(config) as client:
        for name in REMOVED_MEMBERS:
            assert not hasattr(client, name), name
            assert not hasattr(PolicyClient, name), name


def test_c21_dir_is_exactly_the_public_surface(config: dict[str, Any]) -> None:
    """C-21: `dir(client)` lists exactly the six public names and nothing else."""
    with dream._open_policy(config) as client:
        assert sorted(dir(client)) == sorted(PUBLIC_SURFACE)


def test_c26_help_shows_no_private_member(config: dict[str, Any]) -> None:
    """C-26: `help(client)` (pydoc's rendering) contains no single-underscore member."""
    with dream._open_policy(config) as client:
        for subject in (client, PolicyClient):
            text = pydoc.render_doc(subject, renderer=pydoc.plaintext)
            assert "get_action" in text
            assert PRIVATE_NAME_RE.search(text) is None, PRIVATE_NAME_RE.findall(text)


def test_c23_policy_client_not_importable_from_dream() -> None:
    """C-23: `dream.PolicyClient` raises AttributeError; the class exists only in `dream.client`."""
    with pytest.raises(AttributeError):
        _ = dream.PolicyClient  # type: ignore[attr-defined]  # the point of the test
    assert "PolicyClient" not in dir(dream)
    assert PolicyClient.__module__ == "dream.client"


def test_c25_no_exception_message_names_a_private_member(
    server: FakeDreamAPI, config: dict[str, Any], observation: dict[str, Any]
) -> None:
    """C-25: every error path's message names an operation, never a `_`-prefixed member."""
    messages = _collect_error_messages(server, config, observation)
    assert len(messages) >= 20
    for message in messages:
        assert message.strip(), "an empty error message"
        assert PRIVATE_NAME_RE.search(message) is None, message


def test_c22_api_json_matches_the_surface(config: dict[str, Any]) -> None:
    """C-22: the internal policy client's discoverable surface is `_PUBLIC` plus the context-manager pair."""
    assert list(PolicyClient._PUBLIC) + ["__enter__", "__exit__"] == PUBLIC_SURFACE


def test_s09_interpreter_shutdown_with_live_client_is_silent(server: FakeDreamAPI) -> None:
    """S-9: a client dropped at interpreter exit without close() writes nothing to stderr."""
    script = "\n".join(
        [
            "import numpy as np",
            "import dream",
            f"client = dream._open_policy({{'host': {server.host!r}, 'port': {server.port}, 'scheme': 'http', "
            f"'timeout_ms': 5000, 'checkpoint': {CHECKPOINT!r}}})",
            "action = client.get_action({'prompt': 'x'})",
            "assert 'q_r_chunk' in action",
            "print('ok')",
        ]
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
    assert result.stderr == ""
    assert len(server.requests) == 1


def test_s10_shared_client_across_threads_fails_fast(
    server: FakeDreamAPI, config: dict[str, Any], observation: dict[str, Any]
) -> None:
    """S-10: a second concurrent `get_action` raises DreamError at once; state stays consistent."""
    server.latency_s = 0.3
    outcomes: dict[str, BaseException | dict[str, Any]] = {}
    start_line = threading.Barrier(2)

    def worker(name: str) -> None:
        start_line.wait()
        try:
            outcomes[name] = client.get_action(observation)
        except BaseException as exc:  # noqa: B036  # thread boundary: every outcome is recorded
            outcomes[name] = exc

    with dream._open_policy(config) as client:
        threads = [threading.Thread(target=worker, args=(name,)) for name in ("a", "b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(THREAD_JOIN_TIMEOUT_S)
            assert not thread.is_alive(), "get_action hung on a shared client"
        server.latency_s = 0.0
        assert client.get_action(observation)["q_r_chunk"]

    results = list(outcomes.values())
    failures = [r for r in results if isinstance(r, errors.DreamError)]
    successes = [r for r in results if isinstance(r, dict)]
    assert len(failures) == 1 and "thread" in str(failures[0])
    assert len(successes) == 1


def _collect_error_messages(server: FakeDreamAPI, config: dict[str, Any], observation: dict[str, Any]) -> list[str]:
    """Triggers every reachable error path once and returns the messages."""
    messages: list[str] = []

    def catch(action: Any) -> None:
        try:
            action()
        except errors.DreamError as exc:
            messages.append(str(exc))
        else:
            pytest.fail("expected an error")

    bad_configs: list[dict[str, Any]] = [
        {"checkpoint": CHECKPOINT, "timeout_ms": "soon"},
        {"checkpoint": CHECKPOINT, "timeout_ms": 0},
        {"checkpoint": CHECKPOINT, "timeout_ms": 2**31},
        {"checkpoint": CHECKPOINT, "port": 70000},
        {"checkpoint": CHECKPOINT, "port": "5555"},
        {"checkpoint": CHECKPOINT, "host": ""},
        {"checkpoint": CHECKPOINT, "strict": 1},
        {},
        {"checkpoint": ""},
        {"checkpoint": CHECKPOINT, "policy": "x"},
        {"checkpoint": CHECKPOINT, "embodiment": None},
        {"checkpoint": CHECKPOINT, "embodiment": 3},
        {"checkpoint": CHECKPOINT, "embodiment": {"state": 3}},
        {"checkpoint": CHECKPOINT, "embodiment": {"state": {"k": 3}}},
        {"checkpoint": CHECKPOINT, "embodiment": {"state": {"k": {"shape": 7}}}},
        {"checkpoint": CHECKPOINT, "embodiment": {"state": {"k": {"dtype": "nope"}}}},
        {"checkpoint": CHECKPOINT, "embodiment": {"state": {"k": {"dtype": "(2,-1)f4"}}}},
    ]
    for bad in bad_configs:
        catch(lambda bad=bad: dream._open_policy(bad))

    with dream._open_policy(config) as client:
        catch(lambda: client.get_action({}))
        catch(lambda: client.get_action(3))  # type: ignore[arg-type]  # the wrong type is the point
        catch(lambda: client.get_action({**observation, "state.arm_r.joint_pos": np.zeros(7, dtype=np.float32)}))
        catch(lambda: client.get_action({**observation, "state.arm_r.joint_pos": np.zeros(6)}))
        catch(lambda: client.get_action({**observation, "state.arm_r.typo": np.zeros(6, dtype=np.float32)}))
        server.horizon = 9
        catch(lambda: client.get_action(observation))
        server.horizon = 4
        for code, reason in (
            ("DA-1000", "unknown_key"),
            ("DA-2000", "pool_full"),
            ("DA-4000", "malformed"),
            ("DA-5000", "timeout"),
            ("DA-6000", "internal"),
            ("DA-9999", None),
        ):
            server.fail_next(code, reason, detail="detail text")
            catch(lambda: client.get_action(observation))
        client.get_action(observation)
        server.restart()
        catch(lambda: client.get_action(observation))
        server.reply_override = (502, b"<html>")
        catch(lambda: client.get_action(observation))
        server.reply_override = (200, b'{"actions": {}}')
        catch(lambda: client.get_action(observation))
        server.hold_next = 1
        with dream._open_policy({**config, "timeout_ms": 200}) as quick:
            catch(lambda: quick.get_action(observation))
    catch(lambda: client.get_action(observation))
    catch(lambda: client.reset_episode())
    dead = dream._open_policy({**config, "port": 1})
    with dead:
        catch(lambda: dead.get_action(make_observation()))
    with dream._open_policy(config) as busy:
        busy._lock.acquire()
        catch(lambda: busy.get_action(observation))
        catch(lambda: busy.reset_episode())
        busy._lock.release()
    assert EMBODIMENT
    return messages

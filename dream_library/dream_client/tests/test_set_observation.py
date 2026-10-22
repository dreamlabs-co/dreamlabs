"""Tests of `dream.DreamClient` inference: set_observation() every camera tick, then get_action() with no argument.

The client keeps the frame window itself: each set_observation() hands it one frame per camera and the rest of the
observation, it keeps frames at `obs_fps` into a ring of `obs_window` per camera, and get_action() sends the latest
observation plus each camera's window to the model. These tests are written against that public behaviour only:
every window is read back from what the in-process `FakeDreamAPI` (`conftest.py`) received on the wire, and the
keeping rule is driven with explicit `timestamp=` values, so nothing depends on the wall clock.
"""

from __future__ import annotations

import copy
import inspect
import math
import random
import subprocess
import sys
import threading
import time
import types
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import pytest
from conftest import CHECKPOINT, FRAME_B64, FakeDreamAPI, make_observation

import dream
from dream import DreamClient, models
from dream.errors import DreamConfigError, DreamError, DreamObservationError

## Start of every explicit timestamp: seconds on a made-up monotonic clock.
T0 = 100.0

## A keep rate whose period is exact in binary, so schedule arithmetic in these tests carries no rounding.
EXACT_FPS = 4

## `1 / EXACT_FPS`, seconds.
EXACT_PERIOD_S = 0.25

## Window long enough that no test of the keeping rule ever drops a frame: the window is then every kept frame.
LONG_WINDOW = 200

## Longest a thread or a held request may take before that is reported as a hang.
THREAD_TIMEOUT_S = 5.0

## How long a held request is held before the fake lets it go on its own (below THREAD_TIMEOUT_S).
HOLD_S = 3.0

## Longest set_observation() may take while a get_action() is in flight before that counts as waiting on it.
PROMPT_S = 0.05

## The two cameras most tests use: a fine-tuned model's `video.<role>` keys.
TOP = "video.top"
WRIST = "video.wrist"

## Every request body carries exactly these keys (the gateway's request model).
BODY_KEYS = {"checkpoint", "session_id", "reset", "observation"}


# -- helpers -------------------------------------------------------------------------------------------------------


def frame(index: int, role: str = "top") -> str:
    """A stand-in frame that names the tick it came from; the client never decodes a frame, only carries the str.

    @param index The tick number.
    @param role The camera, so two cameras' frames of one tick differ.
    @return A non-empty alphanumeric string.
    """
    return f"{role}{index:05d}"


def index_of(value: str) -> int:
    """The tick number a #frame names."""
    return int(value[-5:])


def both(index: int) -> dict[str, str]:
    """The frames of one tick for the two cameras `TOP` and `WRIST`."""
    return {TOP: frame(index, "top"), WRIST: frame(index, "wrist")}


def open_policy(client: DreamClient, **knobs: Any) -> None:
    """Opens a policy with no embodiment checks and no recording, so any frame key and observation key is sent as
    given and nothing dials a platform (recording has its own tests, `test_recorder.py`).

    @param client The client.
    @param knobs `obs_window`, `obs_fps`, or a connection key to override.
    """
    client.policy({"checkpoint": CHECKPOINT, "strict": False, "timeout_ms": 2000, "_record": False, **knobs})


def sent(server: FakeDreamAPI, index: int = -1) -> dict[str, Any]:
    """The `observation` of one request as the fake received it (the last one by default)."""
    observation: dict[str, Any] = server.requests[index].body["observation"]
    return observation


def ticks(hz: float, count: int, start: float = T0, first: int = 0) -> list[tuple[int, float]]:
    """`(index, timestamp)` for `count` calls at a steady `hz`, numbered from `first`."""
    return [(first + k, start + k / hz) for k in range(count)]


def kept(server: FakeDreamAPI, client: DreamClient, stamps: Sequence[tuple[int, float]]) -> list[int]:
    """Calls set_observation once per `(index, timestamp)`, then get_action once; answers the indices it kept.

    Each call carries frame `index` under `TOP` and `{"tick": index}` as its observation. The policy must be open
    with a window of at least as many frames as are kept, so the window sent is every frame kept, oldest first.

    @return The tick numbers of the frames in the window sent.
    """
    for index, stamp in stamps:
        client.set_observation({TOP: frame(index)}, {"tick": index}, timestamp=stamp)
    client.get_action()
    observation = sent(server)
    # The latest observation is always the newest call's, whether its frame was kept or not.
    assert observation["tick"] == stamps[-1][0]
    return [index_of(value) for value in observation[TOP]]


@pytest.fixture(scope="module")
def module_server() -> Iterator[FakeDreamAPI]:
    """One fake gateway for the whole module: stopping one waits out its half-second poll, once here, not per test."""
    fake = FakeDreamAPI()
    yield fake
    fake.close()


@pytest.fixture
def server(module_server: FakeDreamAPI, monkeypatch: pytest.MonkeyPatch) -> FakeDreamAPI:
    """conftest's `server`, shared across this module and wiped before each test as a fresh one would be.

    `DREAM_GATEWAY_URL` names it, as conftest's does. Overriding the name here also hands it to conftest's `config`.
    Every test leaves the fake idle (each client is closed and every held request released before the test ends), so
    wiping the logs and the sessions is enough.
    """
    module_server.requests.clear()
    module_server.ended.clear()
    module_server.other_paths.clear()
    module_server.sessions.clear()
    module_server.handler = None
    module_server.latency_s = 0.0
    module_server.hold_next = 0
    module_server.reply_override = None
    monkeypatch.setenv("DREAM_GATEWAY_URL", f"http://{module_server.host}:{module_server.port}")
    return module_server


@pytest.fixture
def client(server: FakeDreamAPI) -> Iterator[DreamClient]:
    """A client whose gateway is `server`; no policy open yet. Closed on teardown."""
    opened = DreamClient()
    yield opened
    opened.close()


# -- the one inference API ----------------------------------------------------------------------------------------


def test_set_observation_and_a_bare_get_action_are_the_one_way_to_run_a_model() -> None:
    """`dream.DreamClient` is the only client: no `dream.init`, no `dream.experimental`, no public policy object."""
    assert not hasattr(dream, "init")
    assert not hasattr(dream, "PolicyClient")
    result = subprocess.run(
        [sys.executable, "-c", "import dream.experimental"], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode != 0 and "ModuleNotFoundError" in result.stderr


def test_the_signatures_are_the_specified_ones() -> None:
    """set_observation(frames, observation, *, timestamp=None); get_action(*, commanded=None); reset_episode();
    policy(config) answers nothing."""
    parameters = inspect.signature(DreamClient.set_observation).parameters
    assert list(parameters) == ["self", "frames", "observation", "timestamp"]
    assert parameters["frames"].default is inspect.Parameter.empty
    assert parameters["observation"].default is inspect.Parameter.empty
    assert parameters["timestamp"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["timestamp"].default is None
    parameters = inspect.signature(DreamClient.get_action).parameters
    assert list(parameters) == ["self", "commanded"]
    assert parameters["commanded"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["commanded"].default is None
    assert list(inspect.signature(DreamClient.reset_episode).parameters) == ["self"]
    assert inspect.signature(DreamClient.policy).return_annotation in (None, "None")


def test_get_action_with_a_positional_observation_is_a_type_error_and_sends_nothing(
    server: FakeDreamAPI, client: DreamClient
) -> None:
    """A caller still on the old shape gets a TypeError, not a request built from a stale window."""
    open_policy(client)
    client.set_observation({TOP: frame(0)}, {"tick": 0}, timestamp=T0)
    with pytest.raises(TypeError):
        client.get_action({"prompt": "the old shape"})  # type: ignore[call-arg]  # the point of the test
    with pytest.raises(TypeError):
        client.set_observation({TOP: frame(1)})  # type: ignore[call-arg]  # both positionals are required
    with pytest.raises(TypeError):
        client.set_observation({TOP: frame(1)}, {}, T0 + 1)  # type: ignore[misc]  # timestamp is keyword-only
    assert server.requests == []


# -- policy(): the two knobs ---------------------------------------------------------------------------------------


def test_policy_answers_nothing_and_the_client_carries_the_session(server: FakeDreamAPI, client: DreamClient) -> None:
    """policy() returns None; last_timing and session_id are the client's, None until the first get_action()."""
    assert client.last_timing is None and client.session_id is None
    assert open_policy(client) is None  # type: ignore[func-returns-value]  # the point of the test
    assert client.last_timing is None and client.session_id is None
    client.set_observation({TOP: frame(0)}, {"tick": 0}, timestamp=T0)
    action = client.get_action()
    assert np.asarray(action["q_r_chunk"]).shape == (server.horizon, 7)
    assert client.last_timing is not None
    assert client.session_id in server.sessions
    assert len(server.requests) == 1


def test_the_client_names_the_checkpoint_its_policy_runs(server: FakeDreamAPI, client: DreamClient) -> None:
    """`checkpoint` is the model policy() was given (Live View shows it), None before a policy and after close()."""
    assert client.checkpoint is None
    open_policy(client)
    assert client.checkpoint == CHECKPOINT
    client.close()
    assert client.checkpoint is None


def test_the_defaults_are_a_window_of_nine_kept_at_five_frames_a_second(
    server: FakeDreamAPI, client: DreamClient
) -> None:
    """Without the knobs, 30 Hz calls keep every 6th frame (5 fps) and the window holds the last 9 of them."""
    open_policy(client)
    window = kept(server, client, ticks(30, 61))  # frames 0, 6, ..., 60: eleven kept
    assert window == list(range(12, 61, 6))
    assert len(window) == 9


@pytest.mark.parametrize(("obs_window", "obs_fps"), [(1, 0.5), (3, 2.5), (9, 5), (12, 30), (2, 5.0)])
def test_the_knobs_are_consumed_and_never_sent(
    server: FakeDreamAPI, client: DreamClient, obs_window: int, obs_fps: float
) -> None:
    """A valid `obs_window` / `obs_fps` opens the policy and neither key reaches the wire."""
    config = {"checkpoint": CHECKPOINT, "strict": False, "timeout_ms": 2000, "_record": False}
    client.policy({**config, "obs_window": obs_window, "obs_fps": obs_fps})
    client.set_observation({TOP: frame(0)}, {"tick": 0}, timestamp=T0)
    client.get_action()
    request = server.requests[0]
    assert set(request.body) == BODY_KEYS
    assert b"obs_window" not in request.raw and b"obs_fps" not in request.raw
    assert request.body["observation"] == {"tick": 0, TOP: [frame(0)]}


@pytest.mark.parametrize("repeats", [2, 3])
def test_a_loop_faster_than_its_cameras_keeps_the_same_frames(
    server: FakeDreamAPI, client: DreamClient, repeats: int
) -> None:
    """A loop that hands each 30 Hz camera frame `repeats` times, with that frame's capture time, keeps exactly the
    frames a loop at the camera rate keeps: a repeated timestamp never shrinks the slack."""
    open_policy(client, obs_window=LONG_WINDOW)
    stamps = [stamp for stamp in ticks(30, 61) for _ in range(repeats)]
    assert kept(server, client, stamps) == list(range(0, 61, 6))


def test_policy_does_not_change_the_callers_config(server: FakeDreamAPI, client: DreamClient) -> None:
    """The knobs are popped from a copy: the caller's dict, nested values included, is as it was."""
    config: dict[str, Any] = {
        "checkpoint": CHECKPOINT,
        "strict": False,
        "timeout_ms": 2000,
        "obs_window": 4,
        "obs_fps": 10,
    }
    before = copy.deepcopy(config)
    client.policy(config)
    assert config == before


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("obs_window", 0),
        ("obs_window", -1),
        ("obs_window", True),
        ("obs_window", False),
        ("obs_window", "9"),
        ("obs_window", 9.0),
        ("obs_window", None),
        ("obs_fps", 0),
        ("obs_fps", 0.0),
        ("obs_fps", -1),
        ("obs_fps", -0.5),
        ("obs_fps", True),
        ("obs_fps", "5"),
        ("obs_fps", math.nan),
        ("obs_fps", math.inf),
        ("obs_fps", -math.inf),
        ("obs_fps", None),
    ],
)
def test_a_bad_knob_is_a_config_error_naming_it(
    server: FakeDreamAPI, client: DreamClient, key: str, value: Any
) -> None:
    """Wrong type or range: DreamConfigError naming the key; `bool` is not a number here; nothing is sent."""
    with pytest.raises(DreamConfigError, match=key):
        open_policy(client, **{key: value})
    with pytest.raises(DreamConfigError):
        client.get_action()
    assert server.requests == []


def test_a_bad_knob_is_refused_before_anything_else_and_the_open_policy_stays(
    server: FakeDreamAPI, client: DreamClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check runs first: no embodiment lookup, the previous policy keeps its session, and its window."""
    looked_up: list[dict[str, Any]] = []
    monkeypatch.setattr(models, "list_models", lambda **kwargs: looked_up.append(kwargs) or [])
    open_policy(client, obs_fps=EXACT_FPS)
    client.set_observation({TOP: frame(0)}, {"tick": 0}, timestamp=T0)
    client.get_action()
    client.get_action()
    session = server.requests[1].body["session_id"]
    assert session

    # strict is on and there is no embodiment: policy() would look the checkpoint up first.
    with pytest.raises(DreamConfigError, match="obs_fps"):
        client.policy({"checkpoint": "dream-1-ft:alice-towels-nb1", "obs_fps": 0})
    with pytest.raises(DreamConfigError, match="obs_window"):
        client.policy({"checkpoint": "dream-1-ft:alice-towels-nb1", "obs_window": True})

    assert looked_up == []
    client.get_action()
    assert server.requests[2].body["session_id"] == session
    assert sent(server) == {"tick": 0, TOP: [frame(0)]}
    assert server.ended == []


def test_a_second_policy_starts_an_empty_window_with_its_own_knobs(server: FakeDreamAPI, client: DreamClient) -> None:
    """policy() again replaces the first and its window: nothing to send until set_observation(), then new frames
    only."""
    open_policy(client, obs_window=2, obs_fps=EXACT_FPS)
    for index in range(3):
        client.set_observation({TOP: frame(index)}, {"tick": index}, timestamp=T0 + index * EXACT_PERIOD_S)
    client.get_action()
    assert sent(server)[TOP] == [frame(1), frame(2)]

    open_policy(client, obs_window=3, obs_fps=EXACT_FPS)
    with pytest.raises(DreamObservationError, match="set_observation"):
        client.get_action()
    assert len(server.requests) == 1

    # A new window: other cameras, and a clock that restarted, are a fresh start too.
    for index in range(10, 15):
        client.set_observation({WRIST: frame(index, "wrist")}, {"tick": index}, timestamp=index * EXACT_PERIOD_S)
    client.get_action()
    assert sent(server) == {"tick": 14, WRIST: [frame(12, "wrist"), frame(13, "wrist"), frame(14, "wrist")]}
    assert server.requests[1].body["session_id"] is None


# -- set_observation(): what it refuses ----------------------------------------------------------------------------


def test_set_observation_before_policy_is_a_config_error(server: FakeDreamAPI, client: DreamClient) -> None:
    """No policy, no window: DreamConfigError telling the caller to call policy() first."""
    with pytest.raises(DreamConfigError, match=r"call policy\("):
        client.set_observation({TOP: frame(0)}, {"tick": 0}, timestamp=T0)
    assert server.requests == []


## A timestamp well past the next due one: a refused call stamped with it must not move the clock, or the due call
## that follows it would be refused as going back.
LATER = T0 + 5.0

## Calls refused against an episode that already holds frames for TOP and WRIST at T0 (and is due again at
## T0 + EXACT_PERIOD_S): `(frames, observation, timestamp, a string the message must name or None)`.
REFUSED_CALLS = [
    pytest.param([(TOP, frame(1))], {"tick": 1}, LATER, None, id="frames-a-list"),
    pytest.param(TOP, {"tick": 1}, LATER, None, id="frames-a-str"),
    pytest.param(None, {"tick": 1}, LATER, None, id="frames-none"),
    pytest.param({}, {"tick": 1}, LATER, None, id="frames-empty"),
    pytest.param({TOP: "", WRIST: frame(1, "wrist")}, {"tick": 1}, LATER, TOP, id="frame-empty"),
    pytest.param({TOP: None, WRIST: frame(1, "wrist")}, {"tick": 1}, LATER, TOP, id="frame-none"),
    pytest.param({TOP: 7, WRIST: frame(1, "wrist")}, {"tick": 1}, LATER, TOP, id="frame-a-number"),
    pytest.param({TOP: b"\xff\xd8", WRIST: frame(1, "wrist")}, {"tick": 1}, LATER, TOP, id="frame-bytes"),
    pytest.param({TOP: [frame(1)], WRIST: frame(1, "wrist")}, {"tick": 1}, LATER, TOP, id="frame-a-window"),
    pytest.param({**both(1), "": frame(1)}, {"tick": 1}, LATER, None, id="frame-key-empty"),
    pytest.param({**both(1), 3: frame(1)}, {"tick": 1}, LATER, None, id="frame-key-a-number"),
    pytest.param(both(1), [("tick", 1)], LATER, None, id="observation-a-list"),
    pytest.param(both(1), None, LATER, None, id="observation-none"),
    pytest.param(both(1), {"tick": 1, TOP: [frame(1)]}, LATER, TOP, id="observation-holds-a-frame-key"),
    pytest.param({TOP: frame(1)}, {"tick": 1}, LATER, WRIST, id="frame-key-missing"),
    pytest.param(
        {**both(1), "video.side": frame(1, "side")},
        {"tick": 1},
        LATER,
        "video.side",
        id="frame-key-extra",
    ),
    pytest.param(
        {TOP: frame(1), "video.side": frame(1, "side")},
        {"tick": 1},
        LATER,
        "video.side",
        id="frame-key-renamed",
    ),
    pytest.param(both(1), {"tick": 1}, T0 - EXACT_PERIOD_S, "timestamp", id="timestamp-goes-back"),
    pytest.param(both(1), {"tick": 1}, T0 - 1e-6, "timestamp", id="timestamp-goes-back-a-microsecond"),
]


@pytest.mark.parametrize(("frames", "observation", "timestamp", "named"), REFUSED_CALLS)
def test_a_refused_set_observation_names_the_key_and_leaves_the_buffer_unchanged(
    server: FakeDreamAPI, client: DreamClient, frames: Any, observation: Any, timestamp: float, named: str | None
) -> None:
    """DreamObservationError, and the window, the latest observation, the clock and the keep schedule are as they
    were."""
    open_policy(client, obs_fps=EXACT_FPS)
    client.set_observation(both(0), {"tick": 0}, timestamp=T0)

    with pytest.raises(DreamObservationError) as raised:
        client.set_observation(frames, observation, timestamp=timestamp)
    if named is not None:
        assert named in str(raised.value)

    client.get_action()
    assert sent(server) == {"tick": 0, TOP: [frame(0, "top")], WRIST: [frame(0, "wrist")]}
    # Neither did the clock or the schedule: the next due tick, earlier than LATER, is accepted and kept.
    client.set_observation(both(1), {"tick": 1}, timestamp=T0 + EXACT_PERIOD_S)
    client.get_action()
    assert sent(server) == {
        "tick": 1,
        TOP: [frame(0, "top"), frame(1, "top")],
        WRIST: [frame(0, "wrist"), frame(1, "wrist")],
    }


@pytest.mark.parametrize(
    ("frames", "observation"),
    [
        pytest.param({}, {"tick": 0}, id="frames-empty"),
        pytest.param([TOP], {"tick": 0}, id="frames-a-list"),
        pytest.param({TOP: ""}, {"tick": 0}, id="frame-empty"),
        pytest.param({TOP: b"jpeg"}, {"tick": 0}, id="frame-bytes"),
        pytest.param({"": frame(0)}, {"tick": 0}, id="frame-key-empty"),
        pytest.param({0: frame(0)}, {"tick": 0}, id="frame-key-a-number"),
        pytest.param({TOP: frame(0)}, "tick", id="observation-a-str"),
        pytest.param({TOP: frame(0)}, {TOP: "x"}, id="observation-holds-a-frame-key"),
    ],
)
def test_a_refused_first_set_observation_stores_nothing(
    server: FakeDreamAPI, client: DreamClient, frames: Any, observation: Any
) -> None:
    """Refused as the episode's first call, nothing is stored: get_action still has nothing to send."""
    open_policy(client)
    with pytest.raises(DreamObservationError):
        client.set_observation(frames, observation, timestamp=T0)
    with pytest.raises(DreamObservationError, match="set_observation"):
        client.get_action()
    assert server.requests == []
    # The refused call fixed no frame keys and no clock either.
    client.set_observation({WRIST: frame(1, "wrist")}, {"tick": 1}, timestamp=T0 - 50.0)
    client.get_action()
    assert sent(server) == {"tick": 1, WRIST: [frame(1, "wrist")]}


def test_a_bytes_frame_is_told_to_send_a_base64_str(client: DreamClient) -> None:
    """Raw JPEG bytes are the likely mistake, so the message says what to send instead."""
    open_policy(client)
    with pytest.raises(DreamObservationError) as raised:
        client.set_observation({TOP: b"\xff\xd8\xff\xe0 raw jpeg"}, {"tick": 0}, timestamp=T0)
    message = str(raised.value)
    assert TOP in message and "base64" in message


def test_a_changed_camera_set_names_the_missing_and_the_extra_keys(client: DreamClient) -> None:
    """The frame keys are fixed for the episode; a different set is refused with both differences named."""
    open_policy(client)
    client.set_observation(both(0), {"tick": 0}, timestamp=T0)
    with pytest.raises(DreamObservationError) as raised:
        client.set_observation({TOP: frame(1), "video.side": frame(1, "side")}, {"tick": 1}, timestamp=T0 + 1)
    assert WRIST in str(raised.value) and "video.side" in str(raised.value)


def test_an_equal_timestamp_is_accepted_but_keeps_no_frame(server: FakeDreamAPI, client: DreamClient) -> None:
    """Only an earlier timestamp is refused; a repeat of the last one replaces the observation and keeps nothing."""
    open_policy(client, obs_fps=EXACT_FPS)
    client.set_observation({TOP: frame(0)}, {"tick": 0}, timestamp=T0)
    client.set_observation({TOP: frame(1)}, {"tick": 1}, timestamp=T0)
    client.get_action()
    assert sent(server) == {"tick": 1, TOP: [frame(0)]}


def test_the_default_timestamp_is_the_monotonic_clock_at_the_call(server: FakeDreamAPI, client: DreamClient) -> None:
    """With no timestamp the call is stamped `time.monotonic()`: an explicit earlier one is then refused, a later one
    kept."""
    open_policy(client)
    assert client.set_observation({TOP: frame(0)}, {"tick": 0}) is None
    with pytest.raises(DreamObservationError, match="timestamp"):
        client.set_observation({TOP: frame(1)}, {"tick": 1}, timestamp=time.monotonic() - 60.0)
    client.set_observation({TOP: frame(2)}, {"tick": 2}, timestamp=time.monotonic() + 1.0)
    client.get_action()
    assert sent(server) == {"tick": 2, TOP: [frame(0), frame(2)]}


def test_any_mapping_is_accepted_for_the_frames_and_the_observation(server: FakeDreamAPI, client: DreamClient) -> None:
    """`Mapping`, not only `dict`: a read-only mapping proxy works for both arguments."""
    open_policy(client)
    frames: Mapping[str, str] = types.MappingProxyType({TOP: frame(0)})
    observation: Mapping[str, Any] = types.MappingProxyType({"tick": 0, "prompt": "fold the towel"})
    client.set_observation(frames, observation, timestamp=T0)
    client.get_action()
    assert sent(server) == {"tick": 0, "prompt": "fold the towel", TOP: [frame(0)]}


# -- get_action(): what is sent ------------------------------------------------------------------------------------


def test_get_action_before_policy_is_a_config_error_naming_policy(server: FakeDreamAPI, client: DreamClient) -> None:
    """Before policy(), every inference call and property is a DreamConfigError telling the caller to call it first."""
    for call in (
        client.get_action,
        client.reset_episode,
        lambda: client.set_observation({TOP: frame(0)}, {}),
        lambda: client.refill_due(0, 0),
        lambda: client.execution_horizon,
        lambda: client.action_hz,
        lambda: client.mode,
    ):
        with pytest.raises(DreamConfigError, match=r"call policy\("):
            call()
    assert server.requests == []


def test_get_action_before_any_set_observation_is_an_observation_error(
    server: FakeDreamAPI, client: DreamClient
) -> None:
    """A policy but no observation yet: DreamObservationError telling the caller what to call; nothing is sent."""
    open_policy(client)
    with pytest.raises(DreamObservationError, match=r"set_observation\("):
        client.get_action()
    assert server.requests == []


def test_the_request_is_the_latest_observation_plus_each_window(server: FakeDreamAPI, client: DreamClient) -> None:
    """The newest observation replaces the previous one whole (no merge); each camera's window rides beside it."""
    open_policy(client, obs_fps=EXACT_FPS)
    client.set_observation(both(0), {"state.arm": [0.0, 0.0], "prompt": "fold", "tick": 0}, timestamp=T0)
    client.set_observation(both(1), {"prompt": "fold the towel", "tick": 1}, timestamp=T0 + EXACT_PERIOD_S)
    client.get_action()
    assert sent(server) == {
        "prompt": "fold the towel",
        "tick": 1,
        TOP: [frame(0, "top"), frame(1, "top")],
        WRIST: [frame(0, "wrist"), frame(1, "wrist")],
    }


def test_the_observation_is_stored_as_a_shallow_copy(server: FakeDreamAPI, client: DreamClient) -> None:
    """Changing the caller's dicts after set_observation() changes nothing that is sent."""
    open_policy(client)
    frames = {TOP: frame(0)}
    observation: dict[str, Any] = {"prompt": "fold", "tick": 0}
    client.set_observation(frames, observation, timestamp=T0)
    observation["prompt"] = "changed after the call"
    observation["late"] = "added after the call"
    del observation["tick"]
    frames[TOP] = "replaced after the call"
    frames[WRIST] = "added after the call"
    client.get_action()
    assert sent(server) == {"prompt": "fold", "tick": 0, TOP: [frame(0)]}


def test_the_window_grows_to_obs_window_then_drops_the_oldest(server: FakeDreamAPI, client: DreamClient) -> None:
    """Never empty, shorter until full, then a ring: oldest first, both cameras aligned frame for frame."""
    open_policy(client, obs_window=3, obs_fps=EXACT_FPS)
    for index in range(6):
        client.set_observation(both(index), {"tick": index}, timestamp=T0 + index * EXACT_PERIOD_S)
        client.get_action()
        expected = list(range(max(0, index - 2), index + 1))
        observation = sent(server)
        assert isinstance(observation[TOP], list) and isinstance(observation[WRIST], list)
        assert observation[TOP] == [frame(i, "top") for i in expected]
        assert observation[WRIST] == [frame(i, "wrist") for i in expected]
        assert observation["tick"] == index


def test_a_window_of_one_is_the_latest_kept_frame(server: FakeDreamAPI, client: DreamClient) -> None:
    """`obs_window=1` sends one frame: the newest kept, not the newest seen."""
    open_policy(client, obs_window=1, obs_fps=EXACT_FPS)
    client.set_observation({TOP: frame(0)}, {"tick": 0}, timestamp=T0)
    client.set_observation({TOP: frame(1)}, {"tick": 1}, timestamp=T0 + EXACT_PERIOD_S)
    client.set_observation({TOP: frame(2)}, {"tick": 2}, timestamp=T0 + EXACT_PERIOD_S + 0.01)
    client.get_action()
    assert sent(server) == {"tick": 2, TOP: [frame(1)]}


def test_get_action_forwards_commanded(server: FakeDreamAPI, client: DreamClient) -> None:
    """`commanded` reaches the inference client: a valid one is used, a bad one refused there before sending."""
    open_policy(client)
    client.set_observation({TOP: frame(0)}, {"tick": 0}, timestamp=T0)
    with pytest.raises(DreamObservationError, match="commanded"):
        client.get_action(commanded={"not_a_chunk_key": [0.0] * 6})
    assert server.requests == []
    action = client.get_action(commanded={"q_r_chunk": [0.0] * 6})
    assert np.asarray(action["q_r_chunk"]).shape == (server.horizon, 7)
    assert "commanded" not in sent(server)


def test_a_yam_body_passes_the_strict_checks_with_windows_built_by_the_client(
    server: FakeDreamAPI, config: dict[str, Any]
) -> None:
    """With the embodiment declared, the built observation is checked like any other: `image_window.*` windows
    and state arrays pass; a missing state key is refused before anything is sent."""
    full = make_observation()
    frames = {key: full.pop(key)[0] for key in ("image_window.top", "image_window.left", "image_window.right")}
    with DreamClient() as yam:
        yam.policy({key: value for key, value in config.items() if key not in ("host", "port", "scheme")})
        yam.set_observation(frames, full, timestamp=T0)
        action = yam.get_action()
        assert np.asarray(action["q_r_chunk"]).shape == (server.horizon, 7)
        observation = sent(server)
        for key in frames:
            assert observation[key] == [FRAME_B64]
        assert observation["state.arm_r.joint_pos"] == [0.0] * 6
        assert observation["prompt"] == "pick up the cup"

        partial = {key: value for key, value in full.items() if key != "state.arm_l.gripper_pos"}
        yam.set_observation(frames, partial, timestamp=T0 + 1.0)
        with pytest.raises(DreamObservationError, match="state.arm_l.gripper_pos"):
            yam.get_action()
    assert len(server.requests) == 1


# -- the keeping rule ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("obs_fps", "call_hz", "every"),
    [
        (5, 30, 6),
        (5, 15, 3),
        (5, 10, 2),
        (5, 5, 1),
        (5, 4, 1),
        (10, 30, 3),
        (2.5, 10, 4),
        (1, 30, 30),
        (30, 30, 1),
    ],
)
def test_steady_calls_keep_one_frame_per_period(
    server: FakeDreamAPI, client: DreamClient, obs_fps: float, call_hz: float, every: int
) -> None:
    """At a steady call rate the kept frames are every `call_hz / obs_fps`-th call from the first; a caller slower
    than `obs_fps` keeps every call."""
    open_policy(client, obs_window=LONG_WINDOW, obs_fps=obs_fps)
    window = kept(server, client, ticks(call_hz, 8 * every + 1))
    assert window == list(range(0, 8 * every + 1, every))


## Per-call jitter, seconds, by tick modulo 6, at its worst for 30 Hz calls kept at 5 fps: the tick just before a
## due one arrives 4 ms late (closest to looking due) and the due one 4 ms early (furthest from it).
WORST_JITTER_S = (-0.004, 0.002, -0.003, 0.004, -0.002, 0.004)


def _seeded_jitter() -> list[float]:
    generator = random.Random(588)
    return [generator.uniform(-0.004, 0.004) for _ in range(121)]


@pytest.mark.parametrize(
    "jitter",
    [
        pytest.param([WORST_JITTER_S[k % 6] for k in range(121)], id="worst-case"),
        pytest.param(_seeded_jitter(), id="seeded-uniform"),
    ],
)
def test_30_hz_calls_with_jitter_keep_exactly_every_sixth_at_5_fps(
    server: FakeDreamAPI, client: DreamClient, jitter: list[float]
) -> None:
    """A few milliseconds of jitter on 30 Hz calls neither drops a due frame nor keeps one a tick early."""
    open_policy(client, obs_window=LONG_WINDOW, obs_fps=5)
    stamps = [(k, T0 + k / 30 + jitter[k]) for k in range(121)]
    assert kept(server, client, stamps) == list(range(0, 121, 6))


def test_a_frame_one_tick_early_is_not_kept(server: FakeDreamAPI, client: DreamClient) -> None:
    """With the slack at its largest (half the call interval), the tick before the due one is still not kept."""
    open_policy(client, obs_window=LONG_WINDOW, obs_fps=5)
    # 10 Hz calls at 5 fps: the slack is 50 ms, a tick is 100 ms; the odd ticks are each one tick early.
    assert kept(server, client, ticks(10, 21)) == list(range(0, 21, 2))


def test_the_slack_is_at_most_half_a_period(server: FakeDreamAPI, client: DreamClient) -> None:
    """A slow caller's half-interval slack (500 ms at 1 Hz) is capped at half a period (125 ms at 4 fps): an extra
    call 100 ms after a kept one, 150 ms before the next due time, is not kept; the call at the due time is."""
    open_policy(client, obs_window=LONG_WINDOW, obs_fps=EXACT_FPS)
    stamps = [(k, T0 + k) for k in range(8)]  # 1 Hz, slower than obs_fps: every call kept, each one re-anchoring
    stamps += [(8, T0 + 7.1), (9, T0 + 7.0 + EXACT_PERIOD_S)]
    assert kept(server, client, stamps) == [0, 1, 2, 3, 4, 5, 6, 7, 9]


def test_the_slack_follows_the_last_few_call_intervals(server: FakeDreamAPI, client: DreamClient) -> None:
    """The slack is sized from the last eight calls only, so it follows a caller that slows from 30 Hz to 10 Hz:
    once the 10 Hz calls fill those eight, a due tick 30 ms early is within their 50 ms slack and is kept, where
    the 30 Hz slack (17 ms) would have refused it and kept the tick after."""
    open_policy(client, obs_window=LONG_WINDOW, obs_fps=5)
    fast = ticks(30, 60)  # 0, 6, ..., 54 kept; next due at T0 + 2.0
    # Then 10 Hz from T0 + 2.0, on the schedule's ticks; from the tenth call on, each due tick arrives 30 ms early.
    slow = [(60 + m, T0 + 2.0 + m / 10 - (0.03 if m >= 10 and m % 2 == 0 else 0.0)) for m in range(30)]
    assert kept(server, client, fast + slow) == list(range(0, 60, 6)) + list(range(60, 90, 2))


def test_one_long_interval_does_not_widen_the_slack(server: FakeDreamAPI, client: DreamClient) -> None:
    """The slack is half the median of the recent intervals, so one 133 ms gap among 33 ms ones leaves it at 17 ms:
    the tick right after the gap, one tick before the due time, is not kept; the due tick is."""
    open_policy(client, obs_window=LONG_WINDOW, obs_fps=5)
    stamps = [(k, T0 + k / 30) for k in range(31) if k not in (14, 15, 16)]  # ticks 14 to 16 lost
    assert kept(server, client, stamps) == [0, 6, 12, 18, 24, 30]


def test_after_a_long_pause_the_schedule_re_anchors_on_the_next_call(
    server: FakeDreamAPI, client: DreamClient
) -> None:
    """A call a period or more late is kept and starts the schedule over from itself, rather than every call after
    the pause being kept while the old schedule catches up."""
    open_policy(client, obs_window=LONG_WINDOW, obs_fps=5)
    before = ticks(30, 13)  # 0, 6, 12 kept; due next at T0 + 0.6
    after = ticks(30, 13, start=T0 + 3.0, first=13)  # the caller paused 2.6 s
    assert kept(server, client, before + after) == [0, 6, 12, 13, 19, 25]


def test_a_frame_late_by_less_than_half_a_period_keeps_the_schedule(server: FakeDreamAPI, client: DreamClient) -> None:
    """Two camera ticks lost: the next call is kept, and the one after is kept on the original schedule."""
    open_policy(client, obs_window=LONG_WINDOW, obs_fps=5)
    stamps = [(k, T0 + k / 30) for k in range(25) if k not in (6, 7)]
    # Tick 8 (T0 + 0.267) is the first at or past due (T0 + 0.2) and within half a period of it: next due T0 + 0.4.
    assert kept(server, client, stamps) == [0, 8, 12, 18, 24]


def test_calls_that_keep_no_frame_still_replace_the_observation(server: FakeDreamAPI, client: DreamClient) -> None:
    """State is always the newest: the observation sent is the last call's even though its frame was not kept."""
    open_policy(client, obs_window=LONG_WINDOW, obs_fps=5)
    assert kept(server, client, ticks(30, 9)) == [0, 6]  # `kept` checks the observation is tick 8's


# -- reset_episode() and close() -----------------------------------------------------------------------------------


def test_reset_episode_before_policy_is_a_config_error(server: FakeDreamAPI, client: DreamClient) -> None:
    """No policy to reset."""
    with pytest.raises(DreamConfigError):
        client.reset_episode()
    assert server.requests == [] and server.ended == []


def test_reset_episode_clears_the_window_and_the_observation(server: FakeDreamAPI, client: DreamClient) -> None:
    """After a reset there is nothing to send until set_observation() is called again."""
    open_policy(client)
    client.set_observation({TOP: frame(0)}, {"tick": 0}, timestamp=T0)
    client.get_action()
    assert client.reset_episode() is None
    with pytest.raises(DreamObservationError, match="set_observation"):
        client.get_action()
    assert len(server.requests) == 1


def test_reset_episode_starts_a_new_session(server: FakeDreamAPI, client: DreamClient) -> None:
    """The reset is forwarded to the inference client: its session is released and the next request carries none."""
    open_policy(client)
    client.set_observation({TOP: frame(0)}, {"tick": 0}, timestamp=T0)
    client.get_action()
    client.get_action()
    session = server.requests[1].body["session_id"]
    assert session
    client.reset_episode()
    assert [request.body["session_id"] for request in server.ended] == [session]
    client.set_observation({TOP: frame(1)}, {"tick": 1}, timestamp=T0 + 10.0)
    client.get_action()
    assert server.requests[2].body["session_id"] is None
    assert sent(server) == {"tick": 1, TOP: [frame(1)]}


def test_reset_episode_clears_the_clock_and_the_camera_set(server: FakeDreamAPI, client: DreamClient) -> None:
    """A new episode may use other cameras and a clock that starts earlier; its first call is kept, the schedule
    runs from it."""
    open_policy(client, obs_window=LONG_WINDOW, obs_fps=EXACT_FPS)
    client.set_observation({TOP: frame(0)}, {"tick": 0}, timestamp=T0 + 5.0)
    client.reset_episode()
    client.set_observation({WRIST: frame(1, "wrist")}, {"tick": 1}, timestamp=T0)
    client.set_observation({WRIST: frame(2, "wrist")}, {"tick": 2}, timestamp=T0 + 0.1)
    client.set_observation({WRIST: frame(3, "wrist")}, {"tick": 3}, timestamp=T0 + EXACT_PERIOD_S)
    client.get_action()
    assert sent(server) == {"tick": 3, WRIST: [frame(1, "wrist"), frame(3, "wrist")]}


def test_close_releases_the_session_and_is_idempotent(server: FakeDreamAPI, client: DreamClient) -> None:
    """close() ends the session; again does nothing; afterwards every call needs a policy again."""
    open_policy(client)
    client.set_observation({TOP: frame(0)}, {"tick": 0}, timestamp=T0)
    client.get_action()
    client.get_action()
    session = server.requests[1].body["session_id"]
    client.close()
    client.close()
    assert [request.body["session_id"] for request in server.ended] == [session]
    with pytest.raises(DreamConfigError):
        client.get_action()
    with pytest.raises(DreamConfigError):
        client.set_observation({TOP: frame(1)}, {"tick": 1}, timestamp=T0 + 1.0)
    with pytest.raises(DreamConfigError):
        client.reset_episode()
    assert len(server.requests) == 2


def test_close_before_policy_does_nothing(server: FakeDreamAPI) -> None:
    """Nothing open, nothing to release, no error."""
    unopened = DreamClient()
    unopened.close()
    unopened.close()
    assert server.requests == [] and server.ended == []


def test_a_policy_after_close_starts_from_an_empty_window(server: FakeDreamAPI, client: DreamClient) -> None:
    """Nothing of the closed policy's window survives into the next one."""
    open_policy(client)
    client.set_observation({TOP: frame(0)}, {"tick": 0}, timestamp=T0)
    client.close()
    open_policy(client)
    with pytest.raises(DreamObservationError, match="set_observation"):
        client.get_action()
    assert server.requests == []


def test_it_is_a_context_manager_that_closes(server: FakeDreamAPI) -> None:
    """`with` answers the client itself and closes it on the way out."""
    with DreamClient() as managed:
        assert isinstance(managed, DreamClient)
        open_policy(managed)
        managed.set_observation({TOP: frame(0)}, {"tick": 0}, timestamp=T0)
        managed.get_action()
        managed.get_action()
    assert len(server.ended) == 1
    with pytest.raises(DreamConfigError):
        managed.get_action()


# -- threading -----------------------------------------------------------------------------------------------------


def _hold_requests(server: FakeDreamAPI) -> tuple[threading.Event, threading.Event]:
    """Makes the fake hold every get_action until released (or HOLD_S passes); answers `(arrived, release)`."""
    arrived = threading.Event()
    release = threading.Event()

    def held(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        arrived.set()
        release.wait(HOLD_S)
        return server.default_reply(body)

    server.handler = held
    return arrived, release


def _start_get_action(client: DreamClient, outcome: dict[str, Any]) -> threading.Thread:
    """Runs client.get_action() on a worker thread, recording its result or error under `outcome['result']`."""

    def work() -> None:
        try:
            outcome["result"] = client.get_action()
        except BaseException as exc:  # noqa: B036  # thread boundary: every outcome is recorded
            outcome["result"] = exc

    worker = threading.Thread(target=work, name="get-action")
    worker.start()
    return worker


def test_set_observation_never_waits_for_a_get_action_in_flight(server: FakeDreamAPI, client: DreamClient) -> None:
    """While a request is held on the gateway, set_observation() returns at once; the held request carries the
    window as it was when get_action() was called, and the next one carries the frames added meanwhile."""
    arrived, release = _hold_requests(server)
    open_policy(client, obs_window=LONG_WINDOW, obs_fps=EXACT_FPS, timeout_ms=10_000)
    client.set_observation(both(0), {"tick": 0}, timestamp=T0)
    outcome: dict[str, Any] = {}
    worker = _start_get_action(client, outcome)
    durations: list[float] = []
    try:
        assert arrived.wait(THREAD_TIMEOUT_S), "the request never reached the gateway"
        for index in range(1, 6):
            started = time.perf_counter()
            client.set_observation(both(index), {"tick": index}, timestamp=T0 + index * EXACT_PERIOD_S)
            durations.append(time.perf_counter() - started)
    finally:
        release.set()
        worker.join(THREAD_TIMEOUT_S)
    assert not worker.is_alive(), "get_action hung"
    assert isinstance(outcome["result"], dict), outcome["result"]
    assert max(durations) < PROMPT_S, durations

    assert sent(server, 0) == {"tick": 0, TOP: [frame(0, "top")], WRIST: [frame(0, "wrist")]}
    server.handler = None
    client.get_action()
    assert sent(server) == {
        "tick": 5,
        TOP: [frame(i, "top") for i in range(6)],
        WRIST: [frame(i, "wrist") for i in range(6)],
    }


def test_a_second_get_action_while_one_is_in_flight_is_refused_at_once(
    server: FakeDreamAPI, client: DreamClient
) -> None:
    """Two concurrent get_action() calls: the inference client refuses the second, promptly."""
    arrived, release = _hold_requests(server)
    open_policy(client, timeout_ms=10_000)
    client.set_observation({TOP: frame(0)}, {"tick": 0}, timestamp=T0)
    outcome: dict[str, Any] = {}
    worker = _start_get_action(client, outcome)
    try:
        assert arrived.wait(THREAD_TIMEOUT_S), "the request never reached the gateway"
        started = time.perf_counter()
        with pytest.raises(DreamError, match="another"):
            client.get_action()
        assert time.perf_counter() - started < 0.5
    finally:
        release.set()
        worker.join(THREAD_TIMEOUT_S)
    assert not worker.is_alive(), "get_action hung"
    assert isinstance(outcome["result"], dict), outcome["result"]
    assert len(server.requests) == 1


def test_a_reset_refused_during_a_get_action_leaves_the_session_and_the_window(
    server: FakeDreamAPI, client: DreamClient
) -> None:
    """reset_episode() while a get_action() is in flight is refused by the inference client and changes nothing: the
    episode goes on, its session and its window whole, and a reset once the call has returned ends it."""
    arrived, release = _hold_requests(server)
    open_policy(client, obs_window=LONG_WINDOW, obs_fps=EXACT_FPS, timeout_ms=10_000)
    client.set_observation({TOP: frame(0)}, {"tick": 0}, timestamp=T0)
    outcome: dict[str, Any] = {}
    worker = _start_get_action(client, outcome)
    try:
        assert arrived.wait(THREAD_TIMEOUT_S), "the request never reached the gateway"
        with pytest.raises(DreamError, match="in flight"):
            client.reset_episode()
    finally:
        release.set()
        worker.join(THREAD_TIMEOUT_S)
    assert not worker.is_alive(), "get_action hung"
    assert isinstance(outcome["result"], dict), outcome["result"]

    server.handler = None
    client.set_observation({TOP: frame(1)}, {"tick": 1}, timestamp=T0 + EXACT_PERIOD_S)
    client.get_action()
    session = server.requests[1].body["session_id"]
    assert session
    assert sent(server) == {"tick": 1, TOP: [frame(0), frame(1)]}
    assert server.ended == []

    client.reset_episode()
    assert [request.body["session_id"] for request in server.ended] == [session]
    with pytest.raises(DreamObservationError):
        client.get_action()


def test_windows_stay_aligned_while_another_thread_keeps_adding_frames(
    server: FakeDreamAPI, client: DreamClient
) -> None:
    """A producer thread calls set_observation() as fast as it can while get_action() runs over and over: every
    request's cameras hold the same ticks, consecutive, newest matching the observation sent with them."""
    open_policy(client, obs_window=5, obs_fps=EXACT_FPS)
    client.set_observation(both(0), {"tick": 0}, timestamp=T0)
    stop = threading.Event()
    failures: list[BaseException] = []

    def produce() -> None:
        index = 1
        try:
            while not stop.is_set():
                client.set_observation(both(index), {"tick": index}, timestamp=T0 + index * EXACT_PERIOD_S)
                index += 1
                time.sleep(0)
        except BaseException as exc:  # noqa: B036  # thread boundary: the failure is re-raised below
            failures.append(exc)

    producer = threading.Thread(target=produce, name="producer")
    producer.start()
    try:
        for _ in range(25):
            client.get_action()
    finally:
        stop.set()
        producer.join(THREAD_TIMEOUT_S)
    assert not producer.is_alive(), "set_observation hung"
    assert failures == []
    for request in server.requests:
        observation = request.body["observation"]
        top = [index_of(value) for value in observation[TOP]]
        wrist = [index_of(value) for value in observation[WRIST]]
        assert top == wrist
        newest = top[-1]
        assert top == list(range(max(0, newest - 4), newest + 1))
        assert observation["tick"] == newest


# -- only Dream errors ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("frames", "observation", "timestamp"),
    [
        pytest.param(object(), {}, T0 + 1.0, id="frames-an-object"),
        pytest.param({TOP: frame(1)}, object(), T0 + 1.0, id="observation-an-object"),
        pytest.param({TOP: frame(1)}, {}, "soon", id="timestamp-a-str"),
        pytest.param({TOP: frame(1)}, {}, [T0 + 1.0], id="timestamp-a-list"),
        pytest.param({b"video.top": frame(1)}, {}, T0 + 1.0, id="frame-key-bytes"),
        pytest.param({(TOP, 0): frame(1)}, {}, T0 + 1.0, id="frame-key-a-tuple"),
        pytest.param({TOP: bytearray(b"jpeg")}, {}, T0 + 1.0, id="frame-a-bytearray"),
        pytest.param({TOP: memoryview(b"jpeg")}, {}, T0 + 1.0, id="frame-a-memoryview"),
        pytest.param({TOP: np.frombuffer(b"jpeg", dtype=np.uint8)}, {}, T0 + 1.0, id="frame-an-array"),
        pytest.param({TOP: frame(1)}, {TOP: frame(1)}, T0 + 1.0, id="observation-holds-a-frame-key"),
    ],
)
def test_set_observation_raises_only_dream_errors(
    server: FakeDreamAPI, client: DreamClient, frames: Any, observation: Any, timestamp: Any
) -> None:
    """Whatever the caller passes, what escapes is a DreamError, and the client keeps working after it."""
    open_policy(client)
    client.set_observation({TOP: frame(0)}, {"tick": 0}, timestamp=T0)
    with pytest.raises(DreamError):
        client.set_observation(frames, observation, timestamp=timestamp)
    client.get_action()
    assert sent(server) == {"tick": 0, TOP: [frame(0)]}

"""Tests of DreamClient.on_round_trip() and `dream.DebugView`: each get_action() is handed over as one RoundTrip, a
refusal included, and the page on 127.0.0.1 serves the kept round trips and nothing else.

The policy is a stand-in (#FakePolicy) opened through `dream._open_policy`, so no request leaves the process; the
page is fetched over a real socket on 127.0.0.1.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from typing import Any

import numpy as np
import pytest

import dream
from dream import debug_view

## The policy config every test opens: no model lookup (`strict` off) and no recording, so nothing is dialed.
CONFIG = {"checkpoint": "c", "strict": False, "_record": False}

## What the stand-in policy reports as the last round trip's timing.
TIMING = {"rtt_ms": 41.5, "gateway_ms": 30, "serve_infer_ms": 25}


class FakePolicy:
    """A stand-in policy client: answers `reply`, or raises `refusal` when it is set."""

    def __init__(self) -> None:
        self.reply: dict[str, Any] = {"q_r_chunk": np.zeros((48, 7))}
        self.refusal: dream.DreamError | None = None
        self.last_timing: dict[str, Any] | None = None
        self._session_id = None

    def get_action(self, observation: dict[str, Any], *, commanded: Any = None) -> dict[str, Any]:
        self.last_timing = dict(TIMING)
        if self.refusal is not None:
            raise self.refusal
        return self.reply

    def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def isolated_credentials(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """No credentials file and no DREAM_* in the environment, so nothing of this computer's account is read."""
    monkeypatch.setenv("DREAM_CREDENTIALS", str(tmp_path / "no-credentials"))
    for name in ("DREAM_API_KEY", "DREAM_GATEWAY_URL", "DREAM_API_URL", "DREAM_DATA_URL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakePolicy:
    """The policy every DreamClient in the test opens."""
    policy = FakePolicy()
    monkeypatch.setattr(dream, "_open_policy", lambda config, api_key=None: policy)
    return policy


@pytest.fixture
def client(fake: FakePolicy) -> Any:
    """A client with the stand-in policy open and two ticks in its window."""
    opened = dream.DreamClient("dl_live_x")
    opened.policy(dict(CONFIG))
    opened.set_observation({"video.top": "frame1"}, {"prompt": "pick", "state.arm": [0.5, 0.0]}, timestamp=1.0)
    opened.set_observation({"video.top": "frame2"}, {"prompt": "pick", "state.arm": [0.5, 0.25]}, timestamp=1.2)
    yield opened
    opened.close()


def _get(url: str) -> tuple[int, bytes]:
    """The status and body of a GET to the page's server."""
    try:
        with urllib.request.urlopen(url, timeout=5) as answer:
            return answer.status, answer.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_each_get_action_hands_over_what_was_sent_and_what_came_back(client: Any, fake: FakePolicy) -> None:
    """The frames the window sent are split from the state and the prompt; the action and the timing come along."""
    trips: list[dream.RoundTrip] = []
    client.on_round_trip(trips.append)
    action = client.get_action()

    assert len(trips) == 1
    trip = trips[0]
    assert trip.frames == {"video.top": ["frame1", "frame2"]}, "the window, oldest first"
    assert trip.observation == {"prompt": "pick", "state.arm": [0.5, 0.25]}, "the newest state, without the frames"
    assert trip.action is action
    assert trip.timing == TIMING
    assert trip.error is None
    assert trip.sent_s > 0


def test_a_refusal_is_handed_over_and_still_raised(client: Any, fake: FakePolicy) -> None:
    """A refused call reaches the callable with its error and no action, then raises as before."""
    trips: list[dream.RoundTrip] = []
    client.on_round_trip(trips.append)
    fake.refusal = dream.DreamServerError("DreamAPI refused", code="DA-2000", reason="pool_warming", retryable=True)

    with pytest.raises(dream.DreamServerError) as raised:
        client.get_action()

    assert raised.value is fake.refusal
    assert [(trip.action, trip.error) for trip in trips] == [(None, fake.refusal)]


def test_no_observation_is_not_a_round_trip(fake: FakePolicy) -> None:
    """A get_action() with nothing to send raises before a request, and hands nothing over."""
    trips: list[dream.RoundTrip] = []
    with dream.DreamClient("dl_live_x") as opened:
        opened.policy(dict(CONFIG))
        opened.on_round_trip(trips.append)
        with pytest.raises(dream.DreamObservationError):
            opened.get_action()
    assert trips == []


def test_an_error_in_the_callable_comes_out_of_get_action(client: Any) -> None:
    """The callable is the caller's code: what it raises is not swallowed."""

    def broken(trip: dream.RoundTrip) -> None:
        raise RuntimeError("callable failed")

    client.on_round_trip(broken)
    with pytest.raises(RuntimeError, match="callable failed"):
        client.get_action()


def test_the_page_serves_the_kept_round_trips_on_this_computer_only(client: Any, fake: FakePolicy) -> None:
    """The page, the list and each round trip are served on 127.0.0.1; a NaN goes by name, a refusal by its code."""
    with dream.DebugView(client) as view:
        assert view.url.startswith("http://127.0.0.1:")
        assert view._server.server_address[0] == "127.0.0.1"
        assert _get(view.url + "trips") == (200, b"[]")

        client.get_action()
        client.set_observation({"video.top": "frame3"}, {"state.arm": np.array([0.5, math.nan])}, timestamp=1.4)
        fake.refusal = dream.DreamServerError("DreamAPI refused", code="DA-2000", reason="pool_warming")
        with pytest.raises(dream.DreamServerError):
            client.get_action()

        status, page = _get(view.url)
        assert status == 200 and b"What the model sees" in page
        listed = json.loads(_get(view.url + "trips")[1])
        assert [(trip["n"], trip["error"], trip["rtt_ms"]) for trip in listed] == [
            (1, None, 41.5),
            (2, "DreamServerError", 41.5),
        ]
        first = json.loads(_get(view.url + "trip/1")[1])
        assert first["frames"] == {"video.top": ["frame1", "frame2"]}
        assert first["action"]["q_r_chunk"] == [[0.0] * 7] * 48
        refused = json.loads(_get(view.url + "trip/2")[1])
        assert refused["observation"] == {"state.arm": [0.5, "nan"]}
        assert first["units"] == [] and refused["units"] == [] and first["session"] == refused["session"] == []
        assert refused["action"] is None
        assert refused["error"] == {
            "type": "DreamServerError",
            "code": "DA-2000",
            "reason": "pool_warming",
            "message": "DreamAPI refused",
        }
        assert _get(view.url + "trip/3")[0] == 404
        assert _get(view.url + "trip/x")[0] == 404
        assert _get(view.url + "other")[0] == 404
        assert view.url in view._repr_html_()


def test_the_page_keeps_only_the_newest_round_trips(client: Any) -> None:
    """Past #KEEP round trips the oldest are dropped, so memory stays bounded however long the loop runs."""
    with dream.DebugView(client) as view:
        for _ in range(debug_view.KEEP + 3):
            client.get_action()
        listed = json.loads(_get(view.url + "trips")[1])
    assert [trip["n"] for trip in listed] == list(range(4, debug_view.KEEP + 4))


def test_values_become_what_json_carries() -> None:
    """Arrays become lists and a non-finite number its name, so the page can show a NaN state instead of failing."""
    shown = debug_view._jsonable({"a": np.array([1.0, math.nan]), "b": np.float32(math.inf), "c": (1, "x")})
    assert shown == {"a": [1.0, "nan"], "b": "inf", "c": [1, "x"]}
    assert json.dumps(shown, allow_nan=False)


def test_a_state_in_other_units_is_named_and_one_in_radians_is_not() -> None:
    """Degrees or motor counts (beyond 2 pi) and a gripper outside 0 to 1 are named on the page; radians, metres, a
    NaN (named apart) and the prompt are not."""
    assert (
        debug_view._units(
            {
                "prompt": "pick",
                "state.right_arm": np.array([0.0, 1.5, -3.1, 6.2, 0.1, 0.2]),
                "state.right_gripper": [1.0],
                "state.right_eef_xyz": [0.3, -0.2, 0.4],
                "state.left_arm": [math.nan, 0.0],
                "state.left_gripper": np.array([0.03]),
            }
        )
        == []
    )
    assert debug_view._units(
        {
            "state.right_arm": [0.0, 90.0, -45.0],
            "state.left_arm": [2048.0],
            "state.right_gripper": [-1.0],
            "state.left_gripper": [0.0, 80.0],
            "state.labels": ["a"],
        }
    ) == [
        "state.right_arm reaches -45 to 90: beyond 2 pi, so not radians; degrees or motor counts?",
        "state.left_arm reaches 2.05e+03: beyond 2 pi, so not radians; degrees or motor counts?",
        "state.right_gripper is -1: a gripper is 0 closed to 1 open",
        "state.left_gripper is 0 to 80: a gripper is 0 closed to 1 open",
    ]


def _trip(n: int, observation: dict[str, Any], action: dict[str, Any] | None) -> tuple[int, debug_view.RoundTrip]:
    """Round trip `n` with this state and answer, as the view keeps it."""
    return n, debug_view.RoundTrip(
        sent_s=float(n), frames={}, observation=observation, action=action, timing=None, error=None
    )


def test_a_gripper_that_stays_open_while_its_targets_close_is_named() -> None:
    """A gripper read as 1.0 on every call while the model's grip targets fall from 0.55 to 0.25 is named once, with
    the dry-run caveat; one that follows its targets is not, and neither is anything before #MIN_CALLS calls."""

    def calls(grip: list[float]) -> list[tuple[int, debug_view.RoundTrip]]:
        return [
            _trip(
                i + 1,
                {"prompt": "pick", "state.right_gripper": [grip[i]], "state.right_arm": [0.1 * i] * 6},
                {
                    "per_key": {"action.right_grip": np.linspace(0.55, 0.25, 48)},
                    "q_r_chunk": np.full((48, 7), 0.1 * i),
                },
            )
            for i in range(len(grip))
        ]

    [line] = debug_view._still_parts(calls([1.0] * 5))
    assert line.startswith("state.right_gripper stayed the same over the last 5 calls while the model's targets")
    assert "in a dry run nothing moves, so there this is expected" in line
    assert debug_view._still_parts(calls([1.0, 0.8, 0.6, 0.5, 0.4])) == []
    assert debug_view._still_parts(calls([1.0] * 3)) == []


def test_q_chunks_stand_in_for_per_key_and_still_joints_with_still_targets_pass() -> None:
    """Without per_key, a side's gripper is the last column of its q chunk and its arm the columns before; joints
    held still under still targets are not named."""
    held = np.zeros((48, 7))
    held[:, 6] = np.linspace(1.0, 0.2, 48)
    trips = [
        _trip(i + 1, {"state.left_gripper": [1.0], "state.left_arm": [0.3] * 6}, {"q_l_chunk": held}) for i in range(4)
    ]
    assert debug_view._still_parts(trips) == [
        "state.left_gripper stayed the same over the last 4 calls while the model's targets for it moved: in a run "
        "that moves the arms, your loop is not applying those targets or not reading the part back live (in a dry run "
        "nothing moves, so there this is expected)"
    ]


def test_one_arm_sent_as_two_is_named_with_the_call_it_was_copied_from() -> None:
    """A left arm frozen at the right arm's first pose while the right arm moves is named, with that call; two arms
    moving apart, or both still, are not; two arms sending the same moving state are named as a copy."""
    right = [[0.30 + 0.01 * i, -0.20, 0.40 + 0.01 * i] for i in range(6)]
    copied = [_trip(i + 1, {"state.right_eef_xyz": right[i], "state.left_eef_xyz": right[0]}, None) for i in range(6)]
    assert debug_view._still_arm(copied) == [
        "the left arm's state has not changed at all over the last 6 calls while the right arm's has, and it "
        "matches the right arm's state on call 1: is the left arm read live? One arm sent as two, or a placeholder "
        "for a missing arm, looks like this"
    ]
    apart = [
        _trip(i + 1, {"state.right_eef_xyz": right[i], "state.left_eef_xyz": [0.3, 0.2, 0.4 - 0.01 * i]}, None)
        for i in range(6)
    ]
    assert debug_view._still_arm(apart) == []
    still = [_trip(i + 1, {"state.right_arm": [0.0] * 6, "state.left_arm": [0.0] * 6}, None) for i in range(6)]
    assert debug_view._still_arm(still) == []
    same = [_trip(i + 1, {"state.right_eef_xyz": right[i], "state.left_eef_xyz": right[i]}, None) for i in range(6)]
    assert debug_view._still_arm(same) == [
        "the right and left arms sent the same state on each of the last 6 calls while moving: is one arm's reading "
        "copied to the other?"
    ]


def test_a_loop_on_the_http_api_hands_each_call_to_add() -> None:
    """With no client, add() takes the JSON body sent and the JSON reply: frames from each `video.` key (one JPEG or a
    list), the rest as the state, the reply's `actions` and `timing`; a refusal by its code; no reply as a failure."""
    body = {
        "checkpoint": "c",
        "session_id": None,
        "observation": {"video.top": "frame1", "video.wrist": ["f1", "f2"], "prompt": "pick", "state.arm": [0.5]},
    }
    answer = {
        "session_id": "s",
        "status": {"code": "DA-0001", "name": "OK_NEW_SESSION", "reason": "new"},
        "actions": {"per_key": {"action.arm": [[0.5]] * 48}},
        "timing": {"serve_infer_ms": 25, "gateway_ms": 30},
    }
    refusal = {"status": {"code": "DA-2000", "name": "POOL_WARMING", "reason": "pool_warming"}}
    with dream.DebugView(None) as view:
        view.add(body, answer)
        view.add(body, refusal)
        view.add(body, None)
        listed = json.loads(_get(view.url + "trips")[1])
        assert [(trip["n"], trip["error"], trip["rtt_ms"]) for trip in listed] == [
            (1, None, None),
            (2, "DreamServerError", None),
            (3, "DreamError", None),
        ]
        first = json.loads(_get(view.url + "trip/1")[1])
        assert first["frames"] == {"video.top": ["frame1"], "video.wrist": ["f1", "f2"]}
        assert first["observation"] == {"prompt": "pick", "state.arm": [0.5]}
        assert first["action"] == {"per_key": {"action.arm": [[0.5]] * 48}}
        assert first["timing"] == {"serve_infer_ms": 25, "gateway_ms": 30} and first["error"] is None
        refused = json.loads(_get(view.url + "trip/2")[1])
        assert refused["action"] is None and refused["error"] == {
            "type": "DreamServerError",
            "code": "DA-2000",
            "reason": "pool_warming",
            "message": "DreamAPI refused the inference request: DA-2000 POOL_WARMING (pool_warming)",
        }
        lost = json.loads(_get(view.url + "trip/3")[1])
        assert lost["timing"] is None and lost["error"]["message"].startswith("no reply")
        with pytest.raises(TypeError, match="'observation' dict"):
            view.add({"prompt": "pick"}, answer)


def test_the_arms_are_drawn_only_from_a_motion_handed_to_show_arms() -> None:
    """Without show_arms() there is no motion (404), so the page imports nothing; with it, the motion is served as it
    stands, null before the loop's first call, beside the page's script and the YAM model and joints it draws."""
    motion: list[dict[str, Any] | None] = [None]
    with dream.DebugView(None) as view:
        assert _get(view.url + "arms")[0] == 404
        view.show_arms(lambda: motion[0])
        assert _get(view.url + "arms") == (200, b"null")
        motion[0] = {"now": 1.5, "calls": []}
        assert json.loads(_get(view.url + "arms")[1]) == motion[0]
        status, model = _get(view.url + "yam.glb")
        assert status == 200 and model[:4] == b"glTF"
        joints = json.loads(_get(view.url + "yam.joints.json")[1])
        assert [body["id"] for body in joints["bodies"]] == ["yam_st_0", "yam_st_1"]
        status, script = _get(view.url + "arms.js")
        assert status == 200 and b"export async function start" in script
        assert b"https://cdn.jsdelivr.net/npm/three@0.160.0/" in _get(view.url)[1]

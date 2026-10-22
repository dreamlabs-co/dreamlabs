"""The `robot` config key: the robot the model runs on, sent on every request as the observation's `embodiment`.

`robot` is `"yam"` or `"other_embodiment"`. The client adds it to the observation after the strict checks, whichever
path built the observation (the inference client's get_action(observation), or DreamClient's set_observation() and
get_action()), and refuses an observation that carries `embodiment` itself. On `"other_embodiment"` the serve answers
the joint head as null and packs no joint chunks, and strict mode takes that reply. Left out, `robot` is sent only
where `ik` sends a model trained on YAM arms zeros for another arm's joints (`test_client_ik.py`); everywhere else
nothing is sent and the body is byte for byte the one without `robot`. These tests run against `FakeDreamAPI` and
need no IK.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from conftest import CHECKPOINT, FRAME_B64, FakeDreamAPI

import dream
from dream import transport
from dream.errors import DreamActionError, DreamConfigError, DreamObservationError

## Rows of every reply's parts here.
ROWS = 4

## Arm sides, right first, as the serve names them.
SIDES = ("right", "left")

## The cameras a model trained on YAM arms reads.
CAMERAS = ("overhead", "wrist_l", "wrist_r")


def _spans(parts: tuple[tuple[str, int], ...]) -> dict[str, list[int]]:
    """Each side's parts laid end to end, right first: `<side>_<part>` to its `[start, end)` span."""
    out: dict[str, list[int]] = {}
    start = 0
    for side in SIDES:
        for part, width in parts:
            out[f"{side}_{part}"] = [start, start + width]
            start += width
    return out


## A loop on its own IK (no `ik` key) running a model trained on YAM arms on another arm: the model's six joint slots
## (sent as zeros), the gripper as both `<side>_gripper` and `<side>_grip`, and each hand's pose from the loop's own
## FK; the action parts are the joint head (null on `other_embodiment`), the grip and each hand's pose.
OWN_IK_BODY: dict[str, Any] = {
    "state": _spans((("arm", 6), ("gripper", 1), ("grip", 1), ("eef_xyz", 3), ("eef_rot6", 6))),
    "action": _spans((("arm", 6), ("grip", 1), ("eef_xyz", 3), ("eef_rot6", 6))),
    "video": {role: f"observation.images.{role}" for role in CAMERAS},
}


def own_ik_observation() -> dict[str, Any]:
    """What that loop sends: zero joints, its grip twice, its own FK's hand poses, the three cameras and a prompt."""
    observation: dict[str, Any] = {"prompt": "pick up the cup"}
    for side in SIDES:
        observation[f"state.{side}_arm"] = np.zeros(6, dtype=np.float32)
        observation[f"state.{side}_gripper"] = np.array([0.4], dtype=np.float32)
        observation[f"state.{side}_grip"] = np.array([0.4], dtype=np.float32)
        observation[f"state.{side}_eef_xyz"] = np.array([0.3, 0.1, 0.2], dtype=np.float32)
        observation[f"state.{side}_eef_rot6"] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    observation.update({f"video.{role}": [FRAME_B64] for role in CAMERAS})
    return observation


def withheld_reply() -> dict[str, Any]:
    """The serve's answer on `other_embodiment`: each hand's poses and grip, the joint head null, no joint chunks."""
    per_key: dict[str, Any] = {}
    for side in SIDES:
        per_key[f"action.{side}_eef_xyz"] = np.tile([0.3, 0.1, 0.2], (ROWS, 1)).tolist()
        per_key[f"action.{side}_eef_rot6"] = np.tile([1.0, 0, 0, 0, 1, 0], (ROWS, 1)).tolist()
        per_key[f"action.{side}_grip"] = [0.5] * ROWS
        per_key[f"action.{side}_arm"] = None
    return {"per_key": per_key}


def serve(server: FakeDreamAPI, actions: dict[str, Any]) -> None:
    """Makes the fake answer every request with its usual session handling and `actions` as the action dict."""

    def handler(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        status, reply = server.default_reply(body)
        reply["actions"] = copy.deepcopy(actions)
        return status, reply

    server.handler = handler


def test_without_robot_the_request_is_byte_for_byte_as_before(
    server: FakeDreamAPI, config: dict[str, Any], observation: dict[str, Any]
) -> None:
    """No `robot` and no `ik` (a YAM station's joint loop): the observation on the wire is the caller's own."""
    with dream._open_policy(config) as client:
        client.get_action(observation)

    assert "embodiment" not in server.requests[0].body["observation"]
    assert b'"observation":' + transport.encode(observation) + b"}" in server.requests[0].raw


@pytest.mark.parametrize("robot", ["yam", "other_embodiment"])
def test_a_robot_given_rides_in_every_request_and_strict_mode_does_not_refuse_it(
    server: FakeDreamAPI, config: dict[str, Any], observation: dict[str, Any], robot: str
) -> None:
    """Strict mode on a shape spec: the observation passes as the caller built it, and every request carries `robot`
    as `embodiment` beside the caller's keys, whose own dict is not changed."""
    with dream._open_policy({**config, "robot": robot}) as client:
        assert client._checks is not None
        client.get_action(observation)
        client.get_action(observation)

    for request in server.requests:
        sent = request.body["observation"]
        assert sent["embodiment"] == robot
        assert set(sent) == set(observation) | {"embodiment"}
        assert sent["prompt"] == observation["prompt"]
    assert "embodiment" not in observation


def test_on_your_own_ik_other_embodiment_takes_the_null_joint_head_in_strict_mode(server: FakeDreamAPI) -> None:
    """Route B (no `ik`): with `robot` `other_embodiment` the strict checks do not require the joint head the serve
    answers as null, and the reply comes back as it came, the hand poses and no joint chunks; without `robot` the same
    reply is refused, naming the joint head, as a mis-shaped part."""
    serve(server, withheld_reply())
    config = {
        "host": server.host,
        "port": server.port,
        "scheme": "http",
        "timeout_ms": 2000,
        "checkpoint": CHECKPOINT,
        "embodiment": copy.deepcopy(OWN_IK_BODY),
    }
    with dream._open_policy({**config, "robot": "other_embodiment"}) as client:
        action = client.get_action(own_ik_observation())

    assert server.requests[0].body["observation"]["embodiment"] == "other_embodiment"
    assert "q_r_chunk" not in action and "q_l_chunk" not in action
    for side in SIDES:
        assert action["per_key"][f"action.{side}_arm"] is None
        assert np.asarray(action["per_key"][f"action.{side}_eef_xyz"]).shape == (ROWS, 3)
    with dream._open_policy(config) as client, pytest.raises(DreamActionError, match=r"action\.(right|left)_arm"):
        client.get_action(own_ik_observation())


@pytest.mark.parametrize("value", ["YAM", "other", "", None, 1, ["yam"], {"robot": "yam"}])
def test_a_robot_that_is_neither_value_is_a_config_error_at_policy(server: FakeDreamAPI, value: Any) -> None:
    """Any other value, an explicit None included, is refused at policy() naming the key and the two it takes,
    before anything is sent."""
    message = r"config key 'robot' must be one of \['yam', 'other_embodiment'\]"
    with pytest.raises(DreamConfigError, match=message):
        dream._open_policy({"checkpoint": CHECKPOINT, "robot": value})
    with dream.DreamClient() as client, pytest.raises(DreamConfigError, match=message):
        client.policy({"checkpoint": CHECKPOINT, "strict": False, "_record": False, "robot": value})
    assert server.requests == []


@pytest.mark.parametrize("robot", [None, "yam", "other_embodiment"])
def test_an_observation_that_carries_embodiment_is_refused_naming_robot(
    server: FakeDreamAPI, config: dict[str, Any], observation: dict[str, Any], robot: str | None
) -> None:
    """`embodiment` is the client's to send, with or without a `robot`: the caller's is refused before anything is
    sent, and the message points at `robot`."""
    keys = config if robot is None else {**config, "robot": robot}
    with dream._open_policy(keys) as client:
        for value in ("yam", "other_embodiment"):
            with pytest.raises(DreamObservationError, match=r"observation key 'embodiment'.*config key 'robot'"):
                client.get_action({**observation, "embodiment": value})
    assert server.requests == []


def test_set_observation_and_get_action_carry_the_robot_and_refuse_the_callers_embodiment(
    server: FakeDreamAPI,
) -> None:
    """The facade path: every get_action() sends the window and the latest observation with `robot` as `embodiment`;
    an observation handed to set_observation() with `embodiment` is refused at get_action(), naming `robot`."""
    state = {"prompt": "pick up the cup", "state.right_arm": [0.0] * 6}
    with dream.DreamClient() as client:
        client.policy(
            {
                "checkpoint": CHECKPOINT,
                "strict": False,
                "timeout_ms": 2000,
                "_record": False,
                "robot": "other_embodiment",
            }
        )
        client.set_observation({"video.overhead": FRAME_B64}, state, timestamp=100.0)
        client.get_action()
        client.set_observation({"video.overhead": FRAME_B64}, state, timestamp=101.0)
        client.get_action()
        client.set_observation({"video.overhead": FRAME_B64}, {**state, "embodiment": "yam"}, timestamp=102.0)
        with pytest.raises(DreamObservationError, match=r"observation key 'embodiment'.*config key 'robot'"):
            client.get_action()

    assert len(server.requests) == 2
    for request in server.requests:
        sent = request.body["observation"]
        assert sent["embodiment"] == "other_embodiment"
        assert sent["video.overhead"] and sent["state.right_arm"] == [0.0] * 6

"""Contract tests C-7k..C-7w: strict mode driven by an embodiment config, the one `add_embodiment` takes.

A checkpoint fine-tuned on a body is called with the keys of that body's config: `state.<part>` arrays as wide as each
part's span, `video.<role>` frame windows, `prompt`; its action comes back as `per_key` rows, one `action.<part>` per
action part. Passing the config as `embodiment` makes the client check exactly that, locally. Every shape spec keeps
its meaning, so the YAM station keys the shipped control loop sends are checked as a spec.
"""

from __future__ import annotations

import copy
import datetime
import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from conftest import EMBODIMENT, FRAME_B64, HORIZON, FakeDreamAPI
from shared.datasets import conversion, embodiments

import dream
from dream import client as dream_client
from dream import datasets, models
from dream.errors import DreamActionError, DreamConfigError, DreamObservationError

## The 14-D bimanual YAM body the tutorial registers (six joints and a gripper per arm, three cameras).
YAM_CONFIG: dict[str, Any] = {
    "state": {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]},
    "action": {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]},
    "video": {
        "overhead": "observation.images.overhead",
        "wrist_l": "observation.images.wrist_l",
        "wrist_r": "observation.images.wrist_r",
    },
    "head_camera": "overhead",
}

## A one-arm body with names no YAM key shares: the shape of any custom robot.
ARM_CONFIG: dict[str, Any] = {
    "state": {"arm": [0, 6], "gripper": [6, 7]},
    "action": {"arm": [0, 6], "gripper": [6, 7]},
    "video": {"top": "observation.images.top", "wrist": "observation.images.wrist"},
}

## The `embodiment` spec the shipped YAM control loop (dream-robots, `adapter.embodiment_spec`) builds for 48 rows.
STATION_SPEC: dict[str, Any] = {
    "state": {
        "arm_r.joint_pos": {"shape": [6], "dtype": "float32"},
        "arm_r.gripper_pos": {"shape": [1], "dtype": "float32"},
        "arm_l.joint_pos": {"shape": [6], "dtype": "float32"},
        "arm_l.gripper_pos": {"shape": [1], "dtype": "float32"},
    },
    "image_window": {"top": {}, "left": {}, "right": {}},
    "action": {
        "q_r_chunk": {"shape": [48, 7], "dtype": "float32"},
        "q_l_chunk": {"shape": [48, 7], "dtype": "float32"},
    },
}


def contract_observation(config: dict[str, Any]) -> dict[str, Any]:
    """An observation keyed by an embodiment config: float32 state per part, a one-frame window per role, a prompt."""
    observation: dict[str, Any] = {}
    for part, (start, end) in config["state"].items():
        observation[f"state.{part}"] = np.zeros(end - start, dtype=np.float32)
    for role in config["video"]:
        observation[f"video.{role}"] = [FRAME_B64]
    observation["prompt"] = "pick up the cup"
    return observation


def parts_reply(config: dict[str, Any], rows: int = HORIZON) -> dict[str, Any]:
    """The `per_key` a serve answers for a body: one `action.<part>` per action part, a width-1 part flat as `[H]`."""
    per_key: dict[str, Any] = {}
    for part, (start, end) in config["action"].items():
        width = end - start
        per_key[f"action.{part}"] = [0.5] * rows if width == 1 else [[0.25] * width for _ in range(rows)]
    return {"per_key": per_key}


def serve_actions(server: FakeDreamAPI, actions: Callable[[], dict[str, Any]]) -> None:
    """Makes the fake answer every request with its usual session handling and `actions()` as the action dict."""

    def handler(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        status, reply = server.default_reply(body)
        reply["actions"] = actions()
        return status, reply

    server.handler = handler


@pytest.fixture
def connection(server: FakeDreamAPI) -> dict[str, Any]:
    """The connection keys and the checkpoint against `server`, without an embodiment."""
    return {"host": server.host, "port": server.port, "scheme": "http", "timeout_ms": 2000, "checkpoint": "ft"}


@pytest.mark.parametrize("config", [YAM_CONFIG, ARM_CONFIG], ids=["yam", "one_arm"])
def test_c07k_embodiment_config_drives_the_checks(
    server: FakeDreamAPI, connection: dict[str, Any], config: dict[str, Any]
) -> None:
    """C-7k: a config as `embodiment` passes its own keys; the body is what `strict=False` sends; per_key returns."""
    serve_actions(server, lambda: parts_reply(config))
    observation = contract_observation(config)

    with dream._open_policy({**connection, "embodiment": config}) as strict_client:
        action = strict_client.get_action(observation)
    with dream._open_policy({**connection, "embodiment": config, "strict": False}) as loose_client:
        loose_client.get_action(observation)

    assert server.requests[0].body["observation"] == server.requests[1].body["observation"]
    assert set(server.requests[0].body["observation"]) == set(observation)
    assert action == parts_reply(config), "a reply without q_r_chunk / q_l_chunk passes through untouched"


@pytest.mark.parametrize("key", ["state.gripper", "video.wrist"])
def test_c07l_a_missing_part_or_role_is_named_and_nothing_is_sent(
    server: FakeDreamAPI, connection: dict[str, Any], key: str
) -> None:
    """C-7l: every part and every role of the config is required; the error names the one missing."""
    observation = contract_observation(ARM_CONFIG)
    del observation[key]

    with dream._open_policy({**connection, "embodiment": ARM_CONFIG}) as client:
        with pytest.raises(DreamObservationError, match=re.escape(key)):
            client.get_action(observation)
    assert server.requests == []


def test_c07m_a_wrong_width_or_a_bare_frame_is_refused(server: FakeDreamAPI, connection: dict[str, Any]) -> None:
    """C-7m: a state part is as wide as its span; a camera is a frame window, never one bare base64 string."""
    with dream._open_policy({**connection, "embodiment": ARM_CONFIG}) as client:
        wide = {**contract_observation(ARM_CONFIG), "state.arm": np.zeros(7, dtype=np.float32)}
        with pytest.raises(DreamObservationError, match=r"state\.arm.*\(7,\).*\(6,\)"):
            client.get_action(wide)
        scalar = {**contract_observation(ARM_CONFIG), "state.gripper": 0.5}
        with pytest.raises(DreamObservationError, match=r"state\.gripper.*\(\).*\(1,\)"):
            client.get_action(scalar)
        bare = {**contract_observation(ARM_CONFIG), "video.top": FRAME_B64}
        with pytest.raises(DreamObservationError, match=r"video\.top.*frame window"):
            client.get_action(bare)
    assert server.requests == []


@pytest.mark.parametrize("station_key", ["state.arm_r.joint_pos", "image_window.top"])
def test_c07n_station_keys_are_never_mixed_into_a_config_observation(
    server: FakeDreamAPI, connection: dict[str, Any], station_key: str
) -> None:
    """C-7n: a YAM station key beside the config's keys is refused locally; the serve never sees the two mixed."""
    observation = {**contract_observation(YAM_CONFIG), station_key: [FRAME_B64]}

    with dream._open_policy({**connection, "embodiment": YAM_CONFIG}) as client:
        with pytest.raises(DreamObservationError, match=re.escape(station_key)):
            client.get_action(observation)
    assert server.requests == []


@pytest.mark.parametrize(
    ("actions", "message"),
    [
        ({"q_r_chunk": [[0.0] * 7] * HORIZON}, "no 'per_key' dict"),
        ({"per_key": {"action.arm": [[0.0] * 6] * HORIZON}}, r"lacks parts.*action\.gripper"),
        (
            {"per_key": {"action.arm": [[0.0] * 5] * HORIZON, "action.gripper": [0.5] * HORIZON}},
            rf"\({HORIZON}, 5\).*\(H, 6\)",
        ),
        ({"per_key": {"action.arm": [[0.0] * 6, [0.0] * 5], "action.gripper": [0.5] * 2}}, "rows of one width"),
        ({"per_key": {"action.arm": [[0.0] * 6] * HORIZON, "action.gripper": [0.5] * 3}}, "3 rows where"),
        ({"per_key": {"action.arm": [], "action.gripper": []}}, r"action\.arm.*\(0,\)"),
    ],
    ids=["no_per_key", "missing_part", "wrong_width", "ragged", "rows_disagree", "empty"],
)
def test_c07o_per_key_is_checked_against_the_action_parts(
    server: FakeDreamAPI, connection: dict[str, Any], actions: dict[str, Any], message: str
) -> None:
    """C-7o: the reply's per_key must carry every action part as rows of its width; otherwise nothing is returned."""
    serve_actions(server, lambda: actions)

    with dream._open_policy({**connection, "embodiment": ARM_CONFIG}) as client:
        with pytest.raises(DreamActionError, match=message):
            client.get_action(contract_observation(ARM_CONFIG))


def test_c07p_width_one_parts_extra_entries_and_yam_chunks_pass(
    server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """C-7p: a gripper as `[H]` or `[H, 1]`, per_key entries beyond the config, and the YAM chunks all pass."""

    def reply() -> dict[str, Any]:
        actions = parts_reply(YAM_CONFIG)
        actions["per_key"]["action.left_gripper"] = [[0.5]] * HORIZON
        actions["per_key"]["action.rng_probe"] = [0.0]
        actions["q_r_chunk"] = [[0.0] * 7 for _ in range(HORIZON)]
        actions["q_l_chunk"] = [[0.1] * 7 for _ in range(HORIZON)]
        return actions

    serve_actions(server, reply)

    with dream._open_policy({**connection, "embodiment": YAM_CONFIG}) as client:
        action = client.get_action(contract_observation(YAM_CONFIG))

    assert action["per_key"] == reply()["per_key"]
    assert np.shape(action["q_r_chunk"]) == (HORIZON, 7) and np.shape(action["q_l_chunk"]) == (HORIZON, 7)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"state": {"arm": [0, 6], "gripper": [7, 8]}}, r"nothing covers \[6, 7\)"),
        ({"fps": 30}, "Unknown key 'fps'"),
        ({"video": {"Top": "observation.images.top"}}, "role 'Top' is not a name"),
        ({"head_camera": "side"}, "'side' is not one of the video roles"),
    ],
    ids=["gap", "unknown_key", "bad_role", "bad_head_camera"],
)
def test_c07q_a_bad_embodiment_config_is_a_config_error_at_init(
    connection: dict[str, Any], change: dict[str, Any], message: str
) -> None:
    """C-7q: a bad config fails at init with the line `add_embodiment` would refuse it with; strict off reads none."""
    config = {**ARM_CONFIG, **change}

    with pytest.raises(DreamConfigError, match=f"read as an embodiment config.*{message}"):
        dream._open_policy({**connection, "embodiment": config})
    dream._open_policy({**connection, "embodiment": config, "strict": False}).close()


## ROBO-T1 as its body JSON lays it out: two 7-joint arms, each with a parallel gripper, and a 2-joint head.
ROBO_T1_CONFIG: dict[str, Any] = {
    "state": {
        "right_arm": [0, 7],
        "right_gripper": [7, 8],
        "left_arm": [8, 15],
        "left_gripper": [15, 16],
        "head": [16, 18],
    },
    "action": {
        "right_arm": [0, 7],
        "right_gripper": [7, 8],
        "left_arm": [8, 15],
        "left_gripper": [15, 16],
        "head": [16, 18],
    },
    "video": {
        "overhead": "observation.images.overhead",
        "wrist_l": "observation.images.wrist_l",
        "wrist_r": "observation.images.wrist_r",
    },
}


def test_c07v_a_body_of_seven_joint_arms_and_a_head_runs_on_each_arms_own_joints(
    server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """C-7v: ROBO-T1's parts go out as wide as their spans; `commanded` takes each arm's seven joints or its written
    row of eight and refuses six, naming seven; the arms' chunks come back joined, `[H, 7 + 1]`; the head's rows pass
    through untouched."""

    def reply() -> dict[str, Any]:
        actions = parts_reply(ROBO_T1_CONFIG)
        actions["per_key"]["action.head"] = [[0.1 * row, -0.1 * row] for row in range(HORIZON)]
        actions["q_r_chunk"] = [[0.1] * 8 for _ in range(HORIZON)]
        actions["q_l_chunk"] = [[0.2] * 8 for _ in range(HORIZON)]
        return actions

    serve_actions(server, reply)
    observation = contract_observation(ROBO_T1_CONFIG)
    observation["state.right_arm"] = np.full(7, 0.1, dtype=np.float32)
    observation["state.left_arm"] = np.full(7, 0.2, dtype=np.float32)

    with dream._open_policy({**connection, "embodiment": ROBO_T1_CONFIG}) as client:
        first = client.get_action(observation)
        commanded = {"q_r_chunk": np.full(8, 0.1), "q_l_chunk": np.full(7, 0.2)}
        second = client.get_action(observation, commanded=commanded)
        refused = r"commanded 'q_r_chunk' must be the arm's 7 joint angles, or its written row of 8"
        with pytest.raises(DreamObservationError, match=refused):
            client.get_action(observation, commanded={"q_r_chunk": np.zeros(6)})

    assert len(server.requests) == 2
    sent = server.requests[0].body["observation"]
    assert (len(sent["state.right_arm"]), len(sent["state.left_arm"]), len(sent["state.head"])) == (7, 7, 2)
    for action in (first, second):
        assert np.allclose(action["q_r_chunk"], 0.1) and np.allclose(action["q_l_chunk"], 0.2)
        assert np.shape(action["q_r_chunk"]) == np.shape(action["q_l_chunk"]) == (HORIZON, 7 + 1)
        assert action["per_key"]["action.head"] == reply()["per_key"]["action.head"]


## A customer's own body, its parts named freely: a 7-joint arm named `arm`, a 6-value hand, a 2-joint torso, a
## one-value grip and a quaternion, none of them a YAM's `<side>_arm`.
OWN_BODY: dict[str, Any] = {
    "state": {"arm": [0, 7], "hand": [7, 13], "torso": [13, 15], "grip": [15, 16], "tool_quat": [16, 20]},
    "action": {"arm": [0, 7], "hand": [7, 13], "torso": [13, 15], "grip": [15, 16], "tool_quat": [16, 20]},
    "video": {"overhead": "observation.images.overhead"},
}


def test_c07w_a_customer_bodys_parts_come_back_as_the_model_sent_them_chunk_after_chunk(
    server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """C-7w: no part of a body's `per_key` is joined or filtered, whatever its width or name, also when the measured
    pose trails the rows: a join anchored there would step a joint back, take a grip past 0..1 and a quaternion off
    unit length."""
    served: list[dict[str, Any]] = []

    def reply() -> dict[str, Any]:
        k = len(served)
        actions = parts_reply(OWN_BODY)
        actions["per_key"]["action.arm"] = [[0.1 * (k * HORIZON + row)] * 7 for row in range(HORIZON)]
        actions["per_key"]["action.grip"] = [1.0, 1.0, 0.0, 0.0] if k % 2 else [0.0, 0.0, 1.0, 1.0]
        actions["per_key"]["action.tool_quat"] = [[0.0, 0.0, np.sin(0.1 * r), np.cos(0.1 * r)] for r in range(HORIZON)]
        served.append(copy.deepcopy(actions["per_key"]))
        return actions

    serve_actions(server, reply)
    observation = contract_observation(OWN_BODY)
    with dream._open_policy({**connection, "embodiment": OWN_BODY}) as client:
        for k in range(3):
            observation["state.arm"] = np.full(7, 0.1 * (k * HORIZON - 3), dtype=np.float32)  # three rows behind
            assert client.get_action(observation)["per_key"] == served[k]
    assert len(served) == 3


def test_c07r_every_shape_spec_keeps_its_meaning(server: FakeDreamAPI, connection: dict[str, Any]) -> None:
    """C-7r: a spec is never read as a config: the station spec, the fixtures' spec, `{}`, and a spec whose groups are
    `state`, `action` and `video` (where `video.*` stays an array, with no frame-window rule)."""
    video_spec = {
        "state": {"arm": {"shape": [6]}},
        "action": {"q_r_chunk": {}},
        "video": {"top": {"shape": [2, 2, 3], "dtype": "uint8"}},
    }
    for spec in (STATION_SPEC, EMBODIMENT, {}, video_spec):
        assert not dream_client._is_embodiment_config(spec)
    for config in (YAM_CONFIG, ARM_CONFIG, {**ARM_CONFIG, "fps": 30}):
        assert dream_client._is_embodiment_config(config)

    with dream._open_policy({**connection, "embodiment": video_spec}) as client:
        client.get_action({"state.arm": np.zeros(6), "video.top": np.zeros((2, 2, 3), dtype=np.uint8)})
    with dream._open_policy({**connection, "embodiment": {}}) as client:
        client.get_action({"anything.at_all": [1]})
    assert len(server.requests) == 2


## A fine-tune of the one-arm body, as get_models() answers it: the body's config rides on the model.
FT_CHECKPOINT = "dream-1-ft:1d845dea-arm"
FT_EMBODIMENT_ID = "emb_0123456789abcdef"


def _account(
    monkeypatch: pytest.MonkeyPatch,
    *,
    owner: str,
    config: dict[str, Any] | None,
    calls: list[str] | None = None,
) -> None:
    """Makes the account's models the one-arm fine-tune, trained on a body with `config` (None: a builtin body); the
    account owns no embodiment, as for a company running a fine-tune whose body Dream Labs registered. Records each
    list call in `calls`."""
    made = datetime.datetime(2026, 10, 1, tzinfo=datetime.timezone.utc)

    def list_models(**kwargs: Any) -> list[models.Model]:
        if calls is not None:
            calls.append("models")
        roles = None if config is None else list(config["video"])
        return [
            models.Model(FT_CHECKPOINT, owner, "dreamlabs/dream-1-beta", FT_EMBODIMENT_ID, True, made, roles, config)
        ]

    def list_embodiments(**kwargs: Any) -> list[datasets.Embodiment]:
        if calls is not None:
            calls.append("embodiments")
        return []

    monkeypatch.setattr(models, "list_models", list_models)
    monkeypatch.setattr(datasets, "list_embodiments", list_embodiments)


def _send(client: dream.DreamClient, observation: dict[str, Any]) -> dict[str, Any]:
    """Hands `observation` to the client as one tick (each frame key's last frame as the frame) and asks for an action.

    @param client The client, its policy open.
    @param observation An observation as the model reads it, frame keys holding windows; a tick with no frame key gets
        one stand-in camera, since set_observation() takes at least one.
    @return What get_action() returned.
    """
    frames = {key: value[-1] for key, value in observation.items() if key.startswith(("video.", "image_window."))}
    rest = {key: value for key, value in observation.items() if key not in frames}
    client.set_observation(frames or {"video.stand_in": FRAME_B64}, rest)
    return client.get_action()


def test_c07s_policy_loads_the_fine_tunes_config_and_checks_before_sending(
    server: FakeDreamAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C-7s: policy() on a fine-tune the account runs loads its embodiment config once, from the model, and keeps it
    for the session, also when another account holds the body: an observation missing a part is refused locally with
    nothing sent, a correct one goes through, and the lookup is one control-plane call at policy(), cached per
    checkpoint, none per get_action()."""
    calls: list[str] = []
    _account(monkeypatch, owner="mine", config=ARM_CONFIG, calls=calls)
    serve_actions(server, lambda: parts_reply(ARM_CONFIG))
    client = dream.DreamClient("dl_live_x")

    client.policy({"checkpoint": FT_CHECKPOINT})
    wrong = contract_observation(ARM_CONFIG)
    del wrong["state.gripper"]
    with pytest.raises(DreamObservationError, match=re.escape("state.gripper")):
        _send(client, wrong)
    assert server.requests == [], "the bad observation never left the machine"

    assert _send(client, contract_observation(ARM_CONFIG)) == parts_reply(ARM_CONFIG)
    client.policy({"checkpoint": FT_CHECKPOINT})
    client.close()
    assert calls == ["models"], "looked up once, at the first policy(), and never in the account's own bodies"
    assert len(server.requests) == 1


def test_c07t_explicit_config_strict_off_public_models_and_failed_lookups(
    server: FakeDreamAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C-7t: an explicit `embodiment` or `strict=False` is never looked up; a model of a builtin body has no config
    to load, so the checks stay off as before."""
    calls: list[str] = []
    _account(monkeypatch, owner="mine", config=ARM_CONFIG, calls=calls)
    serve_actions(server, lambda: parts_reply(YAM_CONFIG))  # the later calls run unchecked, so any reply passes
    yam_keys = contract_observation(YAM_CONFIG)

    client = dream.DreamClient("dl_live_x")
    client.policy({"checkpoint": FT_CHECKPOINT, "embodiment": YAM_CONFIG})
    _send(client, yam_keys)  # checked against the explicit config, not the one-arm body
    client.policy({"checkpoint": FT_CHECKPOINT, "strict": False})
    _send(client, {"anything": 1})
    client.close()
    assert calls == [], "an explicit config and strict=False skip the lookup"

    public: list[str] = []
    _account(monkeypatch, owner="public", config=None, calls=public)
    client = dream.DreamClient("dl_live_x")
    client.policy({"checkpoint": FT_CHECKPOINT})
    _send(client, {"anything": 1})
    client.close()
    assert public == ["models"], "a builtin body has no config: nothing to check against"
    assert len(server.requests) == 3


@pytest.mark.parametrize(
    "failure",
    [
        dream.DreamServerError("platform unreachable", code="unavailable", retryable=True),
        dream.DreamProtocolError("could not reach https://platform.test: Connection refused"),
    ],
    ids=["refused", "unreachable"],
)
def test_a_failed_model_lookup_with_strict_on_raises_and_sends_nothing(
    server: FakeDreamAPI, monkeypatch: pytest.MonkeyPatch, failure: dream.DreamError
) -> None:
    """A parts lookup failure must not run unconverted: GET /v1/models failing at policy() (a platform restarting
    during a deploy) must not warn and run unchecked and unconverted, a robot with parts getting the trained 0-1
    grip. With `strict` on it raises, nothing opens and nothing is sent; strict=False still sends unchecked."""
    serve_actions(server, lambda: parts_reply(ARM_CONFIG))

    def failing(**kwargs: Any) -> list[models.Model]:
        raise failure

    monkeypatch.setattr(models, "list_models", failing)
    client = dream.DreamClient("dl_live_x")
    with pytest.raises(dream.DreamProtocolError, match="could not look up checkpoint") as raised:
        client.policy({"checkpoint": FT_CHECKPOINT})
    assert raised.value.__cause__ is failure
    assert "call policy() again" in str(raised.value)
    with pytest.raises(dream.DreamError):
        _send(client, {"anything": 1})
    assert server.requests == [], "nothing is sent"
    client.policy({"checkpoint": FT_CHECKPOINT, "strict": False})
    _send(client, {"anything": 1})
    client.close()
    assert len(server.requests) == 1, "strict=False skips the lookup and sends unchecked, as documented"


def test_a_checkpoint_the_listing_does_not_name_with_strict_on_raises_and_sends_nothing(
    server: FakeDreamAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint absent from GET /v1/models got no config and ran unchecked, so a
    robot with parts unconverted. With `strict` on it raises, naming the checkpoint, and nothing is sent; strict=False
    still sends unchecked. Every deployed gateway serves a key only what its listing shows, so nobody loses a model."""
    serve_actions(server, lambda: parts_reply(ARM_CONFIG))
    calls: list[str] = []
    _account(monkeypatch, owner="mine", config=ARM_CONFIG, calls=calls)
    unlisted = "dream-1-ft:00000000-not-listed"
    client = dream.DreamClient("dl_live_x")
    with pytest.raises(dream.DreamConfigError, match=f"checkpoint '{unlisted}' is not among the models"):
        client.policy({"checkpoint": unlisted})
    with pytest.raises(dream.DreamError):
        _send(client, {"anything": 1})
    assert server.requests == [], "nothing is sent"
    with pytest.raises(dream.DreamConfigError, match="is not among the models"):
        client.policy({"checkpoint": unlisted})
    assert calls == ["models", "models"], "nothing is cached, so the next policy() looks again"
    client.policy({"checkpoint": unlisted, "strict": False})
    _send(client, {"anything": 1})
    client.close()
    assert len(server.requests) == 1, "strict=False skips the lookup and sends unchecked, as documented"


def _parts_config(format: str) -> dict[str, Any]:
    """A two-arm robot with `parts` whose gripper poses train: per side its grasp point, its rotation in `format`, a
    jaw closed at 0.08 and open at 0.01 and six joints, then a 2-wide `other` part; the same parts in state and
    action."""
    width = embodiments.ROTATION_FORMATS[format]
    widths = {"xyz": 3, "rot": width, "jaw": 1, "joints": 6}
    layout: dict[str, list[int]] = {}
    parts: dict[str, Any] = {}
    for side in ("left", "right"):
        for name, part_width in widths.items():
            start = max((end for _, end in layout.values()), default=0)
            layout[f"{side[0]}_{name}"] = [start, start + part_width]
        parts[f"{side[0]}_xyz"] = {"kind": "position", "side": side, "point": "grasp", "frame": "arm_base"}
        parts[f"{side[0]}_rot"] = {"kind": "rotation", "side": side, "format": format}
        parts[f"{side[0]}_jaw"] = {"kind": "gripper", "side": side, "closed": 0.08, "open": 0.01}
        parts[f"{side[0]}_joints"] = {"kind": "arm", "side": side}
    end = max(end for _, end in layout.values())
    layout["base"] = [end, end + 2]
    parts["base"] = {"kind": "other"}
    return {
        "state": layout,
        "action": dict(layout),
        "video": {"top": "observation.images.top"},
        "parts": {"state": parts, "action": dict(parts)},
        "actions": "absolute",
    }


def _answer(body: embodiments.Embodiment) -> dict[str, list[list[float]]]:
    """Two rows of every action slot the plan trains: 0.5 in each value, but a 6D the identity's first two columns
    (0.5 in all six is two parallel vectors, which is no rotation)."""
    assert body.plan is not None
    return {
        f"action.{slot.name}": [
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] if slot.width == 6 and slot.name.endswith("_rot6") else [0.5] * slot.width
        ]
        * 2
        for slot in body.plan.action
    }


def _own_rows(config: dict[str, Any], vector: str, rows: int) -> dict[str, np.ndarray]:
    """`rows` frames of each part of the robot's `vector` in its own units: Euler angles in their one form, jaws
    between their ends, the rest seeded numbers."""
    rng = np.random.default_rng(11)
    out: dict[str, np.ndarray] = {}
    for part, (start, end) in config[vector].items():
        values = rng.uniform(-1, 1, size=(rows, end - start))
        if part.endswith("_rot"):
            values = rng.uniform([-3, -1.5, -3], [3, 1.5, 3], size=(rows, 3))
        if part.endswith("_jaw"):
            values = rng.uniform(0.01, 0.08, size=(rows, 1))
        out[part] = values
    return out


@pytest.mark.parametrize("strict", [True, False])
def test_c07u_a_robot_with_parts_sends_the_plans_slots_and_gets_its_own_parts_back(
    server: FakeDreamAPI, connection: dict[str, Any], strict: bool
) -> None:
    """C-7u: the caller sends and gets the robot's own parts; the model sees and answers the slots its plan trains
    (`shared.datasets.conversion`), a gripper's `[H]` included, strict or not."""
    config = _parts_config("euler_xyz")
    body = embodiments.parse(config)
    assert body.plan is not None and body.plan.use_eef
    wanted = _own_rows(config, "action", HORIZON)
    answer = {
        f"action.{name}": (rows[:, 0] if rows.shape[1] == 1 else rows).tolist()
        for name, rows in conversion.to_trained(body, "action", wanted).items()
    }
    serve_actions(server, lambda: {"per_key": {**answer, "action.rng_probe": [0.0]}, "note": "kept"})
    state = {part: rows[0] for part, rows in _own_rows(config, "state", 1).items()}
    observation = {
        **{f"state.{part}": values for part, values in state.items()},
        "video.top": [FRAME_B64],
        "prompt": "go",
    }

    with dream._open_policy({**connection, "embodiment": config, "strict": strict}) as client:
        action = client.get_action(observation)

    trained = conversion.to_trained(body, "state", {part: values[None] for part, values in state.items()})
    sent = server.requests[0].body["observation"]
    assert set(sent) == {f"state.{slot.name}" for slot in body.plan.state} | {"video.top", "prompt"}
    for name, values in trained.items():
        assert sent[f"state.{name}"] == pytest.approx(values[0].tolist()), name
    assert sent["state.left_grip"] == sent["state.left_gripper"] == pytest.approx([(state["l_jaw"][0] - 0.08) / -0.07])
    assert action["note"] == "kept"
    assert set(action["per_key"]) == {f"action.{part}" for part in config["action"]} | {"action.rng_probe"}
    for part, rows in wanted.items():
        assert np.array(action["per_key"][f"action.{part}"]) == pytest.approx(rows, abs=1e-9), part


def test_c07v_a_pose_the_plan_records_is_not_sent_and_not_answered(
    server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """C-7v: a flange position trains no gripper pose, so the poses neither travel nor come back; the joints, the
    jaws (as `<side>_gripper`) and the `other` part do."""
    config = _parts_config("quat_xyzw")
    for side in ("l", "r"):
        config["parts"]["state"][f"{side}_xyz"] = config["parts"]["action"][f"{side}_xyz"] = {
            "kind": "position",
            "side": "left" if side == "l" else "right",
            "point": "flange",
            "frame": "arm_base",
        }
    body = embodiments.parse(config)
    assert body.plan is not None and not body.plan.use_eef
    serve_actions(server, lambda: {"per_key": _answer(body)})
    observation = {f"state.{part}": [0.5] * (end - start) for part, (start, end) in config["state"].items()}

    with dream._open_policy({**connection, "embodiment": config}) as client:
        action = client.get_action({**observation, "video.top": [FRAME_B64], "prompt": "go"})

    sent = server.requests[0].body["observation"]
    assert sorted(key for key in sent if key.startswith("state.")) == sorted(
        f"state.{name}" for name in ("left_gripper", "left_arm", "right_gripper", "right_arm", "other_base")
    )
    assert sorted(action["per_key"]) == sorted(
        f"action.{part}" for part in ("l_jaw", "l_joints", "r_jaw", "r_joints", "base")
    )
    assert np.array(action["per_key"]["action.l_jaw"]) == pytest.approx(np.full((2, 1), 0.5 * (0.01 - 0.08) + 0.08))


def test_c07v_a_part_marked_not_trained_is_not_read_not_sent_and_not_answered(
    server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """C-7v: the observation carries a part marked not trained like any part the robot records, but nothing of it
    travels, and the answer has every other action part and not it: the robot holds it itself."""
    config = _parts_config("euler_xyz")
    config["parts"]["state"]["base"] = config["parts"]["action"]["base"] = {"kind": "not_trained"}
    body = embodiments.parse(config)
    assert body.plan is not None and body.plan.use_eef
    serve_actions(server, lambda: {"per_key": _answer(body)})
    state = {part: rows[0] for part, rows in _own_rows(config, "state", 1).items()}

    with dream._open_policy({**connection, "embodiment": config}) as client:
        action = client.get_action(
            {**{f"state.{part}": v for part, v in state.items()}, "video.top": [FRAME_B64], "prompt": "go"}
        )

    sent = server.requests[0].body["observation"]
    assert set(sent) == {f"state.{slot.name}" for slot in body.plan.state} | {"video.top", "prompt"}
    assert not any("base" in key for key in sent)
    assert set(action["per_key"]) == {f"action.{part}" for part in config["action"] if part != "base"}


def test_c07u_a_delta_parts_targets_come_back_as_the_change_each_row_makes(
    server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """C-7u: a part whose values are deltas trains as its targets, so the model answers targets; the caller gets the
    change each row makes, the first from the state it sent and each later one from the row before, and every other
    part as before."""
    config = _parts_config("quat_wxyz")
    for part in ("l_xyz", "l_joints"):
        config["parts"]["action"][part] = {**config["parts"]["action"][part], "delta": True}
    body = embodiments.parse(config)
    assert body.plan is not None
    answer = _answer(body)
    answer["action.left_arm"] = [[0.2] * 6, [0.7] * 6]
    answer["action.left_eef_xyz"] = [[1.0, 2.0, 3.0], [1.5, 2.0, 2.0]]
    serve_actions(server, lambda: {"per_key": answer})
    observation = {f"state.{part}": [0.5] * (end - start) for part, (start, end) in config["state"].items()}
    observation = {**observation, "state.l_joints": [0.1] * 6, "state.l_xyz": [1.0, 1.0, 1.0]}

    with dream._open_policy({**connection, "embodiment": config}) as client:
        action = client.get_action({**observation, "video.top": [FRAME_B64], "prompt": "go"})

    assert np.array(action["per_key"]["action.l_joints"]) == pytest.approx(np.array([[0.1] * 6, [0.5] * 6]))
    assert np.array(action["per_key"]["action.l_xyz"]) == pytest.approx(np.array([[0.0, 1.0, 2.0], [0.5, 0.0, -1.0]]))
    assert np.array(action["per_key"]["action.r_joints"]) == pytest.approx(np.full((2, 6), 0.5))
    assert np.array(action["per_key"]["action.base"]) == pytest.approx(np.full((2, 2), 0.5))


def test_c07w_a_rotation_that_is_not_one_and_an_answer_without_a_slot_are_refused(
    server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """C-7w: a zero quaternion is refused before anything is sent; an answer lacking a slot, or with a 6D of length
    zero, is refused before it reaches the caller, strict or not; and a robot with parts takes no `ik`."""
    config = _parts_config("quat_wxyz")
    body = embodiments.parse(config)
    assert body.plan is not None
    observation = {f"state.{part}": [0.5] * (end - start) for part, (start, end) in config["state"].items()}
    observation = {**observation, "video.top": [FRAME_B64], "prompt": "go"}
    full = _answer(body)

    with dream._open_policy({**connection, "embodiment": config}) as client:
        with pytest.raises(
            DreamObservationError, match=r"^observation state part 'l_rot': a quaternion has length zero"
        ):
            client.get_action({**observation, "state.l_rot": [0, 0, 0, 0]})
    assert server.requests == []

    serve_actions(
        server, lambda: {"per_key": {key: rows for key, rows in full.items() if key != "action.left_eef_rot6"}}
    )
    with dream._open_policy({**connection, "embodiment": config, "strict": False}) as client:
        with pytest.raises(DreamActionError, match=r"'action\.left_eef_rot6' is not rows of 6 numbers"):
            client.get_action(observation)
    serve_actions(server, lambda: {"per_key": {**full, "action.right_eef_rot6": [[0.0] * 6] * 2}})
    with dream._open_policy({**connection, "embodiment": config}) as client:
        with pytest.raises(DreamActionError, match="slot 'right_eef_rot6': a 6D rotation has a vector of length zero"):
            client.get_action(observation)

    with pytest.raises(DreamConfigError, match="takes no ik"):
        dream_client.PolicyClient(
            session_config={"checkpoint": "ft", "embodiment": config}, ik={"arm": "metal", "head": "eef"}
        )


def test_c07u_a_listed_config_this_library_cannot_read_raises_and_sends_nothing(
    server: FakeDreamAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C-7u: a model whose listed config the parser refuses (a key a later platform lists) raises at policy() with
    `strict` on, since its calls could be neither checked nor converted."""
    _account(monkeypatch, owner="mine", config={**ARM_CONFIG, "parts": {"arm": {"kind": "arm"}}})
    serve_actions(server, lambda: parts_reply(ARM_CONFIG))
    client = dream.DreamClient("dl_live_x")
    refused = r"is not one this library reads, so its calls could be neither checked nor converted; install the "
    with pytest.raises(DreamConfigError, match=refused + r"current dream client \(pip install --upgrade dreamlabs\)"):
        client.policy({"checkpoint": FT_CHECKPOINT})
    client.close()
    assert server.requests == []


def test_c07u_an_owners_robot_with_parts_run_by_checkpoint_alone_gets_its_own_parts_back(
    server: FakeDreamAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C-7u: policy() by checkpoint alone, on a fine-tune of the account's own robot with `parts` and a URDF, loads
    the config its model lists to the body's owner (`parts`, `actions` and `urdf`, `GET /v1/models`), sends the slots
    its plan trains and returns the robot's own parts, the poses the URDF leaves recorded dropped."""
    config = _parts_config("euler_xyz")
    # With a URDF and joint actions the poses are recorded, so they name no convention (`embodiments.ANSWER_KEYS`).
    for answers in config["parts"].values():
        for part, answer in answers.items():
            if answer["kind"] in ("position", "rotation"):
                answers[part] = {"kind": answer["kind"], "side": answer["side"]}
    config = {
        **config,
        "urdf": {
            "sha256": "c" * 64,
            "joints": {f"{s}_joints": [f"{s}{i}" for i in range(6)] for s in ("l", "r")},
            "reversed": {},
            "grasp": {
                side: {"link": f"{side}_tool", "xyz": [0, 0, 0.1], "rpy": [0, 0, 0]} for side in ("left", "right")
            },
        },
    }
    body = embodiments.parse(config)
    assert body.plan is not None and not body.plan.use_eef, "with the URDF the poses are recorded, not trained"
    trained_parts = {slot.part for slot in body.plan.action}
    wanted = {part: rows for part, rows in _own_rows(config, "action", HORIZON).items() if part in trained_parts}
    answer = {
        f"action.{name}": (rows[:, 0] if rows.shape[1] == 1 else rows).tolist()
        for name, rows in conversion.to_trained(body, "action", wanted).items()
    }
    serve_actions(server, lambda: {"per_key": answer})
    _account(monkeypatch, owner="mine", config=config)
    state = {part: rows[0] for part, rows in _own_rows(config, "state", 1).items()}
    client = dream.DreamClient("dl_live_x")

    client.policy({"checkpoint": FT_CHECKPOINT})
    action = _send(client, {**{f"state.{p}": v for p, v in state.items()}, "video.top": [FRAME_B64], "prompt": "go"})
    client.close()

    sent = server.requests[0].body["observation"]
    assert {key for key in sent if key.startswith("state.")} == {f"state.{slot.name}" for slot in body.plan.state}
    assert set(action["per_key"]) == {f"action.{part}" for part in wanted}
    for part, rows in wanted.items():
        assert np.array(action["per_key"][f"action.{part}"]) == pytest.approx(rows, abs=1e-9), part


@pytest.mark.parametrize("owner", ["shared", "public", "mine"])
def test_c07u_a_robot_whose_parts_are_withheld_is_refused_at_policy_naming_it_and_the_fix(
    server: FakeDreamAPI, monkeypatch: pytest.MonkeyPatch, owner: str
) -> None:
    """C-7u: a fine-tune of a robot with `parts` that another account registered, run by checkpoint alone (a domain
    grant is `shared`, a dev account `public`, the caller's own fine-tune of that robot `mine`), is listed without the
    parts (`Model.parts_withheld`): policy() refuses it with a line naming the robot and `config['embodiment']`, before
    anything is sent, rather than every reply being refused as lacking parts. Given the robot's config, it runs and
    converts."""
    config = _parts_config("euler_xyz")
    listed = {key: config[key] for key in ("state", "action", "video")}
    made = datetime.datetime(2026, 10, 1, tzinfo=datetime.timezone.utc)
    model = models.Model(
        FT_CHECKPOINT, owner, "dreamlabs/dream-1-beta", FT_EMBODIMENT_ID, True, made, ["top"], listed, True
    )
    monkeypatch.setattr(models, "list_models", lambda **kwargs: [model])
    client = dream.DreamClient("dl_live_x")

    with pytest.raises(DreamConfigError) as refused:
        client.policy({"checkpoint": FT_CHECKPOINT})
    assert str(refused.value) == (
        f"checkpoint {FT_CHECKPOINT!r} was trained on robot {FT_EMBODIMENT_ID!r}, registered with parts (what each "
        "part of its state and action is), which the platform lists only to the account that registered it, on a "
        "current client; pass that robot's embodiment config, the dict add_embodiment() took, as "
        "config['embodiment'], or run the model from that account on the current client "
        "(pip install --upgrade dreamlabs)"
    )

    body = embodiments.parse(config)
    assert body.plan is not None
    trained_parts = {slot.part for slot in body.plan.action}
    wanted = {part: rows for part, rows in _own_rows(config, "action", HORIZON).items() if part in trained_parts}
    answer = {
        f"action.{name}": (rows[:, 0] if rows.shape[1] == 1 else rows).tolist()
        for name, rows in conversion.to_trained(body, "action", wanted).items()
    }
    serve_actions(server, lambda: {"per_key": answer})
    client.policy({"checkpoint": FT_CHECKPOINT, "embodiment": config})
    state = {part: rows[0] for part, rows in _own_rows(config, "state", 1).items()}
    action = _send(client, {**{f"state.{p}": v for p, v in state.items()}, "video.top": [FRAME_B64], "prompt": "go"})
    client.close()
    sent = server.requests[0].body["observation"]
    assert {key for key in sent if key.startswith("state.")} == {f"state.{slot.name}" for slot in body.plan.state}
    assert set(action["per_key"]) == {f"action.{part}" for part in wanted}

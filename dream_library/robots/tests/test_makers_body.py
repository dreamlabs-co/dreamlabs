"""The Makers Arm as a body of the station (`robots.bodies`): its config, its wire, its dream config, and the places
the station picks the body's driver.

The wire is checked against the embodiment parser the service and the dream client share
(`shared.datasets.embodiments`); dream's strict mode reading it through the real client is
`test_makers_station.py`'s job. The YAM checks here pin that a config without `body` builds what it built before.
"""

import asyncio
import dataclasses
import pathlib
from typing import Any

import conftest
import numpy as np
import pytest
import test_types_adapter
import yaml
from shared.datasets import embodiments

from robots import (
    adapter,
    bodies,
    cli,
    config,
    errors,
    followers,
    home,
    kinematics,
    makers,
    makers_hold,
    motor_arm,
    run,
    types,
)

WIRE = bodies.MAKERS.wire()
ROLES = ("scene", "wrist_l", "wrist_r")
## A six-joint arm whose gripper part the loop sends: a Makers or Metal arm's layout.
SIX_JOINTS = adapter.ArmLayout(adapter.ALL_JOINTS, 6, client_gripper=False, slots=6)

## The shipped Makers example config; loaded as-is so the file and load() cannot drift apart.
EXAMPLE_PATH = pathlib.Path(__file__).resolve().parent.parent / "configs" / "makers.example.yaml"

## The smallest Makers file: both arms, the three cameras, the required loop keys.
MAKERS_FILE: dict[str, Any] = {
    "body": "makers",
    "follower_l": {"can_channel": "can_left"},
    "follower_r": {"can_channel": "can_right"},
    "cameras": [{"name": name, "device": i, "width": 64, "height": 48} for i, name in enumerate(ROLES)],
    "action_horizon": 48,
    "stale_after_s": 0.2,
    "max_stall_ticks": 10,
    "max_ee_speed_m_s": 0.6,
}

## The state keys the client adds from the joints when its `ik` is set (`dream.client_ik`).
EEF_STATE_KEYS = {f"state.{side}_eef_{part}" for side in ("right", "left") for part in ("xyz", "rot6")}


def load(tmp_path: pathlib.Path, document: dict[str, Any]) -> config.AppConfig:
    """Writes `document` as the config file and loads it."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return config.load(path)


def makers_file(**arm: Any) -> dict[str, Any]:
    """#MAKERS_FILE with `arm` keys added to the right arm's section."""
    return {**MAKERS_FILE, "follower_r": {"can_channel": "can_right", **arm}}


def makers_app(**overrides: Any) -> config.AppConfig:
    """A Makers AppConfig over the three Makers cameras at 30 Hz."""
    cameras = [config.CameraConfig(name=name, device=i, width=64, height=48) for i, name in enumerate(ROLES)]
    return conftest.make_app_config(**{"body": "makers", "cameras": cameras, "control_hz": 30.0, **overrides})


# The config


def test_the_shipped_makers_example_loads_as_a_makers_body() -> None:
    """The example names the body, the three Makers cameras, both adapters, 20 Hz and the grip cap off."""
    cfg = config.load(EXAMPLE_PATH)

    assert cfg.body == "makers" and cfg.control_hz == 20.0 and cfg.grip_force_cap == 0.0
    assert [camera.name for camera in cfg.cameras] == list(ROLES)
    assert (cfg.follower_l.can_channel, cfg.follower_r.can_channel) == ("can_follower_l", "can_follower_r")
    assert cfg.follower_r.gripper_open_deg == -96.0 and cfg.follower_r.gripper_closed_deg == -2.5
    assert cfg.follower_r.gripper_hold_torque_nm == makers_hold.DEFAULT_HOLD_TORQUE_NM
    assert list(run.policy_config(cfg)) == ["embodiment", "action_hz", "sync_above_ms", "ik"]


def test_the_body_defaults_to_the_yam_and_names_outside_the_list_are_refused(tmp_path: pathlib.Path) -> None:
    """A file without `body` is a YAM at 20 Hz; a Makers file without `control_hz` runs at 20 Hz."""
    yam_file = {**MAKERS_FILE, "cameras": [{"name": "top", "device": 0, "width": 64, "height": 48}]}
    del yam_file["body"]
    yam = load(tmp_path, yam_file)
    assert yam.body == "yam" and yam.control_hz == 20.0
    assert load(tmp_path, MAKERS_FILE).control_hz == config.MAKERS_CONTROL_HZ == 20.0
    assert load(tmp_path, {**MAKERS_FILE, "control_hz": 30}).control_hz == 30.0

    with pytest.raises(
        errors.ConfigError, match=r"'body' must be one of \['yam', 'makers', 'metal', 'so101', 'custom'\], got 'so100'"
    ):
        load(tmp_path, {**MAKERS_FILE, "body": "so100"})


def test_every_config_body_has_a_body_description() -> None:
    """`robots.bodies` and the config agree on the names; the SO-101's body is built from its config's embodiment."""
    assert (*bodies.BODIES, "so101") == config.BODY_NAMES
    assert bodies.BODIES["yam"] is bodies.YAM and bodies.YAM.ik is None
    assert bodies.MAKERS.checkpoint is None and bodies.MAKERS.ik == {"arm": "makers", "head": "eef"}


def test_a_makers_arm_reads_its_gripper_keys_and_refuses_the_yams(tmp_path: pathlib.Path) -> None:
    """The gripper's stroke and hold are read; an i2rt key in a Makers section, or the reverse, fails by name."""
    cfg = load(tmp_path, makers_file(gripper_open_deg=-100.5, gripper_closed_deg=-3, gripper_hold_torque_nm=None))
    assert cfg.follower_r.gripper_open_deg == -100.5 and cfg.follower_r.gripper_closed_deg == -3.0
    assert cfg.follower_r.gripper_hold_torque_nm is None
    assert cfg.follower_l.gripper_hold_torque_nm == makers_hold.DEFAULT_HOLD_TORQUE_NM
    assert cfg.follower_l.home == (0.0,) * 6, "the folded calibration pose, the Makers Arm's zero"
    assert load(tmp_path, makers_file(home=[0, -0.5, 1, 0, 0, 0])).follower_r.home == (0.0, -0.5, 1.0, 0.0, 0.0, 0.0)

    with pytest.raises(errors.ConfigError, match=r"unknown keys \['gripper'\] under 'follower_r'"):
        load(tmp_path, makers_file(gripper="linear_4310"))
    yam_file = {**MAKERS_FILE, "body": "yam", "follower_r": {"can_channel": "can_right", "gripper_open_deg": -96}}
    with pytest.raises(errors.ConfigError, match=r"unknown keys \['gripper_open_deg'\] under 'follower_r'"):
        load(tmp_path, yam_file)


@pytest.mark.parametrize(
    ("arm", "message"),
    [
        ({"can_channel": "gs_usb:206A33A35643"}, r"raw-USB candleLight adapter"),
        ({"gripper_open_deg": -2.5, "gripper_closed_deg": -96}, r"gripper_open_deg' \(-2.5\) must be below"),
        ({"gripper_hold_torque_nm": 2.5}, r"within 0.1 to 2 N.m"),
        ({"gripper_hold_torque_nm": 0.05}, r"within 0.1 to 2 N.m"),
        ({"gripper_open_deg": True}, r"finite number"),
        ({"gripper_closed_deg": float("nan")}, r"finite number"),
    ],
)
def test_a_makers_arm_refuses_what_the_driver_cannot_run(tmp_path: pathlib.Path, arm: Any, message: str) -> None:
    """A raw-USB adapter, a stroke the wrong way round, a hold outside the vendor's range, or a value that is not a
    number fails at load."""
    with pytest.raises(errors.ConfigError, match=message):
        load(tmp_path, makers_file(**arm))


# The wire


def test_the_makers_wire_sends_each_arms_joints_stroke_and_grip_and_a_video_window_per_camera() -> None:
    """Right arm then left: `[6]` joints, the gripper as mm of stroke and as the loop's fraction, all float32; each
    camera's frame under `video.<role>`, whose window the client sends; no tool pose (the client adds those) and none
    of the YAM's keys."""
    observation = test_types_adapter.make_observation(camera_names=ROLES)

    payload = test_types_adapter.sent(WIRE.to_dream(observation, conftest.TASK))

    assert list(payload) == [
        "prompt",
        "state.right_arm",
        "state.right_gripper",
        "state.right_grip",
        "state.left_arm",
        "state.left_gripper",
        "state.left_grip",
        "video.scene",
        "video.wrist_l",
        "video.wrist_r",
    ]
    assert payload["prompt"] == conftest.TASK
    for side, arm in (("right", observation.state.right), ("left", observation.state.left)):
        assert all(payload[f"state.{side}_{part}"].dtype == np.float32 for part in ("arm", "gripper", "grip"))
        np.testing.assert_array_equal(payload[f"state.{side}_arm"], arm.positions_rad.astype(np.float32))
        assert payload[f"state.{side}_gripper"].tolist() == [arm.gripper * makers.GRIPPER_MAX_MM]
        assert payload[f"state.{side}_grip"].tolist() == [arm.gripper]
    for name in ROLES:
        assert payload[f"video.{name}"] == [observation.images[name].jpeg]


def test_the_makers_wire_refuses_a_yam_camera_and_a_frame_without_an_encoded_frame() -> None:
    """A camera not named after a Makers role, or a hand-built frame no reader encoded, fails the tick."""
    with pytest.raises(errors.ControlLoopError, match=r"camera 'top' is not a makers camera role"):
        WIRE.to_dream(test_types_adapter.make_observation(camera_names=("scene", "top")), conftest.TASK)

    observation = test_types_adapter.make_observation(camera_names=("scene",))
    bare = types.Frame(data=observation.images["scene"].data, camera="scene", timestamp_s=1.0)
    with pytest.raises(errors.ControlLoopError, match=r"camera 'scene' frame carries no encoded frame"):
        WIRE.to_dream(types.Observation(observation.state, {"scene": bare}, 1.0), conftest.TASK)


def test_the_makers_embodiment_parses_and_declares_exactly_the_wire_plus_the_clients_tool_poses() -> None:
    """The block is an embodiment config the shared parser takes: 34 state and 32 action dims, the video keys of
    the Makers datasets; its state parts are what the wire sends plus the four the client computes."""
    names = list(ROLES)
    block = WIRE.embodiment(names, 48)
    body = embodiments.parse(block)

    assert (body.state_dim, body.action_dim) == (34, 32)
    assert body.video == {
        "scene": "observation.images.scene",
        "wrist_l": "observation.images.left_wrist",
        "wrist_r": "observation.images.right_wrist",
    }
    payload = test_types_adapter.sent(
        WIRE.to_dream(test_types_adapter.make_observation(camera_names=tuple(names)), conftest.TASK)
    )
    sent_state = {key for key in payload if key.startswith("state.")}
    assert {f"state.{part}" for part in body.state} == sent_state | EEF_STATE_KEYS
    assert {key for key in payload if key.startswith("video.")} == {f"video.{name}" for name in names}
    for part, (start, end) in body.state.items():
        if f"state.{part}" in payload:
            assert payload[f"state.{part}"].shape == (end - start,), part
    assert {f"action.{side}_{part}" for side in ("right", "left") for part in ("eef_xyz", "eef_rot6", "grip")} <= {
        f"action.{part}" for part in body.action
    }
    assert WIRE.embodiment(["scene"], 48)["video"] == {"scene": "observation.images.scene"}


@pytest.mark.parametrize(
    "names",
    [("scene", "wrist_l", "wrist_r"), ("overhead", "wrist_l", "wrist_r"), ("head", "left_wrist", "right_wrist")],
)
def test_a_makermods_station_names_its_cameras_what_its_model_reads(names: tuple[str, ...]) -> None:
    block = WIRE.embodiment(list(names), 48)
    payload = test_types_adapter.sent(
        WIRE.to_dream(test_types_adapter.make_observation(camera_names=names), conftest.TASK)
    )

    assert list(block["video"]) == list(names)
    assert list(block["video"].values()) == [
        "observation.images.scene",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    ]
    assert {key for key in payload if key.startswith("video.")} == {f"video.{name}" for name in names}
    with pytest.raises(errors.ConfigError, match="camera twice"):
        WIRE.embodiment([names[0], "scene" if names[0] != "scene" else "overhead", "wrist_l"], 48)


def test_a_wire_sends_only_the_parts_its_embodiment_records() -> None:
    """An embodiment with no `<side>_gripper` gets the joints and the grip, in the same order, and needs no scale."""
    config_ = {
        "state": {"right_arm": [0, 6], "right_grip": [6, 7], "left_arm": [7, 13], "left_grip": [13, 14]},
        "action": {"right_arm": [0, 6], "left_arm": [6, 12]},
        "video": {"scene": "observation.images.scene"},
    }
    wire = adapter.EefWire(config_, None, "custom", SIX_JOINTS)

    payload = test_types_adapter.sent(
        wire.to_dream(test_types_adapter.make_observation(camera_names=("scene",)), conftest.TASK)
    )

    assert list(payload) == [
        "prompt",
        "state.right_arm",
        "state.right_grip",
        "state.left_arm",
        "state.left_grip",
        "video.scene",
    ]
    assert wire.camera_roles == ("scene",)
    assert embodiments.parse(wire.embodiment(["scene"], 48)).state_dim == 14


@pytest.mark.parametrize(
    ("state", "grip_scale", "message"),
    [
        ({"right_wrist": [0, 1]}, 80.0, r"state part 'right_wrist' is not one the loop fills"),
        ({"middle_arm": [0, 6]}, 80.0, r"state part 'middle_arm' is not one the loop fills"),
        ({"right_arm": [0, 7]}, 80.0, r"state part 'right_arm' spans \[0, 7\], expected 6 wide"),
        ({"right_gripper": [0, 1]}, None, r"records <side>_gripper: give its value with the gripper fully open"),
    ],
)
def test_a_wire_refuses_an_embodiment_the_loop_cannot_fill(
    state: dict[str, list[int]], grip_scale: float | None, message: str
) -> None:
    """A part the loop has no value for, a part of the wrong width, or a gripper in an unknown unit fails at build."""
    with pytest.raises(errors.ConfigError, match=message):
        adapter.EefWire({"state": state, "action": {}, "video": {}}, grip_scale, "custom", SIX_JOINTS)


def test_a_wire_on_an_arm_whose_gripper_the_client_makes_sends_its_joints_and_its_grip() -> None:
    """An SO-101 sends its five joints and its grip: the dream client makes `<side>_gripper` from the grip and puts
    the held sixth joint back (`dream.client_ik`), so the embodiment's six-wide arm and its gripper part are right."""
    config_ = {
        "state": {
            "left_arm": [0, 6],
            "left_gripper": [6, 7],
            "right_arm": [7, 13],
            "right_gripper": [13, 14],
            "right_grip": [14, 15],
            "left_grip": [15, 16],
        },
        "action": {"right_grip": [0, 1], "left_grip": [1, 2]},
        "video": {"top": "observation.images.top"},
    }
    wire = adapter.EefWire(config_, None, "so101", adapter.ArmLayout((0, 1, 2, 3, 4), 6, client_gripper=True, slots=6))
    observation = test_types_adapter.make_observation(camera_names=("top",))

    payload = test_types_adapter.sent(wire.to_dream(observation, conftest.TASK))

    assert list(payload) == [
        "prompt",
        "state.right_arm",
        "state.right_grip",
        "state.left_arm",
        "state.left_grip",
        "video.top",
    ]
    np.testing.assert_array_equal(payload["state.right_arm"], observation.state.right.positions_rad[:5])


def test_arm_layout_is_dreams_spec_of_the_arm() -> None:
    """The MakerMods arms drive six joints and send their own gripper part; the SO-101 drives five (dream holds its
    sixth) and the client makes its gripper part."""
    pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")

    assert kinematics.arm_layout("makers") == SIX_JOINTS
    assert kinematics.arm_layout("metal") == SIX_JOINTS
    assert kinematics.arm_layout("so101_faas") == adapter.ArmLayout((0, 1, 2, 3, 4), 6, client_gripper=True, slots=6)
    with pytest.raises(errors.ConfigError, match="dream's IK has no arm 'so100'"):
        kinematics.arm_layout("so100")


# The dream config


def test_policy_config_of_a_makers_arm_carries_the_ik_and_no_default_model() -> None:
    """The embodiment config, the loop's rate and the client's IK; the model only when a run names one."""
    app = makers_app()

    built = run.policy_config(app)
    named = run.policy_config(app, "dreamlabs/makers-eef-ft")

    assert built == {
        "embodiment": WIRE.embodiment(list(ROLES), 48),
        "action_hz": 30.0,
        "sync_above_ms": run.SYNC_ABOVE_MS,
        "ik": {"arm": "makers", "head": "eef", "solve": "traj"},
    }
    assert list(named) == ["checkpoint", "embodiment", "action_hz", "sync_above_ms", "ik"]
    assert named["checkpoint"] == "dreamlabs/makers-eef-ft"
    built["ik"]["head"] = "joint"
    assert bodies.MAKERS.ik == {"arm": "makers", "head": "eef"}, "the body's ik must not be shared with a caller"


def test_policy_config_of_a_yam_carries_its_own_embodiment_config() -> None:
    """A YAM config builds the keys it built before bodies existed, in the same order, plus the sync threshold, with
    the YAM's EEF embodiment config (not the deprecated station spec) as the block, then the YAM's EEF head `ik`."""
    app = conftest.make_app_config()

    assert run.policy_config(app) == {
        "checkpoint": "dreamlabs/dream-1-beta",
        "embodiment": adapter.YAM_POSE_WIRE.embodiment(list(conftest.CAMERA_NAMES), app.action_horizon),
        "action_hz": app.control_hz,
        "sync_above_ms": run.SYNC_ABOVE_MS,
        "ik": {"arm": "yam", "head": "eef", "solve": "traj"},
    }
    assert list(run.policy_config(app)) == ["checkpoint", "embodiment", "action_hz", "sync_above_ms", "ik"]
    assert run.policy_config(app)["embodiment"] != adapter.embodiment_spec(
        list(conftest.CAMERA_NAMES), app.action_horizon
    )
    assert run.DEFAULT_CHECKPOINT == bodies.YAM_CHECKPOINT
    assert run.loop_settings(app)["wire"] is adapter.YAM_VIDEO_WIRE


def test_every_station_goes_sync_above_900_ms_whatever_its_rate_and_chunk() -> None:
    """The client goes sync above 900 ms instead of its 700 ms default on every body, rate and chunk (a Maker Arm runs
    at 30 Hz by default, where 18 rows carry 0.6 s); System 0 adds its 100 ms."""
    yam = dataclasses.replace(conftest.make_app_config(), action_horizon=48, control_hz=20)
    assert run.policy_config(yam)["sync_above_ms"] == run.SYNC_ABOVE_MS == 900
    assert run.policy_config(yam, sys0=True)["sync_above_ms"] == 900 + run.SYS0_SYNC_EXTRA_MS == 1000
    for other in (makers_app(), dataclasses.replace(yam, action_horizon=40)):
        assert run.policy_config(other)["sync_above_ms"] == 900


def test_policy_config_names_the_makers_roles_when_a_camera_is_not_one() -> None:
    """A YAM camera name on a Makers Arm fails before anything opens, listing the roles to use."""
    cameras = [config.CameraConfig(name="top", device=0, width=64, height=48)]

    with pytest.raises(errors.ConfigError, match=r"\['top'\].*\['scene', 'overhead', 'head', 'wrist_l', 'left_wrist'"):
        run.policy_config(makers_app(cameras=cameras))


def test_a_one_shot_makers_run_without_a_checkpoint_fails_before_opening_an_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`robots.run.main` has no model to fall back on for a Makers Arm: it says to pass `--checkpoint`."""

    def no_arm(spec: motor_arm.ArmSpec, cfg: config.ArmConfig, name: str, **_: Any) -> followers.FollowerArm:
        raise AssertionError("no arm may be opened")

    monkeypatch.setattr(motor_arm, "open_arm", no_arm)

    with pytest.raises(errors.ConfigError, match=r"no checkpoint: a makers body has no default model"):
        asyncio.run(run.main(makers_app()))


# Where the body's driver is picked


def test_home_opens_the_bodys_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    """`robots.home.main` on a Makers config opens both arms on the Makers table."""
    opened: list[tuple[str, str, str]] = []

    def open_arm(spec: motor_arm.ArmSpec, cfg: config.ArmConfig, name: str, **_: Any) -> followers.FollowerArm:
        opened.append((spec.name, cfg.can_channel, name))
        return conftest.yam_arm(name, conftest.FakeRobot())

    monkeypatch.setattr(motor_arm, "open_arm", open_arm)
    monkeypatch.setattr(home.time, "sleep", lambda dt_s: None)

    assert home.main(makers_app(), duration_s=0.1) == 0
    assert opened == [("makers", "can_left", "left"), ("makers", "can_right", "right")]


def test_the_cli_checks_a_canable_and_brings_up_a_socketcan_link(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "config.yaml"
    document = {
        **MAKERS_FILE,
        "follower_l": {"can_channel": "/dev/serial/by-id/usb-CANable_left"},
        "follower_r": {"can_channel": "can_right"},
    }
    path.write_text(yaml.safe_dump(document), encoding="utf-8")

    assert cli._can_channels(path) == ["/dev/serial/by-id/usb-CANable_left", "can_right"]


def test_makers_kinematics_is_dreams_forward_kinematics_of_the_tool() -> None:
    """The speed cap's end effector is the tool centre point of dream's Makers model, where the tool poses are."""
    pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")
    from dream_ik.solver import ChunkIK
    from dream_ik.specs import SERVE

    model = kinematics.DreamKinematics.both_arms("makers", 30.0)["right"]
    reference = ChunkIK("makers", SERVE.tool, 30.0)
    poses = [np.asarray(reference.rest, dtype=np.float64), np.array([0.3, -0.9, 1.1, 0.2, -0.1, 0.25])]

    for q in poses:
        np.testing.assert_allclose(model.ee_position(q), reference.fk(q)[:3, 3], atol=1e-9)
    assert model.ee_position(poses[0]).dtype == np.float64
    assert np.linalg.norm(model.ee_position(poses[1]) - model.ee_position(poses[0])) > 0.01

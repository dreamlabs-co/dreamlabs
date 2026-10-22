"""The Metal arm as a body of the station (`robots.bodies`): its config, its wire, its dream config, and the places
the station picks the body's driver.

The wire is the Makers Arm's with the Metal grip; it is checked against the embodiment parser the service and the
dream client share (`shared.datasets.embodiments`). The station end to end is `test_metal_station.py`.
"""

import asyncio
import dataclasses
import math
import pathlib
import time
from typing import Any
from xml.etree import ElementTree

import conftest
import metal_fakes
import numpy as np
import pytest
import test_types_adapter
import yaml
from shared.datasets import embodiments

from robots import (
    adapter,
    bodies,
    cameras,
    config,
    errors,
    followers,
    home,
    kinematics,
    makers_hold,
    metal,
    motor_arm,
    robstride,
    run,
)

WIRE = bodies.METAL.wire()
ROLES = ("scene", "wrist_l", "wrist_r")

## The shipped Metal example config; loaded as-is so the file and load() cannot drift apart.
EXAMPLE_PATH = pathlib.Path(__file__).resolve().parent.parent / "configs" / "metal.example.yaml"

## A Metal home pose as a file gives it, radians: shoulder lifted, elbow folded.
HOME = [0.0, -0.5, 1.2, 0.0, 0.0, 0.0]

## The smallest Metal file: both arms with their gripper ends and homes, the three cameras, the required loop keys.
METAL_FILE: dict[str, Any] = {
    "body": "metal",
    "follower_l": {"can_channel": "can_left", **metal_fakes.GRIPPER_ENDS, "home": HOME},
    "follower_r": {"can_channel": "can_right", **metal_fakes.GRIPPER_ENDS, "home": HOME},
    "cameras": [{"name": name, "device": i, "width": 64, "height": 48} for i, name in enumerate(ROLES)],
    "action_horizon": 48,
    "stale_after_s": 0.2,
    "max_stall_ticks": 10,
    "max_ee_speed_m_s": 0.6,
}


def load(tmp_path: pathlib.Path, document: dict[str, Any]) -> config.AppConfig:
    """Writes `document` as the config file and loads it."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return config.load(path)


def metal_file(**arm: Any) -> dict[str, Any]:
    """#METAL_FILE with the right arm's section replaced by `arm` over its channel, gripper ends and home."""
    right = {"can_channel": "can_right", **metal_fakes.GRIPPER_ENDS, "home": HOME, **arm}
    return {**METAL_FILE, "follower_r": right}


def metal_app(**overrides: Any) -> config.AppConfig:
    """A Metal AppConfig over the three Metal cameras at 30 Hz."""
    cameras = [config.CameraConfig(name=name, device=i, width=64, height=48) for i, name in enumerate(ROLES)]
    arms = {
        side: config.ArmConfig(can_channel=f"can_{side}", **metal_fakes.GRIPPER_ENDS) for side in ("left", "right")
    }
    return conftest.make_app_config(
        **{
            "body": "metal",
            "follower_l": arms["left"],
            "follower_r": arms["right"],
            "cameras": cameras,
            "control_hz": 30.0,
            **overrides,
        }
    )


# The config


def test_the_shipped_metal_example_refuses_to_load_until_its_gripper_ends_and_homes_are_filled(
    tmp_path: pathlib.Path,
) -> None:
    """The example's gripper ends are placeholders that fail by name; with real angles it loads as a Metal body: the
    three cameras, both adapters, 20 Hz, the grip cap off and MakerMods' hold torque."""
    with pytest.raises(errors.ConfigError, match=r"follower_l.gripper_open_deg' must be a finite number"):
        config.load(EXAMPLE_PATH)

    document = yaml.safe_load(EXAMPLE_PATH.read_text(encoding="utf-8"))
    for key in ("follower_l", "follower_r"):
        document[key].update(metal_fakes.GRIPPER_ENDS)
    with pytest.raises(errors.ConfigError, match=r"follower_l.home' must be 1 to 16 finite joint angles"):
        load(tmp_path, document)
    for key in ("follower_l", "follower_r"):
        document[key]["home"] = HOME
    cfg = load(tmp_path, document)

    assert cfg.body == "metal" and cfg.control_hz == 20.0 and cfg.grip_force_cap == 0.0
    assert [camera.name for camera in cfg.cameras] == list(ROLES)
    assert (cfg.follower_l.can_channel, cfg.follower_r.can_channel) == ("can_follower_l", "can_follower_r")
    assert (cfg.follower_r.gripper_open_deg, cfg.follower_r.gripper_closed_deg) == (120.0, 0.0)
    assert cfg.follower_r.gripper_hold_torque_nm == makers_hold.DEFAULT_HOLD_TORQUE_NM
    assert list(run.policy_config(cfg)) == ["embodiment", "action_hz", "sync_above_ms", "ik"]


def test_a_metal_file_without_control_hz_runs_at_20(tmp_path: pathlib.Path) -> None:
    """A Metal arm plays its rows at 20 Hz, as a YAM does, its driver ticking at 30; a file may set its own rate."""
    assert load(tmp_path, METAL_FILE).control_hz == config.METAL_CONTROL_HZ == 20.0
    assert metal.ARM.tick_hz == 30.0
    assert load(tmp_path, {**METAL_FILE, "control_hz": 30}).control_hz == 30.0


def test_every_config_body_has_a_body_description() -> None:
    """`robots.bodies` and the config agree on the names; the Metal arm has no default model and the client's IK."""
    assert (*bodies.BODIES, "so101") == config.BODY_NAMES == ("yam", "makers", "metal", "so101")
    assert bodies.BODIES["metal"] is bodies.METAL
    assert bodies.METAL.checkpoint is None and bodies.METAL.ik == {"arm": "metal", "head": "eef"}
    assert WIRE.camera_roles == tuple(adapter.MAKERMODS_EMBODIMENT["video"]) and set(ROLES) <= set(WIRE.camera_roles)


@pytest.mark.parametrize("given", [{}, {"gripper_open_deg": 120.0}, {"gripper_closed_deg": 0.0}])
def test_a_metal_arm_needs_its_gripper_ends_and_says_what_to_supply(
    tmp_path: pathlib.Path, given: dict[str, float]
) -> None:
    """No gripper ends, no load: the error names the section, both keys, and who records them."""
    document = {**METAL_FILE, "follower_r": {"can_channel": "can_right", "home": HOME, **given}}

    with pytest.raises(errors.ConfigError) as raised:
        load(tmp_path, document)

    message = str(raised.value)
    assert "config section 'follower_r' needs 'gripper_open_deg' and 'gripper_closed_deg'" in message
    assert "linearly" in message and "setup wizard" in message


def test_a_metal_arm_reads_its_keys_and_refuses_the_other_bodies(tmp_path: pathlib.Path) -> None:
    """The gripper ends and the hold torque are read; a YAM key, or the retired stroke table, fails by name."""
    cfg = load(tmp_path, metal_file(gripper_hold_torque_nm=0.8, gripper_open_deg=110.5))
    assert cfg.follower_r.gripper_hold_torque_nm == 0.8
    assert cfg.follower_r.gripper_open_deg == 110.5
    assert cfg.follower_l.gripper_hold_torque_nm == makers_hold.DEFAULT_HOLD_TORQUE_NM
    assert (cfg.follower_l.gripper_open_deg, cfg.follower_l.gripper_closed_deg) == (120.0, 0.0)

    with pytest.raises(errors.ConfigError, match=r"unknown keys \['gripper_stroke_table'\] under 'follower_r'"):
        load(tmp_path, metal_file(gripper_stroke_table=[[0, 0], [120, 100]]))
    with pytest.raises(errors.ConfigError, match=r"unknown keys \['gripper'\] under 'follower_r'"):
        load(tmp_path, metal_file(gripper="linear_4310"))
    makers_file = {**METAL_FILE, "body": "makers", "follower_r": {"can_channel": "can_right"}}
    makers_file["follower_l"] = {"can_channel": "can_left", "gripper_stroke_table": [[0, 0], [120, 100]]}
    with pytest.raises(errors.ConfigError, match=r"unknown keys \['gripper_stroke_table'\] under 'follower_l'"):
        load(tmp_path, makers_file)


def test_a_metal_arm_needs_its_home_and_says_why(tmp_path: pathlib.Path) -> None:
    """No home, no load: a Metal arm's zero pose stands it straight up, so there is no default to fall back to."""
    document = {**METAL_FILE, "follower_r": {"can_channel": "can_right", **metal_fakes.GRIPPER_ENDS}}

    with pytest.raises(errors.ConfigError) as raised:
        load(tmp_path, document)

    message = str(raised.value)
    assert "config section 'follower_r' needs 'home'" in message
    assert "straight up" in message and "dreamcli setup" in message
    assert load(tmp_path, METAL_FILE).follower_r.home == tuple(HOME)


@pytest.mark.parametrize(
    ("arm", "message"),
    [
        ({"can_channel": "gs_usb:206A33A35643"}, r"raw-USB candleLight adapter"),
        ({"gripper_open_deg": 140.0}, r"within the gripper's soft limits 0 to 137.5, closed below open"),
        ({"gripper_closed_deg": -5.0}, r"within the gripper's soft limits 0 to 137.5, closed below open"),
        ({"gripper_open_deg": 10.0, "gripper_closed_deg": 90.0}, r"closed below open"),
        ({"gripper_open_deg": 0.0}, r"closed below open"),
        ({"gripper_open_deg": None}, r"follower_r.gripper_open_deg' must be a finite number"),
        ({"gripper_closed_deg": float("nan")}, r"follower_r.gripper_closed_deg' must be a finite number"),
        ({"gripper_hold_torque_nm": 2.5}, r"within 0.1 to 2 N.m"),
        ({"gripper_hold_torque_nm": 0.05}, r"within 0.1 to 2 N.m"),
        ({"gripper_hold_torque_nm": None}, r"gripper_hold_torque_nm' must be a positive finite number"),
        ({"home": [0.0] * 5 + [float("inf")]}, r"follower_r.home' must be 1 to 16 finite joint angles in radians"),
        ({"home": "rest"}, r"follower_r.home' must be 1 to 16 finite joint angles in radians"),
    ],
)
def test_a_metal_arm_refuses_what_the_driver_cannot_run(tmp_path: pathlib.Path, arm: Any, message: str) -> None:
    """A raw-USB adapter, gripper ends outside the gripper's range or not closed below open, or a hold outside the
    vendor's range (or null: a Metal gripper always holds) fails at load."""
    with pytest.raises(errors.ConfigError, match=message):
        load(tmp_path, metal_file(**arm))


def test_a_metal_arms_keep_out_is_read_and_reaches_the_clients_ik(tmp_path: pathlib.Path) -> None:
    mount = {"kind": "box", "center": [-0.1, 0.33, 0.3], "size": [0.04, 0.04, 0.6]}
    pole = {"kind": "capsule", "center": [0, 0.3, 0.2], "radius": 0.02, "height": 1}

    cfg = load(tmp_path, metal_file(keep_out=[mount, pole]))

    pole_read = {"kind": "capsule", "center": [0.0, 0.3, 0.2], "radius": 0.02, "height": 1.0}
    assert cfg.follower_r.keep_out == (mount, pole_read)
    assert cfg.follower_l.keep_out is None
    ik = run.policy_config(cfg)["ik"]
    assert ik == {"arm": "metal", "head": "eef", "keep_out": {"right": [mount, pole_read]}, "solve": "traj"}
    assert run.policy_config(load(tmp_path, metal_file(keep_out=[])))["ik"]["keep_out"] == {"right": []}


@pytest.mark.parametrize(
    ("keep_out", "message"),
    [
        ({"kind": "box"}, r"'follower_r.keep_out' must be a list of obstacles"),
        ([{"kind": "cone"}], r"'follower_r.keep_out\[0\]' must be a mapping whose 'kind' is 'box' or 'capsule'"),
        ([{"kind": "box", "center": [0, 0, 0]}], r"is a box: it takes exactly the keys \['center', 'kind', 'size'\]"),
        (
            [{"kind": "box", "center": [0, 0, 0], "size": [0.1, 0.1, 0.1], "yaw": 0.3}],
            r"is a box: it takes exactly the keys",
        ),
        (
            [{"kind": "box", "center": [0, 0, 0], "size": [0.1, 0, 0.1]}],
            r"keep_out\[0\].size' must be three numbers above 0",
        ),
        (
            [{"kind": "capsule", "center": [0, 0], "radius": 0.02, "height": 0.5}],
            r"keep_out\[0\].center' must be three finite numbers",
        ),
        (
            [{"kind": "capsule", "center": [0, 0, 0], "radius": -0.02, "height": 0.5}],
            r"keep_out\[0\].radius' must be a number above 0",
        ),
    ],
)
def test_a_metal_arms_keep_out_must_be_obstacles(tmp_path: pathlib.Path, keep_out: Any, message: str) -> None:
    with pytest.raises(errors.ConfigError, match=message):
        load(tmp_path, metal_file(keep_out=keep_out))


def test_a_metal_arms_reach_is_read_and_reaches_the_clients_ik_and_a_file_without_it_is_unchanged(
    tmp_path: pathlib.Path,
) -> None:
    cfg = load(tmp_path, metal_file(reach=[[0.3, -0.1], [0, 0.25]]))

    assert cfg.follower_r.reach == ((0.3, -0.1), (0.0, 0.25))
    assert cfg.follower_l.reach is None
    assert run.policy_config(cfg)["ik"] == {
        "arm": "metal",
        "head": "eef",
        "reach": {"right": [[0.3, -0.1], [0.0, 0.25]]},
        "solve": "traj",
    }
    assert run.policy_config(load(tmp_path, metal_file()))["ik"] == {"arm": "metal", "head": "eef", "solve": "traj"}


@pytest.mark.parametrize(
    ("reach", "message"),
    [
        ([0.3, 0.1], r"'follower_r.reach\[0\]' must be \[x, y\], two finite numbers in metres, got 0.3"),
        ([], r"'follower_r.reach' must be a list of \[x, y\] points in metres, at least one"),
        ({"x": 0.3}, r"'follower_r.reach' must be a list of \[x, y\] points"),
        ([[0.3, 0.1, 0.2]], r"'follower_r.reach\[0\]' must be \[x, y\]"),
        ([[0.3, 0.1], [0.3, "far"]], r"'follower_r.reach\[1\]' must be \[x, y\]"),
        ([[0.3, True]], r"'follower_r.reach\[0\]' must be \[x, y\]"),
        ([[0.3, float("nan")]], r"'follower_r.reach\[0\]' must be \[x, y\]"),
    ],
)
def test_a_metal_arms_reach_must_be_points(tmp_path: pathlib.Path, reach: Any, message: str) -> None:
    with pytest.raises(errors.ConfigError, match=message):
        load(tmp_path, metal_file(reach=reach))


def test_a_metal_arms_region_is_read_and_reaches_the_clients_ik_and_a_file_without_it_is_unchanged(
    tmp_path: pathlib.Path,
) -> None:
    cfg = load(tmp_path, metal_file(region={"lo": [None, None, -0.05], "hi": [0.6, 0.3, None]}))

    assert cfg.follower_r.region == {"lo": [None, None, -0.05], "hi": [0.6, 0.3, None]}
    assert cfg.follower_l.region is None
    assert run.policy_config(cfg)["ik"] == {
        "arm": "metal",
        "head": "eef",
        "region": {"right": {"lo": [None, None, -0.05], "hi": [0.6, 0.3, None]}},
        "solve": "traj",
    }
    floor = load(tmp_path, metal_file(region={"lo": [None, None, 0]}))
    assert run.policy_config(floor)["ik"]["region"] == {"right": {"lo": [None, None, 0.0]}}
    assert "region" not in run.policy_config(load(tmp_path, metal_file()))["ik"]


@pytest.mark.parametrize(
    ("region", "message"),
    [
        ([0, 0, 0], r"'follower_r.region' must be a mapping with 'lo' and or 'hi'"),
        ({}, r"'follower_r.region' must be a mapping with 'lo' and or 'hi'"),
        ({"low": [0, 0, 0]}, r"'follower_r.region' must be a mapping with 'lo' and or 'hi'"),
        ({"lo": [0, 0]}, r"'follower_r.region.lo' must be \[x, y, z\], three finite numbers in metres or null"),
        ({"hi": [0, "far", 0]}, r"'follower_r.region.hi' must be \[x, y, z\]"),
        ({"hi": [0, True, 0]}, r"'follower_r.region.hi' must be \[x, y, z\]"),
        ({"lo": [float("nan"), 0, 0]}, r"'follower_r.region.lo' must be \[x, y, z\]"),
        ({"lo": [0.5, None, None], "hi": [0.2, None, None]}, r"'follower_r.region' has lo 0.5 not below hi 0.2 on x"),
    ],
)
def test_a_metal_arms_region_must_be_a_box(tmp_path: pathlib.Path, region: Any, message: str) -> None:
    with pytest.raises(errors.ConfigError, match=message):
        load(tmp_path, metal_file(region=region))


def test_ik_solve_reaches_the_clients_ik_and_every_body_defaults_to_traj(tmp_path: pathlib.Path) -> None:
    """The file's `ik_solve` reaches the client's `ik`; without it a Metal and a YAM both solve with
    `config.DEFAULT_IK_SOLVE`, traj, and `tick`, named, is the row-by-row solve."""
    tick = load(tmp_path, {**metal_file(), "ik_solve": "tick"})

    assert tick.ik_solve == "tick"
    assert run.policy_config(tick)["ik"] == {"arm": "metal", "head": "eef", "solve": "tick"}
    absent = load(tmp_path, metal_file())
    assert absent.ik_solve is None and run.policy_config(absent)["ik"]["solve"] == config.DEFAULT_IK_SOLVE == "traj"
    yam = run.with_action_space(conftest.make_app_config(), "eef")
    assert run.policy_config(yam)["ik"]["solve"] == "traj"
    with pytest.raises(
        errors.ConfigError, match=r"config key 'ik_solve' must be one of \['tick', 'traj'\], got 'fast'"
    ):
        load(tmp_path, {**metal_file(), "ik_solve": "fast"})


# The wire


def test_the_metal_wire_is_the_makers_wire_with_the_metal_grip() -> None:
    """Right arm then left: `[6]` joints, the stroke in mm (grip x 100) and the grip, all float32; each camera's
    frame under `video.<role>`, whose window the client sends; no tool pose (the client adds those)."""
    observation = test_types_adapter.make_observation(camera_names=ROLES)

    payload = test_types_adapter.sent(WIRE.to_dream(observation, conftest.TASK))

    assert list(payload) == list(test_types_adapter.sent(bodies.MAKERS.wire().to_dream(observation, conftest.TASK)))
    for side, arm in (("right", observation.state.right), ("left", observation.state.left)):
        assert all(payload[f"state.{side}_{part}"].dtype == np.float32 for part in ("arm", "gripper", "grip"))
        np.testing.assert_array_equal(payload[f"state.{side}_arm"], arm.positions_rad.astype(np.float32))
        assert payload[f"state.{side}_gripper"].tolist() == [np.float32(arm.gripper * metal.GRIP_SCALE_MM)]
        assert payload[f"state.{side}_grip"].tolist() == [np.float32(arm.gripper)]
    for name in ROLES:
        assert payload[f"video.{name}"] == [observation.images[name].jpeg]


def test_the_metal_wire_names_the_metal_roles_when_a_camera_is_not_one() -> None:
    """A YAM camera name fails the tick with the Metal arm named."""
    with pytest.raises(errors.ControlLoopError, match=r"camera 'top' is not a metal camera role"):
        WIRE.to_dream(test_types_adapter.make_observation(camera_names=("scene", "top")), conftest.TASK)


def test_the_metal_embodiment_is_the_makers_one_and_parses() -> None:
    """The Metal datasets have the Makers datasets' parts and video keys: the same block, which the shared parser
    takes, and a new one on every call."""
    names = list(ROLES)
    block = WIRE.embodiment(names, 48)
    body = embodiments.parse(block)

    assert block == bodies.MAKERS.wire().embodiment(names, 48)
    assert (body.state_dim, body.action_dim) == (34, 32)
    block["state"]["right_arm"][0] = -1
    assert WIRE.embodiment(names, 48) == bodies.MAKERS.wire().embodiment(names, 48)


# The dream config


@pytest.mark.parametrize("ik", [None, {"arm": "metal", "head": "joint"}])
def test_a_metal_arm_never_runs_a_model_that_answers_in_joints(
    monkeypatch: pytest.MonkeyPatch, ik: dict[str, str] | None
) -> None:
    """Joint-space actions on any robot but the YAM are refused before anything is built: EEF checkpoints only."""
    monkeypatch.setitem(bodies.BODIES, "metal", dataclasses.replace(bodies.METAL, ik=ik))

    with pytest.raises(errors.ConfigError, match=r"a metal body runs EEF checkpoints only"):
        run.policy_config(metal_app(), "dreamlabs/test-yam-joints")


def test_policy_config_of_a_metal_arm_carries_the_metal_ik_and_no_default_model() -> None:
    """The embodiment config, the loop's rate and the client's IK of the Metal arm; the model only when named."""
    app = metal_app()

    built = run.policy_config(app)
    named = run.policy_config(app, "dreamlabs/metal-eef-ft")

    assert built == {
        "embodiment": WIRE.embodiment(list(ROLES), 48),
        "action_hz": 30.0,
        "sync_above_ms": run.SYNC_ABOVE_MS,
        "ik": {"arm": "metal", "head": "eef", "solve": "traj"},
    }
    assert list(named) == ["checkpoint", "embodiment", "action_hz", "sync_above_ms", "ik"]
    assert run.loop_settings(app)["wire"] is WIRE


def test_the_station_files_checkpoint_is_the_model_unless_a_run_names_another(tmp_path: pathlib.Path) -> None:
    app = load(tmp_path, {**METAL_FILE, "checkpoint": "dream-1-ft:tester-metal"})

    assert app.checkpoint == "dream-1-ft:tester-metal"
    assert run.policy_config(app)["checkpoint"] == "dream-1-ft:tester-metal"
    assert run.policy_config(app, "dream-1-ft:other")["checkpoint"] == "dream-1-ft:other"


def test_a_one_shot_metal_run_without_a_checkpoint_fails_before_opening_an_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`robots.run.main` has no model to fall back on for a Metal arm: it says to pass `--checkpoint`."""

    def no_arm(spec: motor_arm.ArmSpec, cfg: config.ArmConfig, name: str, **_: Any) -> followers.FollowerArm:
        raise AssertionError("no arm may be opened")

    monkeypatch.setattr(motor_arm, "open_arm", no_arm)

    with pytest.raises(errors.ConfigError, match=r"no checkpoint: a metal body has no default model"):
        asyncio.run(run.main(metal_app()))


def test_a_metal_home_of_seven_angles_is_refused_before_an_arm_opens(monkeypatch: pytest.MonkeyPatch) -> None:
    """A home of seven angles (the last 0) loads, but dream's model of the Metal arm has six joints: the run refuses
    it before an arm opens or moves home, not once the loop binds its wire to the arms."""

    def no_arm(spec: motor_arm.ArmSpec, cfg: config.ArmConfig, name: str, **_: Any) -> followers.FollowerArm:
        raise AssertionError("no arm may be opened")

    monkeypatch.setattr(motor_arm, "open_arm", no_arm)
    seven = config.ArmConfig(can_channel="can_left", home=(*HOME, 0.0), **metal_fakes.GRIPPER_ENDS)

    with pytest.raises(errors.ConfigError, match=r"arm 'left' has 7 joints .* the metal arm has 6"):
        asyncio.run(run.main(metal_app(follower_l=seven), "dream-1-ft:metal"))


ENABLE_FRAME = b"\xff" * 7 + b"\xfc"


async def _dry_run(app: config.AppConfig, client: conftest.FakeClient) -> None:
    """`robots.run.main` as `dreamcli robot run --dry-run` runs it, until the policy has answered twice."""
    stop = asyncio.Event()
    main = asyncio.create_task(run.main(app, "dream-1-ft:metal", dry_run=True, stop=stop, home_duration_s=0.0))
    deadline_s = time.monotonic() + 30.0
    while client.calls < 2 and not main.done() and time.monotonic() < deadline_s:
        await asyncio.sleep(0.01)
    stop.set()
    await main


def _fake_cameras(monkeypatch: pytest.MonkeyPatch) -> None:
    """The run's cameras over fake capture sources."""

    def readers(cfg: config.AppConfig) -> list[cameras.CameraReader]:
        source = conftest.FakeCaptureSource
        return [cameras.CameraReader(c.name, source(), cfg.stale_after_s, face_blur=False) for c in cfg.cameras]

    monkeypatch.setattr(run, "make_cameras", readers)


def test_a_dry_run_switches_no_metal_motor_on_so_the_next_run_opens_the_arms_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")
    factory = metal_fakes.FakeCanFactory()
    monkeypatch.setattr(robstride, "open_can_bus", factory)
    app = metal_app()
    _fake_cameras(monkeypatch)
    for attempt in (1, 2):
        client = conftest.FakeClient(horizon=app.action_horizon)
        monkeypatch.setattr(run, "open_client", lambda dream_config, client=client: client)
        asyncio.run(_dry_run(app, client))
        assert client.calls >= 2, f"dry run {attempt} did not run the policy"
        for channel in ("can_left", "can_right"):
            bus = factory.buses[channel]
            assert bus.closed and ENABLE_FRAME not in [data for _, data in bus.frames], (attempt, channel)
            assert not any(motor.enabled for motor in bus.motors.values()), (attempt, channel)


def test_a_warm_motor_refuses_a_real_run_but_not_a_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")
    factory = metal_fakes.FakeCanFactory()
    monkeypatch.setattr(robstride, "open_can_bus", factory)
    factory.bus("can_right").motors[3].temp_rotor_c = int(metal.ARM.temp_warn_c) + 3
    app = metal_app()
    _fake_cameras(monkeypatch)
    client = conftest.FakeClient(horizon=app.action_horizon)
    monkeypatch.setattr(run, "open_client", lambda dream_config: client)

    asyncio.run(_dry_run(app, client))
    assert client.calls >= 2, "the dry run ran the policy on the warm arm"
    with pytest.raises(
        errors.ArmError, match=r"the right arm's elbow_flex is at 58 C \(coil\): let it cool below 55 C"
    ):
        asyncio.run(run.main(app, "dream-1-ft:metal", stop=asyncio.Event(), home_duration_s=0.0))
    for channel in ("can_left", "can_right"):
        assert factory.buses[channel].closed, channel
        assert not any(motor.enabled for motor in factory.buses[channel].motors.values()), channel


# Where the body's driver is picked


def test_home_opens_the_bodys_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    """`robots.home.main` on a Metal config opens both arms on the Metal table, each with its gripper ends."""
    opened: list[tuple[str, str, str, Any]] = []

    def open_arm(spec: motor_arm.ArmSpec, cfg: config.ArmConfig, name: str, **_: Any) -> followers.FollowerArm:
        opened.append((spec.name, cfg.can_channel, name, (cfg.gripper_open_deg, cfg.gripper_closed_deg)))
        return conftest.yam_arm(name, conftest.FakeRobot())

    monkeypatch.setattr(motor_arm, "open_arm", open_arm)
    monkeypatch.setattr(home.time, "sleep", lambda dt_s: None)

    assert home.main(metal_app(), duration_s=0.1) == 0
    assert opened == [
        ("metal", "can_left", "left", (120.0, 0.0)),
        ("metal", "can_right", "right", (120.0, 0.0)),
    ]


def test_metal_kinematics_is_dreams_forward_kinematics_of_the_tool() -> None:
    """The speed cap's end effector is the tool centre point of dream's Metal model, where the tool poses are."""
    pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")
    from dream_ik.solver import ChunkIK
    from dream_ik.specs import SERVE

    models = bodies.METAL.kinematics(metal_app())
    reference = ChunkIK("metal", SERVE.tool, 30.0)
    poses = [np.asarray(reference.rest, dtype=np.float64), np.array([0.3, -0.9, 1.1, 0.2, -0.1, 0.25])]

    assert set(models) == {"left", "right"}
    for q in poses:
        np.testing.assert_allclose(models["right"].ee_position(q), reference.fk(q)[:3, 3], atol=1e-9)
    assert isinstance(models["left"], kinematics.DreamKinematics)


def test_the_driver_frame_is_dreams_metal_model() -> None:
    """The driver's soft limits are dream's Metal URDF limits to the degree, joint for joint."""
    pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")
    from dream_ik.specs import METAL_ARM

    urdf = ElementTree.parse(METAL_ARM.urdf).getroot()
    for joint, row in zip(METAL_ARM.joints, metal.JOINTS[:6], strict=True):
        element = urdf.find(f"joint[@name='{joint}']/limit")
        assert element is not None, joint
        urdf_deg = tuple(round(math.degrees(float(element.get(end, "nan")))) for end in ("lower", "upper"))
        assert urdf_deg == row.limits_deg, row.name

"""The SO-101 (`robots.so101`): its config, its servos on LeRobot's Feetech bus (`robots.feetech`, on the fake bus of
`so101_fakes`), the shared bus-arm thread on its five joints, and its body's wire with the dream client's IK.

Most tests step the thread's tick by hand on a fake clock, so rates are exact; the park before torque-off runs the
thread on the wall clock.
"""

import asyncio
import datetime
import math
import pathlib
import sys
import time
from typing import Any

import conftest
import makers_fakes
import numpy as np
import pytest
import so101_fakes
import test_types_adapter
import yaml

from robots import (
    adapter,
    bodies,
    cameras,
    config,
    errors,
    feetech,
    followers,
    home,
    motor_arm,
    run,
    so101,
    station,
    types,
)

## The right arm's serial port, by id.
PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B8E114915-if00"

## A home as a file gives it, radians: five joints, the sixth 0.
HOME = [0.1, -1.6, 1.4, 1.0, -1.4, 0.0]

OPEN_GOAL_TICKS = 3358

## The shipped SO-101 example config.
EXAMPLE_PATH = pathlib.Path(__file__).resolve().parent.parent / "configs" / "so101.example.yaml"

## An SO-101 fine-tune's embodiment config, the layout of an SO-101 station's datasets: six-wide arms (five joints and
## the zero pad, in the checkpoint's units), the gripper in LeRobot's 0..100, the tool poses and the grips.
SO101_EMBODIMENT: dict[str, Any] = {
    "state": {
        "left_arm": [0, 6],
        "left_gripper": [6, 7],
        "right_arm": [7, 13],
        "right_gripper": [13, 14],
        "right_eef_xyz": [14, 17],
        "right_eef_rot6": [17, 23],
        "right_grip": [23, 24],
        "left_eef_xyz": [24, 27],
        "left_eef_rot6": [27, 33],
        "left_grip": [33, 34],
    },
    "action": {
        "right_eef_xyz": [0, 3],
        "right_eef_rot6": [3, 9],
        "right_grip": [9, 10],
        "left_eef_xyz": [10, 13],
        "left_eef_rot6": [13, 19],
        "left_grip": [19, 20],
        "left_arm": [20, 26],
        "right_arm": [26, 32],
    },
    "video": {
        "top": "observation.images.top",
        "left_wrist": "observation.images.left_wrist",
        "right_wrist": "observation.images.right_wrist",
    },
}


def so101_file(tmp_path: pathlib.Path, **right: Any) -> dict[str, Any]:
    """The smallest SO-101 file, with both arms' calibration files written next to it and the right arm's section
    updated by `right`."""
    for side in ("left", "right"):
        so101_fakes.write_calibration(tmp_path / f"{side}.json")
    return {
        "body": "so101",
        "embodiment": "tester/so101-pick",
        "follower_l": {"can_channel": PORT.replace("114915", "113054"), "calibration": "left.json", "home": HOME},
        "follower_r": {"can_channel": PORT, "calibration": "right.json", "home": HOME, **right},
        "cameras": [
            {"name": name, "device": i, "width": 64, "height": 48} for i, name in enumerate(SO101_EMBODIMENT["video"])
        ],
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


def arm_config(tmp_path: pathlib.Path) -> config.ArmConfig:
    """The right arm's config over a written calibration file, as load() builds it."""
    path = str(so101_fakes.write_calibration(tmp_path / "right.json"))
    open_deg, closed_deg = so101.gripper_ends(path)
    return config.ArmConfig(
        can_channel=PORT, gripper_open_deg=open_deg, gripper_closed_deg=closed_deg, calibration=path, home=tuple(HOME)
    )


class Rig:
    """One SO-101 arm on the fake bus: the factory and the arm's thread over its chain, on a fake clock."""

    def __init__(self, tmp_path: pathlib.Path, present: dict[str, int] | None = None) -> None:
        self.clock = makers_fakes.FakeClock()
        self.factory = so101_fakes.FakeFeetechFactory()
        self.factory.present.update(present or {})
        self.arm = motor_arm.MotorArm(
            so101.ARM, "right", arm_config(tmp_path), opener=self.factory, clock=self.clock, sleep=self.clock.sleep
        )

    @property
    def bus(self) -> so101_fakes.FakeFeetechBus:
        """The newest bus."""
        return self.factory.bus(PORT)

    def open(self) -> "Rig":
        """Opens and energizes the arm, then runs its first tick."""
        self.arm.chain.open()
        self.arm.chain.enable()
        self.tick()
        return self

    def tick(self, count: int = 1) -> None:
        """Runs ticks at the arm's rate."""
        for _ in range(count):
            self.arm.tick()
            self.clock.sleep(self.arm.period)


# The config


def test_an_so101_file_loads_its_ports_calibrations_and_homes(tmp_path: pathlib.Path) -> None:
    """The calibration path is read from the file's directory and stored absolute; the gripper's ends are its
    calibrated range; the embodiment is the file's; the loop runs at 20 Hz."""
    cfg = load(tmp_path, so101_file(tmp_path))

    assert cfg.body == "so101" and cfg.embodiment == "tester/so101-pick"
    assert cfg.control_hz == config.SO101_CONTROL_HZ == 20.0
    assert cfg.follower_r.can_channel == PORT
    assert cfg.follower_r.calibration == str(tmp_path / "right.json")
    assert cfg.follower_r.gripper_closed_deg == pytest.approx((2000 - 2048) * 360 / 4096)
    assert cfg.follower_r.gripper_open_deg == pytest.approx((3400 - 2048) * 360 / 4096)
    assert cfg.follower_r.home == tuple(HOME)


def test_an_so101_port_may_be_named_by_its_usb_port(tmp_path: pathlib.Path) -> None:
    """Two adapters that report one serial share one by-id link, so the wizard names each by its USB port."""
    port = "/dev/serial/by-path/pci-0000:00:14.0-usb-0:2:1.0"

    cfg = load(tmp_path, so101_file(tmp_path, can_channel=port))

    assert cfg.follower_r.can_channel == port


@pytest.mark.parametrize(
    ("right", "message"),
    [
        ({"can_channel": "/dev/ttyACM0"}, r"name the arm's serial port by id"),
        ({"home": [0.1, -1.6, 1.4, 1.0, -1.4, 0.2]}, r"follower_r.home' ends in 0.2: an SO-101 has five joints"),
        ({"calibration": "nowhere.json"}, r"nowhere.json is not there: calibrate the arm with dreamcli setup"),
        ({"gripper_open_deg": 90.0}, r"unknown"),
    ],
)
def test_an_so101_file_refuses_what_its_arm_cannot_run(tmp_path: pathlib.Path, right: Any, message: str) -> None:
    """A port by name, a sixth joint off zero, a calibration file that is not there, a Metal key."""
    with pytest.raises(errors.ConfigError, match=message):
        load(tmp_path, so101_file(tmp_path, **right))


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"elbow_flex": (4, 0, 860, 3090)}, r"gives 'elbow_flex' id 4 and drive mode 0; an SO-101 has it at id 3"),
        ({"gripper": (6, 0, 3400, 2000)}, r"gives 'gripper' the range 3400 to 2000, not a range of ticks"),
    ],
)
def test_a_calibration_file_that_does_not_fit_the_arm_is_refused(
    tmp_path: pathlib.Path, entry: dict[str, tuple[int, int, int, int]], message: str
) -> None:
    """A servo at another id or a range that is no range fails at load, naming the file."""
    document = so101_file(tmp_path)
    so101_fakes.write_calibration(tmp_path / "right.json", **entry)

    with pytest.raises(errors.ConfigError, match=message):
        load(tmp_path, document)


def test_the_embodiment_key_is_the_so101s_alone(tmp_path: pathlib.Path) -> None:
    """An SO-101 file must name its embodiment; any other body's file may not."""
    document = so101_file(tmp_path)
    del document["embodiment"]
    with pytest.raises(errors.ConfigError, match=r"'embodiment'"):
        load(tmp_path, document)

    yam = {**{k: v for k, v in so101_file(tmp_path).items() if k != "body"}, "embodiment": "tester/so101-pick"}
    yam["follower_l"] = {"can_channel": "can_left"}
    yam["follower_r"] = {"can_channel": "can_right"}
    with pytest.raises(errors.ConfigError, match=r"'embodiment' is for 'body: so101' only; this file's body is 'yam'"):
        load(tmp_path, yam)


def test_the_shipped_so101_example_loads_once_its_files_and_homes_are_filled(tmp_path: pathlib.Path) -> None:
    """The example's keys are the ones load() takes: with real calibration files and homes it loads."""
    document = yaml.safe_load(EXAMPLE_PATH.read_text(encoding="utf-8"))
    for side, key in (("left", "follower_l"), ("right", "follower_r")):
        document[key]["calibration"] = str(so101_fakes.write_calibration(tmp_path / f"{side}.json"))
        document[key]["home"] = HOME
    cfg = load(tmp_path, document)

    assert cfg.body == "so101" and cfg.control_hz == 20.0 and cfg.grip_force_cap == 0.0
    assert [camera.name for camera in cfg.cameras] == list(SO101_EMBODIMENT["video"])


# The servos


def test_ticks_are_degrees_from_the_middle_pose() -> None:
    """LeRobot's calibration puts the middle pose at 2048 ticks: the URDF's zero."""
    assert feetech.ticks_to_deg(2048) == 0.0
    assert feetech.ticks_to_deg(3072) == 90.0
    assert feetech.deg_to_ticks(-90.0) == 1024
    assert feetech.deg_to_ticks(feetech.ticks_to_deg(2777)) == 2777


def test_an_arm_whose_servos_hold_another_calibration_does_not_start(tmp_path: pathlib.Path) -> None:
    """The file of another arm (or a recalibration since) is refused before anything is written or switched."""
    rig = Rig(tmp_path)
    rig.factory.held = {"elbow_flex": feetech.Calibration(3, 0, 0, 860, 3090)}

    with pytest.raises(errors.ArmError, match=r"right.json: elbow_flex homing_offset 0 \(file -77\)\. Is that file"):
        rig.arm.chain.open()

    assert not rig.bus.connected and not rig.bus.configured and rig.bus.torque_log == []


def test_each_arm_is_matched_to_the_calibration_file_its_servos_hold(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    factory = so101_fakes.FakeFeetechFactory()
    monkeypatch.setattr(feetech, "open_bus", factory)
    factory.eeprom["/dev/b"] = {"elbow_flex": feetech.Calibration(3, 0, 40, 900, 3000)}
    a = str(so101_fakes.write_calibration(tmp_path / "a.json"))
    a_copy = str(so101_fakes.write_calibration(tmp_path / "a_copy.json"))
    b = str(so101_fakes.write_calibration(tmp_path / "b.json", elbow_flex=(3, 40, 900, 3000)))

    assert so101.calibration_for("/dev/a", [b, a]) == a
    assert so101.calibration_for("/dev/b", [a, b]) == b
    assert so101.calibration_for("/dev/a", [a_copy, a, b]) == a_copy
    assert all(bus.torque_log == [] and not bus.connected for buses in factory.buses.values() for bus in buses)
    with pytest.raises(errors.ArmError, match=r"none of your calibration files \(a.json\) is this arm's"):
        so101.calibration_for("/dev/b", [a])


def test_lerobots_missing_servo_text_names_each_servo() -> None:
    message = "\n".join(
        [
            "FeetechMotorsBus motor check failed on port '/dev/x':",
            "\nMissing motor IDs:",
            "  - 3 (expected model: 777)",
            "  - 9 (expected model: 777)",
            "\nMotors with incorrect model numbers:",
            "  - 4 (wrist_flex): expected 777, found 1",
            "\nFull expected motor list (id: model_number):",
            "{1: 777, 2: 777}",
        ]
    )
    assert feetech.missing_servos(message, so101.JOINTS) == ["servo 3 (elbow_flex)", "servo 9"]
    assert feetech.missing_servos("Failed to write", so101.JOINTS) == []


def test_open_configures_the_servos_as_lerobots_follower_does(tmp_path: pathlib.Path) -> None:
    """Off while configured; position mode, P 16, I 0, D 32 on every servo; the gripper's torque caps."""
    rig = Rig(tmp_path)
    rig.arm.chain.open()

    assert rig.bus.configured and rig.bus.torque_log == ["off"]
    for joint in so101.JOINTS:
        assert [rig.bus.registers[(name, joint.name)] for name in ("Operating_Mode", "P_Coefficient")] == [0, 16]
        assert [rig.bus.registers[(name, joint.name)] for name in ("I_Coefficient", "D_Coefficient")] == [0, 32]
    assert rig.bus.registers[("Max_Torque_Limit", "gripper")] == 500
    assert rig.bus.registers[("Protection_Current", "gripper")] == 250
    assert rig.bus.registers[("Overload_Torque", "gripper")] == 25


def test_enable_holds_each_servo_where_it_is(tmp_path: pathlib.Path) -> None:
    """The goal registers are set to the readings before torque goes on, so nothing jumps."""
    present = {"shoulder_pan": 2100, "elbow_flex": 2900, "gripper": 2500}
    rig = Rig(tmp_path, present)
    rig.arm.chain.open()
    rig.arm.chain.enable()

    assert rig.bus.torque_log == ["off", "on"]
    assert rig.bus.goals == [{**dict.fromkeys(rig.bus.names, 2048), **present}]


def test_the_thread_drives_five_joints_and_holds_the_sixth_at_zero(tmp_path: pathlib.Path) -> None:
    """The arm reads and is driven in the loop's six slots; the sixth reads 0 and a target there is clamped away."""
    rig = Rig(tmp_path).open()
    target = np.array([0.2, -0.3, 0.4, 0.1, -0.5, 0.7])

    for _ in range(300):
        rig.arm.write(target, 1.0)
        rig.tick()

    reading = rig.arm.read()
    np.testing.assert_allclose(reading.positions_rad[:5], target[:5], atol=math.radians(0.1))
    assert reading.positions_rad[5] == 0.0 and reading.velocities_rad_s.shape == (6,)
    assert reading.gripper == pytest.approx(1.0 - so101.ARM.grip_margin_mm / so101.GRIP_SCALE)
    assert all(list(goal) == list(rig.bus.names) for goal in rig.bus.goals)
    assert rig.bus.present["gripper"] == OPEN_GOAL_TICKS
    assert rig.bus.present["shoulder_pan"] == feetech.deg_to_ticks(math.degrees(0.2))
    assert rig.arm.alarm() is None


@pytest.mark.parametrize(("ticks", "grip"), [(2000, 0.0), (3400, 1.0), (2700, 0.5), (1900, 0.0)])
def test_the_gripper_runs_linearly_between_its_calibrated_ends(
    tmp_path: pathlib.Path, ticks: int, grip: float
) -> None:
    """Closed at the range's low end, open at its high end: LeRobot's 0..100 over 100."""
    rig = Rig(tmp_path, {"gripper": ticks}).open()

    assert rig.arm.read().gripper == pytest.approx(grip, abs=1e-3)


def test_a_missed_read_holds_the_arm_and_a_long_silence_latches_it(tmp_path: pathlib.Path) -> None:
    """A lost status packet costs one tick with nothing written; #robots.feetech.MAX_MISSED_S of them, the arm."""
    rig = Rig(tmp_path).open()
    written = len(rig.bus.goals)

    rig.bus.fail_reads = 2
    rig.tick(2)
    assert len(rig.bus.goals) == written and not rig.arm.fault
    rig.tick()
    assert len(rig.bus.goals) == written + 1 and rig.arm.chain.drops() == 2

    rig.bus.fail_reads = 1000
    rig.tick(int(feetech.MAX_MISSED_S * so101.ARM.tick_hz) + 2)
    assert "no position from" in rig.arm.fault


def test_a_servo_in_overload_does_not_keep_the_others_on_and_is_named_with_what_to_do(tmp_path: pathlib.Path) -> None:
    rig = Rig(tmp_path).open()
    bus = rig.bus
    bus.overload = {"gripper"}

    with pytest.raises(errors.ArmError, match=r"gripper \(id 6\): it reports an overload") as raised:
        rig.arm.chain.close(disable_torque=True)

    assert "the other servos are off" in str(raised.value) and "switch its power off and on" in str(raised.value)
    assert not any(bus.torque[name] for name in bus.names[:-1])
    assert not bus.connected


def test_servo_temperatures_are_read_once_a_second_and_a_hot_servo_raises_the_arms_alarm(
    tmp_path: pathlib.Path,
) -> None:
    rig = Rig(tmp_path).open()
    rig.bus.temperature["elbow_flex"] = 61

    rig.tick(int(2.5 * so101.ARM.tick_hz))

    assert 2 <= rig.bus.temperature_reads <= 4
    alarm = rig.arm.alarm()
    assert alarm is not None and alarm.startswith("elbow_flex servo reached 61 C (the run stops at 60 C)")
    assert rig.arm.read().motor_temps_c == (35.0, 35.0, 61.0, 35.0, 35.0, 35.0)


def test_a_gripper_closed_on_an_object_is_held_in_a_gentle_squeeze_and_an_open_lets_go(tmp_path: pathlib.Path) -> None:
    rig = Rig(tmp_path).open()
    joints = np.zeros(6)
    for _ in range(120):
        rig.arm.write(joints, 1.0)
        rig.tick()
    rig.bus.floor["gripper"] = 2700

    for _ in range(120):
        rig.arm.write(joints, 0.0)
        rig.tick()

    ticks_per_unit = (3400 - 2000) / so101.GRIP_SCALE
    assert rig.bus.goals[-1]["gripper"] == pytest.approx(2700 - 3 * ticks_per_unit, abs=2)
    for _ in range(120):
        rig.arm.write(joints, 1.0)
        rig.tick()
    assert rig.bus.goals[-1]["gripper"] == OPEN_GOAL_TICKS
    assert rig.arm.alarm() is None


def test_no_gripper_goal_comes_within_its_margin_of_range_max(tmp_path: pathlib.Path) -> None:
    rig = Rig(tmp_path, {"gripper": 3400}).open()
    for _ in range(120):
        rig.arm.write(np.zeros(6), home.HOME_GRIPPER)
        rig.tick()
    rig.arm.request_rest()
    while not rig.arm.rested.is_set():
        rig.tick()

    sent = [goal["gripper"] for goal in rig.bus.goals[1:]]
    assert max(sent) == sent[-1] == OPEN_GOAL_TICKS
    assert rig.arm.rest_result == (True, "at rest")


def test_torque_off_parks_where_the_arm_was_opened_then_switches_it_off(tmp_path: pathlib.Path) -> None:
    """The arm has no brakes: it travels back to its opening pose, gripper open, before the servos go off."""
    factory = so101_fakes.FakeFeetechFactory()
    factory.present = {"shoulder_lift": 1200, "elbow_flex": 2900}
    arm = motor_arm.open_arm(so101.ARM, arm_config(tmp_path), "right", opener=factory, clock=time.monotonic)
    bus = factory.bus(PORT)
    opened = arm.read().positions_rad.copy()

    moved = opened + np.array([0.1, 0.1, -0.1, 0.0, 0.0, 0.0])
    deadline = time.monotonic() + 2.0
    while abs(arm.read().positions_rad[1] - moved[1]) > 0.01:
        assert time.monotonic() < deadline, "the arm never reached its target"
        arm.write(types.ArmCommand(positions_rad=moved, gripper=1.0))
        time.sleep(0.02)
    arm.torque_off()

    assert bus.present["shoulder_lift"] == pytest.approx(1200, abs=60)
    assert bus.torque_log[-1] == "off" and not bus.connected


def test_torque_off_on_an_arm_whose_reads_stopped_leaves_its_servos_on_and_says_so(tmp_path: pathlib.Path) -> None:
    rig = Rig(tmp_path).open()
    rig.arm.start()
    rig.bus.fail_reads = 10_000
    deadline = time.monotonic() + 5.0
    while rig.arm.is_alive():
        assert time.monotonic() < deadline, "the arm never latched"
        time.sleep(0.01)
    arm = followers.FollowerArm("right", rig.arm, home_rad=np.zeros(6))

    with pytest.raises(errors.ArmError, match=r"^arm 'right' stopped answering while it held \(.*no position from"):
        arm.torque_off()

    assert all(rig.bus.torque.values()) and rig.bus.torque_log[-1] == "on"
    assert not rig.bus.connected


def test_robot_down_leaves_an_so101_arm_whose_reads_stopped_holding_and_never_logs_it_switched_off(
    tmp_path: pathlib.Path, monkeypatch: Any
) -> None:
    factory = so101_fakes.FakeFeetechFactory()
    factory.present.update(shoulder_lift=1100, elbow_flex=3000, wrist_flex=2800)
    monkeypatch.setattr(feetech, "open_bus", factory)
    bodies.so101_body.cache_clear()
    monkeypatch.setattr(bodies, "embodiment_config", lambda name, key: SO101_EMBODIMENT)
    monkeypatch.setattr(
        cameras.CameraReader,
        "from_config",
        lambda cfg, stale_after_s, **_encoding: cameras.CameraReader(
            cfg.name, conftest.FakeCaptureSource(), stale_after_s
        ),
    )
    controller = station.StationController.from_config(
        load(tmp_path, {**so101_file(tmp_path), "task": conftest.TASK}), inference_log_path=None
    )
    right, left = factory.bus(PORT), factory.bus(PORT.replace("114915", "113054"))

    async def scenario() -> bool:
        right.fail_reads = 10**9
        deadline = time.monotonic() + 10.0
        while controller.arms_alive()["right"]:
            assert time.monotonic() < deadline, "the right arm never latched"
            await asyncio.sleep(0.02)
        return await controller.close()

    assert asyncio.run(scenario()) is False
    assert all(right.torque.values()) and right.torque_log[-1] == "on" and not right.connected
    assert left.torque_log[-1] == "off" and not left.connected


# The body


def test_the_so101_body_runs_eef_only_with_dreams_so101_ik(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    """The client solves with `so101_faas`; the wire is the account's embodiment, read when first asked for."""
    asked: list[tuple[str, str]] = []
    bodies.so101_body.cache_clear()
    monkeypatch.setattr(bodies, "embodiment_config", lambda name, key: asked.append((name, key)) or SO101_EMBODIMENT)
    cfg = load(tmp_path, so101_file(tmp_path))
    body = bodies.for_config(cfg)

    assert body is bodies.so101_body("tester/so101-pick") and body.name == "so101" and body.checkpoint is None
    assert body.ik == {"arm": "so101_faas", "head": "eef"}
    run.refuse_joint_space(body)
    assert asked == []
    pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")
    assert body.wire().camera_roles == tuple(SO101_EMBODIMENT["video"])
    assert asked == [("tester/so101-pick", "embodiment")]


def test_the_body_opens_an_arm_on_lerobots_bus(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    """The body's open_arm looks the bus opener up when called, and the arm holds the config's home."""
    factory = so101_fakes.FakeFeetechFactory()
    monkeypatch.setattr(feetech, "open_bus", factory)
    cfg = load(tmp_path, so101_file(tmp_path))

    arm = bodies.for_config(cfg).open_arm(cfg.follower_r, "right")
    try:
        assert isinstance(arm, followers.FollowerArm) and factory.bus(PORT).torque_log == ["off", "on"]
        np.testing.assert_array_equal(arm.home_rad, HOME)
    finally:
        arm.close()


def test_the_dream_client_takes_the_wire_and_makes_the_checkpoints_units() -> None:
    """The loop sends five joints in radians and the grip; dream's client adds the tool poses, puts the held sixth
    joint back, converts the joints to the checkpoint's units and makes the gripper part, so the observation is the
    embodiment's, and a chunk of five joints and the grip comes back as the loop's six slots."""
    pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")
    from dream import client_ik, models
    from dream_ik.specs import EMBODIMENTS

    wire = adapter.EefWire(SO101_EMBODIMENT, None, "so101", bodies.kinematics.arm_layout("so101_faas"))
    fine_tune = models.Model(
        id="dream-1-ft:367786a6-so101-eef-14b2300",
        owner="public",
        base_model="dreamlabs/dream-1-beta",
        embodiment="emb_29cd2ba8bb9b70b5",
        servable=True,
        created_at=datetime.datetime(2026, 10, 10, tzinfo=datetime.timezone.utc),
        cameras=list(SO101_EMBODIMENT["video"]),
    )
    ik = client_ik.ClientIK({"arm": "so101_faas", "head": "eef"}, 30.0, SO101_EMBODIMENT, fine_tune)
    observation = test_types_adapter.make_observation(camera_names=tuple(SO101_EMBODIMENT["video"]))

    sent = ik.observe(test_types_adapter.sent(wire.to_dream(observation, conftest.TASK)))

    for side, arm in (("right", observation.state.right), ("left", observation.state.left)):
        for part, (start, end) in SO101_EMBODIMENT["state"].items():
            if part.startswith(side):
                assert np.asarray(sent[f"state.{part}"]).shape == (end - start,), part
        scale, offset = (np.asarray(u) for u in EMBODIMENTS["so101_faas"].side_arm(side).joint_units)
        units = (np.append(arm.positions_rad[:5], 0.0) - offset) / scale
        np.testing.assert_allclose(sent[f"state.{side}_arm"], units)
        np.testing.assert_allclose(sent[f"state.{side}_gripper"], [arm.gripper * 100.0])

    rows = np.column_stack([np.full((4, 5), 0.3), np.full(4, 0.6)])
    chunk = wire.from_dream({"q_r_chunk": rows, "q_l_chunk": rows}, 0.0, 4)
    np.testing.assert_array_equal(chunk.actions[0].right.positions_rad, [0.3] * 5 + [0.0])


class So101Client(conftest.FakeClient):
    """#conftest.FakeClient answering as the dream client does for an SO-101: five joints (small angles) and the
    gripper per row."""

    def get_action(self, *, commanded: Any = None) -> dict[str, Any]:
        chunk = super().get_action(commanded=commanded)
        return {key: [[0.001 * x for x in row[:5]] + row[6:] for row in rows] for key, rows in chunk.items()}


def test_a_dry_run_runs_the_policy_on_both_arms_and_moves_neither(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    """`dreamcli robot run --dry-run` on an SO-101: the arms open on LeRobot's bus read only, the policy gets five
    joints per arm, and no servo is switched on or sent a goal; the arms are only closed at the end."""
    pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")
    factory = so101_fakes.FakeFeetechFactory()
    factory.present.update(shoulder_lift=1100, elbow_flex=3000, wrist_flex=2800)
    monkeypatch.setattr(feetech, "open_bus", factory)
    bodies.so101_body.cache_clear()
    monkeypatch.setattr(bodies, "embodiment_config", lambda name, key: SO101_EMBODIMENT)
    cfg = load(tmp_path, {**so101_file(tmp_path), "task": conftest.TASK})
    client = So101Client(horizon=cfg.action_horizon)
    readers = [
        cameras.CameraReader(name, conftest.FakeCaptureSource(), cfg.stale_after_s, face_blur=False)
        for name in SO101_EMBODIMENT["video"]
    ]
    monkeypatch.setattr(run, "make_cameras", lambda app: readers)
    monkeypatch.setattr(run, "open_client", lambda dream_config: client)

    async def scenario() -> None:
        stop = asyncio.Event()
        main = asyncio.create_task(run.main(cfg, "dream-1-ft:so101", dry_run=True, stop=stop, home_duration_s=0.0))
        deadline_s = time.monotonic() + 30.0
        while client.calls < 2 and not main.done() and time.monotonic() < deadline_s:
            await asyncio.sleep(0.01)
        stop.set()
        await main

    asyncio.run(scenario())

    assert client.calls >= 2
    sent = client.observations[-1]
    assert np.asarray(sent["state.right_arm"]).shape == (5,) and "state.right_gripper" not in sent
    for port in (PORT, PORT.replace("114915", "113054")):
        bus = factory.bus(port)
        assert bus.goals == [], f"{port}: a goal reached a servo"
        assert bus.torque_log == ["off"] and not bus.connected


def test_a_missing_lerobot_names_the_real_install_command(monkeypatch: Any) -> None:
    monkeypatch.setitem(sys.modules, "lerobot", None)
    monkeypatch.setitem(sys.modules, "lerobot.motors", None)
    with pytest.raises(errors.ConfigError) as refused:
        feetech.open_bus("/dev/null", (), {})
    text = str(refused.value)
    assert 'pip install "dreamlabs[so101]"' in text and "~/dream-env" in text and "dream-robots" not in text

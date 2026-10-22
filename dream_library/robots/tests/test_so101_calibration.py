"""The SO-101's calibration in the setup wizard (`robots.so101_calibration`), on a fake arm whose servos read their
true angle less their homing offset (so101_fakes.CalibratingBus): the middle pose is set by a hand a few degrees off,
every joint is moved to its stops, and the calibration written puts each joint's zero where the model's arm has it."""

import datetime
import math
import pathlib

import pytest
import so101_fakes

from robots import errors, feetech, setup_model, setup_session, so101, so101_calibration

## Where each servo's horn sits on its spline, ticks: what the raw angle adds to the joint's own angle. Any values do.
HORN = {
    "shoulder_pan": 310,
    "shoulder_lift": -220,
    "elbow_flex": 95,
    "wrist_flex": -400,
    "wrist_roll": 700,
    "gripper": 0,
}
## How far the hand's middle pose is from the model arm's, degrees, per joint.
HAND_ERROR_DEG = {"shoulder_pan": 4.0, "shoulder_lift": -6.5, "elbow_flex": 9.0, "wrist_flex": -2.0}
## The gripper's ends, raw ticks.
GRIP_CLOSED, GRIP_OPEN = 2000, 3400

NOW = datetime.datetime(2026, 10, 16, 9, 30, 0)


@pytest.fixture(autouse=True)
def _dream_ik() -> None:
    if so101.ARM.ik not in setup_model.spec_names():
        pytest.skip(f"this dream IK has no {so101.ARM.ik}")


def ticks(deg: float) -> float:
    return deg * feetech.TICKS_PER_TURN / 360.0


class Hand:
    """The operator's hands on one arm: joints put at angles of the model arm's calibration, in its ticks."""

    def __init__(self, side: str) -> None:
        self.ranges = so101_calibration.reference(side)
        middle = {name: so101_calibration.HALF_TURN_TICK + ticks(HAND_ERROR_DEG.get(name, 0.0)) for name in HORN}
        middle["gripper"] = GRIP_CLOSED
        self.bus = so101_fakes.CalibratingBus(self._raw(middle))
        self.middle = middle

    def _raw(self, joints: dict[str, float]) -> dict[str, int]:
        return {name: round(joints[name]) + HORN[name] for name in joints}

    def put(self, name: str, model_ticks: float) -> None:
        """Puts one joint at an angle of the model arm's calibration (the gripper: at raw ticks)."""
        self.bus.actual[name] = round(model_ticks) + HORN[name]

    def sweep(self, calibration: so101_calibration.ArmCalibration, skip: tuple[str, ...] = ()) -> None:
        """Moves each joint with stops to both of them, the gripper open and closed, then back to the middle."""
        for name, ends in self.ranges.items():
            if name in skip:
                continue
            for end in (ends.low, ends.high, self.middle[name]):
                self.put(name, end)
                calibration.poll()
        for end in (GRIP_OPEN, GRIP_CLOSED):
            self.put("gripper", end)
            calibration.poll()


def test_the_reference_is_the_model_arms_calibrated_ranges() -> None:
    """Each joint with stops turns about 190 to 220 degrees end to end on the model's arm, its middle near the half
    turn; the wrist roll and the gripper have none."""
    for side in ("left", "right"):
        ranges = so101_calibration.reference(side)
        assert set(ranges) == {"shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"}
        for name, found in ranges.items():
            assert 180.0 < found.span_deg < 230.0, (side, name, found)
            assert abs(found.middle - so101_calibration.HALF_TURN_TICK) < ticks(20.0), (side, name, found)


def test_a_seated_home_past_the_models_range_is_named_and_the_drawn_home_is_inside_it() -> None:
    """The homes saved seated put both arms' shoulder and elbow past the range dream's
    model was recorded on, so every run's commands stopped about 2 degrees inside it, short of home. The drawn home is
    inside it on both sides."""
    seated = {"left": [-0.089, -1.83, 1.6843, -0.0445, -1.5785], "right": [-0.0568, -1.8162, 1.695, -0.0092, -1.6091]}
    drawn = [math.radians(d) for d in setup_session.SO101_HOME_DEG[:5]]
    for side, home in seated.items():
        assert so101_calibration.past_reference(side, home) == [1, 2], side
        assert so101_calibration.past_reference(side, drawn) == [], side


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_middle_pose_set_off_by_hand_is_moved_to_the_model_arms_and_written_to_the_servos_and_file(
    side: str, tmp_path: pathlib.Path
) -> None:
    hand = Hand(side)
    calibration = so101_calibration.ArmCalibration(side, "/dev/ttyACM0", hand.bus, hand.ranges)
    assert hand.bus.torque_off and all(hand.bus.registers[("Operating_Mode", n)] == 0 for n in HORN)
    assert calibration.progress() == []
    calibration.set_middle()
    hand.sweep(calibration)
    assert all(p.done for p in calibration.progress())

    result = calibration.finish(tmp_path, f"{side}_ABC123", lambda: NOW)

    assert pathlib.Path(result.path) == tmp_path / f"{side}_ABC123.json"
    for name, deg in HAND_ERROR_DEG.items():
        assert result.moved_deg[name] == pytest.approx(deg, abs=0.2), (name, result.moved_deg)
    for name, ends in hand.ranges.items():
        for end in (ends.low, ends.high):
            hand.put(name, end)
            read = hand.bus.sync_read("Present_Position", normalize=False)[name]
            assert abs(read - end) <= 1.0, f"{name} at the model arm's {end:.0f} reads {read}"
    held = hand.bus.held()
    assert held["wrist_roll"].range_min == 0 and held["wrist_roll"].range_max == so101_calibration.MAX_TICK
    gripper = held["gripper"]
    assert (gripper.range_min, gripper.range_max) == (
        GRIP_CLOSED - gripper.homing_offset,
        GRIP_OPEN - gripper.homing_offset,
    )
    assert feetech.read_calibration(result.path, so101.JOINTS) == held, "the file is what the servos hold"
    assert feetech.calibration_differences(feetech.read_calibration(result.path, so101.JOINTS), held) == []


def test_finish_waits_for_every_joint_to_go_end_to_end(tmp_path: pathlib.Path) -> None:
    hand = Hand("left")
    calibration = so101_calibration.ArmCalibration("left", "/dev/ttyACM0", hand.bus, hand.ranges)
    calibration.set_middle()
    hand.sweep(calibration, skip=("elbow_flex",))
    with pytest.raises(ValueError, match="move elbow all the way"):
        calibration.finish(tmp_path, "left_A", lambda: NOW)
    assert not list(tmp_path.iterdir()), "nothing is written before every joint went end to end"
    hand.sweep(calibration)
    calibration.finish(tmp_path, "left_A", lambda: NOW)


def test_a_middle_pose_far_off_is_set_again(tmp_path: pathlib.Path) -> None:
    hand = Hand("left")
    hand.middle["shoulder_lift"] = so101_calibration.HALF_TURN_TICK + ticks(40.0)
    hand.put("shoulder_lift", hand.middle["shoulder_lift"])
    calibration = so101_calibration.ArmCalibration("left", "/dev/ttyACM0", hand.bus, hand.ranges)
    calibration.set_middle()
    hand.sweep(calibration)
    with pytest.raises(ValueError, match="shoulder was 40 degrees from its middle"):
        calibration.finish(tmp_path, "left_A", lambda: NOW)


def test_a_file_already_there_is_kept_beside_the_new_one(tmp_path: pathlib.Path) -> None:
    old = so101_fakes.write_calibration(tmp_path / "left_A.json")
    hand = Hand("left")
    calibration = so101_calibration.ArmCalibration("left", "/dev/ttyACM0", hand.bus, hand.ranges)
    calibration.set_middle()
    hand.sweep(calibration)
    result = calibration.finish(tmp_path, "left_A", lambda: NOW)
    backup = tmp_path / "left_A.json.20261016-093000.bak"
    assert backup.is_file() and pathlib.Path(result.path) == old
    assert any("left_A.json.20261016-093000.bak" in note for note in result.notes)
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["left_A.json"], "LeRobot's folder lists one file"


def test_the_file_name_is_the_side_and_the_adapters_serial() -> None:
    assert so101_calibration.file_name("left", {"serial": "5A7A/0185", "id": "x"}) == "left_5A7A-0185"
    assert so101_calibration.file_name("right", {"id": "serial-by-id-usb"}) == "right_serial-by-id-usb"


def test_a_servo_that_cannot_be_zeroed_is_named() -> None:
    """A raw angle at the very end of the turn needs a homing offset past what the servo holds."""
    hand = Hand("left")
    calibration = so101_calibration.ArmCalibration("left", "/dev/ttyACM0", hand.bus, hand.ranges)
    hand.bus.actual["elbow_flex"] = so101_calibration.MAX_TICK
    with pytest.raises(errors.ArmError, match="elbow_flex cannot be zeroed here"):
        calibration.set_middle()

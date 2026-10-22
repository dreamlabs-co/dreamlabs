"""Calibrating an SO-101 arm in the setup wizard, in place of LeRobot's `lerobot-calibrate`.

LeRobot's calibration of an SO-101 follower (`SOFollower.calibrate()` in LeRobot 0.6) is two things the operator does
by hand with the servos switched off: put the arm in its middle pose, which becomes every servo's zero (its homing
offset, so the pose reads LeRobot's half turn), then move every joint but the wrist roll from end to end, which records
each servo's range. The servos keep both in their EEPROM and LeRobot writes the same numbers to a calibration file.
This module does the same with the wizard's page in place of the terminal (#ArmCalibration), and writes the same file
in the same place, so LeRobot's own tools read it and the wizard's arms step opens the arm with it.

One thing it adds. A middle pose put by hand is a few degrees off, differently on every arm, and every model trained
on an SO-101 reads the arm's joints relative to the calibration of the arm it was recorded on: an arm zeroed elsewhere
sends every joint off by its own error, and dream's IK puts the hand off by as much, with no error anywhere. The ends a
joint is moved between are the arm's own mechanical stops, the same on every SO-101, so the middle of the recorded
range says where the operator's zero is against the reference arm's (#reference(), the arm dream's `so101_faas` model
was recorded on): the joint's zero is moved by the difference, whatever pose the hand found. The wrist roll turns all
the way round and has no stops, so its zero stays the hand's; the gripper's ends are its own range, read as such.

@see ArmCalibration, reference(), write_file()
"""

import dataclasses
import datetime
import json
import math
import pathlib
from collections.abc import Callable
from typing import Any

from robots import errors, feetech, so101

## The tick LeRobot's calibration puts the middle pose at: half of 4095 (`MotorsBus._get_half_turn_homings`).
HALF_TURN_TICK = (feetech.TICKS_PER_TURN - 1) // 2
## The highest tick a servo reads.
MAX_TICK = feetech.TICKS_PER_TURN - 1
## The servo with no stops: LeRobot gives it the whole turn.
FULL_TURN_JOINT = "wrist_roll"
## The gripper: its range is its own, so its zero is never moved.
GRIPPER_JOINT = "gripper"
## The largest homing offset an STS3215 holds (sign and 11 bits), ticks.
MAX_HOMING_OFFSET = 2047
## How much shorter than the reference's a joint's recorded range may be and still count as end to end, degrees.
SHORT_RANGE_DEG = 10.0
## How much longer than the reference's a joint's range may be before it is not the same arm's, degrees: its zero is
## then left where the hand put it.
LONG_RANGE_DEG = 10.0
## The most a middle pose put by hand is moved to the reference's, degrees; more means the arm was far from the middle
## pose when Set was pressed, and the operator sets it again.
MAX_SHIFT_DEG = 25.0
## The least range a joint must be moved through before Finish, degrees: the gripper and any joint with no reference.
MIN_RANGE_DEG = 30.0
## What the page calls each servo's joint (`setup_wizard.js`, JOINT_WORDS), so a message names the bar it means.
JOINT_WORDS = {
    "shoulder_pan": "base turn",
    "shoulder_lift": "shoulder",
    "elbow_flex": "elbow",
    "wrist_flex": "wrist bend",
    "wrist_roll": "wrist roll",
    "gripper": "gripper",
}
## The servos' position mode (`lerobot.motors.feetech.OperatingMode.POSITION`).
_POSITION_MODE = 0

_DEG_PER_TICK = 360.0 / feetech.TICKS_PER_TURN


@dataclasses.dataclass(frozen=True)
class Range:
    """One joint's range on the reference arm, in ticks of its own calibration.

    Attributes:
    - low: The low end, ticks.
    - high: The high end, ticks.
    """

    low: float
    high: float

    @property
    def middle(self) -> float:
        """The middle of the range, ticks."""
        return (self.low + self.high) / 2.0

    @property
    def span_deg(self) -> float:
        """How far the joint turns from end to end, degrees."""
        return (self.high - self.low) * _DEG_PER_TICK


def reference(side: str) -> dict[str, Range]:
    """The ranges of the arm dream's `so101_faas` model was recorded on, per joint with stops, for one side.

    The model's arm spec (`dream_ik.specs.FAAS_ARMS`, `so101_faas`) carries them as the map from that arm's LeRobot
    units, -100 to 100 across its calibrated range, to radians from its zero: `rad = offset + scale * unit`. The
    range's ends are then `offset - 100 * scale` and `offset + 100 * scale`. The gripper and the wrist roll are left
    out (#FULL_TURN_JOINT, #GRIPPER_JOINT).

    @param side `left` or `right`.
    @return Per joint name, its #Range in ticks.
    @throws ConfigError dream's IK is not installed, or has no `so101_faas` arm.
    """
    try:
        from dream_ik import specs
    except ImportError as exc:
        raise errors.ConfigError(
            f"calibrating an SO-101 needs dream's IK (dream_ik), which is not installed: {exc}"
        ) from exc
    spec = getattr(specs, "FAAS_ARMS", {}).get(so101.ARM.ik, {}).get(side)
    if spec is None or spec.joint_units is None:
        raise errors.ConfigError(f"this dream IK has no {so101.ARM.ik} {side} arm with its joint units")
    scale, offset = spec.joint_units
    ranges = {}
    for i, joint in enumerate(so101.JOINTS[:-1]):
        if joint.name == FULL_TURN_JOINT:
            continue
        middle = HALF_TURN_TICK + offset[i] / math.radians(_DEG_PER_TICK)
        half = 100.0 * scale[i] / math.radians(_DEG_PER_TICK)
        ranges[joint.name] = Range(middle - half, middle + half)
    return ranges


def reference_ends(side: str) -> dict[int, tuple[float, float]]:
    """#reference()'s ranges as radians from the middle pose, low end first.

    @param side `left` or `right`.
    @return Per joint with a range, 0-based base to wrist, its ends.
    @throws ConfigError dream's IK is not installed, or has no `so101_faas` arm.
    """
    rad = math.radians(_DEG_PER_TICK)
    ends = {
        name: sorted(((r.low - HALF_TURN_TICK) * rad, (r.high - HALF_TURN_TICK) * rad))
        for name, r in reference(side).items()
    }
    return {i: (ends[j.name][0], ends[j.name][1]) for i, j in enumerate(so101.JOINTS[:-1]) if j.name in ends}


def past_reference(side: str, joints: list[float]) -> list[int]:
    """The joints a pose puts past #reference(), where a model never sends them.

    A home there leaves the arm no room: a seated SO-101 elbow sits about 7 degrees past it.

    @param side `left` or `right`.
    @param joints The pose, radians from the middle pose, base to wrist.
    @return The joints past their range, 0-based.
    @throws ConfigError dream's IK is not installed, or has no `so101_faas` arm.
    """
    return [i for i, (low, high) in reference_ends(side).items() if not low <= joints[i] <= high]


@dataclasses.dataclass(frozen=True)
class JointProgress:
    """One joint as the page draws it while the operator moves it.

    Attributes:
    - name: The servo's name.
    - now_deg: Where it is, degrees from the middle pose set.
    - low_deg: The lowest it has been since, degrees.
    - high_deg: The highest, degrees.
    - target_deg: How far it should turn end to end (the reference's), or None for one with no reference.
    - done: Whether it has been moved far enough.
    """

    name: str
    now_deg: float
    low_deg: float
    high_deg: float
    target_deg: float | None
    done: bool


@dataclasses.dataclass(frozen=True)
class Result:
    """A finished calibration.

    Attributes:
    - path: The calibration file written.
    - entries: Each servo's #robots.feetech.Calibration, as written to the servos and the file.
    - moved_deg: Per joint whose zero was moved to the reference's, how far the hand's middle pose was from it,
      degrees (positive: past the reference's middle, the way the joint's readings rise).
    - notes: What the operator should know: a joint whose range is not the reference's, a backup made.
    """

    path: str
    entries: dict[str, feetech.Calibration]
    moved_deg: dict[str, float]
    notes: tuple[str, ...]


class ArmCalibration:
    """One SO-101 arm being calibrated: its servos off, its middle pose set, its ranges recorded as the hand moves it.

    The stages, in order: `middle` (the servos are off and the operator puts the arm in its middle pose; set_middle()),
    `range` (poll() records each joint's lowest and highest reading as the operator moves it; finish()), then `done`.
    Not thread-safe: the wizard's session lock guards it.

    Attributes:
    - side: `left` or `right`.
    - port: The arm's serial device.
    - stage: `middle`, `range` or `done`.
    """

    def __init__(self, side: str, port: str, bus: Any, ranges: dict[str, Range]) -> None:
        """Switches every servo off and puts it in position mode, as `lerobot-calibrate` does.

        @param side `left` or `right`.
        @param port The arm's serial device.
        @param bus The arm's LeRobot Feetech bus, opened without a calibration (#robots.feetech.open_bus).
        @param ranges The reference arm's ranges for this side (#reference()).
        """
        self.side = side
        self.port = port
        self.stage = "middle"
        self._bus = bus
        self._ranges = ranges
        self._names = [joint.name for joint in so101.JOINTS]
        self._homing: dict[str, int] = {}
        self._low: dict[str, int] = {}
        self._high: dict[str, int] = {}
        self._now: dict[str, int] = {}
        bus.disable_torque()
        for name in self._names:
            bus.write("Operating_Mode", name, _POSITION_MODE)

    def _read(self) -> dict[str, int]:
        return {name: int(ticks) for name, ticks in self._bus.sync_read("Present_Position", normalize=False).items()}

    def set_middle(self) -> None:
        """The arm is in its middle pose: each servo's zero becomes where it is now.

        As LeRobot's `set_half_turn_homings()`: every homing offset and range is reset to the factory's, the raw
        positions are read, and each servo's homing offset makes its position read #HALF_TURN_TICK.

        @throws ArmError A servo is so far round that no homing offset puts it at the half turn.
        """
        for name in self._names:
            self._bus.write("Homing_Offset", name, 0, normalize=False)
            self._bus.write("Min_Position_Limit", name, 0, normalize=False)
            self._bus.write("Max_Position_Limit", name, MAX_TICK, normalize=False)
        raw = self._read()
        homing = {name: raw[name] - HALF_TURN_TICK for name in self._names}
        far = [name for name, offset in homing.items() if abs(offset) > MAX_HOMING_OFFSET]
        if far:
            raise errors.ArmError(
                f"{', '.join(far)} cannot be zeroed here: put the arm in its middle pose and set it again"
            )
        for name, offset in homing.items():
            self._bus.write("Homing_Offset", name, offset, normalize=False)
        self._homing = homing
        self._now = self._read()
        self._low, self._high = dict(self._now), dict(self._now)
        self.stage = "range"

    def poll(self) -> None:
        """Reads every servo once and widens each joint's recorded range; a read that fails is skipped."""
        if self.stage != "range":
            return
        try:
            now = self._read()
        except (ConnectionError, OSError, RuntimeError):
            return
        self._now = now
        for name, ticks in now.items():
            self._low[name] = min(self._low[name], ticks)
            self._high[name] = max(self._high[name], ticks)

    def progress(self) -> list[JointProgress]:
        """Each joint's range so far, for the page; empty before the middle pose is set.

        @return One #JointProgress per servo but the wrist roll, base to gripper.
        """
        if self.stage == "middle":
            return []
        out = []
        for name in self._names:
            if name == FULL_TURN_JOINT:
                continue
            span = (self._high[name] - self._low[name]) * _DEG_PER_TICK
            target = self._ranges[name].span_deg if name in self._ranges else None
            enough = MIN_RANGE_DEG if target is None else target - SHORT_RANGE_DEG
            out.append(
                JointProgress(
                    name=name,
                    now_deg=round((self._now[name] - HALF_TURN_TICK) * _DEG_PER_TICK, 1),
                    low_deg=round((self._low[name] - HALF_TURN_TICK) * _DEG_PER_TICK, 1),
                    high_deg=round((self._high[name] - HALF_TURN_TICK) * _DEG_PER_TICK, 1),
                    target_deg=None if target is None else round(target, 1),
                    done=span >= enough,
                )
            )
        return out

    def finish(self, directory: pathlib.Path, name: str, now: Callable[[], datetime.datetime]) -> Result:
        """Writes the calibration to the servos and to a LeRobot calibration file, once every joint went end to end.

        Each joint with a reference has its zero moved by how far the middle of its recorded range is from the
        reference's, when the two ranges are the same length (#SHORT_RANGE_DEG, #LONG_RANGE_DEG) and the move is at
        most #MAX_SHIFT_DEG. The wrist roll gets the whole turn, as LeRobot gives it.

        @param directory Where to write the file (#robots.setup_session.SO101_CALIBRATION_DIR).
        @param name The file's name without `.json`: LeRobot's robot id.
        @param now The wall clock, for the name of a backup of a file already there.
        @return The #Result.
        @throws ValueError A joint has not been moved far enough, or its middle pose was set too far off.
        @throws ArmError A servo refused a write.
        """
        if self.stage != "range":
            raise ValueError("set the middle pose first")
        self.poll()
        short = [JOINT_WORDS[p.name] for p in self.progress() if not p.done]
        if short:
            raise ValueError(
                f"move {', '.join(short)} all the way from one end to the other: each bar fills when it has gone far "
                "enough"
            )
        entries: dict[str, feetech.Calibration] = {}
        moved: dict[str, float] = {}
        notes: list[str] = []
        for joint in so101.JOINTS:
            low, high, homing = self._low[joint.name], self._high[joint.name], self._homing[joint.name]
            if joint.name == FULL_TURN_JOINT:
                low, high = 0, MAX_TICK
            elif joint.name in self._ranges:
                ref = self._ranges[joint.name]
                span = (high - low) * _DEG_PER_TICK
                shift = round((low + high) / 2.0 - ref.middle)
                if span > ref.span_deg + LONG_RANGE_DEG:
                    notes.append(
                        f"{JOINT_WORDS[joint.name]} turns {span:.0f} degrees end to end, more than an SO-101's "
                        f"{ref.span_deg:.0f}: its zero is where you set it, not matched to the model's arm"
                    )
                elif abs(shift) * _DEG_PER_TICK > MAX_SHIFT_DEG:
                    raise ValueError(
                        f"{JOINT_WORDS[joint.name]} was {abs(shift) * _DEG_PER_TICK:.0f} degrees from its middle "
                        "when you set the middle pose: put the arm in the middle pose shown and press Set middle pose "
                        "again"
                    )
                elif shift:
                    # Present = actual - homing: raising the homing by the shift moves every reading down by it.
                    homing, low, high = homing + shift, low - shift, high - shift
                    # The hand's middle was where the range's middle is not: the shift's other way round.
                    moved[joint.name] = round(-shift * _DEG_PER_TICK, 1)
            if abs(homing) > MAX_HOMING_OFFSET or not 0 <= low < high <= MAX_TICK:
                raise errors.ArmError(
                    f"{JOINT_WORDS[joint.name]} cannot hold this calibration (homing offset {homing}, range "
                    f"{low} to {high}): put the arm in the middle pose shown and press Set middle pose again"
                )
            entries[joint.name] = feetech.Calibration(joint.motor_id, 0, homing, low, high)
        for joint_name, entry in entries.items():
            self._bus.write("Homing_Offset", joint_name, entry.homing_offset, normalize=False)
            self._bus.write("Min_Position_Limit", joint_name, entry.range_min, normalize=False)
            self._bus.write("Max_Position_Limit", joint_name, entry.range_max, normalize=False)
        path, backup = write_file(directory, name, entries, now)
        if backup is not None:
            notes.append(f"the file this arm had before is kept as {backup.name}")
        self.stage = "done"
        return Result(path=str(path), entries=entries, moved_deg=moved, notes=tuple(notes))

    def close(self) -> None:
        """Closes the port, the servos left off."""
        try:
            self._bus.disconnect(disable_torque=False)
        except Exception:
            pass


def write_file(
    directory: pathlib.Path, name: str, entries: dict[str, feetech.Calibration], now: Callable[[], datetime.datetime]
) -> tuple[pathlib.Path, pathlib.Path | None]:
    """Writes a LeRobot calibration file, as `lerobot-calibrate` does; a file already there is kept beside it first.

    @param directory The folder.
    @param name The file's name without `.json`.
    @param entries Each servo's calibration.
    @param now The wall clock, for the backup's name.
    @return The file written, and the backup made (None when there was no file).
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.json"
    backup = None
    if path.exists():
        backup = path.with_name(f"{path.name}.{now().strftime('%Y%m%d-%H%M%S')}.bak")
        path.replace(backup)
    document = {joint: dataclasses.asdict(entry) for joint, entry in entries.items()}
    path.write_text(json.dumps(document, indent=4) + "\n", encoding="utf-8")
    return path, backup


def file_name(side: str, device: dict[str, Any]) -> str:
    """The calibration file's name for an arm: its side and its adapter's serial number (else its device id).

    @param side `left` or `right`.
    @param device The arm's adapter (#robots.setup_devices.arm_adapters entry).
    @return A name of letters, digits, `_` and `-`.
    """
    tag = str(device.get("serial") or device.get("id") or "arm")
    return f"{side}_" + "".join(c if c.isalnum() or c in "-_" else "-" for c in tag).strip("-")

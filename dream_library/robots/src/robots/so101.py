"""An SO-101 arm, as a table for #robots.motor_arm; the only module that knows the SO-101.

The SO-101 (The Robot Studio and LeRobot's arm) has five joints and a gripper on six Feetech STS3215 servos, one
serial bus per arm. Its driver is the shared bus-arm thread (#robots.motor_arm.MotorArm) over a
#robots.feetech.FeetechChain on LeRobot's own Feetech bus (the `so101` extra): the SO-101 reference follower and
policy bridge, with their checks, rates and gains.

Units: joints in radians in the URDF's frame (dream's `so101_faas` model, one URDF per side), 0 at the LeRobot
calibration's middle pose (#robots.feetech says why that holds on every calibrated arm). The arm fills the loop's
first five joint slots; the sixth reads 0, where dream's IK holds the model's dummy sixth joint. The gripper runs
linearly between the ends of its calibrated range, closed at `range_min` (the model's 0) and open at `range_max`
(1): LeRobot's 0 to 100 gripper, the unit the checkpoint reads and the dream client makes from the grip.

Calibration: each arm's own LeRobot calibration file (`calibration` in the config, written by `lerobot-calibrate`
with the arm in its middle pose). An arm whose servos hold another calibration than the file does not start.

@see ARM, JOINTS
"""

import pathlib
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from robots import errors, feetech, motor_arm

if TYPE_CHECKING:
    # Annotations only: config reads the gripper's ends from here.
    from robots import config

## The servo's whole turn, degrees: a joint's limits are its calibrated range (#robots.feetech.FeetechChain).
_FULL_TURN_DEG = (-180.0, 180.0)

## The six servos, base to gripper, the same on both arms: name (LeRobot's), id, model, and the P and D registers
## LeRobot's SO-101 follower sets (16 and 32: its default P of 32 shakes).
JOINTS = (
    motor_arm.Joint("shoulder_pan", 1, "sts3215", 16.0, 32.0, _FULL_TURN_DEG),
    motor_arm.Joint("shoulder_lift", 2, "sts3215", 16.0, 32.0, _FULL_TURN_DEG),
    motor_arm.Joint("elbow_flex", 3, "sts3215", 16.0, 32.0, _FULL_TURN_DEG),
    motor_arm.Joint("wrist_flex", 4, "sts3215", 16.0, 32.0, _FULL_TURN_DEG),
    motor_arm.Joint("wrist_roll", 5, "sts3215", 16.0, 32.0, _FULL_TURN_DEG),
    motor_arm.Joint("gripper", 6, "sts3215", 16.0, 32.0, _FULL_TURN_DEG),
)
## The gripper's 0..1 as LeRobot's 0..100, the unit the thread ramps it in.
GRIP_SCALE = 100.0


def gripper_ends(calibration_path: str) -> tuple[float, float]:
    """The gripper's ends from an arm's calibration file, for the config's `gripper_open_deg` and `gripper_closed_deg`.

    @param calibration_path The arm's LeRobot calibration file.
    @return `(open_deg, closed_deg)`: the calibrated range's high and low end.
    @throws ConfigError The file does not fit the arm (#robots.feetech.read_calibration).
    """
    gripper = feetech.read_calibration(calibration_path, JOINTS)[JOINTS[-1].name]
    return feetech.ticks_to_deg(gripper.range_max), feetech.ticks_to_deg(gripper.range_min)


def calibration_for(port: str, files: Sequence[str]) -> str:
    """Which of the operator's calibration files an arm's servos hold, read once from their EEPROM.

    @param port The arm's serial device.
    @param files The calibration files to choose from.
    @return The first file the servos agree with.
    """
    held = feetech.held_calibration(port, JOINTS)
    for path in files:
        if not feetech.calibration_differences(feetech.read_calibration(path, JOINTS), held):
            return path
    names = ", ".join(pathlib.Path(path).name for path in files)
    raise errors.ArmError(
        f"none of your calibration files ({names}) is this arm's: its servos hold other homing offsets and ranges. "
        "Calibrate this arm with lerobot-calibrate, add the file it writes in step 0, then look again"
    )


def _chain(
    side: str,
    cfg: "config.ArmConfig",
    *,
    opener: Callable[..., Any],
    clock: Callable[[], float],
    sleep: Callable[[float], Any],
) -> feetech.FeetechChain:
    """The arm's servos on its serial port, with its calibration file.

    @throws ConfigError The config names no calibration file.
    """
    if cfg.calibration is None:
        raise errors.ConfigError(f"arm '{side}': an SO-101 needs its LeRobot calibration file ('calibration')")
    return feetech.FeetechChain(
        side, cfg.can_channel, JOINTS, cfg.calibration, opener=opener, clock=clock, sleep=sleep
    )


## The SO-101: a 60 Hz bus rate, parked before torque-off where it was opened (its middle pose
## stands up, which is no place to go limp) at 20 deg/s (about the follower's return speed), the gripper
## open.
ARM = motor_arm.ArmSpec(
    name="so101",
    ik="so101_faas",
    tick_hz=60.0,
    rest_speed_deg_s=20.0,
    rest_rad=None,
    grip_scale_mm=GRIP_SCALE,
    grip_margin_mm=3.0,
    temp_warn_c=50.0,
    temp_stop_c=60.0,
    chain=_chain,
)

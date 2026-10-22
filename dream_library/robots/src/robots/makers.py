"""A MakerMods Makers Arm, as a table for #robots.motor_arm; the only module that knows the Makers Arm.

The Makers Arm is the vendor's brakeless arm on seven RobStride motors (six joints and the gripper), one CAN adapter
per arm. Its driver is the shared bus-arm thread (#robots.motor_arm.MotorArm) over a #robots.robstride.RobstrideChain:
together they are a port of the Makers Arm's reference driver (its follower, station and command stepper
at `a7da8d99`, the commit `tests/data/makers_parity.json` records), with every safety check it has;
`tests/test_makers_parity.py` holds them to its frames byte for byte.

Not ported: teleoperation and the Star leaders, calibration (setting the motors' zero), the adapter reopen after an
I/O failure (here an adapter that fails latches the arm down, and the station's Reset reopens it), and the raw-USB
candleLight transport (#robots.robstride says why).

Units: joints are the turn-corrected motor angles in radians with no sign flip, the frame the Makers datasets were
recorded in and dream's IK model reads (`dream_ik`'s `maker_arm_ik.urdf`). The gripper's stroke runs from 0 closed to
#GRIPPER_MAX_MM open, linear in the motor angle between the config's `gripper_closed_deg` and `gripper_open_deg`.

@see ARM, JOINTS
"""

from collections.abc import Callable
from typing import Any

from robots import config, motor_arm, robstride

## The six joints, base to wrist, the same on both arms: name, CAN id, motor model (RS02 `"O1"` on the shoulder lift
## and the elbow, RS00 `"O0"` elsewhere), the vendor's bench MIT `kp` and `kd`, and soft limits in degrees from the
## folded calibration pose (the vendor's mechanical-stop capture).
JOINTS = (
    motor_arm.Joint("shoulder_pan", 1, "O0", 60.0, 4.0, (-158.2, 156.1)),
    motor_arm.Joint("shoulder_lift", 2, "O1", 150.0, 4.5, (-175.3, -3.2)),
    motor_arm.Joint("elbow_flex", 3, "O1", 90.0, 3.0, (2.8, 236.1)),
    motor_arm.Joint("wrist_flex", 4, "O0", 30.0, 2.0, (-65.2, 104.0)),
    motor_arm.Joint("wrist_yaw", 5, "O0", 30.0, 2.0, (-97.6, 78.0)),
    motor_arm.Joint("wrist_roll", 6, "O0", 30.0, 2.0, (-153.2, 152.0)),
)
## The gripper's stroke, mm: the Makers datasets record the gripper in it, and the loop's 0..1 is its
## fraction.
GRIPPER_MAX_MM = 80.0


def _gripper(cfg: config.ArmConfig) -> motor_arm.Joint:
    """The gripper's motor: CAN id 7, its `kp` low so grip force stays compliant, its soft limits the stroke's ends."""
    return motor_arm.Joint("gripper", 7, "O0", 20.0, 0.5, (cfg.gripper_open_deg, cfg.gripper_closed_deg))


def _chain(
    side: str,
    cfg: config.ArmConfig,
    *,
    opener: Callable[[str], Any],
    clock: Callable[[], float],
    sleep: Callable[[float], Any],
) -> robstride.RobstrideChain:
    """The arm's RobStride motors, the gripper through its hold (`gripper_hold_torque_nm`)."""
    joints = (*JOINTS, _gripper(cfg))
    return robstride.RobstrideChain(
        side, cfg.can_channel, joints, cfg.gripper_hold_torque_nm, opener=opener, clock=clock, sleep=sleep
    )


## The Makers Arm: a 60 Hz policy tick, parked before torque-off at its folded calibration pose
## (every joint at zero) at 30 deg/s (the vendor's MAKER_RETURN_SPEED_DEG_S), the gripper open.
ARM = motor_arm.ArmSpec(
    name="makers",
    ik="makers",
    tick_hz=60.0,
    rest_speed_deg_s=30.0,
    rest_rad=(0.0,) * 6,
    grip_scale_mm=GRIPPER_MAX_MM,
    grip_margin_mm=2.0,
    temp_warn_c=55.0,
    temp_stop_c=65.0,
    chain=_chain,
)

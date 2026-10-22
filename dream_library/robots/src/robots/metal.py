"""A MakerMods Metal arm, as a table for #robots.motor_arm; the only module that knows the Metal arm.

The Metal arm is MakerMods' brakeless arm on seven Damiao motors (six joints and the gripper), one CAN adapter per
arm. Its driver is the shared bus-arm thread (#robots.motor_arm.MotorArm) over a #robots.damiao.DamiaoChain, which is
MakerMods' own follower for it, from the control loop they run policies on it with (`metal-control-loop`:
`MetalFollower` of their LeRobot fork at `5ce2fe4` and the `MetalRobot` wrapper of their console), with every check
it makes; `tests/test_metal_parity.py` holds it to their frames byte for byte. The gripper is
#robots.metal_gripper.

Not ported: teleoperation, calibration (setting the motors' zero), the console's planner, its tracking-error check
(the park before torque-off has the shared thread's stall and time limits instead), and the gripper's current-limited
mode and holding thread (#robots.metal_gripper says why).

Units: joints are the motor angles in radians, sign +1, no offset: MakerMods' frame, which is dream's Metal IK model
(`dream_ik`'s Metal URDF, whose limits are these to the degree). The gripper runs linearly in its motor angle from the
arm's `gripper_closed_deg` (0) to its `gripper_open_deg` (#GRIP_SCALE_MM, the loop's 1.0), both required and recorded
per arm by the setup wizard (in place of MakerMods' measured stroke table). The loop's gripper is the
Metal grip, the stroke in mm over #GRIP_SCALE_MM, what dream's Metal fine-tunes read and answer.

Calibration: MakerMods' LeRobot calibration file carries nothing a driver reads. Their follower checks only that one
exists (its homing offsets are 0, its ranges the soft limits); the zero lives in the motors, stored by LeRobot's
calibration with the arm at its zero pose (upright, every joint at 0, the gripper closed). An arm needs its motors
zeroed at that pose, and no file.

@see ARM, JOINTS
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from robots import config, damiao, errors, metal_gripper, motor_arm

## The seven motors, base to gripper, the same on both arms: name, CAN id, Damiao motor model (#robots.damiao
## .MOTOR_LIMITS: MakerMods flash each with the ranges of its joint), MakerMods' follow `kp` and `kd`, and soft limits
## in degrees from the zero pose (the vendor's motor configuration). Every joint also carries #gravity_nm, as
## MakerMods' SDK follower does (with no gravity torque the shoulder sits 0.1 rad under its target
## and a lifted hand stays on the table). Their SDK's stiffer gains are not taken: they go with its 120 Hz smoothed
## command stream, and on this 30 Hz tick they made the arm jerky.
JOINTS = (
    motor_arm.Joint("shoulder_pan", 1, "metal_jlo", 100.0, 5.0, (-160.0, 160.0)),
    motor_arm.Joint("shoulder_lift", 2, "metal_j2", 100.0, 5.0, (-180.0, 0.0)),
    motor_arm.Joint("elbow_flex", 3, "metal_jlo", 100.0, 5.0, (0.0, 180.0)),
    motor_arm.Joint("wrist_flex", 4, "metal_jlo", 50.0, 5.0, (-123.0, 81.0)),
    motor_arm.Joint("wrist_yaw", 5, "metal_jhi", 20.0, 0.6, (-85.0, 85.0)),
    motor_arm.Joint("wrist_roll", 6, "metal_jhi", 20.0, 0.6, (-145.0, 145.0)),
    motor_arm.Joint(metal_gripper.NAME, 7, "metal_jhi", 20.0, 0.6, metal_gripper.LIMITS_DEG),
)
## The arm's links for #gravity_nm, from MakerMods' `metal_with_gripper.urdf` (the SDK's): per joint, base to wrist,
## its origin in the link before it (xyz m, rpy rad), its axis, and the link it turns: mass kg, centre of mass m.
LINKS = (
    ((0.0, 0.0, 0.06675), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.893, (0.000693, -0.001221, 0.050197)),
    ((0.00075, 0.0, 0.055), (1.5708, 0.0, 0.0), (0.0, 0.0, 1.0), 1.028, (-0.185969, -0.001045, -0.000524)),
    ((-0.28, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.838, (0.13281, 0.052976, -0.000025)),
    ((0.215, 0.063, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.444, (0.067326, 0.059806, 0.005043)),
    ((0.080817, 0.05072, 0.00025), (1.5708, 0.0, -0.015918), (0.0, 0.0, 1.0), 0.363, (0.002596, 0.000018, 0.039895)),
    ((0.0235, 0.0, 0.044), (-3.14159265330655, 0.0, 0.0), (1.0, 0.0, 0.0), 0.667, (0.05958, 0.0, -0.006779)),
)
## The SDK's factor on each joint's gravity torque (KdlSolver::FeedforwardTorqueCompensation).
GRAVITY_FACTORS = (1.2, 1.15, 1.1, 1.15, 1.0, 1.0)
GRAVITY_M_S2 = 9.81
## The Metal grip, the loop's gripper unit and the one dream's Metal fine-tunes read and answer in, is the stroke in
## mm divided by this.
GRIP_SCALE_MM = 100.0


def _rotation(axis: npt.NDArray[np.float64], angle: float) -> npt.NDArray[np.float64]:
    """The rotation by `angle` rad about the unit `axis` (Rodrigues).

    @param axis The unit axis, 3 numbers.
    @param angle The angle, radians.
    @return The 3x3 rotation matrix.
    """
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    rotation: npt.NDArray[np.float64] = np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * k @ k
    return rotation


def _rpy(roll: float, pitch: float, yaw: float) -> npt.NDArray[np.float64]:
    """A URDF origin's rotation: yaw about z, then pitch about y, then roll about x, fixed axes.

    @param roll About x, radians.
    @param pitch About y, radians.
    @param yaw About z, radians.
    @return The 3x3 rotation matrix.
    """
    x, y, z = np.eye(3)
    rotation: npt.NDArray[np.float64] = _rotation(z, yaw) @ _rotation(y, pitch) @ _rotation(x, roll)
    return rotation


def gravity_nm(q_deg: tuple[float, ...]) -> tuple[float, ...]:
    """The torque each joint holds the arm against gravity with, times the SDK's #GRAVITY_FACTORS.

    Per joint: its axis dotted with the moment of the weights of every link it carries.

    @param q_deg The arm's joints, degrees, base to wrist.
    @return Per joint, base to wrist, its gravity torque, N.m.
    """
    rotation, origin = np.eye(3), np.zeros(3)
    axes, pivots, weights = [], [], []
    for (xyz, rpy, axis, mass, com), angle in zip(LINKS, q_deg, strict=True):
        origin = origin + rotation @ np.asarray(xyz)
        rotation = rotation @ _rpy(*rpy)
        axes.append(rotation @ np.asarray(axis))
        pivots.append(origin)
        rotation = rotation @ _rotation(np.asarray(axis), math.radians(angle))
        weights.append((origin + rotation @ np.asarray(com), mass * GRAVITY_M_S2 * np.array([0.0, 0.0, 1.0])))
    return tuple(
        factor * float(sum(axes[i] @ np.cross(com - pivots[i], weight) for com, weight in weights[i:]))
        for i, factor in enumerate(GRAVITY_FACTORS)
    )


def _chain(
    side: str,
    cfg: config.ArmConfig,
    *,
    opener: Callable[[str], Any],
    clock: Callable[[], float],
    sleep: Callable[[float], Any],
) -> damiao.DamiaoChain:
    """The arm's Damiao motors, the gripper holding at `gripper_hold_torque_nm`.

    @throws ConfigError The config has no hold torque.
    """
    del sleep
    hold_torque_nm = cfg.gripper_hold_torque_nm
    if hold_torque_nm is None:
        raise errors.ConfigError(f"arm '{side}': a Metal arm's gripper always holds; set gripper_hold_torque_nm")
    gripper = JOINTS[-1]

    def build_gripper(bus: damiao.DamiaoBus) -> metal_gripper.MetalGripper:
        return metal_gripper.MetalGripper(
            bus, gains=(gripper.kp, gripper.kd), hold_torque_nm=hold_torque_nm, clock=clock
        )

    return damiao.DamiaoChain(
        side, cfg.can_channel, JOINTS, build_gripper, gravity=gravity_nm, opener=opener, clock=clock
    )


## The Metal arm: 30 Hz (the Metal datasets' rate), parked before torque-off where it was opened, at 15 deg/s
## (MakerMods' console), the gripper open: its zero pose stands straight up, which is no place to go limp.
ARM = motor_arm.ArmSpec(
    name="metal",
    ik="metal",
    tick_hz=30.0,
    rest_speed_deg_s=15.0,
    rest_rad=None,
    grip_scale_mm=GRIP_SCALE_MM,
    grip_margin_mm=2.0,
    temp_warn_c=55.0,
    temp_stop_c=65.0,
    chain=_chain,
)

"""The seam join: how each chunk takes over from the motion the arm is on; internal to `dream.smoothing`.

The join replaces the rows of the new plan from the row the loop installs it at (the clock row) through
#RAMP_ROWS after it. Those rows start exactly where the old motion goes next, carry its velocity, blend it into the
new plan's velocity, and land exactly on the new plan's row at the end of the window, so the commanded trajectory
is continuous in position and velocity across the seam and the arm is back on the model's plan after the window.

The velocity blend, per step `k` of `N`: `u_k = (1 - w_k) * v_old + w_k * (plan[k + 1] - plan[k])` with
`w_k = (k + 1) / (N + 1)`, integrated from the old motion's next position; summed velocities alone miss the plan's
end row, so the miss is removed along the quintic smoothstep `S(s) = 10s^3 - 15s^4 + 6s^5`, whose velocity and
acceleration are zero at both ends. The old motion's position and velocity are a least-squares quadratic through
the last commanded row and up to three rows the old plan would command next (#fit_motion).

The blend runs in joint space (#velocity_blend) or, when the caller gives the arm's kinematics (#Kinematics), on
the grasp site's pose (#eef_window): position in metres and orientation as a world-frame rotation vector, then
solved back to joints row by row with damped least squares. The end-effector path is what the gripper actually
traces, so a seam that is smooth there is smooth where it matters; the joint-space blend is the fallback whenever
the solve fails or flips branch near a singularity.
"""

from typing import Protocol

import numpy as np
import numpy.typing as npt

## Rows the join spans after the clock row: the window the old motion's velocity blends into the plan's over. It
## must stay below the rows the loop executes between two installs, or the next seam cuts the window off early.
RAMP_ROWS = 16

## Damping of the least-squares IK step, `dq = J^T (J J^T + lambda^2 I)^-1 e`.
IK_DAMPING = 1e-3

## Most Gauss-Newton iterations per row; seeded from the previous row's solution, a row converges in two or three.
IK_MAX_ITERATIONS = 8

## A row is solved once its position error is under this many metres and its rotation error under the next.
IK_TOLERANCE_M = 1e-5
IK_TOLERANCE_RAD = 1e-4

## A row whose final error exceeds either of these fails the solve, and the arm's seam falls back to joint space.
FALLBACK_RESIDUAL_M = 1e-3
FALLBACK_RESIDUAL_RAD = 0.01

## Largest move of any joint, radians, from a row's seed to its solution: beyond it the solve jumped to another IK
## branch (near a singularity), which the arm would follow as a whip, so the seam falls back to joint space.
FALLBACK_SEED_JUMP_RAD = 0.25

## Times, in rows relative to the seam, of the old-motion rows `fit_motion` takes: the last commanded row, then up to
## three the old plan would command next.
OLD_PATH_TIMES = (-1.0, 0.0, 1.0, 2.0)


class Kinematics(Protocol):
    """One arm's forward kinematics at the end effector the seam is joined on (the grasp site).

    The robots package implements it on the arm's MuJoCo model; the client only calls it. The calls come from the
    thread that runs `get_action`, so an implementation the caller also uses elsewhere must not share mutable state
    with that use.
    """

    def pose(self, joints_rad: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The end effector's pose for the arm's joint angles.

        @param joints_rad Joint angles, shape `[joints]`, radians.
        @return `(position, rotation)`: shape `[3]` in metres and `[3, 3]` rotation matrix, in the arm's base frame.
        """
        ...

    def jacobian(self, joints_rad: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The end effector's Jacobian at the joint angles, in the frame `pose` reports.

        @param joints_rad Joint angles, shape `[joints]`, radians.
        @return Shape `[6, joints]`: rows 0..2 the linear velocity of the end-effector point, rows 3..5 the angular
            velocity, per unit joint velocity.
        """
        ...


def smoothstep5(s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The quintic smoothstep `10s^3 - 15s^4 + 6s^5`: 0 at 0, 1 at 1, zero velocity and acceleration at both ends.

    @param s Points in `[0, 1]`.
    @return The weights, same shape.
    """
    return s * s * s * (10.0 + s * (-15.0 + 6.0 * s))


def fit_motion(
    rows: npt.NDArray[np.float64], times: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The old motion's position and per-row velocity at time 0, from a least-squares fit through its rows.

    A quadratic through three or more rows, a line through two, a constant through one (an arm at rest).

    @param rows `[count, columns]` values at `times`.
    @param times `[count]` times, rows relative to the seam.
    @return `(now, velocity)`, each `[columns]`: the fit's value and slope at time 0.
    """
    count = rows.shape[0]
    if count == 1:
        return rows[0].copy(), np.zeros(rows.shape[1])
    basis = np.vander(times, min(3, count), increasing=True)
    coefficients = np.linalg.lstsq(basis, rows, rcond=None)[0]
    return coefficients[0], coefficients[1]


def velocity_blend(
    now: npt.NDArray[np.float64],
    velocity: npt.NDArray[np.float64],
    plan: npt.NDArray[np.float64],
    landing: bool = True,
) -> npt.NDArray[np.float64]:
    """Joins the old motion into `plan` over its rows: row 0 is `now`, the last row is `plan`'s exactly.

    @param now The old motion's next row, `[columns]`.
    @param velocity The old motion's velocity per row, `[columns]`.
    @param plan The plan's window rows, `[N + 1, columns]`, `N` at least 1.
    @param landing Remove the miss along the smoothstep; off only to check the blended steps on their own.
    @return `[N + 1, columns]`.
    """
    steps = blended_steps(velocity, np.diff(plan, axis=0))
    path = np.vstack([now, now + np.cumsum(steps, axis=0)])
    if landing:
        count = plan.shape[0] - 1
        path -= smoothstep5(np.arange(count + 1) / count)[:, None] * (path[-1] - plan[-1])
    return path


def blended_steps(velocity: npt.NDArray[np.float64], plan_steps: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Step `k` of `N` is `(1 - w_k) * velocity + w_k * plan_steps[k]`, `w_k = (k + 1) / (N + 1)`.

    @param velocity The old motion's velocity per row, `[columns]`.
    @param plan_steps The plan's row-to-row steps, `[N, columns]`.
    @return `[N, columns]`.
    """
    count = plan_steps.shape[0]
    weight = (np.arange(1, count + 1, dtype=np.float64) / (count + 1))[:, None]
    steps: npt.NDArray[np.float64] = (1.0 - weight) * velocity + weight * plan_steps
    return steps


def eef_window(
    old_path: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    plan: npt.NDArray[np.float64],
    joint_blend: npt.NDArray[np.float64],
    kinematics: Kinematics,
) -> npt.NDArray[np.float64] | None:
    """The velocity blend done on the end effector's pose and solved back to joints, or None when the solve fails.

    The old motion and the plan window go through `kinematics.pose`; positions and world-frame rotation vectors
    (relative to the old motion's pose at time 0) are fitted and blended as in #velocity_blend, the orientation
    integrated by left-multiplying each step's rotation; the miss is removed in position and in rotation along the
    smoothstep. Row 0 takes the old motion's joints and the last row the plan's, both exact; every row between is
    solved by damped least squares (#solve), seeded with the previous row's solution plus the joint blend's step.

    @param old_path `[count, joints]` old-motion rows at `times`.
    @param times `[count]` their times, rows relative to the seam.
    @param plan `[N + 1, joints]` the plan's window rows.
    @param joint_blend `[N + 1, joints]` the same join in joint space: the seeds and row 0.
    @param kinematics The arm's kinematics.
    @return `[N + 1, joints]`, or None when a row's residual exceeds #FALLBACK_RESIDUAL_M or #FALLBACK_RESIDUAL_RAD
        or its solution is more than #FALLBACK_SEED_JUMP_RAD from its seed.
    """
    count = plan.shape[0] - 1
    old_poses = [kinematics.pose(row) for row in old_path]
    reference = old_poses[int(np.argmin(np.abs(times)))][1]
    old_x = np.array([np.concatenate([p, rotvec(r @ reference.T)]) for p, r in old_poses])
    now, velocity = fit_motion(old_x, times)
    plan_poses = [kinematics.pose(row) for row in plan]
    plan_steps = np.array(
        [
            np.concatenate([p1 - p0, rotvec(r1 @ r0.T)])
            for (p0, r0), (p1, r1) in zip(plan_poses[:-1], plan_poses[1:], strict=True)
        ]
    )
    steps = blended_steps(velocity, plan_steps)
    positions = [now[:3]]
    rotations = [rotation_matrix(now[3:]) @ reference]
    for step in steps:
        positions.append(positions[-1] + step[:3])
        rotations.append(rotation_matrix(step[3:]) @ rotations[-1])
    miss_p = positions[-1] - plan_poses[-1][0]
    miss_r = rotvec(rotations[-1] @ plan_poses[-1][1].T)
    weights = smoothstep5(np.arange(count + 1) / count)
    out = joint_blend.copy()
    out[count] = plan[count]
    for k in range(1, count):
        target_p = positions[k] - weights[k] * miss_p
        target_r = rotation_matrix(-weights[k] * miss_r) @ rotations[k]
        seed = out[k - 1] + (joint_blend[k] - joint_blend[k - 1])
        solved = solve(kinematics, seed, target_p, target_r)
        if solved is None or np.max(np.abs(solved - seed)) > FALLBACK_SEED_JUMP_RAD:
            return None
        out[k] = solved
    return out


def solve(
    kinematics: Kinematics,
    seed: npt.NDArray[np.float64],
    target_p: npt.NDArray[np.float64],
    target_r: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64] | None:
    """Joint angles that put the end effector at the target pose, by damped least squares from `seed`.

    Joint limits are not applied: real plans sit on or slightly past the model's limits at rest, and the loop's
    own limit clamp has the last word on what reaches the motors.

    @param kinematics The arm's kinematics.
    @param seed `[joints]` starting angles.
    @param target_p `[3]` target position, metres.
    @param target_r `[3, 3]` target rotation.
    @return `[joints]`, or None when the final error exceeds #FALLBACK_RESIDUAL_M or #FALLBACK_RESIDUAL_RAD.
    """
    joints = seed.copy()
    damping = IK_DAMPING * IK_DAMPING * np.eye(6)
    for iteration in range(IK_MAX_ITERATIONS + 1):
        position, rotation = kinematics.pose(joints)
        error = np.concatenate([target_p - position, rotvec(target_r @ rotation.T)])
        error_p = float(np.linalg.norm(error[:3]))
        error_r = float(np.linalg.norm(error[3:]))
        if error_p < IK_TOLERANCE_M and error_r < IK_TOLERANCE_RAD:
            return joints
        if iteration == IK_MAX_ITERATIONS:
            break
        jacobian = kinematics.jacobian(joints)
        joints = joints + jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping, error)
    if error_p > FALLBACK_RESIDUAL_M or error_r > FALLBACK_RESIDUAL_RAD:
        return None
    return joints


def rotation_matrix(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The rotation of a rotation vector (axis times angle, radians), by Rodrigues' formula.

    @param vector `[3]`.
    @return `[3, 3]`.
    """
    angle = float(np.linalg.norm(vector))
    skew = _skew(vector)
    if angle < 1e-8:
        sine_term, cosine_term = 1.0, 0.5
    else:
        sine_term, cosine_term = np.sin(angle) / angle, (1.0 - np.cos(angle)) / (angle * angle)
    rotation: npt.NDArray[np.float64] = np.eye(3) + sine_term * skew + cosine_term * (skew @ skew)
    return rotation


def rotvec(rotation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The rotation vector (axis times angle, angle in `[0, pi]`) of a rotation matrix, through its unit quaternion.

    @param rotation `[3, 3]`.
    @return `[3]`.
    """
    quaternion = _quaternion(rotation)
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    sine = float(np.linalg.norm(quaternion[1:]))
    scale = 2.0 if sine < 1e-12 else 2.0 * float(np.arctan2(sine, quaternion[0])) / sine
    vector: npt.NDArray[np.float64] = scale * quaternion[1:]
    return vector


def _quaternion(rotation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """`[w, x, y, z]` of a rotation matrix, by the branch on its largest diagonal term (Shepperd's method)."""
    trace = float(np.trace(rotation))
    m = rotation
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    quaternion = np.array(q, dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def _skew(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The cross-product matrix of a 3-vector."""
    x, y, z = (float(value) for value in vector)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])

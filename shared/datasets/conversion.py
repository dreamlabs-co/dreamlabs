"""The values of a body with `parts` both ways: as the robot records them and as its plan trains on them.

A body whose config says what each part is (`shared.datasets.embodiments`, `parts`) trains on its `Plan`: each part
under the name training keys on, a rotation in any recorded format as the 6D of the end-effector contract (the first
two columns of the rotation matrix, `[R00, R10, R20, R01, R11, R21]`), a gripper as `(value - closed) / (open -
closed)`, 0 closed to 1 open and clipped to that range, and a pose part the plan records but does not train left out.
to_trained() goes that
way over a block of rows: the data service rewrites every episode with it, and the dream
client the observation it sends a fine-tune of the body. from_trained() comes back, which the dream client runs on the
fine-tune's answer, so a control loop reads its own parts: each under its own name, a 6D in the part's format, a grip
between the part's closed and open. A part the plan does not train has nothing to come back as and is left out; a
gripper the state trains twice comes back once. A gripper is clipped to 0-1 both ways, so a value past its confirmed
ends (the New robot page proposes them from the 1st and 99th percentile) trains as fully closed or open, and the robot
is never sent a grip past them.

An action part whose values are deltas (`embodiments.Answer.delta`, the change to the next step, `action_t = x(t+1) -
x(t)`, where `x` is the state part that trains under the same slot name) trains as its target, `state_t + action_t`:
to_targets() makes the targets from a recording, before to_trained(), and from_targets() turns a fine-tune's chunk of
targets back into changes, after from_trained(), the first from the observed state and each later one from the target
before it.

A rotation comes back in one form of each format: a unit quaternion with w at least 0; Euler angles in [-pi, pi] with
the middle one in [-pi/2, pi/2] (degrees alike); a rotation vector of angle in [0, pi]; 6D and matrices orthonormal.
A value recorded in that form comes back as it was, up to float error; any other form comes back as the same rotation
in that form.
"""

from collections.abc import Callable, Mapping

import numpy as np
import numpy.typing as npt

from shared.datasets import embodiments

## Every array here: float64 rows.
Floats = npt.NDArray[np.float64]

## The format the plan trains every rotation in: the 6D of `embodiments.EEF_PARTS`.
TRAINED_ROTATION: str = "rot6_cols"

## The shortest a quaternion, a rotation vector's axis or a 6D column may be and still name a rotation.
MIN_NORM: float = 1e-6


class NotARotationError(ValueError):
    """A rotation's value that names no rotation; `str(error)` names the part or slot and says why."""


def to_trained(body: embodiments.Embodiment, vector: str, parts: Mapping[str, Floats]) -> dict[str, Floats]:
    """One vector's parts as the plan trains them, slot by slot.

    @param body The parsed embodiment; it must have a plan.
    @param vector `state` or `action`.
    @param parts Each part of `vector` the plan trains to its rows, `(rows, part width)`; any other part is ignored.
    @return Each slot name to its rows, `(rows, slot width)` float64, in slot order.
    @throws NotARotationError When a rotation part's value is not a rotation.
    """
    assert body.plan is not None and body.parts is not None  # the caller converts only a body with parts
    out: dict[str, Floats] = {}
    for slot in body.plan.state if vector == "state" else body.plan.action:
        answer = body.parts[vector][slot.part]
        values = np.asarray(parts[slot.part], dtype=np.float64)  # (rows, part width)
        if answer.kind == "rotation":
            assert answer.format is not None  # a rotation that trains has its format (the plan)
            out[slot.name] = rotation(values, answer.format, TRAINED_ROTATION, f"part {slot.part!r}")
        elif answer.kind == "gripper":
            assert answer.closed is not None and answer.open is not None
            out[slot.name] = np.clip((values - answer.closed) / (answer.open - answer.closed), 0.0, 1.0)
        else:
            out[slot.name] = values
    return out


def from_trained(body: embodiments.Embodiment, vector: str, slots: Mapping[str, Floats]) -> dict[str, Floats]:
    """One vector's slots as the parts the robot records: to_trained() undone.

    @param body The parsed embodiment; it must have a plan.
    @param vector `state` or `action`.
    @param slots Each slot name of the plan's `vector` to its rows, `(rows, slot width)`.
    @return Each part the plan trains to its rows, `(rows, part width)` float64, in the vector's order; a part the plan
        records but does not train is left out.
    @throws NotARotationError When a 6D slot holds a vector of length zero.
    """
    assert body.plan is not None and body.parts is not None
    out: dict[str, Floats] = {}
    for slot in body.plan.state if vector == "state" else body.plan.action:
        if slot.part in out:
            continue  # a gripper the state trains twice holds one value in both slots
        answer = body.parts[vector][slot.part]
        values = np.asarray(slots[slot.name], dtype=np.float64)  # (rows, slot width)
        if answer.kind == "rotation":
            assert answer.format is not None
            out[slot.part] = rotation(values, TRAINED_ROTATION, answer.format, f"slot {slot.name!r}")
        elif answer.kind == "gripper":
            assert answer.closed is not None and answer.open is not None
            out[slot.part] = np.clip(values, 0.0, 1.0) * (answer.open - answer.closed) + answer.closed
        else:
            out[slot.part] = values
    return out


def to_targets(
    body: embodiments.Embodiment, state: Mapping[str, Floats], action: Mapping[str, Floats]
) -> dict[str, Floats]:
    """The action's parts with each delta part turned into its target on the same row, `state_t + action_t`.

    @param body The parsed embodiment; it must have a plan.
    @param state Each state part to its rows, `(rows, part width)`, as recorded.
    @param action Each action part to its rows, the same rows, as recorded.
    @return Each action part to its rows, float64: a delta part the plan trains as its targets, every other as given.
    """
    assert body.plan is not None and body.parts is not None
    sources = {slot.name: slot.part for slot in body.plan.state}
    out = {part: np.asarray(values, dtype=np.float64) for part, values in action.items()}
    for slot in body.plan.action:
        if body.parts["action"][slot.part].delta:
            # The parser checked the state trains a part under this slot's name at its width.
            out[slot.part] = np.asarray(state[sources[slot.name]], dtype=np.float64) + out[slot.part]  # (rows, width)
    return out


def from_targets(
    body: embodiments.Embodiment, state: Mapping[str, Floats], action: Mapping[str, Floats]
) -> dict[str, Floats]:
    """to_targets() undone over one answer: each delta part's chunk of targets as the change each step makes.

    The robot is at its last target when it takes the next step, so the first change is from the observed state and
    each later one from the target before it.

    @param body The parsed embodiment; it must have a plan.
    @param state Each state part to its observed row, `(1, part width)`, as the robot records it.
    @param action Each action part to its chunk, `(steps, part width)`, as from_trained() gives it.
    @return Each of those parts to its chunk, float64: a delta part's as changes, every other as given.
    """
    assert body.plan is not None and body.parts is not None
    sources = {slot.name: slot.part for slot in body.plan.state}
    out = {part: np.asarray(values, dtype=np.float64) for part, values in action.items()}
    for slot in body.plan.action:
        if body.parts["action"][slot.part].delta:
            targets = out[slot.part]  # (steps, width)
            observed = np.asarray(state[sources[slot.name]], dtype=np.float64)  # (1, width)
            before = np.concatenate([observed, targets[: len(targets) - 1]])  # (steps, width)
            out[slot.part] = targets - before
    return out


def rotation(values: Floats, given: str, wanted: str, what: str) -> Floats:
    """Rotations in one of `embodiments.ROTATION_FORMATS`, written in another, in the form the module docstring says.

    @param values `(rows, width)` in `given`.
    @param given The format `values` is in.
    @param wanted The format to write.
    @param what The part or slot, for the line.
    @return `(rows, wanted width)` float64.
    @throws NotARotationError When a quaternion, a 6D vector or a matrix column is of length zero, or a 6D's two
        vectors or a matrix's first two columns are parallel.
    """
    return _TO[wanted](_FROM[given](np.asarray(values, dtype=np.float64), what))


def _quaternion(wxyz: Floats, what: str) -> Floats:
    """Rotation matrices of `(rows, 4)` quaternions `w, x, y, z`, each normalized first."""
    norm = np.linalg.norm(wxyz, axis=1, keepdims=True)
    if (norm < MIN_NORM).any():
        raise NotARotationError(f"{what}: a quaternion has length zero, so it is not a rotation")
    w, x, y, z = (wxyz / norm).T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1),
        ],
        axis=1,
    )


def _about(axis: int, angle: Floats) -> Floats:
    """`(rows, 3, 3)` rotations by `angle` (radians) about the fixed axis x, y or z (0, 1, 2)."""
    c, s = np.cos(angle), np.sin(angle)
    one, zero = np.ones_like(angle), np.zeros_like(angle)
    rows = {
        0: [[one, zero, zero], [zero, c, -s], [zero, s, c]],
        1: [[c, zero, s], [zero, one, zero], [-s, zero, c]],
        2: [[c, -s, zero], [s, c, zero], [zero, zero, one]],
    }[axis]
    return np.stack([np.stack(row, axis=1) for row in rows], axis=1)


def _euler(order: str, degrees: bool) -> Callable[[Floats, str], Floats]:
    """Rotation matrices of Euler angles about fixed axes, the first value about the first axis named: `xyz` is
    `Rz(v2) Ry(v1) Rx(v0)`, URDF's `rpy`; `zyx` is `Rx(v2) Ry(v1) Rz(v0)`."""

    def read(values: Floats, what: str) -> Floats:
        angles = np.radians(values) if degrees else values
        out: Floats = np.broadcast_to(np.eye(3), (len(values), 3, 3))
        for k, axis in enumerate(order):
            out = _about("xyz".index(axis), angles[:, k]) @ out
        return out

    return read


def _axis_angle(values: Floats, what: str) -> Floats:
    """Rotation matrices of rotation vectors (angle times unit axis, radians): Rodrigues; a zero vector is no turn."""
    angle = np.linalg.norm(values, axis=1)  # (rows,)
    axis = values / np.where(angle < MIN_NORM, 1.0, angle)[:, None]  # (rows, 3), zero where there is no turn
    k: Floats = np.zeros((len(values), 3, 3))
    k[:, 0, 1], k[:, 0, 2], k[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    k = k - np.transpose(k, (0, 2, 1))  # the cross-product matrix of each axis
    s, c = np.sin(angle)[:, None, None], np.cos(angle)[:, None, None]
    out: Floats = np.eye(3) + s * k + (1 - c) * (k @ k)
    return out


def _rot6(columns: bool) -> Callable[[Floats, str], Floats]:
    """Rotation matrices of 6D rotations: the first two columns (`rot6_cols`) or rows (`rot6_rows`), made orthonormal
    and completed by their cross product; refused when either has length zero or the two are parallel."""

    def read(values: Floats, what: str) -> Floats:
        a, b = values[:, :3], values[:, 3:]  # (rows, 3) each
        if (np.linalg.norm(a, axis=1) < MIN_NORM).any() or (np.linalg.norm(b, axis=1) < MIN_NORM).any():
            raise NotARotationError(f"{what}: a 6D rotation has a vector of length zero")
        a = a / np.linalg.norm(a, axis=1, keepdims=True)
        b = b - (a * b).sum(axis=1, keepdims=True) * a
        if (np.linalg.norm(b, axis=1) < MIN_NORM * np.linalg.norm(values[:, 3:], axis=1)).any():
            raise NotARotationError(f"{what}: a 6D rotation's two vectors are parallel, so it is not a rotation")
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        stacked = np.stack([a, b, np.cross(a, b)], axis=2 if columns else 1)  # (rows, 3, 3)
        return stacked

    return read


def _matrix(columns: bool) -> Callable[[Floats, str], Floats]:
    """Rotation matrices of their nine entries, row by row (`matrix_rows`) or column by column (`matrix_cols`);
    refused when a first or second column has length zero or the two are parallel."""

    def read(values: Floats, what: str) -> Floats:
        out: Floats = values.reshape(len(values), 3, 3)
        if columns:
            out = np.transpose(out, (0, 2, 1))
        lengths = np.linalg.norm(out[:, :, :2], axis=1)  # (rows, 2)
        if (lengths < MIN_NORM).any():
            raise NotARotationError(f"{what}: a rotation matrix has a column of length zero")
        if (np.linalg.norm(np.cross(out[:, :, 0], out[:, :, 1]), axis=1) < MIN_NORM * lengths.prod(axis=1)).any():
            raise NotARotationError(
                f"{what}: a rotation matrix's first two columns are parallel, so it is not a rotation"
            )
        return out

    return read


## Each rotation format to the function reading it as `(rows, 3, 3)` matrices, whose columns are the rotated frame's
## axes (`embodiments.ROTATION_FORMATS`; the tests pin the key sets equal).
_FROM: dict[str, Callable[[Floats, str], Floats]] = {
    "quat_xyzw": lambda values, what: _quaternion(values[:, [3, 0, 1, 2]], what),
    "quat_wxyz": _quaternion,
    "euler_xyz": _euler("xyz", degrees=False),
    "euler_zyx": _euler("zyx", degrees=False),
    "euler_xyz_deg": _euler("xyz", degrees=True),
    "euler_zyx_deg": _euler("zyx", degrees=True),
    "axis_angle": _axis_angle,
    "rot6_cols": _rot6(columns=True),
    "rot6_rows": _rot6(columns=False),
    "matrix_rows": _matrix(columns=False),
    "matrix_cols": _matrix(columns=True),
}


def _first_columns(rotations: Floats) -> Floats:
    """The 6D of rotation matrices: the first two columns, made orthonormal (Gram-Schmidt), column-major.

    @param rotations `(rows, 3, 3)`, whose first two columns are not of length zero.
    @return `(rows, 6)`: `[R00, R10, R20, R01, R11, R21]`.
    """
    a = rotations[:, :, 0]  # (rows, 3) first column
    b = rotations[:, :, 1]  # (rows, 3) second column
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b - (a * b).sum(axis=1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.concatenate([a, b], axis=1)


def _to_quaternion(m: Floats) -> Floats:
    """Unit quaternions `w, x, y, z` of `(rows, 3, 3)` rotation matrices, w at least 0, each read off the row of
    `4 q q^T` whose diagonal entry is largest, so never divided by a component near zero (Shepperd)."""
    m00, m11, m22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    wx, wy, wz = m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]  # 4wx, 4wy, 4wz
    xy, xz, yz = m[:, 0, 1] + m[:, 1, 0], m[:, 0, 2] + m[:, 2, 0], m[:, 1, 2] + m[:, 2, 1]  # 4xy, 4xz, 4yz
    outer = np.stack(
        [
            np.stack([1 + m00 + m11 + m22, wx, wy, wz], axis=1),
            np.stack([wx, 1 + m00 - m11 - m22, xy, xz], axis=1),
            np.stack([wy, xy, 1 - m00 + m11 - m22, yz], axis=1),
            np.stack([wz, xz, yz, 1 - m00 - m11 + m22], axis=1),
        ],
        axis=1,
    )  # (rows, 4, 4): 4 q q^T
    rows = np.arange(len(m))
    big = np.argmax(np.diagonal(outer, axis1=1, axis2=2), axis=1)  # (rows,): its entry is at least 1
    q = outer[rows, big] / (2 * np.sqrt(outer[rows, big, big]))[:, None]  # (rows, 4), up to sign
    out: Floats = np.where(q[:, :1] < 0, -q, q)
    return out


def _xyz_angles(m: Floats) -> Floats:
    """`(rows, 3)` angles `v` with `m = Rz(v2) Ry(v1) Rx(v0)`. The yaw is read off `m Rx(-v0)`, so where the pitch is
    a quarter turn and the roll is anything, the yaw takes up the rest."""
    roll = np.arctan2(m[:, 2, 1], m[:, 2, 2])
    pitch = np.arctan2(-m[:, 2, 0], np.hypot(m[:, 0, 0], m[:, 1, 0]))
    s, c = np.sin(roll), np.cos(roll)
    yaw = np.arctan2(s * m[:, 0, 2] - c * m[:, 0, 1], c * m[:, 1, 1] - s * m[:, 1, 2])
    return np.stack([roll, pitch, yaw], axis=1)


def _to_euler(order: str, degrees: bool) -> Callable[[Floats], Floats]:
    """Euler angles of rotation matrices, as _euler() reads them. `zyx`'s `Rx(v2) Ry(v1) Rz(v0)` is the inverse of
    `xyz`'s with the angles negated and reversed, so it is read as `xyz` of the transpose."""

    def write(m: Floats) -> Floats:
        angles = _xyz_angles(m) if order == "xyz" else -_xyz_angles(np.transpose(m, (0, 2, 1)))[:, ::-1]
        return np.degrees(angles) if degrees else angles

    return write


def _to_axis_angle(m: Floats) -> Floats:
    """Rotation vectors of rotation matrices, angle in [0, pi]: twice the half turn of the quaternion with w >= 0."""
    q = _to_quaternion(m)
    sine = np.linalg.norm(q[:, 1:], axis=1)  # (rows,): sin(angle / 2)
    angle = 2 * np.arctan2(sine, q[:, 0])
    small = sine < MIN_NORM
    scale = np.where(small, 2 / q[:, 0], angle / np.where(small, 1.0, sine))  # angle / sin(angle / 2), or its limit
    out: Floats = q[:, 1:] * scale[:, None]
    return out


## Each rotation format to the function writing `(rows, 3, 3)` rotation matrices in it (the key sets are pinned equal).
_TO: dict[str, Callable[[Floats], Floats]] = {
    "quat_xyzw": lambda m: _to_quaternion(m)[:, [1, 2, 3, 0]],
    "quat_wxyz": _to_quaternion,
    "euler_xyz": _to_euler("xyz", degrees=False),
    "euler_zyx": _to_euler("zyx", degrees=False),
    "euler_xyz_deg": _to_euler("xyz", degrees=True),
    "euler_zyx_deg": _to_euler("zyx", degrees=True),
    "axis_angle": _to_axis_angle,
    "rot6_cols": _first_columns,
    "rot6_rows": lambda m: _first_columns(np.transpose(m, (0, 2, 1))),
    "matrix_rows": lambda m: m.reshape(len(m), 9),
    "matrix_cols": lambda m: np.transpose(m, (0, 2, 1)).reshape(len(m), 9),
}

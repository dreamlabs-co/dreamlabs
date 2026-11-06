"""The arm your loop drives, as `python -m dream.check` reads it from the loop's first call; internal.

The check takes no setting for the robot: the loop's first call says which arm it drives (#identify). A call that
carries each hand's pose beside the arm's joints, as the client's `ik` config key sends them, is from the client's arm
(`dream.client_ik.ARMS`) whose tool pose at those joints is that pose. A call to the check also names the `ik` arm,
an arm of your own by its arm file (`dream.arm_file`), which is read the same way once its pose at the joints is the
call's. A call that sends no pose is from a YAM pair when it names the robot `yam`, or names none and carries the
YAM's parts (each `<side>_arm` of 6 joints beside its `<side>_gripper`): a model trained on YAM arms runs without `ik`
only on a YAM. The kinematics are dream's IK's arm models (`dream_ik.specs`), the ones the client solves with: the
hand's pose at any joints (#dream.check_arm.Arm.hand), its tool point's Jacobian (#dream.check_arm.Arm.jacobian), the
joints' limits, and the chain of joints the page draws.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from dream import arm_file, client_ik, datasets
from dream.errors import DreamConfigError

__all__ = ["Arm", "identify", "ik_specs", "turned"]

## The tool pose at the reported joints names the arm when it is this close to the reported pose: metres, and each
## number of the rotation's two columns.
MATCH_M, MATCH_ROT = 0.002, 0.01


class Arm:
    """One side's arm: the chain of joints from its base to its tool point, from its URDF and dream's tool point.

    Attributes:
    - name: dream's name for the arm (`yam`, `metal`, ...), or the path of the arm file the call names.
    - limits: `(joints, 2)` each joint's lower and upper limit, radians; a joint dream's IK holds has both 0.
    - ik_limits: `(joints, 2)` the limits the client's IK keeps each joint inside: each of `limits` the spec's
      `limit_margin` further in (a joint it holds, both 0).
    - chain: From the base link, each URDF joint on the way to the flange: `(origin, axis, col, kind)`, its 4x4 origin
      in its parent link, its unit axis, the column of the arm's joints it moves (None for a fixed joint) and its
      URDF type.
    - tool: The 4x4 pose of the tool point in the flange: the spec's tool the client serves (`dream_ik.specs.SERVE`).
    """

    def __init__(self, name: str, spec: Any) -> None:
        """Reads the arm's URDF.

        @param name dream's name for the arm, or its arm file's path.
        @param spec Its side's `dream_ik.specs.ArmSpec`.
        @throws ValueError A joint between the base and the flange moves but is none of the spec's joints.
        """
        import yourdfpy

        urdf = yourdfpy.URDF.load(str(spec.urdf), build_scene_graph=False, load_meshes=False)
        by_child = {joint.child: joint for joint in urdf.robot.joints}
        cols = {joint: col for col, joint in enumerate(spec.joints)}
        chain: list[tuple[np.ndarray, np.ndarray, int | None, str]] = []
        link = spec.flange
        while link in by_child:
            joint = by_child[link]
            moves = joint.type != "fixed"
            if moves and joint.name not in cols:
                raise ValueError(f"{spec.urdf.name}: {joint.name} moves the flange but is not one of {spec.joints}")
            origin = np.eye(4) if joint.origin is None else np.asarray(joint.origin, dtype=float)
            axis = np.asarray(joint.axis, dtype=float)
            unit = axis / np.linalg.norm(axis) if moves else axis
            chain.append((origin, unit, cols[joint.name] if moves else None, joint.type))
            link = joint.parent
        self.name = name
        self.chain = chain[::-1]
        tool = spec.tools[ik_specs().SERVE.tool]
        self.tool = np.eye(4)
        self.tool[:3, :3] = tool.rotation
        self.tool[:3, 3] = tool.offset
        joints = {joint.name: joint for joint in urdf.robot.joints}
        self.limits = np.array(
            [
                (0.0, 0.0)
                if name in spec.held_joints
                else (-np.inf, np.inf)
                if joints[name].type == "continuous"
                else (joints[name].limit.lower, joints[name].limit.upper)
                for name in spec.joints
            ]
        )
        turns = self.limits[:, 1] > self.limits[:, 0]  # (joints,): the joints dream's IK does not hold
        self.ik_limits = self.limits + np.where(turns, spec.limit_margin, 0.0)[:, None] * np.array([1.0, -1.0])

    def poses(self, q: np.ndarray) -> np.ndarray:
        """The tool pose at each row of joints, in the arm's base frame.

        @param q `(rows, joints)` radians (metres for a sliding joint), in the spec's order.
        @return `(rows, 4, 4)` each row's tool pose.
        """
        out = np.broadcast_to(np.eye(4), (q.shape[0], 4, 4))
        for origin, axis, col, kind in self.chain:
            out = out @ origin
            if col is not None:
                out = out @ _move(axis, q[:, col], kind)
        return np.asarray(out @ self.tool)

    def hand(self, q: np.ndarray) -> np.ndarray:
        """The hand's pose at each row of joints, as the client's `ik` sends it: `<side>_eef_xyz`, `<side>_eef_rot6`.

        @param q `(rows, joints)` in the spec's order.
        @return `(rows, 9)`: the tool point (metres, the base's axes), then the rotation's first two columns
            (`dream_ik.decode.mat_to_rot6d`).
        """
        pose = self.poses(q)  # (rows, 4, 4)
        return np.concatenate([pose[:, :3, 3], pose[:, :3, 0], pose[:, :3, 1]], axis=1)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """How the tool point moves per radian of each joint at `q`, by central differences.

        @param q `(joints,)` in the spec's order.
        @return `(3 xyz, joints)`.
        """
        step, n = 1e-6, q.size
        rows = np.concatenate([q + step * np.eye(n), q - step * np.eye(n)])  # (2 joints, joints)
        at = self.poses(rows)[:, :3, 3]  # (2 joints, 3 xyz)
        return np.asarray(((at[:n] - at[n:]) / (2 * step)).T)  # (3 xyz, joints)


def _move(axis: np.ndarray, q: np.ndarray, kind: str) -> np.ndarray:
    """`(rows, 4, 4)`: a joint's motion at each row, a turn by `q` radians about `axis` (a slide of `q` metres along it
    for a sliding joint)."""
    out = np.broadcast_to(np.eye(4), (q.size, 4, 4)).copy()
    if kind == "prismatic":
        out[:, :3, 3] = q[:, None] * axis  # (rows, 3)
        return out
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])  # (3, 3): axis x v
    s, c = np.sin(q)[:, None, None], np.cos(q)[:, None, None]  # (rows, 1, 1)
    out[:, :3, :3] = np.eye(3) + s * cross + (1 - c) * (cross @ cross)  # Rodrigues
    return out


def turned(rot6: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Each hand rotation as the client's `ik` sends it, as the turn from the one it started at about the base's axes.

    @param rot6 `(rows, 6)` each rotation's first two columns (`<side>_eef_rot6`, `dream_ik.decode.rot6d_to_mat`).
    @param start `(6,)` the rotation it started at, the same way.
    @return `(rows, 3)` each turn's axis times its angle (a rotation vector), radians about the base's x, y and z.
    """
    from dream_ik import decode

    rel = decode.rot6d_to_mat(rot6) @ decode.rot6d_to_mat(start).T  # (rows, 3, 3): from start, in the base's axes
    v = 0.5 * np.stack([rel[:, 2, 1] - rel[:, 1, 2], rel[:, 0, 2] - rel[:, 2, 0], rel[:, 1, 0] - rel[:, 0, 1]], axis=1)
    sin = np.linalg.norm(v, axis=1)  # (rows,): v is the axis times the angle's sine
    angle = np.arctan2(sin, (np.trace(rel, axis1=1, axis2=2) - 1) / 2)  # (rows,)
    return np.asarray(v * (angle / np.where(sin > 0, sin, 1.0))[:, None])


def ik_specs() -> Any:
    """dream's IK's arm models (`dream_ik.specs`), which the library ships.

    @return The module.
    @throws DreamConfigError The library is not installed.
    """
    try:
        from dream_ik import specs
    except ImportError as error:
        raise DreamConfigError(
            f"the check reads your arm's kinematics from dream's IK, which the library ships for Python 3.11 or "
            f"newer ({error}); {datasets._install_hint()}"
        ) from None
    return specs


def identify(state: Mapping[str, np.ndarray], robot: Any, named: str | None) -> dict[str, Arm] | None:
    """The arm on each side, from the loop's first call.

    @param state Per part (without `state.`), the first call's reading.
    @param robot The call's robot (#dream.input_check.ROBOT_KEY).
    @param named The arm the call names (`dream.client_ik.ARM_HEADER`): an arm dream ships, or an arm file's path;
        None when it names none.
    @return Per side (`right`, `left`), its #Arm: the named arm, or else the client's arm, whose tool pose at every
        side's `<side>_arm` is that side's `<side>_eef_xyz` and `<side>_eef_rot6`, or, for a call that sends no pose
        and is a YAM's (the module's docstring), the YAM's; None when no arm matches, or the call sends no pose and is
        no YAM's.
    @throws DreamConfigError dream's IK is not installed (#ik_specs), or the named arm file cannot be read
        (`dream.arm_file.register`).
    """
    specs = ik_specs()
    sides = [side for side in client_ik.SIDES if f"{side}_arm" in state]
    posed = [side for side in sides if f"{side}_eef_xyz" in state and f"{side}_eef_rot6" in state]
    yam = robot == "yam" or (
        robot is None and all(state[f"{side}_arm"].size == 6 and f"{side}_gripper" in state for side in sides)
    )
    if not sides or (not posed and not yam):
        return None
    # (name, dream's IK's name): the named arm first, used, like any other, only where its pose matches the call's.
    # A model trained on YAM arms is sent zeros for the joints, at which no arm's pose is the hand's.
    first = (
        [] if named is None or not posed else [(named, named if named in client_ik.ARMS else arm_file.register(named))]
    )
    for name, key in first + [(name, name) for name in (client_ik.ARMS if posed else ("yam",))]:
        body = specs.EMBODIMENTS[key]
        if not set(sides) <= set(body.sides):
            continue
        arms = {side: body.side_arm(side) for side in sides}
        if any(len(arms[side].joints) != state[f"{side}_arm"].size for side in sides):
            continue
        found = {side: Arm(name, arms[side]) for side in sides}
        if all(_matches(found[side], state, side) for side in posed):
            return found
    return None


def _matches(arm: Arm, state: Mapping[str, np.ndarray], side: str) -> bool:
    """Whether the arm's tool pose at the side's reported joints is the side's reported pose."""
    hand = arm.hand(state[f"{side}_arm"][None])[0]  # (9,)
    xyz, rot6 = state[f"{side}_eef_xyz"], state[f"{side}_eef_rot6"]
    return bool(np.max(np.abs(hand[:3] - xyz)) <= MATCH_M and np.max(np.abs(hand[3:] - rot6)) <= MATCH_ROT)

"""The arm as the setup wizard checks it: dream's own URDF of the arm (`dream_ik`), the model the dream client's IK
solves on, so every check asks the same question: does that URDF, drawn at these numbers, move like the real arm?

#ArmModel gives what the checks and the page need from it: forward kinematics of every link (the poses the page's 3D
view draws the arm at, #robots.setup_view), the tool pose (the tool centre point the datasets' tool poses are in,
`dream_ik.specs.SERVE.tool`), the joint limits, which way each joint turns, and the chunk IK the wizard's end-effector
sweep is solved with (`dream_ik.solver.ChunkIK`, the solver the client runs for an EEF checkpoint). `dream_ik` and JAX
are imported in the constructor, so the rest of the wizard imports without them; building a model compiles its forward
kinematics, a few seconds.

@see ArmModel, spec_names()
"""

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from robots import errors, followers, kinematics

## Rows per second the sweep's IK is solved at and the wizard writes its powered moves at.
MOTION_HZ: float = 30.0

## How far outside the URDF's limits a reading may sit before it counts as out of range, radians (about 11 degrees):
## a driver's soft limits and the URDF's differ by a few degrees, and a motor reads a little past a stop.
LIMIT_SLACK_RAD: float = 0.2

## A joint lifts the arm when turning it moves the tool up or down at least this far per radian, metres (10 cm): a
## Metal's or YAM's shoulder, elbow and wrist pitch (about 0.22 to 0.47), never a base turn or a wrist roll (0).
LIFT_M_PER_RAD: float = 0.1


def spec_names() -> list[str]:
    """The arms dream's IK has a URDF of, by the name a body's `ik["arm"]` uses.

    @return The names, sorted.
    @throws ConfigError dream's IK is not installed.
    """
    specs = _import_specs()
    return sorted(specs.EMBODIMENTS)


def _import_specs() -> Any:
    """`dream_ik.specs`, or a ConfigError saying how to get it."""
    try:
        from dream_ik import specs
    except ImportError as exc:
        raise errors.ConfigError(
            f"the setup checks need dream's IK, which the dream library ships for Python 3.11 or newer "
            f"(the platform's Setup API page installs it with Python 3.12): {exc}"
        ) from exc
    return specs


class ArmModel:
    """dream's URDF of one arm type, for both arms of a station (each in its own base frame).

    An arm dream models per side (the SO-101's `so101_faas`) differs between its sides only in where the base sits and
    the calibrated limits, so both arms are drawn and checked on the first side's URDF; the limits are both sides'
    widest, the rest pose and the tool pose (and so the sweep) each side's own.

    Joint vectors are the spec's order and sign (`ArmSpec.joints`), radians: what a driver reports in URDF units. They
    are as long as the spec has joints (#width, 1 to `robots.followers.MAX_JOINTS`), which is how many the home the
    wizard saves lists, so how many the loop reads and commands. Not thread-safe across calls that compile; the
    wizard builds one per robot choice and calls it from one thread at a time.

    Attributes:
    - spec: dream's name for the arm (`"yam"`, `"makers"`, `"metal"`, ..., or an arm file's `file:` name).
    - sides: The arms in the order the IK solves them, `("right", "left")`.
    - joint_names: The URDF's actuated joints, in the spec's order.
    - width: How many joints dream's IK of the arm has: every joint vector's length.
    - lower: The URDF's lower joint limits, radians, `[width]`.
    - upper: The URDF's upper joint limits, radians, `[width]`.
    - rest: The first side's rest pose, `[width]`, a pose the arm can always reach.
    - rests: Each side's rest pose, `[width]`.
    - held: Joints the arm does not have, 0-based: the ones dream's spec holds at zero (the SO-101's `joint6`). No
      check asks for them.
    - link_names: Every link of the URDF with the tool as a link, in the order poses() gives them.
    - segments: How the links hang together, `(parent link, child link)` index pairs into #link_names: the bones the
      page draws, one to each link a moving joint turns and to the tool, from the nearest such link above it
      (_bones()).
    - tool_link: The index of the tool centre point in #link_names.

    @see poses(), tool(), lifting(), out_of_range(), solve(), chain()
    """

    def __init__(self, spec: str, hz: float = MOTION_HZ) -> None:
        """Builds dream's model of the arm with its tool, and compiles its forward kinematics.

        @param spec dream's name for the arm, one of spec_names(), or an arm file's path
            (#robots.kinematics.dream_arm).
        @param hz The rate the IK's speed caps are per (#MOTION_HZ).
        @throws ConfigError dream's IK is not installed, it has no arm by that name, or the arm file is malformed.
        """
        specs = _import_specs()
        spec = kinematics.dream_arm(spec)
        if spec not in specs.EMBODIMENTS:
            raise errors.ConfigError(f"dream's IK has no arm called {spec!r}; it has {sorted(specs.EMBODIMENTS)}")
        import jax
        from dream_ik.solver import ChunkIK, build_arm

        self.spec = spec
        self._ik = ChunkIK(spec, specs.SERVE.tool, hz)
        self.sides: tuple[str, ...] = tuple(self._ik.sides)
        arm_spec = specs.EMBODIMENTS[spec].side_arm(self.sides[0])
        arm = build_arm(spec, self.sides[0], specs.SERVE.tool)
        robot = arm.robot
        self.joint_names: tuple[str, ...] = tuple(arm_spec.joints)
        self.width: int = len(self.joint_names)
        if self.width > followers.MAX_JOINTS:
            raise errors.ConfigError(
                f"dream's {spec} arm has {self.width} joints; the loop drives at most {followers.MAX_JOINTS}"
            )
        self.lower: npt.NDArray[np.float64] = np.min(self._ik.side_lo, axis=0) - arm_spec.limit_margin
        self.upper: npt.NDArray[np.float64] = np.max(self._ik.side_hi, axis=0) + arm_spec.limit_margin
        self.rests: dict[str, npt.NDArray[np.float64]] = {
            side: np.asarray(rest, dtype=np.float64) for side, rest in zip(self.sides, self._ik.side_rest, strict=True)
        }
        self.rest: npt.NDArray[np.float64] = self.rests[self.sides[0]]
        self.held: tuple[int, ...] = tuple(self.joint_names.index(name) for name in arm_spec.held_joints)
        self._to_pyroki = list(arm.to_pyroki)
        parent_joint = list(robot.links.parent_joint_indices)
        child_of = {joint: link for link, joint in enumerate(parent_joint) if joint >= 0}
        root = parent_joint.index(-1)
        joint_parent = list(robot.joints.parent_indices)
        self.tool_link: int = int(arm.tool_index)
        self.segments: list[tuple[int, int]] = _bones(
            root, child_of, joint_parent, robot.joints.twists, self.tool_link
        )
        joint_index = [list(robot.joints.names).index(name) for name in self.joint_names]
        self._pivot_links = [child_of[j] for j in joint_index]
        self.link_names: tuple[str, ...] = tuple(robot.links.names)
        # A revolute joint's twist is (0, 0, 0, axis), the axis in the frame of the link it turns.
        self._joint_axes = np.asarray(robot.joints.twists, dtype=np.float64)[joint_index, 3:6]
        self._links_fn = jax.jit(robot.forward_kinematics)
        self.poses(self.rest)
        for side in self.sides:
            self.tool(self.rests[side], side)

    def poses(self, joints: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Every link's pose for one set of joints.

        @param joints `[width]` radians, spec order.
        @return `[links, 7]`: per link of #link_names, its orientation quaternion `w, x, y, z` and origin `x, y, z`
            in metres, in the arm's base frame.
        """
        q = np.asarray(joints, dtype=np.float64)[self._to_pyroki]
        return np.asarray(self._links_fn(q), dtype=np.float64)

    def frame(self, joints: npt.ArrayLike) -> list[list[float]] | None:
        """poses() as the page takes them: rounded to 10 micrometres, or None for a reading that is not a number.

        @param joints `[width]` as read.
        @return `[links][7]`, or None.
        """
        q = np.asarray(joints, dtype=np.float64)
        if not np.all(np.isfinite(q)):
            return None
        return [[round(float(v), 5) for v in row] for row in self.poses(q)]

    def tool(self, joints: npt.ArrayLike, side: str | None = None) -> npt.NDArray[np.float64]:
        """The tool centre point's pose for one pose.

        @param joints `[width]` radians, spec order.
        @param side The arm whose base frame the pose is in (what solve() takes for it); None for the first side's.
        @return `[4, 4]` in that arm's base frame, metres.
        """
        q = np.asarray(joints, dtype=np.float64)
        return np.asarray(self._ik.fk(q, self.sides[0] if side is None else side), dtype=np.float64)

    def rise(self, joints: npt.ArrayLike, side: str) -> npt.NDArray[np.float64]:
        """How far turning each joint raises the tool at a pose, metres per radian (negative lowers it).

        Up is the arm's base frame's +z, which is up for an arm standing on a table.

        @param joints `[width]` radians, spec order.
        @param side The arm, as for tool().
        @return `[width]` metres per radian.
        """
        q = np.asarray(joints, dtype=np.float64)
        step = 1e-3
        z = self.tool(q, side)[2, 3]
        return np.array(
            [(self.tool(q + step * np.eye(len(q))[joint], side)[2, 3] - z) / step for joint in range(len(q))]
        )

    def lifting(self, joints: npt.ArrayLike, side: str) -> frozenset[int]:
        """The joints that lift the arm at a pose: turning them moves the tool up or down, so they carry its weight.

        @param joints `[width]` radians, spec order.
        @param side The arm, as for tool().
        @return Joint indices, 0-based, whose turn moves the tool at least #LIFT_M_PER_RAD up or down per radian.
        """
        rise = self.rise(joints, side)
        return frozenset(joint for joint, m_per_rad in enumerate(rise) if abs(m_per_rad) >= LIFT_M_PER_RAD)

    def out_of_range(self, joints: npt.ArrayLike, slack: float = LIMIT_SLACK_RAD) -> list[int]:
        """The joints a reading puts past the URDF's limits by more than `slack`.

        @param joints `[width]` as read.
        @param slack How far past a limit still counts as in range, radians.
        @return Joint indices, 0-based; a non-finite value counts as out of range.
        """
        q = np.asarray(joints, dtype=np.float64)
        bad = ~np.isfinite(q) | (q < self.lower - slack) | (q > self.upper + slack)
        return [int(i) for i in np.flatnonzero(bad)]

    def solve(
        self, targets: dict[str, npt.NDArray[np.float64]], start: dict[str, npt.NDArray[np.float64]]
    ) -> dict[str, dict[str, npt.NDArray[np.float64]]]:
        """Solves tool pose paths to joints with dream's chunk IK, both arms in one call, as the client does.

        @param targets Per side, `[ticks, 4, 4]` tool poses in that arm's base frame; every side of #sides, same
            length.
        @param start Per side, the joints the path starts from, `[width]`.
        @return Per side, `{"q": [ticks, width], "pos_miss": [ticks] m, "ang_miss": [ticks] rad}`: the commanded joints
            (inside the margined limits and under the arm's speed caps) and how far their tool pose is from the target.
        """
        result = self._ik.solve(
            np.stack([targets[side] for side in self.sides]),
            np.stack([np.asarray(start[side], dtype=np.float64) for side in self.sides]),
        )
        q = np.asarray(result.q, dtype=np.float64)
        return {
            side: {"q": q[i], "pos_miss": result.pos_miss[i], "ang_miss": result.ang_miss[i]}
            for i, side in enumerate(self.sides)
        }

    def chain(self) -> dict[str, Any]:
        """What the page's 3D view needs once: the links, how they hang together, and the joints.

        @return `{"links", "segments", "tool_link", "joints"}`; `joints` is per joint in spec order `{"name", "link",
            "axis"}`: the link it turns and its axis in that link's frame.
        """
        return {
            "links": list(self.link_names),
            "segments": [list(pair) for pair in self.segments],
            "tool_link": self.tool_link,
            "joints": [
                {"name": name, "link": link, "axis": axis.round(6).tolist()}
                for name, link, axis in zip(self.joint_names, self._pivot_links, self._joint_axes, strict=True)
            ],
        }


def _bones(
    root: int, child_of: dict[int, int], joint_parent: list[int], twists: npt.ArrayLike, tool_link: int
) -> list[tuple[int, int]]:
    """The links' bones as the page draws them: from each moving joint's link, and the tool, up to the nearest link
    above it that a moving joint turns, or the root.

    A link hung by a fixed joint is passed through, not drawn to: a CAD export (Onshape) hangs every joint off a servo
    link fixed to the body whose origin is far from the servo, at the base in every pose for one, and bones through
    those origins draw a straight arm zig-zagged (a Sourccey, for one). A moving joint's link has its origin on
    the joint, so the bones follow the joints.

    @param root The root link's index.
    @param child_of Per joint index, the link it moves (its child).
    @param joint_parent Per joint index, its parent joint's index, -1 for one hung off the root.
    @param twists `[joints, 6]` per joint; all zero for a fixed joint.
    @param tool_link The tool's link index.
    @return `(parent link, child link)` pairs, in joint order.
    """
    moving = {joint for joint, twist in enumerate(np.asarray(twists, dtype=np.float64)) if np.any(twist != 0.0)}
    joint_of = {link: joint for joint, link in child_of.items()}
    drawn = {child_of[joint] for joint in moving if joint in child_of} | {tool_link}

    def anchor(link: int) -> int:
        joint = joint_of.get(link)
        while joint is not None:
            above = joint_parent[joint]
            if above < 0:
                return root
            if above in moving:
                return child_of[above]
            joint = above
        return root

    return [(anchor(link), link) for joint, link in sorted(child_of.items()) if link in drawn and link != root]


def pose_offset(
    pose: npt.NDArray[np.float64], shift_m: npt.ArrayLike, turn_rad: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """A tool pose moved by a shift and turned about the base frame's axes through the tool point.

    @param pose `[4, 4]` tool pose.
    @param shift_m `[3]` metres, base frame.
    @param turn_rad `[3]` a rotation vector, radians, base frame.
    @return The new `[4, 4]` pose.
    """
    out = np.array(pose, dtype=np.float64)
    out[:3, :3] = _rotvec_matrix(np.asarray(turn_rad, dtype=np.float64)) @ out[:3, :3]
    out[:3, 3] = out[:3, 3] + np.asarray(shift_m, dtype=np.float64)
    return out


def _rotvec_matrix(rotvec: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Rodrigues' formula: the rotation matrix of a rotation vector.

    @param rotvec `[3]` radians.
    @return `[3, 3]`.
    """
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3)
    k = rotvec / angle
    skew = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.asarray(np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * skew @ skew, dtype=np.float64)


def rotation_angle(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
    """The angle between two rotations.

    @param a `[3, 3]`.
    @param b `[3, 3]`.
    @return Radians, 0 to pi.
    """
    cosine = (float(np.trace(a @ b.T)) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, cosine)))


def rotation_vector(m: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """The rotation vector of a rotation matrix, the inverse of #pose_offset's turn (well inside half a turn).

    @param m `[3, 3]`.
    @return `[3]` radians: the axis times the angle.
    """
    angle = rotation_angle(m, np.eye(3))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]) / (2.0 * math.sin(angle))
    return axis * angle

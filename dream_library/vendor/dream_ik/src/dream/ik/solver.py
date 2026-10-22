"""The one IK: a PyRoKi solve per tick, a whole action chunk for every arm in one compiled call.

Per tick and arm: reach the target tool pose (position 50 : angle 10, as our teleop),
the joint-limit margin inside the solve, a light pull toward the rest pose, a pull toward the last
pose, and a stiff push away from fixed obstacles. The solution is clipped to the margined limits,
then the command moves toward it under the per-joint speed caps (the arm's, or the caller's where
lower), direction preserved. Ticks run in a ``lax.scan`` (each warm-starts from and is pulled toward
the previous solution), arms sharing one ArmSpec in a ``vmap`` (an ArmSpec per side: a solve per side);
shapes are fixed, so one compile per (arm, tool, chunk length).
"""

import functools
import io
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import yourdfpy
from einops import rearrange
from scipy.spatial.transform import Rotation as Rot

from .specs import EMBODIMENTS, ArmSpec, Box, Capsule, Embodiment, Obstacle, Region, ToolPose

assert jax.default_backend() == "cpu", (
    f"dream.ik must run JAX on the CPU (got {jax.default_backend()!r}): import dream.ik before jax, "
    "or set JAX_PLATFORMS=cpu"
)

TOOL_LINK = "ik_tool"
# A collision sphere this close to an obstacle (m) turns the collision cost on for that tick: well past the 2 cm margin
# plus a tick's motion (a hand at 0.6 m turning 4.5 rad/s moves 9 cm per tick at 30 Hz). Farther, the cost is 0 anyway.
NEAR_M = 0.15


@jaxls.Cost.factory
def _limit_cost(vals: jaxls.VarValues, var: jaxls.Var, lo: jax.Array, hi: jax.Array, weight: jax.Array) -> jax.Array:
    """Stiff push back inside the margined limits. Stiff costs, not jaxls constraints: with a limit
    active the augmented Lagrangian ran to its 100-step cap on a third of the replay's solves."""
    q = vals[var]  # (dof,) pyroki order
    return weight * jnp.concatenate([jnp.maximum(0.0, q - hi), jnp.maximum(0.0, lo - q)])


def _near(arm: "_Arm", obstacles: dict[str, jax.Array], q: jax.Array) -> jax.Array:
    """() True when a collision sphere of the arm at ``q`` is within ``NEAR_M`` of one of its obstacles."""
    Ts = arm.robot._link_poses_from_joint_poses(arm.robot._forward_kinematics_joints(q))
    centers = jaxlie.SE3(Ts[arm.sphere_link]) @ arm.sphere_local  # (S, 3)
    return (jax.vmap(_sphere_distance, in_axes=(0, 0, None))(centers, arm.sphere_radius, obstacles) < NEAR_M).any()


def _sphere_distance(center: jax.Array, radius: jax.Array, obstacles: dict[str, jax.Array]) -> jax.Array:
    """Signed distance (m) from one sphere (center (3,)) to every obstacle -> (n,). Capsules (n, 5)
    [cx, cy, cz, radius, height] stand vertical; boxes (n, 6) [cx, cy, cz, sx, sy, sz] are axis-aligned; regions
    (n, 6) [lo xyz, hi xyz] are boxes the sphere's center must stay inside: the distance is from the center to the
    nearest face, from inside (the arm's centreline is what a Region bounds, so a hand can reach a table face)."""
    out = []
    if "capsule" in obstacles:
        c = obstacles["capsule"]  # (n, 5)
        z = jnp.clip(center[2], c[:, 2] - c[:, 4] / 2, c[:, 2] + c[:, 4] / 2)
        axis_pt = jnp.stack([c[:, 0], c[:, 1], z], -1)  # (n, 3) nearest point on the capsule axis
        out.append(jnp.linalg.norm(center - axis_pt, axis=-1) - c[:, 3] - radius)
    if "box" in obstacles:
        b = obstacles["box"]  # (n, 6)
        d = jnp.abs(center - b[:, :3]) - b[:, 3:] / 2  # (n, 3)
        out.append(jnp.linalg.norm(jnp.maximum(d, 0.0), axis=-1) + jnp.minimum(jnp.max(d, axis=-1), 0.0) - radius)
    if "region" in obstacles:
        r = obstacles["region"]  # (n, 6)
        out.append(jnp.minimum(center - r[:, :3], r[:, 3:] - center).min(axis=-1))  # the centreline: no radius
    return jnp.concatenate(out)


def _collision_jac(
    vals, cache, var, robot, sphere_link, sphere_local, sphere_radius, sphere_drive, obstacles, margin, weight
):
    """Analytic Jacobian of ``_collision_cost`` (autodiff through FK was 4x slower per chunk)."""
    del vals, var, sphere_link, sphere_local, sphere_radius, obstacles, margin
    Ts_joint, centers, grad, active = cache  # (J, 7) joint poses, (S, 3) centers, (S, n, 3) d dist / d center, (S*n,)
    T = jaxlie.SE3(Ts_joint)
    twists = robot.joints.twists * robot.joints.mimic_multiplier[:, None]  # (J, 6)
    omega, v = T.rotation() @ twists[:, 3:], T.rotation() @ twists[:, :3]  # (J, 3) each, world frame
    dp = jnp.cross(omega, centers[:, None] - T.translation()) + v  # (S, J, 3) sphere velocity per joint
    jac = -jnp.einsum("snx,sjx,sja->sna", grad, dp, sphere_drive)  # (S, n, actuated)
    return weight * active[:, None] * rearrange(jac, "s n a -> (s n) a")


@jaxls.Cost.factory(jac_custom_with_cache_fn=_collision_jac)
def _collision_cost(
    vals, var, robot, sphere_link, sphere_local, sphere_radius, sphere_drive, obstacles, margin, weight
):
    """Stiff push of every collision sphere out to ``margin`` from every obstacle."""
    Ts_joint = robot._forward_kinematics_joints(vals[var])  # (J, 7)
    centers = jaxlie.SE3(robot._link_poses_from_joint_poses(Ts_joint)[sphere_link]) @ sphere_local  # (S, 3)
    per_sphere = functools.partial(jax.vmap, in_axes=(0, 0, None))
    short = margin - per_sphere(_sphere_distance)(centers, sphere_radius, obstacles)  # (S, n) > 0: too close
    grad = per_sphere(jax.jacfwd(_sphere_distance))(centers, sphere_radius, obstacles)  # (S, n, 3)
    short = rearrange(short, "s n -> (s n)")
    return weight * jnp.maximum(0.0, short), (Ts_joint, centers, grad, (short > 0).astype(short.dtype))


@jdc.pytree_dataclass
class _Arm:
    """One arm type, compiled: kinematics with the tool as a link, limits, rest, speeds, collision spheres."""

    robot: pk.Robot
    sphere_link: jax.Array  # (S,) link index of each collision sphere
    sphere_local: jax.Array  # (S, 3) sphere center in its link frame
    sphere_radius: jax.Array  # (S,)
    sphere_drive: jax.Array  # (S, joints, actuated) 1 where the joint moves the sphere and is driven by that actuator
    sphere_hand: jax.Array  # (S,) 1 on the spheres from the flange to the tool point (the gripper), else 0
    tool_index: jdc.Static[int]
    lo: jax.Array  # (dof,) margined limits, pyroki order
    hi: jax.Array
    rest: jax.Array  # (dof,)
    rest_weight: jax.Array  # (dof,)
    max_speed: jax.Array  # (dof,) rad/s
    to_spec: jdc.Static[tuple[int, ...]]  # q_spec = q_pyroki[to_spec]
    to_pyroki: jdc.Static[tuple[int, ...]]  # q_pyroki = q_spec[to_pyroki]


def _urdf_with_tool(spec: ArmSpec, tool: ToolPose) -> yourdfpy.URDF:
    """The arm URDF plus a fixed ``ik_tool`` link on the flange at the tool pose (no meshes), the spec's
    ``fixed_joints`` made fixed."""
    root = ET.fromstring(spec.urdf.read_text())
    held = [j for j in root.findall("joint") if j.get("name") in spec.fixed_joints]
    assert len(held) == len(spec.fixed_joints), f"{spec.urdf.name}: fixed_joints {spec.fixed_joints} not all in it"
    for j in held:
        j.set("type", "fixed")
    ET.SubElement(root, "link", name=TOOL_LINK)
    j = ET.SubElement(root, "joint", name=f"{TOOL_LINK}_joint", type="fixed")
    ET.SubElement(j, "parent", link=spec.flange)
    ET.SubElement(j, "child", link=TOOL_LINK)
    rpy = Rot.from_matrix(np.array(tool.rotation)).as_euler("xyz")  # URDF fixed-axis rpy
    ET.SubElement(j, "origin", xyz=" ".join(map(str, tool.offset)), rpy=" ".join(map(str, rpy)))
    return yourdfpy.URDF.load(io.StringIO(ET.tostring(root, encoding="unicode")), load_meshes=False)


def _chain_spheres(urdf: yourdfpy.URDF, link_names: tuple[str, ...], radius: float, base: str):
    """Collision spheres of ``radius`` laid on each moving link along its child joint's origin offset,
    spaced at most one radius apart: the arm as a string of beads, from the URDF alone. The base link
    never moves, so it gets none. -> (link index (S,), local center (S, 3), radius (S,))."""
    link, local = [], []
    for joint in urdf.joint_map.values():
        if joint.parent == base:
            continue
        p = joint.origin[:3, 3]  # child joint position in the parent link frame
        for t in np.linspace(0.0, 1.0, max(2, int(np.ceil(np.linalg.norm(p) / radius)) + 1)):
            link.append(link_names.index(joint.parent))
            local.append(t * p)
    return jnp.asarray(link, dtype=jnp.int32), jnp.asarray(np.array(local)), jnp.full(len(link), radius)


def _sphere_drive(robot: pk.Robot, sphere_link: jax.Array) -> jax.Array:
    """(S, joints, actuated): 1 where a joint lies on the chain from the base to sphere s and is driven
    by actuated joint a (itself or through a mimic)."""
    J = robot.joints
    drive = np.zeros((len(sphere_link), J.num_joints, J.num_actuated_joints))
    for s, link in enumerate(np.asarray(sphere_link)):
        j = robot.links.parent_joint_indices[link]
        while j != -1:
            a = J.actuated_indices[j] if J.actuated_indices[j] != -1 else J.mimic_act_indices[j]
            if a != -1:
                drive[s, j, a] = 1.0
            j = J.parent_indices[j]
    return jnp.asarray(drive)


@functools.cache
def build_arm(embodiment: str, side: str, tool: str) -> _Arm:
    spec = EMBODIMENTS[embodiment].side_arm(side)
    assert len(spec.rest_pose) == len(spec.rest_weight) == len(spec.max_speed) == len(spec.joints), embodiment
    urdf = _urdf_with_tool(spec, spec.tools[tool])
    robot = pk.Robot.from_urdf(urdf)
    names = list(robot.joints.actuated_names)
    to_spec = tuple(names.index(j) for j in spec.joints)
    assert sorted(to_spec) == list(range(len(names))), f"{embodiment}: spec joints {spec.joints} != URDF {names}"
    to_pyroki = tuple(int(i) for i in np.argsort(to_spec))
    lo = np.array(robot.joints.lower_limits) + spec.limit_margin
    hi = np.array(robot.joints.upper_limits) - spec.limit_margin
    held = [names.index(j) for j in spec.held_joints]
    lo[held] = hi[held] = 0.0
    sphere_link, sphere_local, sphere_radius = _chain_spheres(urdf, robot.links.names, spec.link_radius, urdf.base_link)
    return _Arm(
        robot=robot,
        sphere_link=sphere_link,
        sphere_local=sphere_local,
        sphere_radius=sphere_radius,
        sphere_drive=_sphere_drive(robot, sphere_link),
        sphere_hand=(sphere_link == list(robot.links.names).index(spec.flange)).astype(jnp.float32),
        tool_index=robot.links.names.index(TOOL_LINK),
        lo=jnp.asarray(lo),
        hi=jnp.asarray(hi),
        rest=jnp.asarray(np.asarray(spec.rest_pose)[list(to_pyroki)]),
        rest_weight=jnp.asarray(np.asarray(spec.rest_weight)[list(to_pyroki)]),
        max_speed=jnp.asarray(np.asarray(spec.max_speed)[list(to_pyroki)]),
        to_spec=to_spec,
        to_pyroki=to_pyroki,
    )


# The traj solve's orientation weight on an arm with fewer than six free joints, as a share of its own (#solve_traj).
TRAJ_FEW_JOINTS_ORI_SCALE = 0.5

# A region face left open (None), and the region a side without one gets: 1 km out, never near.
REGION_OPEN = (-1000.0, -1000.0, -1000.0, 1000.0, 1000.0, 1000.0)


def _region_params(region: Region) -> tuple[float, ...]:
    """[lo xyz, hi xyz] with open faces 1 km out."""
    lo = tuple(REGION_OPEN[i] if v is None else v for i, v in enumerate(region.lo))
    hi = tuple(REGION_OPEN[3 + i] if v is None else v for i, v in enumerate(region.hi))
    return (*lo, *hi)


@jdc.jit
def _region_depth(arm: "_Arm", region: jax.Array, q: jax.Array) -> jax.Array:
    """q (ticks, dof) pyroki order, region (6,) -> (ticks,) m the arm's centreline (its collision sphere centers,
    the tool point among them) is outside the region at its farthest (0 or less: inside)."""
    centers = jax.vmap(lambda qt: _arm_spheres(arm, qt))(q)  # (ticks, S, 3)
    inside = jnp.minimum(centers - region[:3], region[3:] - centers).min(axis=-1)  # (ticks, S)
    return -inside.min(axis=-1)


@jdc.jit
def _keep_out_depth(arm: "_Arm", obstacles: dict[str, jax.Array], q: jax.Array, region_slack) -> jax.Array:
    """q (ticks, dof) pyroki order, one side's obstacles (kind -> (n, params)) -> (ticks,) m the arm's centreline (its
    sphere centers, the grasp point among them) is past what the fence allows at its worst (0 or less: clear): inside
    a capsule or box at all (the station's own guard: no point of the arm in the camera pole), or more than
    ``region_slack`` out of its Region. The spheres' radius is left out: they are fatter than the links (recorded
    YAM teleop has them 33 mm into the pole while the centreline keeps 7 mm clear)."""
    centers = jax.vmap(lambda qt: _arm_spheres(arm, qt))(q)  # (ticks, S, 3)
    zero = jnp.zeros(centers.shape[1])
    depth = -jax.vmap(lambda c: jax.vmap(_sphere_distance, in_axes=(0, 0, None))(c, zero, obstacles))(centers)
    n_region = obstacles["region"].shape[0] if "region" in obstacles else 0  # regions are the last columns
    slack = jnp.concatenate([jnp.zeros(depth.shape[-1] - n_region), jnp.full(n_region, region_slack)])
    return (depth - slack).max(axis=(-1, -2))


def build_obstacles(obstacles: Mapping[str, tuple[Obstacle, ...]], sides: tuple[str, ...]) -> dict[str, jax.Array]:
    """Per obstacle kind present, an array (sides, n, params) of ``sides`` in each arm's base frame (see
    ``_sphere_distance``). The sides are vmapped, so a side with fewer of a kind gets inert ones: size 0, 1 km away."""
    out = {}
    for kind, params, inert in (
        (Capsule, lambda o: (*o.center, o.radius, o.height), (1000.0, 1000.0, 1000.0, 0.0, 0.0)),
        (Box, lambda o: (*o.center, *o.size), (1000.0, 1000.0, 1000.0, 0.0, 0.0, 0.0)),
        (Region, _region_params, REGION_OPEN),
    ):
        per_side = [[params(o) for o in obstacles[s] if isinstance(o, kind)] for s in sides]
        n = max(len(p) for p in per_side)
        if n:
            out[kind.model_fields["kind"].default] = jnp.asarray([p + [inert] * (n - len(p)) for p in per_side])
    return out


def _solve_one(arm: _Arm, obstacles, target, q_prev, lo, hi, w):
    """One arm, one tick. target (7,) wxyz_xyz tool pose; q_prev (dof,) previous solution; lo/hi (dof,)."""
    var = arm.robot.joint_var_cls(0)
    costs = [
        pk.costs.pose_cost_analytic_jac(
            arm.robot,
            var,
            jaxlie.SE3(target),
            jnp.array(arm.tool_index, dtype=jnp.int32),
            pos_weight=w["pos"],
            ori_weight=w["ori"],
        ),
        _limit_cost(var, lo, hi, w["limit"]),
        pk.costs.rest_cost(var, arm.rest, arm.rest_weight),
        pk.costs.rest_cost(var, q_prev, w["last_pose"] * jnp.ones_like(q_prev)),  # the pull toward the last pose
    ]
    if obstacles:
        costs.append(
            _collision_cost(
                var,
                arm.robot,
                arm.sphere_link,
                arm.sphere_local,
                arm.sphere_radius,
                arm.sphere_drive,
                obstacles,
                w["collision_margin"],
                w["collision"],
            )
        )
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=[var])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make([var.with_value(q_prev)]),  # a dict here would drop q_prev
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[var]


def _tick(arm: _Arm, obstacles, w, max_step, lo, hi, carry, target):
    """All arms, one tick. carry = (q_cmd, q_sol) (arms, dof); target (arms, 7); max_step (dof,) rad. The
    command steps toward the solution, scaled by the most-limited joint so its direction is kept (robots
    rate_limit_joint_step). Also returns (arms,) True where the speed cap shortened the step."""
    q_cmd, q_prev = carry

    def solve(obs):
        return jax.vmap(_solve_one, in_axes=(None, 0, 0, 0, 0, 0, None))(arm, obs, target, q_prev, lo, hi, w)

    if obstacles:  # the collision cost only near an obstacle, at the last pose or where the solve without it goes
        near = jax.vmap(_near, in_axes=(None, 0, 0))

        def free_first():
            q_free = solve({})
            return jax.lax.cond(near(arm, obstacles, q_free).any(), lambda: solve(obstacles), lambda: q_free)

        q_sol = jax.lax.cond(near(arm, obstacles, q_prev).any(), lambda: solve(obstacles), free_first)
    else:
        q_sol = solve(obstacles)
    q_sol = jnp.clip(q_sol, lo, hi)  # the stiff limit cost leaves <1e-4 rad; the margin is a hard guarantee
    step = q_sol - q_cmd  # (arms, dof)
    scale = jnp.minimum(1.0, jnp.min(max_step / jnp.maximum(jnp.abs(step), 1e-12), axis=-1, keepdims=True))
    q_new = q_cmd + scale * step
    return (q_new, q_sol), (q_new, q_sol, scale[:, 0] < 1.0)


@jdc.jit
def _solve_chunk(arm: _Arm, obstacles, w, targets, q0, max_step):
    """targets (ticks, arms, 7) wxyz_xyz; q0 (arms, dof) current joints, pyroki order; max_step (dof,) rad per
    tick. -> q_cmd, q_sol (ticks, arms, dof), at the speed cap (ticks, arms) and the commanded tool pose (ticks, arms, 7)."""
    lo = jnp.minimum(arm.lo, q0)  # a joint already past its margin may stay there, never go further
    hi = jnp.maximum(arm.hi, q0)
    step_fn = functools.partial(_tick, arm, obstacles, w, max_step, lo, hi)
    _, (q_cmd, q_sol, at_cap) = jax.lax.scan(step_fn, (q0, q0), targets)
    reached = arm.robot.forward_kinematics(q_cmd)[..., arm.tool_index, :]  # (ticks, arms, 7)
    return q_cmd, q_sol, at_cap, reached


@jdc.jit
def _solve_tick(arm: _Arm, obstacles, w, target, q_cmd, q_prev, lo, hi, max_step):
    """One tick for all arms, same math as a ``_solve_chunk`` step (per-tick calls, for tests and timing)."""
    (q_new, q_sol), _ = _tick(arm, obstacles, w, max_step, lo, hi, (q_cmd, q_prev), target)
    return q_new, q_sol


@jaxls.Cost.factory
def _step_cost(vals: jaxls.VarValues, a: jaxls.Var, b: jaxls.Var, weight: jax.Array) -> jax.Array:
    """Joint velocity between two ticks of one arm."""
    return weight * (vals[a] - vals[b])


@jaxls.Cost.factory
def _accel_cost(vals: jaxls.VarValues, a: jaxls.Var, b: jaxls.Var, c: jaxls.Var, weight: jax.Array) -> jax.Array:
    """Joint acceleration over three ticks of one arm."""
    return weight * (vals[a] - 2.0 * vals[b] + vals[c])


@jaxls.Cost.factory
def _jerk_cost(
    vals: jaxls.VarValues, a: jaxls.Var, b: jaxls.Var, c: jaxls.Var, d: jaxls.Var, weight: jax.Array
) -> jax.Array:
    """Joint jerk over four ticks of one arm, ``a`` the latest."""
    return weight * (vals[a] - 3.0 * vals[b] + 3.0 * vals[c] - vals[d])


@jaxls.Cost.factory
def _history_cost(
    vals: jaxls.VarValues,
    q0: jaxls.Var,
    q1: jaxls.Var,
    q2: jaxls.Var,
    last: jax.Array,
    before: jax.Array,
    before2: jax.Array,
    wv,
    wa,
    wj,
    wj0,
) -> jax.Array:
    """Where the chunk meets the arm's last three commands (``last`` the newest): velocity into tick 0, acceleration
    at ticks 0 and 1, jerk at ticks 0 (weight ``wj0``: 0 when the third row is not known), 1 and 2."""
    a, b, c = vals[q0], vals[q1], vals[q2]
    return jnp.concatenate(
        [
            wv * (a - last),
            wa * (a - 2.0 * last + before),
            wa * (b - 2.0 * a + last),
            wj0 * (a - 3.0 * last + 3.0 * before - before2),
            wj * (b - 3.0 * a + 3.0 * last - before),
            wj * (c - 3.0 * b + 3.0 * a - last),
        ]
    )


def _arm_spheres(arm: _Arm, q: jax.Array) -> jax.Array:
    """(S, 3) collision sphere centers of one arm at q, in its base frame."""
    Ts = arm.robot._link_poses_from_joint_poses(arm.robot._forward_kinematics_joints(q))
    return jaxlie.SE3(Ts[arm.sphere_link]) @ arm.sphere_local


def _interarm_shortfall(arm_a: _Arm, arm_b: _Arm, q1: jax.Array, q2: jax.Array, b_in_a, margin, hand_margin):
    """(Sa, Sb) m each sphere pair of the two arms is inside its margin (> 0: too close): ``hand_margin`` for a pair
    of gripper spheres (negative: the hands may touch, a hand-over), ``margin`` for every other pair."""
    ca, cb = _arm_spheres(arm_a, q1), _arm_spheres(arm_b, q2) + b_in_a
    d = jnp.linalg.norm(ca[:, None] - cb[None], axis=-1) - arm_a.sphere_radius[:, None] - arm_b.sphere_radius[None]
    both_hands = arm_a.sphere_hand[:, None] * arm_b.sphere_hand[None] > 0
    return jnp.where(both_hands, hand_margin, margin) - d


@jaxls.Cost.factory
def _interarm_cost(
    vals: jaxls.VarValues,
    qa: jaxls.Var,
    qb: jaxls.Var,
    arm_a: _Arm,
    arm_b: _Arm,
    b_in_a: jax.Array,
    margin,
    hand_margin,
    weight,
) -> jax.Array:
    """Stiff push of every sphere pair of the two arms out to its margin (``_interarm_shortfall``); ``b_in_a`` (3,)
    is arm b's base in a's."""
    short = _interarm_shortfall(arm_a, arm_b, vals[qa], vals[qb], b_in_a, margin, hand_margin)
    return weight * jnp.maximum(0.0, short).reshape(-1)


@jdc.jit
def _interarm_distance(
    arm_a: _Arm, arm_b: _Arm, qa: jax.Array, qb: jax.Array, b_in_a: jax.Array, margin, hand_margin
) -> jax.Array:
    """Smallest clearance (m) over ticks beyond each sphere pair's margin (``_interarm_shortfall``; < 0: a pair is
    inside its margin): qa, qb (ticks, dof) pyroki order."""
    short = jax.vmap(lambda q1, q2: _interarm_shortfall(arm_a, arm_b, q1, q2, b_in_a, margin, hand_margin))(qa, qb)
    return -short.max()


@jdc.jit
def _interarm_depth_rows(arm: _Arm, qa: jax.Array, qb: jax.Array, b_in_a: jax.Array, hand_margin) -> jax.Array:
    """(ticks,) m the two arms (one ArmSpec) are in contact at their deepest sphere pair per tick, a gripper pair only
    past ``hand_margin``: qa, qb (ticks, dof) pyroki order. Jitted: run on every decode, a retrace each time was
    ~330 ms on YAM (2026-10-17)."""
    short = jax.vmap(lambda q1, q2: _interarm_shortfall(arm, arm, q1, q2, b_in_a, 0.0, hand_margin))(qa, qb)
    return short.max(axis=(-1, -2))


def _batched(tree, n: int):
    """Every array leaf broadcast to a leading axis of n: one cost per tick, vectorized by jaxls."""
    return jax.tree.map(lambda x: jnp.broadcast_to(x, (n, *jnp.shape(x))), tree)


@jdc.jit
def _solve_traj(arms, obstacles, ws, targets, init, last, before, before2, lo, hi, base, traj):
    """Every tick of every arm in one least-squares problem. Per side (tuples, ``Embodiment.sides`` order): its _Arm,
    obstacles (kind -> (n, params)), weights, targets (ticks, 7) wxyz_xyz, init (ticks, dof) warm start, ``last``,
    ``before`` and ``before2`` (dof,) its last three commands, lo/hi (dof,); pyroki order. ``base`` (sides, 3) or
    None; ``traj`` the serve's weights (``jerk0`` 0 when ``before2`` is not known). -> tuple of (ticks, dof)."""
    n = targets[0].shape[0]
    ticks = jnp.arange(n)
    vars_, costs = [], []
    for i, (arm, obs, w, tgt, lo_i, hi_i) in enumerate(zip(arms, obstacles, ws, targets, lo, hi)):
        var = lambda k, i=i, arm=arm: arm.robot.joint_var_cls(k + i * n)  # noqa: E731
        q = var(ticks)
        vars_.append(q)
        costs += [
            pk.costs.pose_cost_analytic_jac(
                _batched(arm.robot, 1),
                q,
                jaxlie.SE3(tgt),
                jnp.full((n,), arm.tool_index, dtype=jnp.int32),
                pos_weight=w["pos"],
                ori_weight=w["ori"],
            ),
            _limit_cost(q, *_batched((lo_i, hi_i, w["limit"]), n)),
            pk.costs.rest_cost(q, *_batched((arm.rest, arm.rest_weight), n)),
            _step_cost(var(ticks[1:]), var(ticks[:-1]), jnp.full((n - 1,), traj["vel"])),
            _accel_cost(var(ticks[2:]), var(ticks[1:-1]), var(ticks[:-2]), jnp.full((n - 2,), traj["acc"])),
            _jerk_cost(
                var(ticks[3:]), var(ticks[2:-1]), var(ticks[1:-2]), var(ticks[:-3]), jnp.full((n - 3,), traj["jerk"])
            ),
            _history_cost(
                var(ticks[:1]),
                var(ticks[1:2]),
                var(ticks[2:3]),
                last[i][None],
                before[i][None],
                before2[i][None],
                traj["hist"] * traj["vel"][None],
                traj["hist"] * traj["acc"][None],
                traj["hist"] * traj["jerk"][None],
                traj["hist"] * traj["jerk0"][None],
            ),
        ]
        if obs:
            costs.append(
                _collision_cost(
                    q,
                    *_batched(
                        (
                            arm.robot,
                            arm.sphere_link,
                            arm.sphere_local,
                            arm.sphere_radius,
                            arm.sphere_drive,
                            obs,
                            w["collision_margin"],
                            w["collision"],
                        ),
                        n,
                    ),
                )
            )
    if base is not None and len(arms) == 2:
        costs.append(
            _interarm_cost(
                vars_[0],
                vars_[1],
                *_batched(
                    (arms[0], arms[1], base[1] - base[0], traj["margin"], traj["hand_margin"], traj["interarm"]), n
                ),
            )
        )
    sol = (
        jaxls.LeastSquaresProblem(costs=costs, variables=vars_)
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make([v.with_value(q0) for v, q0 in zip(vars_, init)]),
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return tuple(sol[v] for v in vars_)


@jdc.jit
def _tool_fk(arm: _Arm, q):
    """q (..., dof) pyroki order -> (..., 4, 4) tool pose in the arm base frame."""
    return jaxlie.SE3(arm.robot.forward_kinematics(q)[..., arm.tool_index, :]).as_matrix()


def poses_to_wxyz_xyz(T: np.ndarray) -> np.ndarray:
    """(..., 4, 4) poses -> (..., 7) [qw, qx, qy, qz, x, y, z]."""
    lead = T.shape[:-2]
    xyzw = Rot.from_matrix(T.reshape(-1, 4, 4)[:, :3, :3]).as_quat().reshape(*lead, 4)
    return np.concatenate([xyzw[..., 3:], xyzw[..., :3], T[..., :3, 3]], axis=-1)


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray  # (arms, ticks, dof) commanded joints, spec order
    q_solution: np.ndarray  # (arms, ticks, dof) the solver's answer before the speed cap
    pos_miss: np.ndarray  # (arms, ticks) m, commanded tool position vs target
    ang_miss: np.ndarray  # (arms, ticks) rad
    at_cap: np.ndarray  # (arms, ticks) bool, the speed cap shortened this tick's step


class _Group:
    """Sides solved in one vmapped call: they share one ArmSpec."""

    def __init__(
        self,
        emb: Embodiment,
        embodiment: str,
        sides: tuple[str, ...],
        first: int,
        tool: str,
        obstacles: Mapping[str, tuple[Obstacle, ...]],
    ):
        spec = emb.side_arm(sides[0])
        self.arms = slice(first, first + len(sides))  # these sides' rows in the embodiment's side order
        self.arm = build_arm(embodiment, sides[0], tool)
        self.obstacles = build_obstacles(obstacles, sides)
        self.w = {k: jnp.asarray(v) for k, v in spec.weights.model_dump().items()}
        self.rest = np.asarray(spec.rest_pose)

    def to_pyroki(self, q: np.ndarray) -> jax.Array:
        return jnp.asarray(q[..., list(self.arm.to_pyroki)])

    def to_spec(self, q: jax.Array) -> np.ndarray:
        return np.asarray(q)[..., list(self.arm.to_spec)]


class ChunkIK:
    """Compiled chunk IK for one embodiment and tool frame. ``solve(targets, q0, speed_cap)`` with targets
    (arms, ticks, 4, 4) tool poses in each arm's base frame (arms in ``Embodiment.sides`` order), q0
    (arms, dof) the current joints and speed_cap (rad/s) the caller's own per-joint limit, applied where
    lower than the arm's; ``solve_traj`` solves the whole chunk at once (``ServeSpec.solve`` "traj"). The first call per chunk
    length compiles; ``warmup`` does it. Per-side values are
    ``side_rest``, ``side_lo``, ``side_hi``, ``side_max_step`` (arms, dof); ``rest``, ``lo``, ``hi``,
    ``max_step`` and ``fk`` without a side are (dof,) for an embodiment with one ArmSpec on every side. ``obstacles``,
    per side in that arm's base frame, replaces the embodiment's own (a station's mounts where they differ), and
    ``side_base`` the embodiment's ``side_base`` (where a station's arms stand)."""

    def __init__(
        self,
        embodiment: str,
        tool: str,
        hz: float,
        obstacles: Mapping[str, tuple[Obstacle, ...]] | None = None,
        side_base: Mapping[str, tuple[float, float, float]] | None = None,
    ):
        emb = EMBODIMENTS[embodiment]
        obstacles = emb.obstacles if obstacles is None else obstacles
        side_base = emb.side_base if side_base is None else side_base
        self.side_base = None if side_base is None else jnp.asarray([side_base[s] for s in emb.sides])
        assert set(obstacles) == set(emb.sides), f"obstacles per side {sorted(obstacles)} != {emb.sides}"
        self.sides = emb.sides
        groups = [(s,) for s in self.sides] if isinstance(emb.arm, dict) else [self.sides]
        self._groups = [_Group(emb, embodiment, g, self.sides.index(g[0]), tool, obstacles) for g in groups]
        self._side_groups = [g for g, sides in zip(self._groups, groups) for _ in sides]  # one per side
        self.dt = jnp.asarray(1.0 / hz)
        self.side_rest = np.stack([g.rest for g in self._side_groups])  # (arms, dof)
        self.side_lo = np.stack([g.to_spec(g.arm.lo) for g in self._side_groups])  # (arms, dof) margined, spec order
        self.side_hi = np.stack([g.to_spec(g.arm.hi) for g in self._side_groups])
        # (arms, 6) [lo xyz, hi xyz] of each side's Region (at most one), open faces 1 km out; None = no side has one
        regions = [[o for o in obstacles[s] if isinstance(o, Region)] for s in self.sides]
        # the solve's collision margin, which holds the centreline this far in from a Region's faces
        self.collision_margin = max(float(g.w["collision_margin"]) for g in self._groups)
        assert all(len(r) <= 1 for r in regions), f"one Region per side, got {regions}"
        self.side_region = (
            np.asarray([_region_params(r[0]) if r else REGION_OPEN for r in regions]) if any(regions) else None
        )

    def _one_arm(self, per_side: np.ndarray) -> np.ndarray:
        assert len(self._groups) == 1, f"the sides {self.sides} have different arms: use the side_ values"
        return per_side[0]

    @property
    def rest(self) -> np.ndarray:
        return self._one_arm(self.side_rest)

    @property
    def lo(self) -> np.ndarray:
        return self._one_arm(self.side_lo)

    @property
    def hi(self) -> np.ndarray:
        return self._one_arm(self.side_hi)

    def warmup(self, n_ticks: int, serve=None) -> float:
        """Compile for chunks of ``n_ticks`` (and the trajectory solve, when ``serve.solve`` is "traj"); returns the
        seconds it took."""
        t0 = time.perf_counter()
        targets = np.broadcast_to(np.eye(4), (len(self.sides), n_ticks, 4, 4))
        self.solve(targets, self.side_rest)
        if serve is not None and serve.solve == "traj":
            self.solve_traj(targets, self.side_rest, self.side_rest, np.inf, serve)
        return time.perf_counter() - t0

    def _max_step(self, g: _Group, speed_cap: float) -> jax.Array:
        """(dof,) rad per tick, pyroki order."""
        assert speed_cap > 0, speed_cap
        return jnp.minimum(g.arm.max_speed, speed_cap) * self.dt

    def side_max_step(self, speed_cap: float) -> np.ndarray:
        """(arms, dof) rad per tick, spec order: the per-tick joint step ``solve`` keeps under."""
        return np.stack([g.to_spec(self._max_step(g, speed_cap)) for g in self._side_groups])

    def max_step(self, speed_cap: float) -> np.ndarray:
        return self._one_arm(self.side_max_step(speed_cap))

    def solve(self, targets: np.ndarray, q0: np.ndarray, speed_cap: float = np.inf) -> IKResult:
        assert targets.shape[0] == q0.shape[0] == len(self.sides), (targets.shape, q0.shape, self.sides)
        tgt = rearrange(poses_to_wxyz_xyz(targets), "arms ticks d -> ticks arms d")
        parts = []
        for g in self._groups:
            out = _solve_chunk(
                g.arm,
                g.obstacles,
                g.w,
                jnp.asarray(tgt[:, g.arms]),
                g.to_pyroki(q0[g.arms]),
                self._max_step(g, speed_cap),
            )
            parts.append((g.to_spec(out[0]), g.to_spec(out[1]), np.asarray(out[2]), np.asarray(out[3])))
        q_cmd, q_sol, at_cap, reached = (np.concatenate(x, axis=1) for x in zip(*parts))  # (ticks, arms, ...)
        reached = rearrange(reached, "ticks arms d -> arms ticks d")  # (arms, ticks, 7)
        pos_miss = np.linalg.norm(reached[..., 4:] - targets[..., :3, 3], axis=-1)
        R_reached = (
            Rot.from_quat(reached[..., [1, 2, 3, 0]].reshape(-1, 4)).as_matrix().reshape(*reached.shape[:-1], 3, 3)
        )
        ang_miss = np.linalg.norm(
            Rot.from_matrix((R_reached @ np.swapaxes(targets[..., :3, :3], -1, -2)).reshape(-1, 3, 3))
            .as_rotvec()
            .reshape(*reached.shape[:-1], 3),
            axis=-1,
        )
        return IKResult(
            q=rearrange(q_cmd, "ticks arms j -> arms ticks j"),
            q_solution=rearrange(q_sol, "ticks arms j -> arms ticks j"),
            pos_miss=pos_miss,
            ang_miss=ang_miss,
            at_cap=rearrange(at_cap, "ticks arms -> arms ticks"),
        )

    def solve_traj(
        self, targets: np.ndarray, q0: np.ndarray, q_before: np.ndarray, speed_cap: float, serve
    ) -> IKResult:
        """The whole chunk, every arm at once: ``solve``'s targets and caps, with q0 (arms, dof) the last command and
        q_before the commands before it, (arms, dof) the one before or (arms, rows, dof) the rows before, newest
        first (two are read: the second sets the jerk into the chunk; q0 again for an arm that was still), and
        ``serve``'s traj weights. Warm started from the tick solve; the answer is clipped to the margined limits and
        then speed capped tick by tick from q0, as ``solve`` is, so the joints the arm gets keep the same
        guarantees."""
        q_before = np.asarray(q_before, dtype=np.float64)
        third = q_before.ndim == 3 and q_before.shape[1] >= 2  # the row before the one before: the jerk into tick 0
        q_before2 = q_before[:, 1] if third else (q_before[:, 0] if q_before.ndim == 3 else q_before)
        q_before = q_before[:, 0] if q_before.ndim == 3 else q_before
        warm = self.solve(targets, q0, np.inf)
        tgt = poses_to_wxyz_xyz(targets)  # (arms, ticks, 7)
        arms, obstacles, ws, init, last, before, before2, lo, hi = [], [], [], [], [], [], [], [], []
        for i, g in enumerate(self._side_groups):
            k = i - g.arms.start  # this side's row among its group's
            arms.append(g.arm)
            obstacles.append({kind: v[k] for kind, v in g.obstacles.items()})
            # An arm with fewer than six free joints cannot make every rotation: the orientation weight halved (10 ->
            # 5) lets the hand reach the position when the rotation asked is off its reach (SO-101, 2026-10-17: a
            # rotation 10/20 deg off the wrist's reach pulled the hand 18/33 mm off at 10, 10/14 mm at 5; noise alone
            # +0.7 deg orientation, same hand error; Metal, six joints, unchanged at 5 but kept at 10)
            free = int(np.sum(np.asarray(g.arm.hi) > np.asarray(g.arm.lo)))
            ws.append(g.w if free >= 6 else {**g.w, "ori": g.w["ori"] * TRAJ_FEW_JOINTS_ORI_SCALE})
            init.append(g.to_pyroki(warm.q_solution[i]))
            last.append(g.to_pyroki(q0[i]))
            before.append(g.to_pyroki(q_before[i]))
            before2.append(g.to_pyroki(q_before2[i]))
            lo.append(jnp.minimum(g.arm.lo, last[-1]))
            hi.append(jnp.maximum(g.arm.hi, last[-1]))
        traj = {
            k: jnp.asarray(v)
            for k, v in (
                ("vel", serve.traj_vel_weight),
                ("acc", serve.traj_acc_weight),
                ("hist", serve.traj_history_weight),
                ("jerk", serve.traj_jerk_weight),
                ("jerk0", serve.traj_jerk_weight if third else 0.0),
                ("interarm", serve.interarm_weight),
                ("margin", serve.interarm_margin),
                ("hand_margin", serve.interarm_hand_margin),
            )
        }
        # The arm-vs-arm cost (most of the solve's time) only when the tick solve brings the arms within NEAR_M,
        # as the obstacle cost is: teleop keeps them 17 cm apart or more (2026-10-17 YAM replays).
        base = self.side_base if serve.interarm_weight > 0 else None
        if base is not None and len(arms) == 2:
            gap = _interarm_distance(
                arms[0], arms[1], init[0], init[1], base[1] - base[0], traj["margin"], traj["hand_margin"]
            )
            base = base if float(gap) < NEAR_M else None
        sol = _solve_traj(
            tuple(arms),
            tuple(obstacles),
            tuple(ws),
            tuple(jnp.asarray(t) for t in tgt),
            tuple(init),
            tuple(last),
            tuple(before),
            tuple(before2),
            tuple(lo),
            tuple(hi),
            base,
            traj,
        )
        q_sol = np.stack(
            [g.to_spec(np.clip(np.asarray(q), lo_i, hi_i)) for g, q, lo_i, hi_i in zip(self._side_groups, sol, lo, hi)]
        )
        max_step = self.side_max_step(speed_cap)[:, None]  # (arms, 1, dof)
        q, prev, at_cap = np.empty_like(q_sol), np.asarray(q0, dtype=q_sol.dtype), np.zeros(q_sol.shape[:2], bool)
        for t in range(q_sol.shape[1]):  # the tick solve's cap: the step scaled by its most-limited joint
            step = q_sol[:, t] - prev
            scale = np.minimum(1.0, np.min(max_step[:, 0] / np.maximum(np.abs(step), 1e-12), axis=-1, keepdims=True))
            prev = prev + scale * step
            q[:, t], at_cap[:, t] = prev, scale[:, 0] < 1.0
        reached = np.stack([self.fk(q[i], side) for i, side in enumerate(self.sides)])  # (arms, ticks, 4, 4)
        pos_miss = np.linalg.norm(reached[..., :3, 3] - targets[..., :3, 3], axis=-1)
        rel = (reached[..., :3, :3] @ np.swapaxes(targets[..., :3, :3], -1, -2)).reshape(-1, 3, 3)
        ang_miss = np.linalg.norm(Rot.from_matrix(rel).as_rotvec(), axis=-1).reshape(pos_miss.shape)
        return IKResult(q=q, q_solution=q_sol, pos_miss=pos_miss, ang_miss=ang_miss, at_cap=at_cap)

    def solve_per_tick(self, targets: np.ndarray, q0: np.ndarray, speed_cap: float = np.inf) -> np.ndarray:
        """Same answer as ``solve(...).q`` from one compiled call per tick (tests and timing)."""
        tgt = jnp.asarray(poses_to_wxyz_xyz(targets))  # (arms, ticks, 7)
        parts = []
        for g in self._groups:
            q_cmd = q_prev = g.to_pyroki(q0[g.arms])
            lo, hi = jnp.minimum(g.arm.lo, q_cmd), jnp.maximum(g.arm.hi, q_cmd)
            max_step = self._max_step(g, speed_cap)
            out = []
            for k in range(targets.shape[1]):
                q_cmd, q_prev = _solve_tick(g.arm, g.obstacles, g.w, tgt[g.arms, k], q_cmd, q_prev, lo, hi, max_step)
                out.append(g.to_spec(q_cmd))
            parts.append(np.stack(out))  # (ticks, arms of g, dof)
        return rearrange(np.concatenate(parts, axis=1), "ticks arms j -> arms ticks j")

    def clip_to_region(self, targets: np.ndarray, margin: float) -> np.ndarray:
        """targets (arms, ticks, 4, 4) with each hand position moved into its side's Region, ``margin`` (m) in from
        every face (rotations kept); unchanged without a Region."""
        if self.side_region is None:
            return targets
        out = np.array(targets, dtype=np.float64)
        lo, hi = self.side_region[:, None, :3] + margin, self.side_region[:, None, 3:] - margin
        out[..., :3, 3] = np.clip(out[..., :3, 3], lo, np.maximum(hi, lo))
        return out

    def region_depth(self, q: np.ndarray, side: str | None = None) -> np.ndarray:
        """q (arms, ticks, dof) spec order -> (arms, ticks) m the arm's centreline is outside its side's Region at
        its farthest (0 or less: inside); zeros without a Region. With ``side``, q (ticks, dof) -> (ticks,)."""
        if side is not None:
            return self.region_depth(self._side_axis(q, side))[self.sides.index(side)]
        if self.side_region is None:
            return np.zeros(q.shape[:2])
        return np.stack(
            [
                np.asarray(_region_depth(g.arm, jnp.asarray(self.side_region[i]), g.to_pyroki(q[i])), dtype=np.float64)
                for i, g in enumerate(self._side_groups)
            ]
        )

    def fence_region(
        self, q: np.ndarray, q_from: np.ndarray, slack: float, side: str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """The hard half of ``Region``: q (arms, ticks, dof) held, per arm, from the first row whose centreline is
        more than ``slack`` (m) outside the region, or farther out than at ``q_from`` (arms, dof) when that already
        was (an arm outside may come back in, never go deeper), at the row before (``q_from`` for the first row),
        for the rest of the chunk. Returns the rows and (arms,) the row each arm holds from, -1 where none. With
        ``side``, one arm: q (ticks, dof), q_from (dof,) -> (ticks, dof) and (1,)."""
        if side is not None:
            i = self.sides.index(side)
            fenced, held = self.fence_region(self._side_axis(q, side), self._side_axis(q_from, side), slack)
            return fenced[i], held[i : i + 1]
        held = np.full(q.shape[0], -1)
        if self.side_region is None:
            return q, held
        depth = self.region_depth(np.concatenate([q_from[:, None], q], axis=1))  # (arms, 1 + ticks)
        q = np.array(q)
        for i in range(q.shape[0]):
            bad = np.nonzero(depth[i, 1:] > max(slack, depth[i, 0]) + 1e-5)[0]  # 1e-5: float32 FK
            if len(bad):
                t = int(bad[0])
                q[i, t:] = q_from[i] if t == 0 else q[i, t - 1]
                held[i] = t
        return q, held

    def interarm_depth(self, q: np.ndarray, hand_margin: float) -> np.ndarray:
        """q (2, ticks, dof) spec order -> (ticks,) m the two arms are in contact at their deepest sphere pair (0 or
        less: apart), a pair of gripper spheres counting only past ``hand_margin`` (negative: a touch is allowed).
        Needs ``side_base`` and two arms with one ArmSpec."""
        assert self.side_base is not None and len(self.sides) == 2 and len(self._groups) == 1, self.sides
        g = self._groups[0]
        qa, qb = g.to_pyroki(q[0]), g.to_pyroki(q[1])
        b_in_a = self.side_base[1] - self.side_base[0]
        return np.asarray(_interarm_depth_rows(g.arm, qa, qb, b_in_a, jnp.asarray(hand_margin)), dtype=np.float64)

    def keep_out_depth(self, q: np.ndarray, region_slack: float = 0.0) -> np.ndarray:
        """q (arms, ticks, dof) spec order -> (arms, ticks) m each arm's centreline is past what the fence allows at
        its worst (``_keep_out_depth``: inside a capsule or box, or more than ``region_slack`` out of its Region;
        0 or less: clear); zeros for a side with no obstacle."""
        out = []
        for i, g in enumerate(self._side_groups):
            k = i - g.arms.start
            obs = {kind: jnp.asarray(v[k]) for kind, v in g.obstacles.items()}
            depth = _keep_out_depth(g.arm, obs, g.to_pyroki(q[i]), region_slack) if obs else np.zeros(q.shape[1])
            out.append(np.asarray(depth, dtype=np.float64))
        return np.stack(out)

    def fence_keep_out(
        self, q: np.ndarray, q_from: np.ndarray, region_slack: float, side: str | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """The hard half of every obstacle cost, the station's guard in the solver: q (arms, ticks, dof) held, per
        arm, at the row before the first whose centreline enters a capsule or box (the camera pole, a mount) or
        leaves its Region by more than ``region_slack`` (m), or goes deeper than at ``q_from`` (arms, dof) when it
        already was, for the rest of the chunk. Returns the rows and (arms,) the row each arm holds from, -1 where
        none. With ``side``, one arm: q (ticks, dof), q_from (dof,) -> (ticks, dof) and (1,)."""
        if side is not None:
            i = self.sides.index(side)
            fenced, held = self.fence_keep_out(self._side_axis(q, side), self._side_axis(q_from, side), region_slack)
            return fenced[i], held[i : i + 1]
        depth = self.keep_out_depth(np.concatenate([q_from[:, None], q], axis=1), region_slack)  # (arms, 1 + ticks)
        held = np.full(q.shape[0], -1)
        q = np.array(q)
        for i in range(q.shape[0]):
            bad = np.nonzero(depth[i, 1:] > max(0.0, depth[i, 0]) + 1e-5)[0]  # 1e-5: float32 FK
            if len(bad):
                t = int(bad[0])
                q[i, t:] = q_from[i] if t == 0 else q[i, t - 1]
                held[i] = t
        return q, held

    def fence_interarm(
        self, q: np.ndarray, q_from: np.ndarray, slack: float, hand_margin: float
    ) -> tuple[np.ndarray, int]:
        """The hard half of the arm-vs-arm cost: both arms of q (2, ticks, dof) held at the row before (``q_from``
        for the first row) from the first row where they are more than ``slack`` (m) in contact (``interarm_depth``),
        or deeper than at ``q_from`` when they already were, for the rest of the chunk. Returns the rows and the row
        they hold from, -1 for none. A no-op without ``side_base``, two arms and one ArmSpec."""
        if self.side_base is None or len(self.sides) != 2 or len(self._groups) != 1:
            return q, -1
        depth = self.interarm_depth(np.concatenate([q_from[:, None], q], axis=1), hand_margin)
        bad = np.nonzero(depth[1:] > max(slack, depth[0]) + 1e-5)[0]
        if not len(bad):
            return q, -1
        t = int(bad[0])
        q = np.array(q)
        q[:, t:] = (q_from if t == 0 else q[:, t - 1])[:, None]
        return q, t

    def _side_axis(self, per_side: np.ndarray, side: str) -> np.ndarray:
        """One side's array -> the (arms, ...) array of every side, the others at their rest pose (never fenced)."""
        out = np.stack([np.broadcast_to(r, per_side.shape) for r in self.side_rest]).astype(np.float64)
        out[self.sides.index(side)] = per_side
        return out

    def fk(self, q: np.ndarray, side: str | None = None) -> np.ndarray:
        """q (..., dof) spec order -> (..., 4, 4) tool pose in the arm base frame (float64 of a float32 FK).
        ``side`` picks the arm; None only with one ArmSpec on every side."""
        if side is None:
            assert len(self._groups) == 1, f"the sides {self.sides} have different arms: fk(q, side)"
            side = self.sides[0]
        g = self._side_groups[self.sides.index(side)]
        return np.asarray(_tool_fk(g.arm, g.to_pyroki(q)), dtype=np.float64)

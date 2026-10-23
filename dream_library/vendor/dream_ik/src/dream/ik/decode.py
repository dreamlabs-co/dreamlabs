"""The EEF head decoded to joints, on any CPU: the torch-free half of the serving contract.

What every consumer of a served EEF head runs for itself (exec ``"eef"``): a robots station through the
``dream-ik`` wheel from packages/dream_ik, our simulator through ``dream.ik``. The serve never solves (since
2026-10-10). Three steps on the head's absolute tool poses: Savitzky-Golay over the hand path, the chunk solve from the
joints the chunk continues at (under the arm's and the caller's speed caps), and the seam blend that eases in from them.
Only numpy, scipy, ``.solver`` and ``.specs`` here: no torch, nothing else of dream.
"""

import time

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as Rot

from .solver import ChunkIK, IKResult
from .specs import SERVE, ServeSpec

FLIP_MIN_STEP = 1e-3  # rad per tick; smaller steps are solver noise, not motion


def rot6d_to_mat(x: np.ndarray) -> np.ndarray:
    """(..., 6) rotation_6d -> (..., 3, 3): Gram-Schmidt on the two column vectors. The same map as
    the model code's rotation_6d decode (a test holds them equal)."""
    a1, a2 = x[..., :3], x[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-1)


def mat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6): the first two columns, as ``dream.data.action_space.enc_rot(R, "rotation_6d")``."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def head_poses(xyz: np.ndarray, rot6: np.ndarray) -> np.ndarray:
    """(..., 3) m and (..., 6) rotation_6d -> (..., 4, 4); Gram-Schmidt puts the two columns back on SO(3)."""
    T = np.broadcast_to(np.eye(4), (*xyz.shape[:-1], 4, 4)).copy()
    T[..., :3, :3] = rot6d_to_mat(rot6)
    T[..., :3, 3] = xyz
    return T


def smooth_poses(T: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay over the ticks of (..., ticks, 4, 4) poses: positions as they are, rotations as rotation
    vectors relative to the first tick. The window shrinks to the chunk (odd); 0 or too short = off."""
    n = T.shape[-3]
    w = min(window | 1, n if n % 2 else n - 1)
    if window <= 0 or w <= polyorder:
        return T
    R0 = T[..., :1, :3, :3]  # (..., 1, 3, 3)
    rel = np.swapaxes(R0, -1, -2) @ T[..., :3, :3]  # (..., ticks, 3, 3)
    rv = Rot.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(*rel.shape[:-2], 3)  # (..., ticks, 3)
    rv = savgol_filter(rv, w, polyorder, axis=-2, mode="interp")
    out = T.copy()
    out[..., :3, :3] = R0 @ Rot.from_rotvec(rv.reshape(-1, 3)).as_matrix().reshape(rel.shape)
    out[..., :3, 3] = savgol_filter(T[..., :3, 3], w, polyorder, axis=-2, mode="interp")
    return out


def extend_poses(T: np.ndarray, rows: int, k: int = 4) -> np.ndarray:
    """(..., ticks, 4, 4) -> (..., ticks + rows, 4, 4): the added rows continue the last ``k`` rows' mean step, the
    position in a line and the rotation about a fixed axis of the hand's own frame. A station pads a chunk it cut at
    its install row with these for the trajectory solve, which would brake into a hold at its last pose."""
    if rows <= 0:
        return T
    k = min(k, T.shape[-3] - 1)
    last, back = T[..., -1, :, :], T[..., -1 - k, :, :]
    dp = (last[..., :3, 3] - back[..., :3, 3]) / k  # (..., 3)
    rel = np.swapaxes(back[..., :3, :3], -1, -2) @ last[..., :3, :3]
    w = Rot.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(*rel.shape[:-2], 3) / k  # (..., 3) per row
    steps = np.arange(1, rows + 1, dtype=np.float64)
    ext = np.broadcast_to(last[..., None, :, :], (*last.shape[:-2], rows, 4, 4)).copy()
    ext[..., :3, 3] = last[..., None, :3, 3] + steps[:, None] * dp[..., None, :]
    turn = Rot.from_rotvec((steps[:, None] * w[..., None, :]).reshape(-1, 3)).as_matrix().reshape(*ext.shape[:-2], 3, 3)
    ext[..., :3, :3] = last[..., None, :3, :3] @ turn
    return np.concatenate([T, ext], axis=-3)


def seam_blend(
    q: np.ndarray, q_from: np.ndarray, ticks: int, lo: np.ndarray, hi: np.ndarray, max_step: np.ndarray
) -> np.ndarray:
    """Ease a chunk in from the last command: the plan is shifted by its start offset (q_from - q[0]) and the
    shift fades to zero over the first ``ticks`` ticks, so the path keeps its speed and shape without the
    jump at the seam. Clipped to the margined limits (lo, hi), and each tick kept within max_step rad of the one
    before, as the solve was; ticks 0 eases nothing but still clips and caps. q (B, arms, T, dof), q_from
    (B, arms, dof); lo, hi (arms, 1, dof) or (dof,), max_step (arms, dof) or (dof,)."""
    w = np.clip(1.0 - np.arange(1, q.shape[2] + 1) / (ticks + 1), 0.0, 1.0)  # (T,) shift fades 1 -> 0
    eased = np.clip(q + w[None, None, :, None] * (q_from - q[:, :, 0])[:, :, None], lo, hi)
    out, prev = np.empty_like(eased), q_from
    for t in range(eased.shape[2]):  # the shift rides on the solver's capped steps: cap again
        prev = np.clip(eased[:, :, t], prev - max_step, prev + max_step)
        out[:, :, t] = prev
    return out


def motion_diag(q: np.ndarray, q_from: np.ndarray) -> dict:
    """One side's commanded motion per chunk, from the last command: joint travel (rad), direction flips of a
    joint moving both ticks (back-and-forth), and the largest first-tick jump (rad). q (B, T, dof), q_from (B, dof)."""
    step = np.diff(np.concatenate([q_from[:, None], q], axis=1), axis=1)  # (B, T, dof)
    moving = np.abs(step) > FLIP_MIN_STEP
    flips = (np.sign(step[:, 1:]) != np.sign(step[:, :-1])) & moving[:, 1:] & moving[:, :-1]
    return {
        "travel_rad": float(np.abs(step).sum(axis=(1, 2)).mean()),
        "flips": float(flips.sum(axis=(1, 2)).mean()),
        "start_jump_rad": float(np.abs(step[:, 0]).max()),
    }


def side_diag(pos: np.ndarray, ang: np.ndarray, cap: np.ndarray) -> dict:
    """One side's misses (m, rad) and speed-cap ticks, each (ticks,) over the batch."""
    return {
        "pos_miss_p50_m": float(np.percentile(pos, 50)),
        "pos_miss_p95_m": float(np.percentile(pos, 95)),
        "ang_miss_p50_rad": float(np.percentile(ang, 50)),
        "ang_miss_p95_rad": float(np.percentile(ang, 95)),
        "ticks_at_cap": int(cap.sum()),
        "ticks": int(cap.size),
    }


def solve_chunk(
    ik: ChunkIK,
    xyz: np.ndarray,
    rot6: np.ndarray,
    q_from: np.ndarray,
    speed_cap: float,
    serve: ServeSpec = SERVE,
    q_before: np.ndarray | None = None,
) -> tuple[np.ndarray, IKResult]:
    """One chunk for one robot, the three steps of the contract: Savitzky-Golay over the hand path, the solve from
    ``q_from`` under the caps, the seam blend from ``q_from``. ``xyz`` (arms, T, 3) m and ``rot6`` (arms, T, 6) are
    the head's absolute tool poses in each arm's base frame, arms in ``ik.sides`` order; ``q_from`` (arms, dof) the
    joints the chunk continues from, in this arm's own sign; ``speed_cap`` rad/s the caller's own per-joint clamp,
    inf = none (the arm's spec caps still hold). Returns the joint commands (arms, T, dof) and the solver's result
    (misses, cap ticks, the unblended solution)."""
    assert xyz.shape[:2] == rot6.shape[:2] and xyz.shape[0] == q_from.shape[0] == len(ik.sides), (
        xyz.shape,
        rot6.shape,
        q_from.shape,
        ik.sides,
    )
    xyz, rot6, q_from = (np.asarray(a, dtype=np.float64) for a in (xyz, rot6, q_from))
    targets = smooth_poses(head_poses(xyz, rot6), serve.smooth_window, serve.smooth_polyorder)
    # A Region: the hands' targets inside it, as far in as the collision cost holds the spheres
    targets = ik.clip_to_region(targets, ik.collision_margin)
    if serve.solve == "traj":  # continuity is in the solve (q_before: the command before q_from): no ease-in
        res = ik.solve_traj(
            targets, q_from, q_from if q_before is None else np.asarray(q_before, dtype=np.float64), speed_cap, serve
        )
        q = res.q
    else:
        assert serve.solve == "tick", serve.solve
        res = ik.solve(targets, q_from, speed_cap)
        lo, hi = ik.side_lo[:, None], ik.side_hi[:, None]  # (arms, 1, dof)
        q = seam_blend(res.q[None], q_from[None], serve.seam_blend_ticks, lo, hi, ik.side_max_step(speed_cap))[0]
    # with ServeSpec.fences only (off by default): the guarantee the costs are not, no point of an arm in the camera
    # pole or a mount, none past a Region's slack, and the arms not in contact; it holds the arm for the rest of the
    # chunk where it would cross
    if serve.fences and (serve.solve == "traj" or ik.side_region is not None):
        q, _held = ik.fence_keep_out(q, q_from, serve.region_slack)
    if serve.fences and serve.solve == "traj" and serve.interarm_weight > 0:  # and the arm-vs-arm cost's
        q, _held_both = ik.fence_interarm(q, q_from, serve.region_slack, serve.interarm_hand_margin)
    return q, res


def decode_chunk(
    ik: ChunkIK,
    xyz: np.ndarray,
    rot6: np.ndarray,
    q_from: np.ndarray,
    speed_cap: float,
    serve: ServeSpec = SERVE,
    rows: int | None = None,
    q_before: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """``solve_chunk`` for a station: the joint commands (arms, rows, dof) plus the flat diag the serve logs as
    ``ik.diag`` (per side: misses, cap ticks, travel, flips, first-tick jump). ``rows`` keeps the first that many
    rows of the solve and scores only them: a station that solves a padded full-length chunk (one compiled shape)
    passes the number of rows it will execute. ``q_before`` is the history the trajectory solve continues
    (``ServeSpec.solve`` "traj"): (arms, dof) the command before ``q_from``, or (arms, rows, dof) the commands before
    it, newest first (two rows also set the jerk into the chunk); None = the arm was still."""
    t0 = time.perf_counter()
    q, res = solve_chunk(ik, xyz, rot6, q_from, speed_cap, serve, q_before)
    diag = {"solve_ms": 1e3 * (time.perf_counter() - t0), "speed_cap_rad_s": float(speed_cap)}
    k = q.shape[1] if rows is None else min(max(1, int(rows)), q.shape[1])
    q_from = np.asarray(q_from, dtype=np.float64)
    for i, s in enumerate(ik.sides):
        diag |= {
            f"{s}_{d}": v for d, v in side_diag(res.pos_miss[i, :k], res.ang_miss[i, :k], res.at_cap[i, :k]).items()
        }
        diag |= {f"{s}_{d}": v for d, v in motion_diag(q[i, None, :k], q_from[i, None]).items()}
    return q[:, :k], diag

"""Seam join tests C-42f..C-42k: the velocity blend, in joint space and on the end effector.

The end-effector tests run on `TestArm`, a six-joint serial arm with exact forward kinematics and geometric Jacobian
written here in numpy, so the IK and the pose arithmetic are checked against a real kinematic chain without mujoco.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import dream
from dream import seam_join, smoothing
from dream.errors import DreamConfigError

## Window rows used throughout: the library's default ramp.
N = seam_join.RAMP_ROWS


class TestArm:
    """A six-joint serial arm: per joint a fixed offset then a rotation about one local axis; the site after it."""

    __test__ = False  # a helper, not a test class

    OFFSETS = np.array(
        [[0.0, 0.0, 0.1], [0.0, 0.0, 0.05], [0.25, 0.0, 0.0], [0.2, 0.0, 0.0], [0.05, 0.0, 0.0], [0.05, 0.0, 0.0]]
    )
    AXES = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float64)
    SITE = np.array([0.1, 0.0, 0.0])

    def __init__(self) -> None:
        """Counts the calls, so a test can see the client used it."""
        self.calls = 0

    def _frames(
        self, joints: npt.NDArray[np.float64]
    ) -> tuple[list[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]], npt.NDArray[np.float64], Any]:
        """Each joint's world origin and axis, then the site's position and rotation."""
        rotation, position = np.eye(3), np.zeros(3)
        origins, axes = [], []
        for offset, axis, angle in zip(self.OFFSETS, self.AXES, joints, strict=True):
            position = position + rotation @ offset
            origins.append(position.copy())
            axes.append(rotation @ axis)
            rotation = rotation @ seam_join.rotation_matrix(axis * angle)
        return origins, axes, position + rotation @ self.SITE, rotation

    def pose(self, joints_rad: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The site's position and rotation."""
        self.calls += 1
        _, _, position, rotation = self._frames(np.asarray(joints_rad, dtype=np.float64))
        return position, rotation

    def jacobian(self, joints_rad: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """The geometric Jacobian at the site: `axis x (site - origin)` over `axis`, per joint."""
        origins, axes, site, _ = self._frames(np.asarray(joints_rad, dtype=np.float64))
        return np.array([np.concatenate([np.cross(a, site - o), a]) for o, a in zip(origins, axes, strict=True)]).T


class BrokenArm(TestArm):
    """The same arm with a Jacobian of zeros: no IK step moves, so every solve fails."""

    __test__ = False

    def jacobian(self, joints_rad: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Zeros."""
        return np.zeros((6, 6))


## A pose away from the arm's singularities, and a plan that moves every joint from it along a smooth curve.
Q0 = np.array([0.1, 0.6, -0.9, 0.3, 0.5, -0.2])


def plan_window(start: npt.NDArray[np.float64], rows: int = N + 1) -> npt.NDArray[np.float64]:
    """`[rows, 6]` joint rows: a gentle curve from `start`, about 0.01 rad per row per joint."""
    t = np.arange(rows, dtype=np.float64)[:, None]
    return start + 0.01 * t * np.array([1.0, -0.5, 0.8, 1.2, -0.7, 0.4]) + 2e-4 * t**2


def old_path(start: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Old-motion rows at times -1..2 moving in a different direction from the plan, and their times."""
    times = np.array(seam_join.OLD_PATH_TIMES)
    return start + times[:, None] * 0.012 * np.array([-0.6, 0.9, 0.2, -1.0, 0.5, 0.8]), times


def test_c42f_row_zero_is_the_old_motions_next_row() -> None:
    """C-42f: the join's row 0 is the old motion's fitted position at the seam, in joint space and on the site."""
    rows, times = old_path(Q0)
    plan = plan_window(Q0 + 0.03)
    now, velocity = seam_join.fit_motion(rows, times)
    assert np.allclose(now, rows[1], atol=1e-12)  # a line through the old rows: its value at 0 is the row at 0
    joints = seam_join.velocity_blend(now, velocity, plan)
    assert np.array_equal(joints[0], now)
    eef = seam_join.eef_window(rows, times, plan, joints, TestArm())
    assert eef is not None
    assert np.array_equal(eef[0], now)


def test_c42g_the_window_lands_exactly_on_the_plan() -> None:
    """C-42g: the last window row is the plan's exactly, in joint space and on the end effector."""
    rows, times = old_path(Q0)
    plan = plan_window(Q0 + 0.03)
    joints = seam_join.velocity_blend(*seam_join.fit_motion(rows, times), plan)
    assert np.allclose(joints[-1], plan[-1], atol=1e-12)
    eef = seam_join.eef_window(rows, times, plan, joints, TestArm())
    assert eef is not None
    assert np.array_equal(eef[-1], plan[-1])


def test_c42g_rows_after_the_window_are_the_filtered_plan() -> None:
    """C-42g: through the smoother, every row from the clock row plus the ramp on is the filtered chunk's own."""
    rows = 48
    previous = np.column_stack([plan_window(Q0, rows), np.zeros(rows)])
    chunk = np.column_stack([plan_window(Q0 + 0.05, rows), np.ones(rows)])
    smoother = smoothing.SeamSmoother(action_hz=20.0, kinematics={"q_r_chunk": TestArm()})
    smoother._previous["q_r_chunk"] = previous
    observation = {"state.arm_r.joint_pos": previous[10, :6]}
    out = np.asarray(smoother.smooth(observation, {"q_r_chunk": chunk.tolist()}, 0.0, 0.3)["q_r_chunk"])
    clock = 6  # 0.3 s at 20 rows per second
    expected = smoothing.filter_from(chunk, clock)
    assert np.array_equal(out[clock + N :], expected[clock + N :])
    assert np.array_equal(out[clock:, 6], expected[clock:, 6])  # the gripper column is the plan's from the first row


def test_c42h_without_the_landing_the_steps_are_the_linear_velocity_blend() -> None:
    """C-42h: with the landing term off, each step is `(1 - w) * v_old + w * plan step`, `w = (k + 1) / (N + 1)`."""
    rows, times = old_path(Q0)
    plan = plan_window(Q0 + 0.03)
    now, velocity = seam_join.fit_motion(rows, times)
    steps = np.diff(seam_join.velocity_blend(now, velocity, plan, landing=False), axis=0)
    weight = (np.arange(1, N + 1) / (N + 1))[:, None]
    assert np.allclose(steps, (1 - weight) * velocity + weight * np.diff(plan, axis=0), atol=1e-12)


def test_c42i_the_site_path_is_the_blend_of_positions_with_the_landing() -> None:
    """C-42i: the solved joints put the grasp site at `p_now + sum of steps - S(k/N) * miss`, within 0.1 mm."""
    arm = TestArm()
    rows, times = old_path(Q0)
    plan = plan_window(Q0 + 0.03)
    joints = seam_join.velocity_blend(*seam_join.fit_motion(rows, times), plan)
    eef = seam_join.eef_window(rows, times, plan, joints, arm)
    assert eef is not None

    p_now, v_old = seam_join.fit_motion(np.array([arm.pose(row)[0] for row in rows]), times)
    plan_p = np.array([arm.pose(row)[0] for row in plan])
    path = np.vstack([p_now, p_now + np.cumsum(seam_join.blended_steps(v_old, np.diff(plan_p, axis=0)), axis=0)])
    path -= seam_join.smoothstep5(np.arange(N + 1) / N)[:, None] * (path[-1] - plan_p[-1])
    reached = np.array([arm.pose(row)[0] for row in eef])
    # Row 0 is the old motion's joints, whose site is the fit through the old sites only to second order.
    assert np.abs(reached[1:] - path[1:]).max() < 1e-4


def test_c42j_from_rest_the_first_step_is_small() -> None:
    """C-42j: an arm at rest starts slowly: the first step is the plan's own scaled by `1 / (N + 1)`."""
    arm = TestArm()
    plan = plan_window(Q0 + 0.03)
    rest, times = Q0[None, :], np.zeros(1)
    joints = seam_join.velocity_blend(*seam_join.fit_motion(rest, times), plan)
    eef = seam_join.eef_window(rest, times, plan, joints, arm)
    assert eef is not None
    first_speed = np.linalg.norm(arm.pose(eef[1])[0] - arm.pose(eef[0])[0])
    plan_speed = np.linalg.norm(arm.pose(plan[1])[0] - arm.pose(plan[0])[0])
    assert first_speed < 0.25 * plan_speed


def test_c42k_a_failed_solve_falls_back_to_the_joint_blend() -> None:
    """C-42k: kinematics the solve cannot use returns None, and the smoother then uses the joint-space join."""
    rows, times = old_path(Q0)
    plan = plan_window(Q0 + 0.03)
    joints = seam_join.velocity_blend(*seam_join.fit_motion(rows, times), plan)
    assert seam_join.eef_window(rows, times, plan, joints, BrokenArm()) is None

    chunk_rows = 48
    previous = np.column_stack([plan_window(Q0, chunk_rows), np.zeros(chunk_rows)])
    chunk = np.column_stack([plan_window(Q0 + 0.05, chunk_rows), np.ones(chunk_rows)])
    observation = {"state.arm_r.joint_pos": previous[10, :6]}
    outputs = []
    for kinematics in ({"q_r_chunk": BrokenArm()}, None):
        smoother = smoothing.SeamSmoother(action_hz=20.0, kinematics=kinematics)
        smoother._previous["q_r_chunk"] = previous
        outputs.append(smoother.smooth(observation, {"q_r_chunk": chunk.tolist()}, 0.0, 0.3)["q_r_chunk"])
    assert outputs[0] == outputs[1]


def test_c42k_a_branch_flip_falls_back_to_the_joint_blend(monkeypatch: pytest.MonkeyPatch) -> None:
    """C-42k: a solve that lands more than the jump limit from its seed is a branch flip, and the window is refused."""
    rows, times = old_path(Q0)
    plan = plan_window(Q0 + 0.03)
    joints = seam_join.velocity_blend(*seam_join.fit_motion(rows, times), plan)
    monkeypatch.setattr(seam_join, "solve", lambda kinematics, seed, target_p, target_r: seed + 0.5)
    assert seam_join.eef_window(rows, times, plan, joints, TestArm()) is None


def test_c42i_the_end_effector_join_is_used_when_kinematics_is_given() -> None:
    """C-42i: with an arm's kinematics the window rows differ from the joint-space join's and come from the IK."""
    rows = 48
    previous = np.column_stack([plan_window(Q0, rows), np.zeros(rows)])
    chunk = np.column_stack([plan_window(Q0 + 0.05, rows), np.ones(rows)])
    observation = {"state.arm_r.joint_pos": previous[10, :6]}
    arm = TestArm()
    outputs = []
    for kinematics in ({"q_r_chunk": arm}, None):
        smoother = smoothing.SeamSmoother(action_hz=20.0, kinematics=kinematics)
        smoother._previous["q_r_chunk"] = previous
        outputs.append(np.asarray(smoother.smooth(observation, {"q_r_chunk": chunk.tolist()}, 0.0, 0.3)["q_r_chunk"]))
    assert arm.calls > 0
    clock = 6
    assert not np.allclose(outputs[0][clock + 1 : clock + N], outputs[1][clock + 1 : clock + N], atol=1e-6)
    assert np.array_equal(outputs[0][clock + N :], outputs[1][clock + N :])


def test_c42i_rotation_vectors_round_trip() -> None:
    """C-42i: `rotvec` inverts `rotation_matrix`, at small angles, large ones and near a half turn."""
    rng = np.random.default_rng(3)
    for angle in (1e-9, 1e-4, 0.3, 2.0, np.pi - 1e-6):
        axis = rng.normal(size=3)
        vector = angle * axis / np.linalg.norm(axis)
        assert np.allclose(seam_join.rotvec(seam_join.rotation_matrix(vector)), vector, atol=1e-6)


def test_c42i_kinematics_config_is_checked(config: dict[str, Any]) -> None:
    """C-42i C-27a: `kinematics` takes the arms' chunk keys to objects with pose() and jacobian(), nothing else."""
    for bad, match in (
        ([TestArm()], "must be a dict"),
        ({"q_x_chunk": TestArm()}, "q_x_chunk"),
        ({"q_r_chunk": object()}, "pose"),
    ):
        config["kinematics"] = bad
        with pytest.raises(DreamConfigError, match=match):
            dream._open_policy(config)
    config["kinematics"] = {"q_r_chunk": TestArm(), "q_l_chunk": TestArm()}
    with dream._open_policy(config) as client:
        assert set(client._smoother._kinematics) == {"q_r_chunk", "q_l_chunk"}

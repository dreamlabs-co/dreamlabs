"""What the arm is told, per kind of model."""

import dataclasses
from collections.abc import Callable, Iterator
from typing import Any

import conftest
import makers_fakes
import metal_fakes
import numpy as np
import pytest
import so101_fakes
import test_end_to_end
import test_so101
import test_types_adapter

from robots import bodies, config, makers, metal, motor_arm, run, so101, types

dream_ik_decode = pytest.importorskip("dream_ik.decode", reason="dream's IK needs Python 3.11 or newer")
dream_ik_solver = pytest.importorskip("dream_ik.solver")
dream_ik_specs = pytest.importorskip("dream_ik.specs")

HORIZON = 48
CONTROL_HZ = 30.0
OFFSETS = np.array([0.10, 0.08, 0.06, 0.05, 0.05, 0.0])
ROWS = (6, 18, 30, 42)
REACH_TOL_DEG = 0.5
MAX_TICKS = 3000

MEASURED_ENDS = {
    "metal": {"gripper_open_deg": 114.0, "gripper_closed_deg": 1.5},
    "makers": {"gripper_open_deg": -96.0, "gripper_closed_deg": -4.0},
}


@dataclasses.dataclass(frozen=True)
class Kind:
    """One kind of model on the body it runs on, with what its checkpoint's metadata.json says it emits."""

    checkpoint: str
    embodiment: str
    cameras: tuple[str, ...]
    body: str
    ik: str
    grip: tuple[float, float]
    arm_max: tuple[float, ...]
    gripper: bool


KINDS = {
    "beta_on_metal": Kind(
        "dreamlabs/dream-1-beta",
        "yam",
        ("overhead", "wrist_l", "wrist_r"),
        "metal",
        "metal",
        (0.0, 1.0),
        (2.51, 3.122, 3.096, 3.507, 3.727, 4.207),
        True,
    ),
    "metal_ft": Kind(
        "dream-1-ft:367786a6-metalarm-eef-14b2300",
        "emb_0a1b2c3d4e5f6a7b",
        ("head", "wrist_l", "wrist_r"),
        "metal",
        "metal",
        (0.0, 1.014),
        (1.701, 1.803, 1.466, 1.819, 1.237, 1.785),
        False,
    ),
    "so101_ft": Kind(
        "dream-1-ft:367786a6-so101-eef-14b2300",
        "emb_29cd2ba8bb9b70b5",
        tuple(test_so101.SO101_EMBODIMENT["video"]),
        "so101",
        "so101_faas",
        (0.0, 0.867),
        (77.973, 167.414, 180.594, 110.658, 82.469, 0.0),
        False,
    ),
    "makers_ft": Kind(
        "dream-1-ft:00000000-makers-eef",
        "emb_0a1b2c3d4e5f6a7b",
        ("head", "wrist_l", "wrist_r"),
        "makers",
        "makers",
        (0.0, 1.014),
        (1.701, 1.803, 1.466, 1.819, 1.237, 1.785),
        False,
    ),
}


class ReplyAPI(test_end_to_end.FakeDreamAPI):
    """The fake gateway and platform: one model's record, and every reply's `actions` as given."""

    def __init__(self, kind: Kind, actions: dict[str, Any]) -> None:
        """@param kind The model. @param actions What every reply carries."""
        super().__init__(HORIZON)
        self.actions = actions
        self.models = [
            {
                "id": kind.checkpoint,
                "owner": "public",
                "base_model": "dream-1-base",
                "embodiment": kind.embodiment,
                "servable": True,
                "created_at": "2026-10-10T00:00:00Z",
                "cameras": list(kind.cameras),
                "embodiment_config": None,
            }
        ]

    def _reply(self, body: dict[str, Any], index: int) -> tuple[int, dict[str, Any]]:
        """The base fake's sessions, with the actions replaced."""
        status, reply = super()._reply(body, index)
        reply["actions"] = self.actions
        return status, reply


@pytest.fixture(scope="module")
def solvers() -> Iterator[dict[str, Any]]:
    """dream's model of each arm at the loop's rate, built when first asked for."""
    built: dict[str, Any] = {}
    yield built
    built.clear()


def solver_of(solvers: dict[str, Any], ik: str) -> Any:
    """dream's model of the arm `ik`, built once."""
    if ik not in solvers:
        solvers[ik] = dream_ik_solver.ChunkIK(ik, dream_ik_specs.SERVE.tool, CONTROL_HZ)
    return solvers[ik]


def target(solver: Any, i: int) -> np.ndarray:
    """Arm `i`'s joint target, radians: its rest pose moved by #OFFSETS, each joint away from its nearer limit."""
    rest = solver.side_rest[i]
    away = np.where(rest - solver.side_lo[i] < solver.side_hi[i] - rest, 1.0, -1.0)
    return rest + away * OFFSETS


def grips(kind: Kind) -> np.ndarray:
    """The reply's grip column, a quarter each."""
    low, high = kind.grip
    return np.repeat([high, 1.0, low, (low + high) / 2], HORIZON // 4)


def reply(kind: Kind, solver: Any) -> dict[str, Any]:
    """The reply: per arm the tool poses of its target and the grips, beside the parts the arm must never get."""
    per_key: dict[str, Any] = {}
    column = grips(kind).tolist()
    for i, side in enumerate(solver.sides):
        pose = solver.fk(target(solver, i), side)
        per_key[f"action.{side}_eef_xyz"] = [pose[:3, 3].tolist()] * HORIZON
        per_key[f"action.{side}_eef_rot6"] = [
            np.asarray(dream_ik_decode.mat_to_rot6d(pose[:3, :3])).tolist()
        ] * HORIZON
        per_key[f"action.{side}_grip"] = column
        per_key[f"action.{side}_arm"] = [list(kind.arm_max)] * HORIZON
        if kind.gripper:
            per_key[f"action.{side}_gripper"] = column
    chunk = [[*kind.arm_max, grip] for grip in column]
    return {"per_key": per_key, "q_r_chunk": chunk, "q_l_chunk": chunk, "q_r": chunk[0], "q_l": chunk[0]}


def station_config(kind: Kind, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> config.AppConfig:
    """The station's config, solving with `tick`: these tests check the wire (units, joint order, the gripper's ends)
    on one reply that steps a still arm to a target and expect it there by row 6, which the per-row solve does; the
    `traj` solve eases into a step from rest over 9-16 rows by design, and dream's own tests cover
    it."""
    if kind.body == "so101":
        bodies.so101_body.cache_clear()
        monkeypatch.setattr(bodies, "embodiment_config", lambda name, key: test_so101.SO101_EMBODIMENT)
        return dataclasses.replace(test_so101.load(tmp_path, test_so101.so101_file(tmp_path)), ik_solve="tick")
    height, width, _ = conftest.FRAME_SHAPE
    roles = kind.cameras
    return conftest.make_app_config(
        body=kind.body,
        follower_l=config.ArmConfig(can_channel="can_left", **MEASURED_ENDS[kind.body]),
        follower_r=config.ArmConfig(can_channel="can_right", **MEASURED_ENDS[kind.body]),
        cameras=[config.CameraConfig(name=name, device=i, width=width, height=height) for i, name in enumerate(roles)],
        control_hz=CONTROL_HZ,
        action_horizon=HORIZON,
        ik_solve="tick",
    )


def observation(solver: Any, camera_names: list[str]) -> types.Observation:
    """Both arms at dream's rest pose, the gripper half open, one frame per camera."""
    arms = [
        types.ArmState(
            positions_rad=np.asarray(rest, dtype=np.float64),
            velocities_rad_s=np.zeros(6),
            gripper=0.5,
            commanded_rad=None,
            commanded_gripper=None,
        )
        for rest in solver.side_rest
    ]
    images = {name: test_types_adapter.make_frame(name) for name in camera_names}
    state = types.RobotState(right=arms[0], left=arms[1], timestamp_s=1.0)
    return types.Observation(state=state, images=images, timestamp_s=1.0)


def right_arm(kind: Kind, cfg: config.ArmConfig, rest: np.ndarray, clock: Any) -> motor_arm.MotorArm:
    """The right arm's thread on a fake bus and `clock`, its joints at `rest`, opened and energized."""
    opener: Callable[..., Any]
    if kind.body == "so101":
        opener, spec = so101_fakes.FakeFeetechFactory(), so101.ARM
    else:
        factories = {"metal": metal_fakes.FakeCanFactory, "makers": makers_fakes.FakeCanFactory}
        tables = {"metal": (metal.ARM, metal.JOINTS[:6]), "makers": (makers.ARM, makers.JOINTS)}
        opener = factories[kind.body](clock)
        spec, joints = tables[kind.body]
        motors = opener.bus(cfg.can_channel).motors
        for joint, rad in zip(joints, rest, strict=True):
            motors[joint.motor_id].position_deg = float(np.degrees(rad))
        motors[7].position_deg = (cfg.gripper_open_deg + cfg.gripper_closed_deg) / 2
    arm = motor_arm.MotorArm(spec, "right", cfg, opener=opener, clock=clock, sleep=clock.sleep)
    arm.chain.open()
    arm.chain.enable()
    return arm


def drive(
    arm: motor_arm.MotorArm, clock: Any, command: types.ArmCommand, gripper_deg: float, joints_deg: np.ndarray
) -> np.ndarray:
    """Writes one row each tick, as the loop does, until the motors hold it; returns their angles, degrees."""
    for _ in range(MAX_TICKS):
        arm.write(command.positions_rad, command.gripper)
        arm.tick()
        clock.sleep(arm.period)
        held = np.asarray(arm.chain.read().q_deg)
        if abs(held[-1] - gripper_deg) < 0.1 and np.allclose(held[: len(joints_deg)], joints_deg, atol=REACH_TOL_DEG):
            break
    return held


@pytest.mark.parametrize("name", list(KINDS))
def test_a_reply_at_the_checkpoints_action_statistics_tells_the_arm_its_solve_and_the_measured_gripper_ends(
    name: str, solvers: dict[str, Any], tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    kind = KINDS[name]
    solver = solver_of(solvers, kind.ik)
    app = station_config(kind, tmp_path, monkeypatch)
    wire = bodies.for_config(app).wire()
    names = [camera.name for camera in app.cameras]
    server = ReplyAPI(kind, reply(kind, solver))
    server.start()
    monkeypatch.setenv("DREAM_GATEWAY_URL", f"http://{server.host}:{server.port}")
    monkeypatch.setenv("DREAM_API_URL", f"http://{server.host}:{server.port}")
    try:
        with run.open_client(run.policy_config(app, kind.checkpoint)) as client:
            frames, state = wire.to_dream(observation(solver, names), conftest.TASK)
            client.set_observation(frames, state)
            action = client.get_action()
    finally:
        server.close()

    chunk = wire.from_dream(action, 0.0, HORIZON)
    column = grips(kind)
    goal = target(solver, 0)
    free = len(OFFSETS) if kind.body != "so101" else 5
    assert "q_r" not in action and "q_l" not in action
    rows = np.asarray(action["q_r_chunk"], dtype=np.float64)
    np.testing.assert_allclose(rows[list(ROWS), -1], column[list(ROWS)], atol=1e-6)
    cfg = app.follower_r
    clock = makers_fakes.FakeClock()
    arm = right_arm(kind, cfg, solver.side_rest[0][:free], clock)
    stroke = motor_arm.LinearStroke(cfg.gripper_open_deg, cfg.gripper_closed_deg, arm.spec.grip_scale_mm)
    open_goal_mm = stroke.open_mm - arm.spec.grip_margin_mm
    try:
        for row in ROWS:
            command = chunk.actions[row].right
            grip = float(np.clip(column[row], 0.0, 1.0))
            assert command.gripper == pytest.approx(grip)
            np.testing.assert_allclose(command.positions_rad[:free], goal[:free], atol=np.radians(REACH_TOL_DEG))
            gripper_deg = stroke.to_deg(min(grip * arm.spec.grip_scale_mm, open_goal_mm))
            held = drive(arm, clock, command, gripper_deg, np.degrees(goal[:free]))
            assert held[-1] == pytest.approx(gripper_deg, abs=0.1), (row, column[row])
            np.testing.assert_allclose(held[:free], np.degrees(goal[:free]), atol=REACH_TOL_DEG)
            if column[row] >= 1.0:
                assert held[-1] == pytest.approx(stroke.to_deg(open_goal_mm), abs=0.1)
                assert held[-1] != pytest.approx(cfg.gripper_open_deg, abs=0.5)
    finally:
        arm.chain.close(disable_torque=True)

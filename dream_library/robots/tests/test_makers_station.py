"""End to end on a Makers Arm: the station, the real dream client with its IK, a fake DreamAPI that answers with tool
poses, and the Makers driver on the fake CAN bus.

`robots.station.StationController.from_config` builds everything from a Makers config, as `serve()` does; only the
CAN adapters (`makers_fakes.FakeCanFactory`, both arms posed at dream's rest pose) and the cameras are fakes. The
gateway is `test_end_to_end.FakeDreamAPI` with one change: every reply carries no joint chunks, only `per_key`, the
tool poses, grips and joint head of each arm, the way a FaaS EEF fine-tune answers. So what is exercised is the whole
path of a Makers run: the adapter's wire through dream's strict mode, the client adding the tool poses from the
joints, the chunk solved back to joints with dream's IK, the loop, the command thread, the driver's tick and its
frames, and on close the return to rest before torque off.

The fake answers poses computed by dream's own forward kinematics from fixed joint targets, so the arms must end
where those targets are. The loop runs on the real clock at 30 Hz.
"""

import asyncio
import time
from collections.abc import Callable, Iterator
from typing import Any

import conftest
import makers_fakes
import numpy as np
import pytest
import test_end_to_end

from robots import cameras, config, errors, makers, motor_arm, robstride, station

## dream's IK needs Python 3.11 or newer (jax), so on 3.10 this file skips.
dream_ik_decode = pytest.importorskip("dream_ik.decode", reason="dream's IK needs Python 3.11 or newer")
dream_ik_solver = pytest.importorskip("dream_ik.solver")
dream_ik_specs = pytest.importorskip("dream_ik.specs")

ROLES = ("scene", "wrist_l", "wrist_r")

## Rows per chunk: the Makers fine-tunes' horizon, which the client's solver is compiled for.
HORIZON = 48

## The loop's rate on a Makers Arm.
CONTROL_HZ = 30.0

## The model the run names; a Makers Arm has no default.
CHECKPOINT = "dreamlabs/makers-eef-ft"

## Each arm's channel.
CHANNELS = {"left": "can_left", "right": "can_right"}

## Joint targets as offsets from dream's rest pose, radians, per arm; the fake answers their tool poses.
TARGET_OFFSETS = {
    "right": np.array([0.10, -0.08, 0.06, 0.05, -0.05, 0.08]),
    "left": np.array([-0.10, 0.05, -0.06, 0.04, 0.05, -0.08]),
}

## The grip each arm is told, the loop's `[0, 1]` (1 open).
GRIPS = {"right": 0.75, "left": 0.40}

## Where the arms start: the gripper's angle, degrees (about half open).
START_GRIPPER_DEG = -50.0
## The Makers gripper's CAN id, and its stroke at the default ends.
GRIPPER_ID = 7
STROKE = motor_arm.LinearStroke(-96.0, -2.5, makers.GRIPPER_MAX_MM)

## How close the arms must come to the targets, degrees: the solve and the encoders' resolution.
REACH_TOL_DEG = 0.5

## Longest a test waits for one condition, seconds.
WAIT_S = 30.0


@pytest.fixture(scope="module")
def solver() -> Any:
    """dream's Makers model at the loop's rate, for the targets' tool poses and the checks."""
    return dream_ik_solver.ChunkIK("makers", dream_ik_specs.SERVE.tool, CONTROL_HZ)


def targets(solver: Any) -> dict[str, np.ndarray]:
    """Each arm's joint target, radians."""
    rest = np.asarray(solver.rest, dtype=np.float64)
    return {side: rest + offset for side, offset in TARGET_OFFSETS.items()}


class EEFDreamAPI(test_end_to_end.FakeDreamAPI):
    """The fake gateway answering like a FaaS EEF fine-tune: no joint chunks, only `per_key`.

    Attributes:
    - per_key: What every reply carries: per arm `action.<side>_eef_xyz` `[H, 3]`, `action.<side>_eef_rot6`
      `[H, 6]`, `action.<side>_grip` `[H]` (the serve drops a width-1 part's axis) and `action.<side>_arm` `[H, 6]`.
    """

    def __init__(self, per_key: dict[str, Any]) -> None:
        """@param per_key The reply's parts."""
        super().__init__(HORIZON)
        self.per_key = per_key
        self.models = [
            {
                "id": CHECKPOINT,
                "owner": "mine",
                "base_model": "dreamlabs/dream-1-beta",
                "embodiment": "emb_0a1b2c3d4e5f6a7b",
                "servable": True,
                "created_at": "2026-10-01T00:00:00Z",
                "cameras": list(ROLES),
                "embodiment_config": None,
            }
        ]

    def _reply(self, body: dict[str, Any], index: int) -> tuple[int, dict[str, Any]]:
        """The base fake's sessions, with the actions replaced."""
        status, reply = super()._reply(body, index)
        reply["actions"] = {"per_key": self.per_key}
        return status, reply


def eef_per_key(solver: Any) -> dict[str, Any]:
    """The tool poses of each arm's target, held for the whole horizon, with its grip and joint head."""
    per_key: dict[str, Any] = {}
    for side, q in targets(solver).items():
        pose = solver.fk(q)
        rows = {
            "eef_xyz": pose[:3, 3].tolist(),
            "eef_rot6": np.asarray(dream_ik_decode.mat_to_rot6d(pose[:3, :3])).tolist(),
            "arm": q.tolist(),
        }
        for part, row in rows.items():
            per_key[f"action.{side}_{part}"] = [row] * HORIZON
        per_key[f"action.{side}_grip"] = [GRIPS[side]] * HORIZON
    return per_key


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch, solver: Any) -> Iterator[EEFDreamAPI]:
    """A started EEF fake named as the credentials' gateway and platform; closed on teardown."""
    fake = EEFDreamAPI(eef_per_key(solver))
    fake.start()
    monkeypatch.setenv("DREAM_GATEWAY_URL", f"http://{fake.host}:{fake.port}")
    monkeypatch.setenv("DREAM_API_URL", f"http://{fake.host}:{fake.port}")
    try:
        yield fake
    finally:
        fake.close()


@pytest.fixture
def bus(monkeypatch: pytest.MonkeyPatch, solver: Any) -> makers_fakes.FakeCanFactory:
    """Both arms' adapters on the fake bus, every joint at dream's rest pose; the Makers driver opens them, and the
    cameras are fakes."""
    factory = makers_fakes.FakeCanFactory()
    rest_deg = np.degrees(np.asarray(solver.rest, dtype=np.float64))
    for channel in CHANNELS.values():
        motors = factory.bus(channel).motors
        for i, joint in enumerate(makers.JOINTS):
            motors[joint.motor_id].position_deg = float(rest_deg[i])
        motors[GRIPPER_ID].position_deg = START_GRIPPER_DEG
    monkeypatch.setattr(robstride, "open_can_bus", factory)

    def fake_reader(cfg: config.CameraConfig, stale_after_s: float, **_encoding: Any) -> cameras.CameraReader:
        return cameras.CameraReader(cfg.name, conftest.FakeCaptureSource(), stale_after_s)

    monkeypatch.setattr(cameras.CameraReader, "from_config", fake_reader)
    return factory


def makers_config() -> config.AppConfig:
    """A Makers station config over the fake cameras, at the Makers horizon and rate."""
    height, width, _ = conftest.FRAME_SHAPE
    return conftest.make_app_config(
        body="makers",
        follower_l=config.ArmConfig(can_channel=CHANNELS["left"]),
        follower_r=config.ArmConfig(can_channel=CHANNELS["right"]),
        cameras=[config.CameraConfig(name=name, device=i, width=width, height=height) for i, name in enumerate(ROLES)],
        control_hz=CONTROL_HZ,
        action_horizon=HORIZON,
        action_timeout_s=3.0,
        max_stall_ticks=60,
    )


def arm_deg(factory: makers_fakes.FakeCanFactory, side: str) -> np.ndarray:
    """One arm's six joints as its motors hold them, degrees."""
    motors = factory.buses[CHANNELS[side]].motors
    return np.array([motors[joint.motor_id].position_deg for joint in makers.JOINTS])


def gripper_deg(factory: makers_fakes.FakeCanFactory, side: str) -> float:
    """One arm's gripper motor angle, degrees."""
    return factory.buses[CHANNELS[side]].motors[GRIPPER_ID].position_deg


async def wait_for(predicate: Callable[[], bool], what: str) -> None:
    """Polls `predicate` on the event loop, failing the test after #WAIT_S."""
    deadline = time.monotonic() + WAIT_S
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {WAIT_S} s waiting for {what}")
        await asyncio.sleep(0.02)


def test_a_makers_station_runs_an_eef_model_through_the_clients_ik_and_parks_on_close(
    server: EEFDreamAPI, bus: makers_fakes.FakeCanFactory, solver: Any
) -> None:
    """Start is refused without a model; with one, the Makers wire passes strict mode, the client adds each arm's
    tool pose, solves the reply's tool poses to joints, and both arms end at the targets those poses came from with
    their grippers at the grips; close returns both arms to rest and switches every motor off."""
    goal = targets(solver)
    controller = station.StationController.from_config(makers_config(), inference_log_path=None)
    controller.open_cameras()

    def at_targets() -> bool:
        return all(
            np.allclose(arm_deg(bus, side), np.degrees(goal[side]), atol=REACH_TOL_DEG)
            and abs(gripper_deg(bus, side) - STROKE.to_deg(GRIPS[side] * 80.0)) < 0.1
            for side in CHANNELS
        )

    async def scenario() -> dict[str, Any]:
        with pytest.raises(errors.StationError, match=r"no checkpoint"):
            await controller.start()
        await controller.start(checkpoint=CHECKPOINT)
        await wait_for(at_targets, "both arms at the targets")
        running = controller.status()
        await controller.close()
        return running

    running = asyncio.run(scenario())

    assert running["state"] == station.RUNNING and running["checkpoint"] == CHECKPOINT
    assert running["chunks"] >= 1 and running["stalls"] == 0 and running["last_error"] is None
    first = server.requests[0].body
    assert first["checkpoint"] == CHECKPOINT
    observation = first["observation"]
    state_parts = ("arm", "gripper", "grip", "eef_xyz", "eef_rot6")
    assert set(observation) == {
        "prompt",
        *(f"state.{side}_{part}" for side in ("right", "left") for part in state_parts),
        *(f"video.{name}" for name in ROLES),
    }
    for side in ("right", "left"):
        joints = np.asarray(observation[f"state.{side}_arm"])
        # A few encoder counts (16 bits over +-12.57 rad, 0.38 mrad each) from the pose the motors were put in.
        np.testing.assert_allclose(joints, solver.rest, atol=2e-3)
        pose = solver.fk(joints)
        np.testing.assert_allclose(observation[f"state.{side}_eef_xyz"], pose[:3, 3], atol=1e-6)
        np.testing.assert_allclose(
            observation[f"state.{side}_eef_rot6"], dream_ik_decode.mat_to_rot6d(pose[:3, :3]), atol=1e-6
        )
        grip = observation[f"state.{side}_grip"][0]
        assert observation[f"state.{side}_gripper"][0] == pytest.approx(grip * makers.GRIPPER_MAX_MM, rel=1e-6)
        assert grip == pytest.approx(STROKE.to_mm(START_GRIPPER_DEG) / 80.0, abs=1e-3)
    assert all(isinstance(observation[f"video.{name}"][0], str) for name in ROLES)

    for side, channel in CHANNELS.items():
        motors = bus.buses[channel].motors
        assert not any(motor.enabled for motor in motors.values()), side
        assert bus.buses[channel].closed, side
        assert np.all(np.abs(arm_deg(bus, side)) <= np.degrees(motor_arm.REST_TOL_RAD)), side
        stroke = STROKE.to_mm(gripper_deg(bus, side))
        assert stroke >= makers.GRIPPER_MAX_MM - makers.ARM.grip_margin_mm - motor_arm.REST_TOL_MM, side

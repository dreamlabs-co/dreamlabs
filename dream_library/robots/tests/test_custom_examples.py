"""The beta robots shipped as robots of your own (`examples/<robot>/`: a driver file and a config each): each config
loads, the shared loop runs its driver on a fake SDK (a dry run never switching a motor on), opening the reBot's arm
commands no motion, and the dream client's IK turns the loop's joints into what the robot's fine-tune reads and solves
its tool poses back to joints.

The embodiments are the two fine-tunes' own (their checkpoints' `state` and `action`, in the checkpoint's order), as
the account would answer them.
"""

import asyncio
import dataclasses
import datetime
import pathlib
import shutil
import sys
import time
import types
from collections.abc import Iterator
from typing import Any

import conftest
import numpy as np
import pytest
import yaml

from robots import bodies, cameras, config, run

EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"


def spans(parts: tuple[tuple[str, int], ...]) -> dict[str, list[int]]:
    """`(name, width)` parts in order as the embodiment config's `[start, end)` spans."""
    out, at = {}, 0
    for name, width in parts:
        out[name] = [at, at + width]
        at += width
    return out


def eef(width: int) -> tuple[tuple[str, int], ...]:
    """Both sides' tool pose and grip, left first, then each side's `<side>_arm` of `width`."""
    pose = tuple((f"{s}_{p}", w) for s in ("left", "right") for p, w in (("eef_xyz", 3), ("eef_rot6", 6), ("grip", 1)))
    return pose + (("left_arm", width), ("right_arm", width))


## Per robot: its example, dream's arm, its fine-tune's embodiment (faas_ddade1a95a3f0031 and faas_1e7f98780a6e4e7f,
## checkpoint-2300) and its camera roles.
ROBOTS: dict[str, dict[str, Any]] = {
    "seeed_rebot": {
        "ik": "seeed_faas",
        "embodiment": {
            "state": spans(
                (("right_arm", 6), ("right_gripper", 1), ("left_arm", 6), ("left_gripper", 1))
                + tuple(
                    (f"{s}_{p}", w)
                    for s in ("left", "right")
                    for p, w in (("eef_xyz", 3), ("eef_rot6", 6), ("grip", 1))
                )
            ),
            "action": spans(eef(6)),
            "video": {
                "head": "observation.images.top",
                "wrist_l": "observation.images.left",
                "wrist_r": "observation.images.right",
            },
        },
    },
    "trossen": {
        "ik": "trossen_wb_faas",
        "embodiment": {
            "state": spans(
                (("right_arm", 7), ("left_arm", 7))
                + tuple(
                    (f"{s}_{p}", w)
                    for s in ("left", "right")
                    for p, w in (("eef_xyz", 3), ("eef_rot6", 6), ("grip", 1))
                )
            ),
            "action": spans(eef(7)),
            "video": {
                "overhead": "observation.images.camera_main",
                "wrist_l": "observation.images.camera_left",
                "wrist_r": "observation.images.camera_right",
            },
        },
    },
}

## Per dream arm, a pose inside its limits and off its wrist singularity (dream's rest is folded onto the limits).
WORK = {"seeed_faas": (0.2, -0.8, -0.8, 0.3, 0.6, 0.2), "trossen_wb_faas": (0.2, 0.8, 0.8, 0.3, 0.6, 0.2)}

## Every SDK call the fake drivers saw, in order: `(robot, call, side or address)`.
CALLS: list[tuple[str, ...]] = []

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")


def fake_seeed() -> dict[str, types.ModuleType]:
    """Seeed's LeRobot package: motors that hold what they are sent, in motor degrees (direction -1 on the elbow), and
    log every target sent with how far it is from where the motor is."""

    class Cfg:
        def __init__(self, port: str, id: str) -> None:
            self.port, self.id, self.dm_serial_baud = port, id, 921600
            # Wide: the fake client's rows run to tens of radians (conftest.FakeClient).
            self.joint_limits = {m: (-1e6, 1e6) for m in MOTORS} | {"gripper": (0.0, 0.0)}
            self.joint_directions = {m: (-1.0 if m == "elbow_flex" else 1.0) for m in (*MOTORS, "gripper")}

    class Motor:
        def __init__(self) -> None:
            self.pos_rad, self.on = 0.0, False

        def request_feedback(self) -> None:
            pass

        def get_state(self) -> Any:
            return types.SimpleNamespace(pos=self.pos_rad, vel=0.0)

        def ensure_mode(self, mode: str) -> None:
            CALLS.append(("seeed", "mode on" if self.on else "mode off"))

        def enable(self) -> None:
            CALLS.append(("seeed", "enable"))
            self.on = True

        def disable(self) -> None:
            CALLS.append(("seeed", "disable"))
            self.on = False

        def send_pos_vel(self, pos_rad: float, vel_rad_s: float) -> None:
            CALLS.append(("seeed", "target", pos_rad - self.pos_rad))

        def send_force_pos(self, pos_rad: float, vel_rad_s: float, ratio: float) -> None:
            CALLS.append(("seeed", "target", pos_rad - self.pos_rad))

        def close(self) -> None:
            pass

    class Bus:
        def poll_feedback_once(self) -> None:
            pass

        def close(self) -> None:
            CALLS.append(("seeed", "bus_close"))

    class Follower:
        def __init__(self, cfg: Cfg) -> None:
            self.config, self.bus = cfg, None
            self.motors = {m: Motor() for m in (*MOTORS, "gripper")}
            self.motors["gripper"].pos_rad = np.radians(-150.0)

        def _add_motors_to_bus(self) -> None:
            pass

        def connect(self, calibrate: bool) -> None:
            CALLS.append(("seeed", "connect", self.config.port))
            self.bus = Bus()

        def send_action(self, action: dict[str, float]) -> None:
            CALLS.append(("seeed", "send", self.config.port))
            for key, deg in action.items():
                name = key.removesuffix(".pos")
                self.motors[name].pos_rad = np.radians(deg * self.config.joint_directions[name])

        def disconnect(self) -> None:
            CALLS.append(("seeed", "disconnect", self.config.port))

    class Bridge:
        @staticmethod
        def from_dm_serial(serial_port: str, baud: int) -> Bus:
            CALLS.append(("seeed", "bus_open", serial_port))
            return Bus()

    root = types.ModuleType("lerobot_robot_seeed_b601")
    mode = types.SimpleNamespace(POS_VEL="pos_vel", FORCE_POS="force_pos")
    modules = {
        "config_seeed_b601_dm_follower": {"SeeedB601DMFollowerConfig": Cfg},
        "seeed_b601_dm_follower": {"SeeedB601DMFollower": Follower},
        "seeed_b601_follower": {"MotorBridgeController": Bridge, "MotorBridgeMode": mode},
    }
    out = {root.__name__: root}
    for name, attrs in modules.items():
        module = types.ModuleType(f"{root.__name__}.{name}")
        module.__dict__.update(attrs)
        out[module.__name__] = module
    return out


def fake_trossen() -> dict[str, types.ModuleType]:
    """Trossen's trossen_arm: an arm that holds what it is sent, the jaw's opening in metres last."""
    module = types.ModuleType("trossen_arm")
    module.Model = types.SimpleNamespace(wxai_v0="wxai_v0")
    module.StandardEndEffector = types.SimpleNamespace(wxai_v0_follower="wxai_v0_follower")
    module.Mode = types.SimpleNamespace(position="position", idle="idle", external_effort="external_effort")

    class Driver:
        def configure(self, model: str, jaw: str, address: str, clear_error: bool) -> None:
            CALLS.append(("trossen", "configure", address))
            self.address, self.positions = address, [0.0] * 6 + [0.02]

        def set_all_modes(self, mode: str) -> None:
            CALLS.append(("trossen", f"mode {mode}", self.address))

        def get_all_positions(self) -> list[float]:
            return list(self.positions)

        def get_all_velocities(self) -> list[float]:
            return [0.0] * 7

        def set_all_positions(self, positions: list[float], goal_time: float, blocking: bool) -> None:
            CALLS.append(("trossen", "send", self.address))
            self.positions = list(positions)

        def set_all_external_efforts(self, efforts: list[float], goal_time: float, blocking: bool) -> None:
            CALLS.append(("trossen", "effort", self.address))

        def cleanup(self) -> None:
            CALLS.append(("trossen", "cleanup", self.address))

    module.TrossenArmDriver = Driver
    return {"trossen_arm": module}


@pytest.fixture(autouse=True)
def sdks(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Both SDKs faked, the call log cleared, each body built afresh."""
    for name, module in {**fake_seeed(), **fake_trossen()}.items():
        monkeypatch.setitem(sys.modules, name, module)
    CALLS.clear()
    bodies.custom_body.cache_clear()
    bodies.load_driver.cache_clear()
    yield
    bodies.custom_body.cache_clear()
    bodies.load_driver.cache_clear()


def rest(ik: str) -> list[list[float]]:
    """dream's rest pose of the arm, right then left: the homes these tests give the shipped configs."""
    from dream_ik.specs import EMBODIMENTS

    spec = EMBODIMENTS[ik]
    return [list(spec.side_arm(side).rest_pose) for side in ("right", "left")]


def load(tmp_path: pathlib.Path, robot: str, driver: str, monkeypatch: pytest.MonkeyPatch) -> config.AppConfig:
    """The shipped config beside its driver file, as a customer fills it in: an embodiment, a home, a driver class."""
    shutil.copy(EXAMPLES / robot / f"{robot}.py", tmp_path)
    document = yaml.safe_load((EXAMPLES / robot / f"{robot}.yaml").read_text(encoding="utf-8"))
    document["custom"] |= {"driver": f"{robot}.py:{driver}", "embodiment": f"alice/{robot}"}
    home_r, home_l = rest(ROBOTS[robot]["ik"])
    document["follower_r"]["home"], document["follower_l"]["home"] = home_r, home_l
    path = tmp_path / f"{robot}.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    embodiment = ROBOTS[robot]["embodiment"]
    monkeypatch.setattr(bodies, "embodiment_config", lambda name, key: embodiment)
    return dataclasses.replace(config.load(path), rollouts=config.RolloutsConfig(dir=str(tmp_path / "rollouts")))


@pytest.mark.parametrize("robot", sorted(ROBOTS))
def test_the_config_runs_dreams_ik_of_the_arm_on_the_fine_tunes_wire(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, robot: str
) -> None:
    """The client gets `ik` with dream's arm and the head `eef`, and strict mode the fine-tune's own parts."""
    cfg = load(tmp_path, robot, "Arm", monkeypatch)
    dream_config = run.policy_config(cfg, "dream-1-ft:test")
    assert dream_config["ik"] == {"arm": ROBOTS[robot]["ik"], "head": "eef", "solve": "traj"}
    assert dream_config["action_hz"] == 20.0
    assert dream_config["embodiment"]["state"] == ROBOTS[robot]["embodiment"]["state"]


@pytest.mark.parametrize("dry_run", [False, True])
@pytest.mark.parametrize("robot", sorted(ROBOTS))
def test_the_shared_loop_drives_the_driver_and_a_dry_run_switches_no_motor_on(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, robot: str, dry_run: bool
) -> None:
    """A run switches the arms on, drives them on the policy's rows, homes them and switches them off; a dry run on the
    read-only driver opens the arms without switching one on, sends nothing and only closes them."""
    cfg = load(tmp_path, robot, "ReadOnlyArm" if dry_run else "Arm", monkeypatch)
    client = conftest.FakeClient(horizon=cfg.action_horizon)
    roles = list(ROBOTS[robot]["embodiment"]["video"])
    readers = [
        cameras.CameraReader(r, conftest.FakeCaptureSource(), cfg.stale_after_s, face_blur=False) for r in roles
    ]
    monkeypatch.setattr(run, "make_cameras", lambda app: readers)
    monkeypatch.setattr(run, "open_client", lambda dream_config: client)

    async def scenario() -> None:
        stop = asyncio.Event()
        main = asyncio.create_task(run.main(cfg, "dream-1-ft:test", dry_run=dry_run, stop=stop, home_duration_s=0.0))
        deadline_s = time.monotonic() + 20.0
        while client.calls < 3 and not main.done() and time.monotonic() < deadline_s:
            await asyncio.sleep(0.01)
        stop.set()
        await main

    asyncio.run(scenario())

    assert client.calls >= 3
    sent = client.observations[0]
    assert sent["state.right_arm"].shape == (6,), "the loop sends the six joints; the client makes the rest"
    assert sent["state.right_grip"].shape == (1,)
    assert "state.right_gripper" not in sent, "the client makes the gripper drive from the grip"
    assert sorted(k for k in sent if k.startswith("video.")) == sorted(f"video.{r}" for r in roles)
    kinds = [call[1] for call in CALLS]
    assert not {"connect", "disconnect"} & set(kinds), "the reBot's driver calls neither: they can move the arm"
    if dry_run:
        assert not {"enable", "target", "send", "mode position", "mode idle"} & set(kinds), kinds
        assert kinds[-2:] == (["bus_close"] * 2 if robot == "seeed_rebot" else ["cleanup"] * 2)
    else:
        switch_on, motors = ("enable", 7) if robot == "seeed_rebot" else ("mode position", 1)
        assert kinds.count(switch_on) == 2 * motors and kinds.count("send") > 10
        off = ["disable"] * 7 + ["bus_close"] if robot == "seeed_rebot" else ["mode idle", "cleanup"]
        assert kinds[-2 * len(off) :] == off * 2


def test_opening_the_rebots_arm_switches_each_motor_on_where_it_is(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Arm opens the bus without the driver's connect(), sets each motor's mode while it is off, and the first target
    each motor gets once on is where it already is: opening the arm commands no motion."""
    cfg = load(tmp_path, "seeed_rebot", "Arm", monkeypatch)
    driver = bodies.load_driver(f"{tmp_path / 'seeed_rebot.py'}:Arm")(cfg.follower_r, "right")
    assert [call[1] for call in CALLS] == ["bus_open"] + ["disable", "mode off", "enable", "target"] * 7
    assert [call[2] for call in CALLS if call[1] == "target"] == [0.0] * 7
    assert all(motor.on for motor in driver.dev.motors.values())


@pytest.mark.parametrize("robot", sorted(ROBOTS))
def test_the_client_sends_what_the_fine_tune_reads_and_solves_its_tool_poses_back_to_the_joints(robot: str) -> None:
    """`dream.client_ik` on the arm the config names: the reBot's gripper goes out as `<side>_gripper`, Trossen's jaw
    opening as the seventh value of `<side>_arm`; tool poses made by FK of a smooth joint path that starts at rest
    solve back to it with `traj`, the default solve, within 5 mm and 1 degree on every row."""
    from dream import client_ik, models, smoothing

    ik_arm, embodiment = ROBOTS[robot]["ik"], ROBOTS[robot]["embodiment"]
    model = models.Model(
        id="dream-1-ft:test",
        owner="mine",
        base_model="dreamlabs/dream-1",
        embodiment="emb_test",
        servable=True,
        created_at=datetime.datetime.now(datetime.UTC),
        cameras=list(embodiment["video"]),
    )
    ik = client_ik.ClientIK({"arm": ik_arm, "head": "eef"}, 30.0, embodiment, model)
    rows = 48
    t = np.arange(rows)[:, None] / 30.0  # [rows 48, 1] s
    # Starts at rest, as the arm does: `traj` continues the arm's motion, so a path moving at row 0 is eased into.
    wave = 0.15 * (1 - np.cos(2 * np.pi * 0.5 * t)) * np.array([1, 1, -1, 1, 1, 1])  # [rows 48, joints 6] rad
    path = {"right": np.asarray(WORK[ik_arm]) + wave, "left": np.asarray(WORK[ik_arm]) - wave}  # [rows 48, joints 6]
    video = {f"video.{r}": ["frame"] for r in embodiment["video"]}
    obs = {
        "state.right_arm": path["right"][0],
        "state.left_arm": path["left"][0],
        "state.right_grip": np.array([0.25]),
        "state.left_grip": np.array([0.75]),
    } | video
    out = ik.observe(obs)
    if robot == "seeed_rebot":
        np.testing.assert_allclose(out["state.right_gripper"], [0.25])
        assert out["state.right_arm"].shape == (6,)
    else:
        assert out["state.right_arm"].shape == (7,) and "state.right_gripper" not in out
        np.testing.assert_allclose(out["state.right_arm"][6], 0.25 * 0.044)
    poses = np.stack(
        [ik._fk(np.stack([path["right"][i], path["left"][i]])) for i in range(rows)]
    )  # [rows, arms 2, 4, 4]
    per_key = {}
    for a, side in enumerate(("right", "left")):
        per_key[f"action.{side}_eef_xyz"] = poses[:, a, :3, 3]
        per_key[f"action.{side}_eef_rot6"] = np.stack([ik._decode.mat_to_rot6d(p) for p in poses[:, a, :3, :3]])
        per_key[f"action.{side}_grip"] = np.full(rows, 0.5)
    # The client solves with `tick` until `traj` has compiled on its thread; the chunk is solved with `traj`.
    assert ik.traj_ready.wait(120)
    reply = ik.solve({"per_key": per_key}, obs, smoothing.SeamSmoother(30.0), 0.0, None)
    assert reply["q_r_chunk"].shape == reply["q_l_chunk"].shape == (rows, 7)
    solved = np.stack([ik._fk(np.stack([reply["q_r_chunk"][i, :6], reply["q_l_chunk"][i, :6]])) for i in range(rows)])
    xyz_mm = 1000 * np.linalg.norm(solved[..., :3, 3] - poses[..., :3, 3], axis=2)  # [rows 48, arms 2]
    turn = np.einsum("raij,raik->rajk", poses[..., :3, :3], solved[..., :3, :3])  # [rows, arms, 3, 3] target^T solved
    rot_deg = np.degrees(np.arccos(np.clip((np.trace(turn, axis1=2, axis2=3) - 1) / 2, -1, 1)))  # [rows 48, arms 2]
    print(f"{robot}: tool pose miss max {xyz_mm.max():.2f} mm, {rot_deg.max():.2f} deg over {rows} rows")
    assert xyz_mm.max() < 5.0 and rot_deg.max() < 1.0

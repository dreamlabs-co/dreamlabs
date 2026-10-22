"""Arms of any joint count, 1 to `followers.MAX_JOINTS`: each arm's width is its own home's, and nothing on the path
assumes six.

A made-up 7-joint and 5-joint arm (arm files of their own URDFs) and ROBO-T1 (dream's `robo_t_faas`, two
7-joint arms whose rest poses differ per side) go through the setup wizard's checks and saved config, the follower's
reads and commands, the loop on a fine-tune's own joints and on a base model zero-shot through the real dream client,
the live view's tick and the rollout journal. A YAM still opens only with six; the Makers and Metal drivers refuse a
home their chain cannot drive.
"""

import asyncio
import json
import pathlib
import time
from collections.abc import Iterator
from typing import Any, ClassVar

import conftest
import makers_fakes
import metal_fakes
import numpy as np
import numpy.typing as npt
import pytest
import setup_fakes
import test_custom_body
import test_end_to_end
import test_rollouts
import test_setup_flow
import test_setup_wizard
import yaml
from dream import live as dream_live
from shared.contracts import live as wire

from robots import (
    adapter,
    bodies,
    cameras,
    config,
    driver_template,
    errors,
    followers,
    kinematics,
    makers,
    metal,
    motor_arm,
    rollout_episode,
    run,
    setup_session,
    setup_wizard,
    station,
    types,
    yam,
)

specs = pytest.importorskip("dream_ik.specs", reason="dream's IK needs Python 3.11 or newer")

## Rows per reply.
HORIZON = 48
## The grip the fake models ask for.
GRIP = 0.7
## The fine-tune of the account's arm (`test_custom_body.Account`), and the embodiment it was trained on.
FINE_TUNE, OWN = "dream-1-ft:aa-my-arm", "alice/my-arm"
## The embodiment the setup wizard registers for an arm with no fine-tune yet.
NEW = "alice/new-arm"


def arm_file(folder: pathlib.Path, joints: int) -> str:
    """Writes a made-up arm of `joints` revolute joints, 0.1 m apart and turning about z and y in turn, and its arm
    file; returns the arm file's path.

    @param folder Where both files go.
    @param joints Its joints.
    @return The arm file's path.
    """
    links = ['<link name="base_link"/>', *(f'<link name="l{i}"/>' for i in range(1, joints + 1))]
    chain = [
        f'<joint name="j{i}" type="revolute"><parent link="{"base_link" if i == 1 else f"l{i - 1}"}"/>'
        f'<child link="l{i}"/><origin xyz="0 0 0.1"/><axis xyz="{"0 0 1" if i % 2 else "0 1 0"}"/>'
        '<limit lower="-2" upper="2" effort="10" velocity="3"/></joint>'
        for i in range(1, joints + 1)
    ]
    (folder / "arm.urdf").write_text(f'<robot name="made_up">{"".join(links + chain)}</robot>')
    arm = {
        "urdf": "arm.urdf",
        "base": "base_link",
        "flange": f"l{joints}",
        "tcp": [0.0, 0.0, 0.05],
        "approach": [0.0, 0.0, 1.0],
        "jaw": [0.0, 1.0, 0.0],
        "rest": [0.3 if i % 2 else 0.0 for i in range(joints)],
        "tcp_calibration": {"status": "calibrated", "at": "2026-10-16T08:00:00+00:00"},
    }
    path = folder / "arm.json"
    path.write_text(json.dumps(arm))
    return str(path)


## The arms under test, by name: what `ik` names for each, built in a test's folder.
ARMS = ("seven", "five", "robo_t_faas")


def ik_of(name: str, folder: pathlib.Path) -> str:
    """The `ik` of the arm #ARMS names: an arm file written to `folder`, or dream's own arm."""
    return {"seven": lambda: arm_file(folder, 7), "five": lambda: arm_file(folder, 5)}.get(name, lambda: name)()


def rests(ik: str) -> dict[str, npt.NDArray[np.float64]]:
    """Per side, the arm's rest pose in dream's spec: each side's own on ROBO-T1."""
    embodiment = specs.EMBODIMENTS[kinematics.dream_arm(ik)]
    return {side: np.asarray(embodiment.side_arm(side).rest_pose, dtype=np.float64) for side in ("left", "right")}


class SidedArm(driver_template.TemplateArm):
    """The template's arm in memory, keeping every target written per side, `[joints, gripper]`."""

    rows: ClassVar[dict[str, list[list[float]]]] = {"left": [], "right": []}

    def write(self, positions_rad: npt.NDArray[np.float64], gripper: float) -> None:
        SidedArm.rows[self._side].append([*positions_rad, gripper])
        super().write(positions_rad, gripper)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No network for the account's embodiments; each body built afresh; no rows from another test."""
    setup_fakes.stub_embodiments(monkeypatch)
    monkeypatch.setattr(driver_template, "SidedArm", SidedArm, raising=False)
    SidedArm.rows = {"left": [], "right": []}
    yield
    bodies.custom_body.cache_clear()


def embodiment_of(ik: str) -> dict[str, dict[str, Any]]:
    """The embodiment the wizard registers for the arm, with a `<side>_gripper` per arm, last, on an arm whose gripper
    drive the dream client makes (ROBO-T1's)."""
    layout = kinematics.arm_layout(ik)
    out = adapter.zero_shot_embodiment(layout.slots)
    if layout.client_gripper:
        for vector in ("state", "action"):
            end = max(span[1] for span in out[vector].values())
            out[vector] |= {"right_gripper": [end, end + 1], "left_gripper": [end + 1, end + 2]}
    return out


def wire_of(monkeypatch: pytest.MonkeyPatch, ik: str) -> None:
    """Every account embodiment is #embodiment_of the arm."""
    monkeypatch.setattr(bodies, "embodiment_config", lambda name, key: embodiment_of(ik))
    bodies.custom_body.cache_clear()


# The follower and the config


@pytest.mark.parametrize("joints", [7, 5, 1, followers.MAX_JOINTS])
def test_an_arm_reads_and_takes_as_many_joints_as_its_home_lists_and_refuses_any_other_count(joints: int) -> None:
    """The follower's width is its home's; a driver that reads another count, or a command of another count, fails
    naming the arm and saying where its count comes from."""
    home = tuple(0.1 * i for i in range(joints))
    driver = driver_template.TemplateArm(config.ArmConfig(can_channel="sim", home=home), "left")
    arm = followers.FollowerArm("left", driver, home_rad=home)
    assert arm.dof == joints and arm.read().positions_rad.shape == (joints,)
    arm.write(types.ArmCommand(np.full(joints, 0.2), 0.5))
    np.testing.assert_array_equal(arm.read().commanded_rad, np.full(joints, 0.2))

    with pytest.raises(errors.ArmError, match=rf"arm 'left' command has shape \({joints + 1},\).*has {joints} joints"):
        arm.write(types.ArmCommand(np.zeros(joints + 1), 0.5))
    wider = driver_template.TemplateArm(config.ArmConfig(can_channel="sim", home=(0.0,) * (joints + 1)), "left")
    with pytest.raises(errors.ArmError, match=r"arm 'left' driver read .* as many as its home in the config lists"):
        followers.FollowerArm("left", wider, home_rad=home).read()


@pytest.mark.parametrize(("joints", "loads"), [(7, True), (5, True), (16, True), (0, False), (17, False)])
def test_a_custom_robots_home_may_list_1_to_16_joints(tmp_path: pathlib.Path, joints: int, loads: bool) -> None:
    """The config takes any count the live view's tick carries (`TICK_MAX_ARM_DOF`), on either arm, and names the
    arm past it."""
    document = test_custom_body.example()
    document["follower_r"]["home"] = [0.0] * joints
    document["follower_l"]["home"] = [0.0] * 6
    if loads:
        loaded = test_custom_body.load(tmp_path, document)
        assert (len(loaded.follower_r.home), len(loaded.follower_l.home)) == (joints, 6)
    else:
        with pytest.raises(errors.ConfigError, match=r"follower_r.*must be 1 to 16 finite joint angles"):
            test_custom_body.load(tmp_path, document)
    assert followers.MAX_JOINTS == wire.TICK_MAX_ARM_DOF == 16


def test_each_driver_refuses_a_home_its_arm_cannot_drive_naming_the_arm() -> None:
    """A YAM opens only with six; a Makers arm drives six and parks at its six-joint rest; a Metal arm drives six and
    holds any slot past them at 0. Each refuses before anything opens."""
    with pytest.raises(errors.ConfigError, match=r"arm 'left': its home in the config lists 7 angles, but a YAM arm"):
        yam.open_arm(config.ArmConfig(can_channel="can0", home=(0.0,) * 7), "left")

    clock = makers_fakes.FakeClock()
    for home, refused in (((0.0,) * 5, "has 6 joints"), ((0.0,) * 7, "parks at 6 \\(its rest pose\\)")):
        with pytest.raises(errors.ConfigError, match=rf"arm 'right': its home in the config lists .*{refused}"):
            motor_arm.MotorArm(
                makers.ARM,
                "right",
                config.ArmConfig(can_channel="can0", home=home),
                opener=makers_fakes.FakeCanFactory(clock),
                clock=clock,
                sleep=clock.sleep,
            )

    clock = metal_fakes.FakeClock()

    def metal_arm(home: tuple[float, ...]) -> motor_arm.MotorArm:
        cfg = config.ArmConfig(can_channel="can0", home=home, **metal_fakes.GRIPPER_ENDS)
        factory = metal_fakes.FakeCanFactory(clock)
        return motor_arm.MotorArm(metal.ARM, "left", cfg, opener=factory, clock=clock, sleep=clock.sleep)

    with pytest.raises(errors.ConfigError, match=r"arm 'left': its home in the config lists 5 angles .*has 6 joints"):
        metal_arm((0.0,) * 5)
    assert metal_arm((0.0,) * 7).slots == 7


# dream's spec and the wire


def test_each_arms_layout_is_as_wide_as_dreams_spec_of_it(tmp_path: pathlib.Path) -> None:
    """ROBO-T1 sends its seven joints and leaves its gripper to the client; an arm file sends every joint its chain
    has, the loop fills the gripper."""
    assert kinematics.arm_layout("robo_t_faas") == adapter.ArmLayout(tuple(range(7)), 7, True, 7)
    for joints in (7, 5):
        folder = tmp_path / str(joints)
        folder.mkdir()
        assert kinematics.arm_layout(arm_file(folder, joints)) == adapter.ArmLayout(
            tuple(range(joints)), joints, False, joints
        )


def test_the_wire_takes_arms_of_dream_specs_joints_and_refuses_another_count_naming_the_arm() -> None:
    """Each command is as wide as dream's spec of the arm; an arm with fewer or more joints is refused at bind, naming
    it, so no joint past dream's model is commanded 0 unseen by the speed clamp (a Makers Arm's six with a seventh in
    the home); the YAM's wire carries only six."""
    layout = adapter.ArmLayout(tuple(range(5)), 5, False, 5)
    wire_ = adapter.EefWire(adapter.zero_shot_embodiment(5), None, config.CUSTOM_BODY, layout)
    chunk = np.tile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], (HORIZON, 1))  # (48, 5 joints + gripper)
    action = {"q_r_chunk": chunk, "q_l_chunk": chunk}

    first = wire_.for_arms({"left": 5, "right": 5}).from_dream(action, 0.0, HORIZON).actions[0]
    np.testing.assert_allclose(first.right.positions_rad, [0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(first.left.positions_rad, [0.1, 0.2, 0.3, 0.4, 0.5])
    assert first.right.gripper == first.left.gripper == pytest.approx(0.6)

    with pytest.raises(errors.ConfigError, match=r"arm 'left' has 4 joints .* the custom arm has 5"):
        wire_.for_arms({"left": 4, "right": 5})
    with pytest.raises(errors.ConfigError, match=r"arm 'right' has 7 joints .* the custom arm has 5"):
        wire_.for_arms({"left": 5, "right": 7})
    makers_arm = kinematics.arm_layout("makers")
    with pytest.raises(errors.ConfigError, match=r"arm 'left' has 7 joints .* the custom arm has 6"):
        adapter.EefWire(adapter.zero_shot_embodiment(6), None, config.CUSTOM_BODY, makers_arm).for_arms(
            {"left": 7, "right": 7}
        )
    with pytest.raises(errors.ConfigError, match=r"arm 'left' has 7 joints .* a YAM's arm has 6"):
        adapter.YAM_WIRE.for_arms({"left": 7, "right": 7})


def test_a_home_padded_to_six_on_a_five_joint_arm_file_is_refused_when_the_run_loads_naming_the_arm(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A robot of your own set up before arms counted their joints: a five-joint arm file, its home the five angles
    then the 0 every home took. policy_config(), which a run calls before an arm opens (`run.main`), refuses it and
    says what to change."""
    ik = arm_file(tmp_path, 5)
    wire_of(monkeypatch, ik)
    document = test_custom_body.example()
    document["custom"]["ik"] = ik
    for side in ("follower_l", "follower_r"):
        document[side]["home"] = [0.2, 0.1, 0.2, 0.1, 0.2, 0.0]
    cfg = test_custom_body.load(tmp_path, document)

    refused = r"arm 'left' has 6 joints .* has 5: give its home one angle per joint"
    with pytest.raises(errors.ConfigError, match=refused):
        run.policy_config(cfg)


# The setup wizard


@pytest.mark.parametrize("name", ARMS)
def test_the_wizard_checks_and_saves_an_arm_of_any_joint_count(
    name: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both arms found and home set, then every powered check passes: each joint nudged one at a time and the
    gripper, and the sweep; the file saved lists the arm's own count, which the config loads."""
    ik = ik_of(name, tmp_path)
    model = test_setup_flow.model(ik)
    joints = model.width
    wire_of(monkeypatch, ik)
    session = test_setup_flow.Station(ik, custom=True)
    session.find_arms()
    assert session.set_home()["out_of_range"] == []

    motion = session.motion()
    assert motion.state == "done", (motion.error, motion.results[-3:])
    assert [row for row in motion.results if row["finding"].state != "pass"] == []
    assert [row["group"] for row in motion.results].count("nudge") == 2 * (joints + 1)

    arms = {
        side: {"can_channel": f"can{i}", "settings": {}, "home": list(session.session.home[side]), "reach": None}
        for i, side in enumerate(("left", "right"))
    }
    rows = [
        setup_wizard._camera_row(dict(row, name=role), i)
        for i, (row, role) in enumerate(zip(test_setup_wizard.CAMERAS, session.robot.roles, strict=True))
    ]
    path = tmp_path / "station.yaml"
    path.write_text(yaml.safe_dump(setup_wizard.build_config(session.robot, arms, rows), sort_keys=False))
    loaded = config.load(path)
    assert (len(loaded.follower_l.home), len(loaded.follower_r.home)) == (joints, joints)
    assert setup_session.custom_robot(ik, "my.driver:MyArm", NEW).roles == session.robot.roles


# A run: the loop, the policy's request and reply, the live tick


class ModelAPI(test_end_to_end.FakeDreamAPI):
    """The fake gateway answering one model's rows on every call: per side the arm's joints, the tool pose and the
    grip, under `per_key`, each part as wide as the embodiment's."""

    def __init__(self, model: dict[str, Any], per_key: dict[str, Any]) -> None:
        """@param model Its record on `GET /v1/models`. @param per_key Per part, one row, repeated #HORIZON times; a
        `q_r_chunk` or `q_l_chunk` goes beside `per_key`."""
        super().__init__(HORIZON)
        self.rows = {key: [row] * HORIZON for key, row in per_key.items()}
        self.models = [{"servable": True, "created_at": "2026-10-13T00:00:00Z", **model}]

    def _reply(self, body: dict[str, Any], index: int) -> tuple[int, dict[str, Any]]:
        """The base fake's sessions, with the actions replaced."""
        status, reply = super()._reply(body, index)
        chunks = {key: rows for key, rows in self.rows.items() if key.endswith("_chunk")}
        reply["actions"] = {"per_key": {k: v for k, v in self.rows.items() if k not in chunks}, **chunks}
        return status, reply


def run_station(
    monkeypatch: pytest.MonkeyPatch,
    fake: ModelAPI,
    ik: str,
    embodiment: str,
    homes: dict[str, npt.NDArray[np.float64]],
    checkpoint: str,
    action_space: str,
    reached: Any,
) -> wire.Pose:
    """Starts a robot of your own on `checkpoint` from the Live view's Start, as a station does, until `reached()`.

    @return The pose the station shows on Live View for that moment, as the contract parses it.
    """
    fake.start()
    monkeypatch.setenv("DREAM_GATEWAY_URL", f"http://{fake.host}:{fake.port}")
    monkeypatch.setenv("DREAM_API_URL", f"http://{fake.host}:{fake.port}")
    monkeypatch.setattr(bodies, "check_model", test_custom_body.CHECK)
    monkeypatch.setattr(bodies.dream, "DreamClient", test_custom_body.Account)
    monkeypatch.setattr(
        cameras.CameraReader,
        "from_config",
        lambda cfg, stale_after_s, **_: cameras.CameraReader(cfg.name, conftest.FakeCaptureSource(), stale_after_s),
    )
    height, width, _ = conftest.FRAME_SHAPE
    cfg = conftest.make_app_config(
        body=config.CUSTOM_BODY,
        custom=config.CustomConfig(driver="robots.driver_template:SidedArm", ik=ik, embodiment=embodiment),
        # The station is what is tested, not the solve: tick needs no compile, which on CI's runners starved the loop.
        ik_solve="tick",
        follower_l=config.ArmConfig(can_channel="sim", home=tuple(homes["left"])),
        follower_r=config.ArmConfig(can_channel="sim", home=tuple(homes["right"])),
        cameras=[
            config.CameraConfig(name=name, device=i, width=width, height=height)
            for i, name in enumerate(adapter.ZERO_SHOT_CAMERAS)
        ],
        control_hz=30.0,
        action_horizon=HORIZON,
        action_timeout_s=3.0,
        max_stall_ticks=60,
    )
    controller = station.StationController.from_config(cfg, inference_log_path=None)
    controller.open_cameras()
    poses: list[dict[str, Any]] = []

    async def scenario() -> None:
        await controller.start(conftest.TASK, checkpoint, action_space)
        # The client compiles dream's IK solver as it is built; on a GitHub runner its two compiles ran past 30 s.
        deadline = time.monotonic() + 120.0
        while not reached():
            assert time.monotonic() < deadline, (controller.state, controller.last_error, SidedArm.rows)
            await asyncio.sleep(0.02)
        assert controller.last_error is None and controller.loop is not None and controller.loop.tick is not None
        pose = dream_live._pose_message(dream_live._checked_arms(station._live_arms(controller, controller.loop)))
        assert pose is not None
        poses.append(pose)
        await controller.close()

    try:
        asyncio.run(scenario())
    finally:
        fake.close()
    return wire.Pose.model_validate_json(json.dumps(poses[0]))


def last_rows(side: str) -> npt.NDArray[np.float64]:
    """The last two targets written to the arm on `side`, `(2, joints + 1)`; empty before two."""
    return np.asarray(SidedArm.rows[side][-2:])


@pytest.mark.parametrize("name", ARMS)
def test_a_fine_tune_of_the_arm_drives_its_own_joints_through_the_real_client_and_the_live_tick_carries_them(
    name: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fine-tune of the arm, started on joints: the real dream client sends each arm's own joints, the model's rows
    of that many joints and the grip reach each arm, every target as wide as the arm, and the live view's tick carries
    every joint of both arms."""
    ik = ik_of(name, tmp_path)
    targets = rests(ik)
    joints = targets["right"].shape[0]
    homes = {side: target + np.eye(joints)[3] * 0.1 for side, target in targets.items()}
    wire_of(monkeypatch, ik)
    # Each part the fine-tune answers, and, as a serve does for an arm file's fine-tune, each arm's rows; ROBO-T1's
    # the client builds from its parts.
    per_key: dict[str, Any] = {}
    if ik.endswith(".json"):
        per_key = {"q_r_chunk": [*targets["right"], GRIP], "q_l_chunk": [*targets["left"], GRIP]}
    for side, target in targets.items():
        per_key |= {
            f"action.{side}_arm": target.tolist(),
            f"action.{side}_eef_xyz": [0.2, 0.0, 0.1],
            f"action.{side}_eef_rot6": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            f"action.{side}_grip": [GRIP],
        }
        if f"{side}_gripper" in embodiment_of(ik)["action"]:
            per_key[f"action.{side}_gripper"] = [1.0 - GRIP]
    model = {"id": FINE_TUNE, "owner": "mine", "base_model": test_custom_body.BASE, "embodiment": "emb_mine"}
    fake = ModelAPI({**model, "cameras": list(adapter.ZERO_SHOT_CAMERAS)}, per_key)

    def reached() -> bool:
        rows = {side: last_rows(side) for side in targets}
        return all(
            len(rows[side]) == 2 and np.allclose(rows[side], [*targets[side], GRIP], atol=1e-6) for side in targets
        )

    pose = run_station(monkeypatch, fake, ik, OWN, homes, FINE_TUNE, "joint", reached)

    observation = fake.requests[0].body["observation"]
    for side in ("left", "right"):
        np.testing.assert_allclose(observation[f"state.{side}_arm"], homes[side], atol=1e-6)
        assert {len(row) for row in SidedArm.rows[side]} == {joints + 1}
        assert len(pose.arms[side].q) == joints
        np.testing.assert_allclose(pose.arms[side].q, targets[side], atol=1e-6)


@pytest.mark.parametrize("name", ARMS)
def test_a_base_model_runs_the_arm_zero_shot_on_tool_poses_solved_to_its_own_joints(
    name: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Dream Labs base model, trained on YAM arms, started on tool poses: the client sends it YAM's six zero joints
    whatever the arm's count, and solves its tool poses on dream's model of the arm, so each hand reaches the pose
    asked for with the grip asked for, on targets as wide as the arm."""
    from dream_ik import decode
    from dream_ik.solver import ChunkIK

    ik = ik_of(name, tmp_path)
    solver = ChunkIK(kinematics.dream_arm(ik), specs.SERVE.tool, 30.0)
    poses = {side: np.asarray(solver.fk(rest, side)) for side, rest in rests(ik).items()}  # (4, 4) each
    joints = rests(ik)["right"].shape[0]
    homes = {side: rest + np.eye(joints)[3] * 0.1 for side, rest in rests(ik).items()}
    wire_of(monkeypatch, ik)
    per_key: dict[str, Any] = {}
    for side, pose in poses.items():
        per_key |= {
            f"action.{side}_eef_xyz": pose[:3, 3].tolist(),
            f"action.{side}_eef_rot6": np.asarray(decode.mat_to_rot6d(pose[:3, :3])).tolist(),
            f"action.{side}_grip": [GRIP],
            f"action.{side}_arm": [0.0] * 6,
        }
    base = test_custom_body.BASE
    model = {"id": base, "owner": "public", "base_model": base, "embodiment": "yam"}
    fake = ModelAPI({**model, "cameras": ["overhead", "wrist_l", "wrist_r"]}, per_key)

    def miss(side: str, row: npt.NDArray[np.float64]) -> float:
        """How far a target puts the tool from the pose asked for, metres."""
        return float(np.linalg.norm(np.asarray(solver.fk(row[:joints], side))[:3, 3] - poses[side][:3, 3]))

    def reached() -> bool:
        rows = {side: last_rows(side) for side in poses}
        return all(
            len(rows[side]) == 2
            and max(miss(side, row) for row in rows[side]) < 2e-3
            and all(abs(row[joints] - GRIP) < 1e-6 for row in rows[side])
            for side in poses
        )

    pose = run_station(monkeypatch, fake, ik, NEW, homes, base, "eef", reached)

    observation = fake.requests[0].body["observation"]
    for side in ("left", "right"):
        assert observation[f"state.{side}_arm"] == [0.0] * 6
        assert {len(row) for row in SidedArm.rows[side]} == {joints + 1}
        assert len(pose.arms[side].q) == joints


# The rollout journal


def test_an_episode_of_a_7_and_a_5_joint_arm_names_each_arms_own_joints(tmp_path: pathlib.Path) -> None:
    """A run of a 7-joint left arm and a 5-joint right arm: `rollout.json` keeps each arm's count, and every row of
    the saved episode holds each arm's joints and gripper under its own names; an episode from before the counts were
    kept reads as two 6-joint arms."""
    recorder = test_rollouts.make_recorder(tmp_path, joints=(7, 5))
    recorder.begin(conftest.TASK)
    for i in range(test_rollouts.TICKS):
        left = types.ArmState(np.full(7, 0.01 * i), np.zeros(7), 0.5, None, None)
        right = types.ArmState(np.full(5, 0.02 * i), np.zeros(5), 0.5, None, None)
        action = types.Action(types.ArmCommand(np.full(7, 0.1), 0.25), types.ArmCommand(np.full(5, 0.2), 0.25))
        frames = test_rollouts.tick_arguments(i)[5]
        recorder.tick(i / 30.0, 1, i % 4, types.RobotState(left, right, i / 30.0), action, frames)
    recorder.end("stopped")
    recorder.wait()

    root = test_rollouts.only_episode(tmp_path)
    assert json.loads((root / "rollout.json").read_text())["joints"] == {"left": 7, "right": 5}
    meta = json.loads((root / "meta/info.json").read_text())
    names = meta["features"]["observation.state"]["names"]
    assert names == [*(f"left_joint_{i}" for i in range(7)), "left_gripper"] + [
        *(f"right_joint_{i}" for i in range(5)),
        "right_gripper",
    ]
    table = test_rollouts.pq.read_table(root / "data/chunk-000/episode_000000.parquet")
    assert len(table["observation.state"][0]) == len(table["action"][0]) == 7 + 1 + 5 + 1
    assert table["action"][0].as_py() == pytest.approx([0.1] * 7 + [0.25] + [0.2] * 5 + [0.25])
    assert len(rollout_episode.vector_names(rollout_episode.JOINTS_BEFORE_WIDTHS)) == 2 * (6 + 1)


def test_a_journal_of_arms_not_both_six_joints_has_its_own_name_so_an_older_release_never_misreads_it(
    tmp_path: pathlib.Path,
) -> None:
    """While two 7-joint arms record, the journal is `ticks_wide.f64`, never `ticks.f64`, a name that is read as
    rows of two six-joint arms; finishing reads it and deletes it."""
    recorder = test_rollouts.make_recorder(tmp_path, joints=(7, 7))
    root = recorder.begin(conftest.TASK)
    for i in range(test_rollouts.TICKS):
        arm = types.ArmState(np.full(7, 0.01 * i), np.zeros(7), 0.5, None, None)
        command = types.ArmCommand(np.full(7, 0.1), 0.25)
        frames = test_rollouts.tick_arguments(i)[5]
        recorder.tick(i / 30.0, 1, i % 4, types.RobotState(arm, arm, i / 30.0), types.Action(command, command), frames)
    deadline_s = time.monotonic() + 10.0
    while not (root / rollout_episode.WIDE_JOURNAL_NAME).exists() and time.monotonic() < deadline_s:
        time.sleep(0.01)
    assert (root / "ticks_wide.f64").exists() and not (root / "ticks.f64").exists()
    recorder.end("stopped")
    recorder.wait()

    info = json.loads((root / "rollout.json").read_text())
    assert info["frames"] > 0 and not (root / "ticks_wide.f64").exists()
    table = test_rollouts.pq.read_table(root / "data/chunk-000/episode_000000.parquet")
    assert table.num_rows == info["frames"] and len(table["action"][0]) == 2 * (7 + 1)

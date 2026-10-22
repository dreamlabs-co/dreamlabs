"""The beta robots (`robots.beta_robots`): what a guided setup saves for each is the layout its fine-tunes train on.

`shared.tests.datasets.beta_bodies` is each robot's working setup: the body
config its fine-tunes train on, that fine-tune's training contract (state and action parts in trained order,
cameras, rate), the gripper drive's ends, and the tool pose the bake's own URDF gives 8 joint samples per arm (the
poses the robot's EEF features were baked with). Unitree R1 has no contract: its registered body stands in, and its
rate is its datasets'. Per robot, the layout the wizard registers is diffed against the body field by field, run
through the data service's registration rules, the station wire and the dream client, and its arm's tool point
through dream's own IK against the bake's. The data service and the training service
check the same layouts in their own tests."""

import dataclasses
import datetime

import numpy as np
import pytest
import test_types_adapter
from dream import client_ik
from dream.models import Model
from shared.datasets import embodiments
from shared.tests.datasets.beta_bodies import WORKING

from robots import beta_robots, bodies, config, kinematics, makers, metal, setup_session

specs = pytest.importorskip("dream_ik.specs", reason="dream's IK needs Python 3.11 or newer")

ROBOTS = list(beta_robots.robots().values())
FIELDS = ("state", "action", "video", "head_camera")
## What `<side>_gripper` reads with a MakerMods gripper fully open; dream's spec of the other arms carries its own.
GRIP_SCALE = {"metal": metal.GRIP_SCALE_MM, "makers": makers.GRIPPER_MAX_MM}
## Tool point tolerance against the bake's URDF: 0.01 mm and 0.001 degree.
TOOL_MM, TOOL_DEG = 0.01, 0.001


def ids(robot: beta_robots.BetaRobot) -> str:
    return robot.id


def test_the_table_carries_every_tester_and_only_them() -> None:
    assert [robot.id for robot in ROBOTS] == list(WORKING)


@pytest.mark.parametrize("robot", ROBOTS, ids=ids)
def test_the_layout_the_other_suites_check_is_this_codes(robot: beta_robots.BetaRobot) -> None:
    """The data and training services' suites cannot import robots, so they check the fixture's copy (`guided`)."""
    assert beta_robots.embodiment_config(robot) == WORKING[robot.id]["guided"]


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("robot", ROBOTS, ids=ids)
def test_each_field_of_the_layout_is_the_testers(robot: beta_robots.BetaRobot, field: str) -> None:
    assert beta_robots.embodiment_config(robot)[field] == WORKING[robot.id]["body"][field]


@pytest.mark.parametrize("robot", ROBOTS, ids=ids)
def test_the_layout_registers_with_a_head_camera(robot: beta_robots.BetaRobot) -> None:
    body = embodiments.check_registration(beta_robots.embodiment_config(robot))
    # The data service's embodiment registration: a head camera, or a role named overhead, or a 422.
    assert body.head_camera is not None or "overhead" in body.video


@pytest.mark.parametrize("robot", ROBOTS, ids=ids)
def test_it_runs_at_the_rate_its_fine_tune_learned(robot: beta_robots.BetaRobot) -> None:
    """FaaS trains at 30 rows a second and resamples a 15 fps dataset up to it (the trainer's frame rate), so a
    fine-tune of AI Worker or Galaxea R1 Lite recordings is played at 30 Hz, not at the rate it was recorded."""
    working = WORKING[robot.id]
    trained = working["training"]["control_hz"] if working["training"] is not None else working["recorded_fps"]
    assert robot.control_hz == trained


@pytest.mark.parametrize("robot", ROBOTS, ids=ids)
def test_its_cameras_are_the_fine_tunes_in_its_order(robot: beta_robots.BetaRobot) -> None:
    working = WORKING[robot.id]
    trained = working["training"]["cameras"] if working["training"] is not None else list(working["body"]["video"])
    assert list(beta_robots.embodiment_config(robot)["video"]) == trained


@pytest.mark.parametrize("robot", ROBOTS, ids=ids)
def test_its_gripper_drive_spans_the_testers_ends(robot: beta_robots.BetaRobot) -> None:
    """The drive dream's IK makes from the grip runs between the ends the tester's data records (closed, open); a
    MakerMods gripper's is the stroke `<side>_gripper` reads, 0 closed to its grip scale open."""
    spec = specs.EMBODIMENTS[robot.ik]
    for side in ("left", "right"):
        jaw = spec.side_arm(side).jaw
        ends = (0.0, GRIP_SCALE[robot.ik]) if jaw is None else jaw
        np.testing.assert_allclose(ends, WORKING[robot.id]["jaw"][side], atol=1e-3, err_msg=side)


@pytest.mark.parametrize("robot", ROBOTS, ids=ids)
def test_its_tool_point_is_the_one_its_data_was_baked_with(robot: beta_robots.BetaRobot) -> None:
    """dream's IK, as the dream client builds it for the arm (`ChunkIK` on the serve tool), puts the tool where the
    bake's own URDF put it for the same joints, so the EEF the client sends is the EEF the fine-tune learned."""
    poses = WORKING[robot.id]["tool_poses"]
    from dream_ik.solver import ChunkIK

    solver = ChunkIK(kinematics.dream_arm(robot.ik), specs.SERVE.tool, robot.control_hz)
    for side in ("left", "right"):
        q, theirs = np.asarray(poses[side]["q"]), np.asarray(poses[side]["pose"])
        mine = np.stack([solver.fk(row, side) for row in q])
        mm = np.linalg.norm(mine[:, :3, 3] - theirs[:, :3, 3], axis=1) * 1000
        rel = np.einsum("nji,njk->nik", mine[:, :3, :3], theirs[:, :3, :3])
        # The angle from both the sine and cosine of the rotation between them: arccos alone loses it near 0.
        sin = np.linalg.norm(rel[:, [2, 0, 1], [1, 2, 0]] - rel[:, [1, 2, 0], [2, 0, 1]], axis=1) / 2
        deg = np.degrees(np.arctan2(sin, (np.trace(rel, axis1=1, axis2=2) - 1) / 2))
        assert mm.max() <= TOOL_MM and deg.max() <= TOOL_DEG, f"{side}: up to {mm.max():.4f} mm, {deg.max():.4f} deg"


def station_body(robot: beta_robots.BetaRobot, monkeypatch: pytest.MonkeyPatch) -> bodies.Body:
    """The body the station runs for the robot after a guided setup: a MakerMods arm's own (its built-in wire), else
    the SO-101's or a robot of your own's on the embodiment the wizard registered, looked up as the station looks it
    up (`bodies.embodiment_config`), here answered with the layout the wizard registers."""
    if robot.body in bodies.BODIES:
        return bodies.BODIES[robot.body]
    name = f"tester/{robot.ik.replace('_', '-')}"
    registered = beta_robots.embodiment_config(robot)
    monkeypatch.setattr(bodies, "embodiment_config", lambda wanted, key: {name: registered}[wanted])
    if robot.body == "so101":
        return bodies.so101_body(name)
    return bodies.custom_body(config.CustomConfig(driver="mine:Open", ik=robot.ik, embodiment=name))


@pytest.mark.parametrize("robot", ROBOTS, ids=ids)
def test_the_station_and_dream_client_send_what_its_fine_tune_trained_on(
    robot: beta_robots.BetaRobot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One tick as the station runs a fine-tune of the robot (`robots.run`): its body's wire, the cameras the wizard
    names, the `embodiment` block the station hands the dream client, and the client's IK on dream's arm. dream
    receives every state part the tester's body names, as wide as its span, and no other (Unitree R1's
    `<side>_gripper` included, the drive the client makes from the grip after both arms), and one camera in each of
    the fine-tune's camera roles."""
    working = WORKING[robot.id]
    trained = working["training"]["cameras"] if working["training"] is not None else list(working["body"]["video"])
    wire = station_body(robot, monkeypatch).wire()
    roles = setup_session.MAKERMODS_ROLES if robot.body in bodies.BODIES else tuple(robot.cameras)
    layout = kinematics.arm_layout(robot.ik)
    observation = test_types_adapter.make_observation(camera_names=roles)
    arms = {
        side: dataclasses.replace(
            getattr(observation.state, side),
            positions_rad=0.1 * np.arange(layout.slots, dtype=np.float64),
            velocities_rad_s=np.zeros(layout.slots),
        )
        for side in ("left", "right")
    }
    observation = dataclasses.replace(observation, state=dataclasses.replace(observation.state, **arms))
    created = datetime.datetime(2026, 10, 17, tzinfo=datetime.timezone.utc)
    fine_tune = Model(
        f"dream-1-ft:{robot.id}", "public", "dreamlabs/dream-1", f"emb_{robot.id}", True, created, trained
    )
    embodiment = wire.embodiment(list(roles), 30)
    ik = client_ik.ClientIK({"arm": robot.ik, "head": "eef"}, robot.control_hz, embodiment, fine_tune)
    sent = ik.observe(test_types_adapter.sent(wire.to_dream(observation, "pick")))
    state = {
        key.removeprefix("state."): np.asarray(value).shape for key, value in sent.items() if key.startswith("state.")
    }
    assert state == {part: (end - start,) for part, (start, end) in working["body"]["state"].items()}
    assert sorted(key for key in sent if key.startswith("video.")) == sorted(f"video.{role}" for role in trained)


def test_only_an_so101_or_a_robot_of_your_own_takes_the_testers_layout() -> None:
    assert beta_robots.by_ik("metal") is None and beta_robots.by_ik("makers") is None
    assert beta_robots.by_ik("so101_faas").id == "so101"
    assert beta_robots.by_ik("unitree_r1_faas").id == "unitree_r1"
    assert beta_robots.by_ik("yam") is None

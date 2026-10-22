"""Tests for the YAM on its model's own wire (`robots.bodies.YAM`, `robots.adapter.YAM_VIDEO_WIRE`).

The body is the YAM's arms, driver, default model and speed clamp with the observation keyed as the model reads it
(`state.<part>`, `video.<role>`) instead of the deprecated station keys (`state.arm_r.*`, `image_window.*`), with
nothing to switch: no flag and no file key. What is pinned here is the wire the station speaks, the embodiment block
and the dream config the client is built with. The join of the chunks on that wire is the client's and is tested there.
"""

import asyncio
import logging
import pathlib
from typing import Any

import conftest
import numpy as np
import pytest
import test_control
import test_end_to_end
import test_sys0
import test_types_adapter
import yaml
from shared.datasets import embodiments

from robots import adapter, bodies, config, errors, run

## The wire's camera names: the YAM station's, mapped onto the model's roles.
ROLES = ("top", "left", "right")

## The smallest YAM file: both arms, the three cameras, the required loop keys.
YAM_FILE: dict[str, Any] = {
    "follower_l": {"can_channel": "can_left"},
    "follower_r": {"can_channel": "can_right"},
    "cameras": [{"name": name, "device": i, "width": 64, "height": 48} for i, name in enumerate(ROLES)],
    "action_horizon": 48,
    "stale_after_s": 0.2,
    "max_stall_ticks": 10,
    "max_ee_speed_m_s": 0.6,
}


def test_a_yam_config_speaks_its_models_own_wire_with_nothing_to_switch(tmp_path: pathlib.Path) -> None:
    """A YAM file is the YAM body on its model's own wire: the same default model (it reads this wire), no `ik` (the
    model answers in the YAM's joints), the YAM's driver and kinematics; `video` is not a key a file may carry."""
    path = tmp_path / "station.yaml"
    path.write_text(yaml.safe_dump(YAM_FILE), encoding="utf-8")
    body = bodies.for_config(config.load(path))
    assert body is bodies.YAM and body.name == "yam"
    assert body.wire() is adapter.YAM_VIDEO_WIRE and body.wire() is not adapter.YAM_WIRE
    assert body.checkpoint == bodies.YAM_CHECKPOINT and body.ik is None
    path.write_text(yaml.safe_dump({**YAM_FILE, "video": True}), encoding="utf-8")
    with pytest.raises(errors.ConfigError, match="video"):
        config.load(path)


def test_the_yam_wire_sends_the_yams_own_parts_under_the_stations_camera_names() -> None:
    """Right arm then left: `[6]` joints and the `[1]` gripper as itself (the loop's `[0, 1]`), all float32; each
    camera's frame under the model's role for the station's name, whose window the client keeps; none of the station
    wire's keys."""
    wire = bodies.for_config(conftest.make_app_config()).wire()
    assert wire.camera_roles == ROLES
    observation = test_types_adapter.make_observation(camera_names=ROLES)
    payload = test_types_adapter.sent(wire.to_dream(observation, conftest.TASK))
    assert list(payload) == [
        "prompt",
        "state.right_arm",
        "state.right_gripper",
        "state.left_arm",
        "state.left_gripper",
        "video.overhead",
        "video.wrist_l",
        "video.wrist_r",
    ]
    assert payload["prompt"] == conftest.TASK
    for side, arm in (("right", observation.state.right), ("left", observation.state.left)):
        np.testing.assert_array_equal(payload[f"state.{side}_arm"], arm.positions_rad.astype(np.float32))
        assert payload[f"state.{side}_arm"].dtype == np.float32
        assert payload[f"state.{side}_gripper"].tolist() == [arm.gripper]
    for name, role in adapter.YAM_VIDEO_ROLES.items():
        assert payload[f"video.{role}"] == [observation.images[name].jpeg]
    station = test_types_adapter.sent(adapter.to_dream(observation, conftest.TASK))
    assert not set(payload) & set(station) - {"prompt"}, "no station key rides along"
    np.testing.assert_array_equal(payload["state.right_arm"], station["state.arm_r.joint_pos"])
    assert payload["video.overhead"] == station["image_window.top"]
    with pytest.raises(errors.ControlLoopError, match=r"camera 'overhead' is not a yam camera role"):
        wire.to_dream(test_types_adapter.make_observation(camera_names=("overhead",)), conftest.TASK)


def test_the_yam_wire_reads_the_joint_chunks_as_the_station_wire_did() -> None:
    """The serve packs the YAM's joint answer into `q_r_chunk` / `q_l_chunk` on either wire, so the chunk and the
    `commanded` joints are what the station wire gave."""
    wire = bodies.for_config(conftest.make_app_config()).wire()
    reply = conftest.make_chunk(48, base=0.5)
    chunk = wire.from_dream(reply, 1.0, 48)
    station = adapter.from_dream(reply, 1.0, 48)
    assert len(chunk.actions) == 48 and chunk.observed_at_s == 1.0
    for ours, theirs in zip(chunk.actions, station.actions, strict=True):
        np.testing.assert_array_equal(ours.right.positions_rad, theirs.right.positions_rad)
        np.testing.assert_array_equal(ours.left.positions_rad, theirs.left.positions_rad)
        assert (ours.right.gripper, ours.left.gripper) == (theirs.right.gripper, theirs.left.gripper)
    action = chunk.actions[3]
    assert {key: value.tolist() for key, value in wire.commanded(action).items()} == {
        key: value.tolist() for key, value in adapter.commanded_joints(action).items()
    }


def test_the_yam_embodiment_is_the_yam_body_the_shared_parser_reads() -> None:
    """The block is the YAM's embodiment config: 14 state and 14 action dims, the model's video roles for the
    cameras named, and nothing the wire does not send."""
    wire = bodies.for_config(conftest.make_app_config()).wire()
    block = wire.embodiment(list(ROLES), 48)
    body = embodiments.parse(block)
    assert (body.state_dim, body.action_dim) == (14, 14)
    assert list(block["video"]) == list(embodiments.YAM_VIDEO_ROLES) == ["overhead", "wrist_l", "wrist_r"]
    assert wire.embodiment(["top"], 48)["video"] == {"overhead": "observation.images.overhead"}
    _, payload = wire.to_dream(test_types_adapter.make_observation(ROLES), "t")
    assert set(block["state"]) == {key.removeprefix("state.") for key in payload if key.startswith("state.")}
    assert set(block["action"]) == set(block["state"])


def test_policy_config_of_a_yam_runs_the_eef_head_and_noeef_the_joint_head() -> None:
    """By default a YAM runs `dreamlabs/dream-1-beta` on the EEF head: the pose embodiment and `ik` arm `yam`, head
    `eef`. The action space `joint` (`dreamcli robot up noeef`) runs the joint head on the YAM's own embodiment. The
    loop's wire is the YAM's on both; the deprecated station spec (`adapter.embodiment_spec`) never goes out."""
    app = conftest.make_app_config()
    built = run.policy_config(app)
    assert built == {
        "checkpoint": "dreamlabs/dream-1-beta",
        "embodiment": adapter.YAM_POSE_WIRE.embodiment(list(ROLES), app.action_horizon),
        "action_hz": app.control_hz,
        "sync_above_ms": run.SYNC_ABOVE_MS,
        "ik": {"arm": "yam", "head": "eef", "solve": "traj"},
    }
    parsed = embodiments.parse(built["embodiment"])
    assert (parsed.state_dim, parsed.action_dim) == (34, 32)
    assert "right_gripper" not in built["embodiment"]["action"], "EEF models answer their grip, not a gripper"
    joint = run.policy_config(run.with_action_space(app, "joint"), "dreamlabs/dream-1-beta")
    assert joint["embodiment"] == adapter.YAM_VIDEO_WIRE.embodiment(list(ROLES), app.action_horizon)
    assert joint["ik"] == {"arm": "yam", "head": "joint", "solve": "traj"}
    assert built["embodiment"] != adapter.embodiment_spec(list(ROLES), app.action_horizon)
    assert run.loop_settings(app)["wire"] is adapter.YAM_VIDEO_WIRE


## Each hand's pose and grip, which the dream client adds on a YAM (`ik` arm `yam`).
POSE_PARTS = {f"{side}_{part}" for side in ("right", "left") for part in ("eef_xyz", "eef_rot6", "grip")}
## Where both fake arms are, radians: away from zero, so the solved chunk is told from the serve's joint chunk.
HELD_RAD = np.array([0.2, 0.4, 0.3, -0.2, 0.1, 0.3])


class YamModelAPI(test_end_to_end.FakeDreamAPI):
    """The fake gateway and platform: the YAM's default model and a joints-only one, and replies as the asked one
    answers them."""

    def __init__(self, model: str) -> None:
        """@param model `dreamlabs/test-yam-joints` (joints and grippers) or `dreamlabs/dream-1-beta` (also each
        hand's pose, held at #HELD_RAD, and its grip in place of the gripper)."""
        super().__init__(48)
        self.model = model

    def _reply(self, body: dict[str, Any], index: int) -> tuple[int, dict[str, Any]]:
        """The base fake's chunks and joints; for the beta its action head (experiment_cfg/conf.yaml): each arm's
        joints, tool pose and grip, and no `<side>_gripper`."""
        from dream_ik.decode import mat_to_rot6d
        from dream_ik.solver import ChunkIK
        from dream_ik.specs import SERVE

        status, reply = super()._reply(body, index)
        if self.model != "dreamlabs/dream-1-beta":
            return status, reply
        per_key = reply["actions"]["per_key"]
        solver = ChunkIK("yam", SERVE.tool, 20.0)
        for side in ("right", "left"):
            pose = np.asarray(solver.fk(HELD_RAD, side))
            per_key[f"action.{side}_grip"] = per_key.pop(f"action.{side}_gripper")
            per_key[f"action.{side}_eef_xyz"] = [pose[:3, 3].tolist()] * self.horizon
            per_key[f"action.{side}_eef_rot6"] = [np.asarray(mat_to_rot6d(pose[:3, :3])).tolist()] * self.horizon
        return status, reply


def held_observation() -> Any:
    """Both arms at #HELD_RAD, one frame per camera."""
    observation = test_types_adapter.make_observation(camera_names=ROLES)
    for arm in (observation.state.right, observation.state.left):
        arm.positions_rad[:] = HELD_RAD
    return observation


@pytest.mark.parametrize(
    ("model", "action_space", "solved"),
    [
        ("dreamlabs/dream-1-beta", "eef", True),
        ("dreamlabs/test-yam-joints", "eef", False),
        ("dreamlabs/dream-1-beta", "joint", False),
    ],
    ids=["beta-eef", "joints-only-has-no-eef-head", "beta-noeef"],
)
def test_a_yam_run_sends_each_hands_pose_and_runs_the_head_the_model_has(
    model: str, action_space: str, solved: bool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The station's own client, built from policy_config(), sends the YAM's parts plus the client's poses and grips
    (every state part the beta reads). On the EEF head the beta's tool poses are solved to joints (here the held
    pose); a model with no EEF head runs its joint chunks, with one warning naming it; `noeef` runs the joint chunks
    of every model."""
    pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")
    caplog.set_level(logging.WARNING, logger="dream.client_ik")
    server = YamModelAPI(model)
    server.start()
    monkeypatch.setenv("DREAM_GATEWAY_URL", f"http://{server.host}:{server.port}")
    monkeypatch.setenv("DREAM_API_URL", f"http://{server.host}:{server.port}")
    app = run.with_action_space(conftest.make_app_config(), action_space)
    wire = run.loop_settings(app)["wire"]
    observation = held_observation()
    try:
        with run.open_client(run.policy_config(app, model)) as client:
            client.set_observation(*wire.to_dream(observation, conftest.TASK))
            action = client.get_action()
    finally:
        server.close()

    sent = server.requests[0].body["observation"]
    yam_parts = set(adapter.YAM_EMBODIMENT["state"])
    assert {key for key in sent if key.startswith("state.")} == {f"state.{part}" for part in yam_parts | POSE_PARTS}
    for side in ("right", "left"):
        np.testing.assert_allclose(sent[f"state.{side}_arm"], HELD_RAD, atol=1e-6)
    chunk = wire.from_dream(action, 1.0, 48)
    served = np.asarray(conftest.make_chunk(48)["q_r_chunk"])[:, :6]
    rows = np.stack([row.right.positions_rad for row in chunk.actions])
    if solved:
        # dream's IK pulls lightly toward rest, a few milliradians off the held pose.
        np.testing.assert_allclose(rows, np.tile(HELD_RAD, (48, 1)), atol=5e-3)
    else:
        np.testing.assert_allclose(rows, served, atol=1e-9)
    warned = [record.getMessage() for record in caplog.records if "has no EEF head" in record.getMessage()]
    assert warned == (
        [f"{model} has no EEF head (it answers no tool poses): running its joint head on this YAM instead"]
        if (action_space, solved) == ("eef", False)
        else []
    )


def test_a_yam_on_the_eef_head_feeds_the_solved_joints_through_system_0(monkeypatch: pytest.MonkeyPatch) -> None:
    """System 0 tracks the joints the loop commands, so on the EEF head it drives the motors to the client's IK
    solution: the beta's held tool poses solved to #HELD_RAD, not the serve's joint chunk."""
    pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")
    server = YamModelAPI("dreamlabs/dream-1-beta")
    server.start()
    monkeypatch.setenv("DREAM_GATEWAY_URL", f"http://{server.host}:{server.port}")
    monkeypatch.setenv("DREAM_API_URL", f"http://{server.host}:{server.port}")
    app = conftest.make_app_config()
    dream_config = run.policy_config(app, sys0=True)
    assert dream_config["ik"] == {"arm": "yam", "head": "eef", "solve": "traj"}
    robots: dict[str, conftest.FakeRobot] = {
        side: test_sys0.FakeMotorRobot(joint_pos=HELD_RAD.copy()) for side in ("left", "right")
    }
    client = run.open_client(dream_config)
    harness = conftest.make_loop(
        robots=robots, client=client, action_horizon=48, sys0=test_sys0._models(), clock=test_control.YieldingClock()
    )

    async def ticks() -> None:
        async with harness.loop:
            for _ in range(400):
                await harness.loop.step(harness.clock())
                harness.clock.now_s += 1 / app.control_hz
                await asyncio.sleep(0.005)
                if harness.loop._commands.written >= 5 and harness.loop._last_action is not None:
                    return

    try:
        asyncio.run(ticks())
    finally:
        server.close()
    assert harness.loop._commands.written >= 5, "System 0 drove the motors"
    last = harness.loop._last_action
    assert last is not None
    np.testing.assert_allclose(last.right.positions_rad, HELD_RAD, atol=1e-2)

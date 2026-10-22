"""Client-side IK (`dream.client_ik`, config key `ik`): the tool poses out, the model's tool poses back as joints.

The client runs against `FakeDreamAPI` with the FaaS body of a MakerMods Metal arm, a Makers Arm or a FaaS beta robot,
and the fake answers with tool poses made by forward kinematics of a known smooth joint path. `FakePlatform` answers
the checkpoint's record, whose training body decides what the model reads (#CATALOG). The tests that build the
solver need dream's IK (`pip install dreamlabs`, Python 3.11 or newer) and skip without it. Every solve is 48 rows
of both arms, the shape the client compiles at init, so each arm compiles once.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import sys
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from conftest import FRAME_B64, PLATFORM_KEY, FakeDreamAPI, FakePlatform, make_observation

import dream
from dream import client_ik, smoothing, transport
from dream.client import PolicyClient
from dream.errors import DreamActionError, DreamConfigError, DreamObservationError

## Rows per chunk: the rows the client compiles the solver for.
ROWS = client_ik.WARMUP_ROWS

## The loop's row rate, the client's default.
HZ = 20.0

## Arm sides in the solver's order, and the chunk key each comes back under.
SIDES = {"right": "q_r_chunk", "left": "q_l_chunk"}

## The FaaS body of a two-arm EEF fine-tune (the datasets' meta/info.json): 34 wide state, the six EEF parts first in
## the action, then the joint head.
FAAS_CONFIG: dict[str, Any] = {
    "state": {
        "left_arm": [0, 6],
        "left_gripper": [6, 7],
        "right_arm": [7, 13],
        "right_gripper": [13, 14],
        "right_eef_xyz": [14, 17],
        "right_eef_rot6": [17, 23],
        "right_grip": [23, 24],
        "left_eef_xyz": [24, 27],
        "left_eef_rot6": [27, 33],
        "left_grip": [33, 34],
    },
    "action": {
        "right_eef_xyz": [0, 3],
        "right_eef_rot6": [3, 9],
        "right_grip": [9, 10],
        "left_eef_xyz": [10, 13],
        "left_eef_rot6": [13, 19],
        "left_grip": [19, 20],
        "left_arm": [20, 26],
        "right_arm": [26, 32],
    },
    "video": {"top": "observation.images.top"},
}

## The state parts the client computes; the caller sends every other one.
POSE_PARTS = ("right_eef_xyz", "right_eef_rot6", "left_eef_xyz", "left_eef_rot6")

FINE_TUNE = "dream-1-ft:0a1b2c3d-metal"

PLAIN = "metalarm-eef"

BASE = "dreamlabs/dream-1-beta"

YAM_FINE_TUNE = "dream-1-ft:1d845dea-yam"

BETA = "dream-1-ft:5e6f7a8b-beta"

YAM_CAMERAS = ["overhead", "wrist_l", "wrist_r"]


def record(checkpoint: str, embodiment: str, cameras: list[str] | None) -> dict[str, Any]:
    """The platform's record of a servable checkpoint trained on `embodiment`."""
    owner = "mine" if checkpoint.startswith("dream-1-ft:") else "public"
    return {
        "id": checkpoint,
        "owner": owner,
        "base_model": "dreamlabs/dream-1-beta",
        "embodiment": embodiment,
        "servable": True,
        "created_at": "2026-10-01T00:00:00Z",
        "cameras": cameras,
        "embodiment_config": None,
    }


CATALOG = [
    record(FINE_TUNE, "emb_0a1b2c3d4e5f6a7b", ["head", "wrist_l", "wrist_r"]),
    record(PLAIN, "makermods_metal", None),
    record(BASE, "yam", YAM_CAMERAS),
    record(YAM_FINE_TUNE, "yam", YAM_CAMERAS),
    record(BETA, "emb_5e6f7a8b9c0d1e2f", ["top"]),
]


@pytest.fixture
def platform(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The fake platform, named by the credentials (the environment) with the key it takes, stopped after the test."""
    fake = FakePlatform(CATALOG)
    monkeypatch.setenv("DREAM_API_URL", fake.url)
    monkeypatch.setenv("DREAM_API_KEY", PLATFORM_KEY)
    yield fake
    fake.close()


@pytest.fixture
def connection(server: FakeDreamAPI, platform: FakePlatform) -> dict[str, Any]:
    """The connection keys, a model trained on the robot and the FaaS body against `server`, without `ik`."""
    return {
        "host": server.host,
        "port": server.port,
        "scheme": "http",
        "timeout_ms": 2000,
        "checkpoint": PLAIN,
        "embodiment": copy.deepcopy(FAAS_CONFIG),
    }


def ik_client(
    connection: dict[str, Any],
    arm: str = "metal",
    head: str = "eef",
    keep_out: Any = None,
    reach: Any = None,
    solve: str | None = "tick",
    region: Any = None,
    **keys: Any,
) -> PolicyClient:
    """A client with `ik` set (`keep_out`, `reach`, `solve`, `region` when given); skips when dream's IK is absent.

    `solve` is `tick` unless a test names another: its rows are the same whatever the machine's speed, where the
    default, `traj`, solves the first chunks with `tick` until it has compiled. None leaves the key out.
    """
    pytest.importorskip("dream_ik")
    ik = {"arm": arm, "head": head} | ({} if keep_out is None else {"keep_out": keep_out})
    ik |= {} if solve is None else {"solve": solve}
    ik |= {} if region is None else {"region": region}
    ik |= {} if reach is None else {"reach": reach}
    return dream._open_policy({**connection, "ik": ik, **keys})


def body_observation(joints: npt.NDArray[np.float64]) -> dict[str, Any]:
    """An observation of the FaaS body with the arms at `joints` (`[2, 6]`, right then left), without the poses."""
    observation: dict[str, Any] = {}
    for part, (start, end) in FAAS_CONFIG["state"].items():
        if part not in POSE_PARTS:
            observation[f"state.{part}"] = np.zeros(end - start, dtype=np.float32)
    observation["state.right_arm"] = joints[0].astype(np.float32)
    observation["state.left_arm"] = joints[1].astype(np.float32)
    observation["video.top"] = [FRAME_B64]
    observation["prompt"] = "pick up the cup"
    return observation


def joint_path(client: PolicyClient, arm: str, seed: int) -> npt.NDArray[np.float64]:
    """`[2, ROWS, 6]` smooth joint paths well inside the arm's limits, at a third of its slowest speed cap or less."""
    from dream_ik.specs import EMBODIMENTS

    solver = client._ik._solver
    speed = min(0.3, min(EMBODIMENTS[arm].arm.max_speed) / 3 / np.pi)
    rng = np.random.default_rng(seed)
    t = np.arange(ROWS) / HZ
    center = np.clip(solver.rest + rng.uniform(-0.3, 0.3, (2, 1, 6)), solver.lo + 0.7, solver.hi - 0.7)
    return center + speed * np.sin(np.pi * t[:, None] + rng.uniform(0, 2 * np.pi, (2, 1, 6)))


def tool_poses(client: PolicyClient, joints: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """`[2, rows, 4, 4]` tool poses of `[2, rows, 6]` joints, one row at a time: the shape the client compiled."""
    solver = client._ik._solver
    return np.stack([solver.fk(joints[:, row]) for row in range(joints.shape[1])], axis=1)


def eef_reply(client: PolicyClient, path: npt.NDArray[np.float64]) -> dict[str, Any]:
    """The FaaS serve's action for an EEF checkpoint whose tool poses are those of `path`.

    `per_key` carries both heads: the tool poses and a grip ramp, and a joint head held at zero; the serve builds
    `q_r_chunk` and `q_l_chunk` from the joint head.
    """
    from dream_ik.decode import mat_to_rot6d

    poses = tool_poses(client, path)
    grip = np.linspace(0.0, 1.0, path.shape[1])
    per_key: dict[str, Any] = {}
    actions: dict[str, Any] = {"per_key": per_key}
    for i, (side, chunk_key) in enumerate(SIDES.items()):
        per_key[f"action.{side}_eef_xyz"] = poses[i, :, :3, 3].tolist()
        per_key[f"action.{side}_eef_rot6"] = mat_to_rot6d(poses[i, :, :3, :3]).tolist()
        per_key[f"action.{side}_grip"] = grip.tolist()
        per_key[f"action.{side}_arm"] = np.zeros((path.shape[1], 6)).tolist()
        actions[chunk_key] = np.column_stack([np.zeros((path.shape[1], 6)), grip]).tolist()
    return actions


def serve_actions(server: FakeDreamAPI, actions: Callable[[], dict[str, Any]]) -> None:
    """Makes the fake answer every request with its usual session handling and `actions()` as the action dict."""

    def handler(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        status, reply = server.default_reply(body)
        reply["actions"] = actions()
        return status, reply

    server.handler = handler


def joints_of(action: dict[str, Any]) -> npt.NDArray[np.float64]:
    """`[2, rows, 6]` the joint columns of the returned chunks, right then left."""
    return np.stack([np.asarray(action[key])[:, :6] for key in SIDES.values()])


def spy_on_solve(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Records the arguments of every `solve_chunk` the client calls; the solve itself still runs."""
    decode = pytest.importorskip("dream_ik.decode")
    calls: list[dict[str, Any]] = []
    real = decode.solve_chunk

    def spy(ik: Any, xyz: Any, rot6: Any, q_from: Any, speed_cap: float, serve: Any, q_before: Any = None) -> Any:
        calls.append(
            {"xyz": xyz, "rot6": rot6, "q_from": q_from, "speed_cap": speed_cap, "serve": serve, "q_before": q_before}
        )
        return real(ik, xyz, rot6, q_from, speed_cap, serve, q_before)

    monkeypatch.setattr(decode, "solve_chunk", spy)
    return calls


@pytest.mark.parametrize("arm", ["yam", "metal", "makers"])
def test_the_tool_poses_go_out_and_come_back_as_joints_on_the_same_path(
    server: FakeDreamAPI, connection: dict[str, Any], arm: str
) -> None:
    """FK on the way out, IK on the way back: the returned joints put each tool on the reply's poses."""
    with ik_client(connection, arm) as client:
        path = joint_path(client, arm, seed=1)
        reply = eef_reply(client, path)
        serve_actions(server, lambda: reply)
        observation = body_observation(path[:, 0])
        action = client.get_action(observation)
        reached = tool_poses(client, joints_of(action))
        target = tool_poses(client, path)

    assert not set(observation) & {f"state.{part}" for part in POSE_PARTS}, "the caller's dict is not changed"
    sent = server.requests[0].body["observation"]
    assert set(sent) == {f"state.{part}" for part in FAAS_CONFIG["state"]} | {"video.top", "prompt"}
    for side in SIDES:
        assert np.allclose(sent[f"state.{side}_eef_xyz"], reply["per_key"][f"action.{side}_eef_xyz"][0], atol=1e-9)
        assert np.allclose(sent[f"state.{side}_eef_rot6"], reply["per_key"][f"action.{side}_eef_rot6"][0], atol=1e-9)
        chunk = np.asarray(action[SIDES[side]])
        assert chunk.shape == (ROWS, 7)
        assert np.allclose(chunk[:, 6], reply["per_key"][f"action.{side}_grip"], atol=1e-9)
    position_miss = np.linalg.norm(reached[..., :3, 3] - target[..., :3, 3], axis=-1)
    relative = np.einsum("...ji,...jk->...ik", reached[..., :3, :3], target[..., :3, :3])
    angle_miss = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0))
    # dream's filter on the hand path (window 15) rounds this path off by up to 2 mm; the solve adds about 1 mm. The
    # YAM's longer reach and rest pull leave up to about 6 mm on this path (measured 6.1 mm); EEF models
    # (they read and answer each hand's end-effector pose) run head `joint` on a YAM, which solves nothing.
    assert position_miss.max() < (8e-3 if arm == "yam" else 3e-3)  # m
    assert angle_miss.max() < np.radians(1.0)


def test_the_solve_starts_where_the_arm_is_commanded_when_the_chunk_takes_over(
    monkeypatch: pytest.MonkeyPatch, server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """The first chunk solves from the measured joints; a later one from the previous chunk at the clock row after the
    joints last commanded, though the measured ones trail them, the rows from that row on of the poses filtered over
    the whole chunk; an arm off its plan from the measured joints. dream's filter is run before the cut, its seam
    blend is off, the speed caps are the arm's own."""
    calls = spy_on_solve(monkeypatch)
    with ik_client(connection) as client:
        path = joint_path(client, "metal", seed=2)
        reply = eef_reply(client, path)
        serve_actions(server, lambda: reply)
        first_action = client.get_action(body_observation(path[:, 0]))
        first = joints_of(first_action)
        server.latency_s = 0.3  # a round trip of about six rows at 20 Hz
        anchor = 12
        commanded = {key: np.asarray(first_action[key])[anchor - 1] for key in SIDES.values()}  # rows as written
        client.get_action(body_observation(first[:, anchor - 3]), commanded=commanded)
        clock = int(round(client._pacer._recent_rtt_s[-1] * HZ))
        off_plan = first[:, anchor] + np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0])  # the path spans 0.6 at most
        client.get_action(body_observation(off_plan))

    assert np.array_equal(calls[0]["q_from"], path[:, 0].astype(np.float32).astype(np.float64))
    assert 0 < clock < ROWS - anchor
    assert np.allclose(calls[1]["q_from"], first[:, anchor + clock], atol=1e-12)
    decode = pytest.importorskip("dream_ik.decode")
    poses = decode.head_poses(
        np.asarray(reply["per_key"]["action.right_eef_xyz"]), np.asarray(reply["per_key"]["action.right_eef_rot6"])
    )
    xyz = decode.smooth_poses(poses, 15, 3)[:, :3, 3]  # filtered over the whole chunk
    assert np.allclose(calls[1]["xyz"][0, : ROWS - clock], xyz[clock:], atol=1e-12)
    assert np.allclose(calls[1]["xyz"][0, ROWS - clock :], xyz[-1], atol=1e-12), "padded with a hold"
    assert np.array_equal(calls[2]["q_from"], off_plan.astype(np.float32).astype(np.float64))
    for call in calls:
        assert call["xyz"].shape == (2, ROWS, 3), "one compiled shape"
        assert call["speed_cap"] == np.inf
        assert call["serve"].seam_blend_ticks == 0
        assert call["serve"].smooth_window == 0


@pytest.mark.parametrize("arm", ["metal", "makers"])
def test_an_arm_measured_past_the_margin_of_a_limit_is_solved_back_inside_it(
    server: FakeDreamAPI, connection: dict[str, Any], arm: str
) -> None:
    """Recorded arms often rest just past the margin dream keeps inside each joint limit; the solve is clipped to the
    margined limits, as the sim eval's decode is, though dream's seam blend is off."""
    with ik_client(connection, arm) as client:
        solver = client._ik._solver
        path = joint_path(client, arm, seed=3)
        path[:, :, 1] = solver.hi[1] + 0.01  # past the 2 degree margin, inside the joint's limit
        serve_actions(server, lambda: eef_reply(client, path))
        action = client.get_action(body_observation(path[:, 0]))

    assert (joints_of(action)[:, :, 1] <= float(solver.hi[1]) + 1e-9).all()


def test_head_joint_keeps_the_models_joint_chunks_and_the_join_anchors_on_the_bodys_joints(
    monkeypatch: pytest.MonkeyPatch, server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """Head `joint` solves nothing: the serve's chunks are joined as any other, anchored on `state.<side>_arm`."""
    decode = pytest.importorskip("dream_ik.decode")
    monkeypatch.setattr(decode, "solve_chunk", None)
    line = np.repeat(np.linspace(0.0, 1.0, ROWS)[:, None], 7, axis=1)
    serve_actions(server, lambda: {"q_r_chunk": line.tolist(), "q_l_chunk": line.tolist(), "per_key": {}})
    with ik_client({**connection, "checkpoint": PLAIN}, head="joint", strict=False) as client:
        first = client.get_action(body_observation(np.zeros((2, 6))))
        server.latency_s = 0.3
        anchor = 12
        second = client.get_action(body_observation(np.repeat(line[anchor : anchor + 1, :6], 2, axis=0)))
        clock = int(round(client._pacer._recent_rtt_s[-1] * HZ))

    assert np.allclose(first["q_r_chunk"], smoothing.filter_from(line, 0), atol=1e-12)
    assert "state.right_eef_xyz" in server.requests[0].body["observation"], "the poses go out for either head"
    assert clock > 0
    for key in SIDES.values():
        # Rows before the clock row are the previous plan's from the anchor on: the join found the arm on it.
        assert np.allclose(np.asarray(second[key])[:clock, :6], line[anchor : anchor + clock, :6], atol=1e-12)


def test_traj_compiles_on_a_thread_and_the_client_solves_with_tick_until_it_is_ready(
    monkeypatch: pytest.MonkeyPatch, server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """`solve` `traj` compiles on a thread (17-20 s on one slow CPU core, long enough to miss a station's start-up
    deadlines): a chunk solved before it is ready uses `tick`, and every chunk after uses `traj`."""
    from dream_ik.solver import ChunkIK

    gate = threading.Event()
    warmup = ChunkIK.warmup

    def slow_warmup(self: Any, rows: int, serve: Any = None) -> None:
        if serve is not None and serve.solve == "traj":
            gate.wait(300)  # held until the test has solved a chunk
        warmup(self, rows, serve)

    monkeypatch.setattr(ChunkIK, "warmup", slow_warmup)
    calls = spy_on_solve(monkeypatch)
    with ik_client(connection, solve="traj") as client:
        path = joint_path(client, "metal", seed=2)
        serve_actions(server, lambda: eef_reply(client, path))
        assert not client._ik.traj_ready.is_set()
        client.get_action(body_observation(path[:, 0]))
        gate.set()
        assert client._ik.traj_ready.wait(300)
        client.get_action(body_observation(path[:, 0]))

    assert [call["serve"].solve for call in calls] == ["tick", "traj"]


def test_solve_traj_continues_the_previous_plans_motion_into_the_takeover_row(
    monkeypatch: pytest.MonkeyPatch, server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """With `solve` `traj` dream's trajectory solve runs, from the same takeover row as `tick`, with the previous
    plan's two rows before it as the motion the arm is on; the first chunk holds the measured joints as that motion.
    The seam join still runs, and the default stays `tick`."""
    calls = spy_on_solve(monkeypatch)
    with ik_client(connection, solve="traj") as client:
        # traj compiles on a thread; the client solves with tick until it is ready (#test_traj_compiles_on_a_thread)
        assert client._ik.traj_ready.wait(300), "the traj solve compiled"
        path = joint_path(client, "metal", seed=2)
        reply = eef_reply(client, path)
        serve_actions(server, lambda: reply)
        first_action = client.get_action(body_observation(path[:, 0]))
        first = joints_of(first_action)
        server.latency_s = 0.3
        anchor = 12
        commanded = {key: np.asarray(first_action[key])[anchor - 1] for key in SIDES.values()}
        client.get_action(body_observation(first[:, anchor - 3]), commanded=commanded)
        clock = int(round(client._pacer._recent_rtt_s[-1] * HZ))

    assert all(call["serve"].solve == "traj" for call in calls)
    assert calls[0]["q_before"].shape == (2, 2, 6), "two rows per arm, newest first"
    assert np.array_equal(calls[0]["q_before"], np.stack([calls[0]["q_from"]] * 2, axis=1)), "no plan: still"
    assert np.allclose(calls[1]["q_from"], first[:, anchor + clock], atol=1e-12)
    assert np.allclose(calls[1]["q_before"][:, 0], first[:, anchor + clock - 1], atol=1e-12)
    assert np.allclose(calls[1]["q_before"][:, 1], first[:, anchor + clock - 2], atol=1e-12)
    for call in calls:
        assert call["serve"].seam_blend_ticks == 0, "the client's seam join joins the chunks, as with tick"


def test_solve_traj_is_the_default(
    monkeypatch: pytest.MonkeyPatch, server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """Without `solve` the chunk solve is dream's trajectory solve, as the robots stations run it: once it has
    compiled, a chunk is solved with `traj`."""
    calls = spy_on_solve(monkeypatch)
    with ik_client(connection, solve=None) as client:
        assert client._ik.traj_ready.wait(300), "the traj solve compiled"
        path = joint_path(client, "metal", seed=3)
        serve_actions(server, lambda: eef_reply(client, path))
        client.get_action(body_observation(path[:, 0]))

    assert calls[0]["serve"].solve == "traj"


def test_traj_compiles_only_for_the_eef_head_on_a_thread_the_program_waits_for(connection: dict[str, Any]) -> None:
    """Head `joint` never solves a pose, so it compiles no traj and is ready at once; head `eef` compiles traj on a
    thread that is not a daemon, so a program that ends mid-compile waits for it instead of aborting in XLA."""
    with ik_client(connection, head="joint", solve="traj") as client:
        assert client._ik.traj_ready.is_set()
        assert not any(t.name == "dream-client-ik-traj-warmup" for t in threading.enumerate())
    with ik_client(connection, solve="traj") as client:
        warming = [t for t in threading.enumerate() if t.name == "dream-client-ik-traj-warmup"]
        assert client._ik.traj_ready.is_set() or (warming and not warming[0].daemon)
        assert client._ik.traj_ready.wait(300)


@pytest.mark.parametrize(
    ("ik", "message"),
    [
        ("metal", "must be a dict"),
        ({"arm": "metal"}, "must be a dict with the keys"),
        ({"arm": "metal", "head": "eef", "hz": 30}, "must be a dict with the keys"),
        ({"arm": "rover", "head": "eef"}, "names the arm 'rover'"),
        ({"arm": "my_arm.urdf", "head": "eef"}, "or the path of an arm file"),
        ({"arm": "metal", "head": "both"}, "names the head 'both'"),
        ({"arm": "metal", "head": "eef", "solve": "fast"}, "names the solve 'fast'"),
    ],
    ids=[
        "not_a_dict",
        "no_head",
        "extra_key",
        "unknown_arm",
        "a_urdf_not_an_arm_file",
        "unknown_head",
        "unknown_solve",
    ],
)
def test_a_bad_ik_key_is_a_config_error_at_init(
    connection: dict[str, Any], platform: FakePlatform, ik: Any, message: str
) -> None:
    """The key is checked before the platform is asked and dream's IK is imported, so this holds on every Python the
    client takes, with no platform in reach."""
    platform.close()
    with pytest.raises(DreamConfigError, match=message):
        dream._open_policy({**connection, "ik": ik})


@pytest.mark.parametrize("checkpoint", [BASE, YAM_FINE_TUNE])
@pytest.mark.parametrize("arm", ["metal", "so101_faas"])
def test_head_joint_on_a_model_trained_on_yam_arms_is_a_config_error(
    monkeypatch: pytest.MonkeyPatch, connection: dict[str, Any], arm: str, checkpoint: str
) -> None:
    monkeypatch.setitem(sys.modules, "dream_ik.solver", None)
    keys = {**connection, "checkpoint": checkpoint, "ik": {"arm": arm, "head": "joint"}}
    with pytest.raises(DreamConfigError, match=f"head 'joint', but '{checkpoint}' was trained on YAM arms"):
        dream._open_policy(keys)


def test_a_checkpoint_the_platform_does_not_list_is_a_config_error_at_init(connection: dict[str, Any]) -> None:
    keys = {**connection, "checkpoint": "dream-1-ft:ffffffff-gone", "ik": {"arm": "metal", "head": "eef"}}
    with pytest.raises(
        DreamConfigError, match=r"'dream-1-ft:ffffffff-gone' is not a model your key may run.*metalarm"
    ):
        dream._open_policy(keys)


STATION_CAMERAS = ("scene", "wrist_l", "wrist_r")


def station_body(cameras: tuple[str, ...] = STATION_CAMERAS) -> dict[str, Any]:
    """A Metal station's body: the FaaS body with `cameras`."""
    return {**copy.deepcopy(FAAS_CONFIG), "video": {role: f"observation.images.{role}" for role in cameras}}


def station_observation(
    joints: npt.NDArray[np.float64], gripper_mm: float, cameras: tuple[str, ...] = STATION_CAMERAS
) -> dict[str, Any]:
    """What a Metal station sends at `joints` (`[2, 6]`, right then left)."""
    observation = {key: value for key, value in body_observation(joints).items() if key != "video.top"}
    for side in SIDES:
        observation[f"state.{side}_gripper"] = np.array([gripper_mm], dtype=np.float32)
        observation[f"state.{side}_grip"] = np.array([gripper_mm / 100.0], dtype=np.float32)
    for role in cameras:
        observation[f"video.{role}"] = [FRAME_B64]
    return observation


## The YAM's own body for a base model that reads each hand's pose beside the joints (the `yam_eef_joint` modality of
## dream-1-beta): YAM's joint parts, then the six EEF parts, read and answered alike.
YAM_EEF_PARTS = {
    "right_arm": [0, 6],
    "right_gripper": [6, 7],
    "left_arm": [7, 13],
    "left_gripper": [13, 14],
    "right_eef_xyz": [14, 17],
    "right_eef_rot6": [17, 23],
    "right_grip": [23, 24],
    "left_eef_xyz": [24, 27],
    "left_eef_rot6": [27, 33],
    "left_grip": [33, 34],
}
YAM_EEF_CONFIG: dict[str, Any] = {
    "state": YAM_EEF_PARTS,
    "action": YAM_EEF_PARTS,
    "video": {role: f"observation.images.{role}" for role in YAM_CAMERAS},
}


def test_a_flagship_on_a_yam_reads_its_own_joints_its_hand_poses_and_its_grip_and_runs_its_joint_answer(
    server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """On a YAM the joints and the gripper are the arm's own, the client adds the poses and the grip, and head
    `joint` returns the model's joint chunks: this arm's joints."""
    with ik_client({**connection, "checkpoint": BASE, "embodiment": YAM_EEF_CONFIG}, "yam", "joint") as client:
        path = joint_path(client, "yam", seed=3)
        reply = eef_reply(client, path)
        for i, (side, chunk_key) in enumerate(SIDES.items()):
            reply["per_key"][f"action.{side}_arm"] = path[i].tolist()
            reply["per_key"][f"action.{side}_gripper"] = reply["per_key"][f"action.{side}_grip"]
            reply[chunk_key] = np.column_stack([path[i], reply["per_key"][f"action.{side}_grip"]]).tolist()
        serve_actions(server, lambda: reply)
        observation: dict[str, Any] = {"prompt": "put apple into the bowl"}
        for i, side in enumerate(SIDES):
            observation[f"state.{side}_arm"] = path[i, 0].astype(np.float32)
            observation[f"state.{side}_gripper"] = np.array([0.4], dtype=np.float32)
        observation.update({f"video.{role}": [FRAME_B64] for role in YAM_CAMERAS})
        action = client.get_action(observation)

    sent = server.requests[0].body["observation"]
    assert set(sent) == {f"state.{part}" for part in YAM_EEF_PARTS} | {f"video.{r}" for r in YAM_CAMERAS} | {"prompt"}
    for i, side in enumerate(SIDES):
        assert np.allclose(sent[f"state.{side}_arm"], path[i, 0], atol=1e-6), "the arm's own joints, not zeros"
        assert np.allclose(sent[f"state.{side}_gripper"], [0.4]) and np.allclose(sent[f"state.{side}_grip"], [0.4])
        assert np.allclose(sent[f"state.{side}_eef_xyz"], reply["per_key"][f"action.{side}_eef_xyz"][0], atol=1e-6)
        assert np.allclose(sent[f"state.{side}_eef_rot6"], reply["per_key"][f"action.{side}_eef_rot6"][0], atol=1e-6)
        assert np.allclose(action[SIDES[side]], smoothing.filter_from(np.asarray(reply[SIDES[side]]), 0), atol=1e-9)
    assert "state.right_grip" not in observation, "the caller's dict is not changed"


## The YAM's own body, which names no pose: what a YAM station sends for every model (`robots.adapter.YAM_EMBODIMENT`).
YAM_PARTS = {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]}
YAM_CONFIG: dict[str, Any] = {"state": YAM_PARTS, "action": YAM_PARTS, "video": YAM_EEF_CONFIG["video"]}


def test_on_a_yam_the_yams_own_body_passes_strict_mode_with_the_poses_added_and_a_grip_for_the_gripper(
    server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """A YAM station declares the YAM's own body for every model: the poses and grips the client adds are read, and
    a base model's answer, which carries `action.<side>_grip` and no `action.<side>_gripper`, is not refused."""
    with ik_client(
        {**connection, "checkpoint": BASE, "embodiment": YAM_CONFIG, "strict": True}, "yam", "joint"
    ) as client:
        path = joint_path(client, "yam", seed=5)
        reply = eef_reply(client, path)
        for i, (side, chunk_key) in enumerate(SIDES.items()):
            reply["per_key"][f"action.{side}_arm"] = path[i].tolist()
            reply[chunk_key] = np.column_stack([path[i], reply["per_key"][f"action.{side}_grip"]]).tolist()
        serve_actions(server, lambda: reply)
        observation: dict[str, Any] = {"prompt": "put apple into the bowl"}
        for i, side in enumerate(SIDES):
            observation[f"state.{side}_arm"] = path[i, 0].astype(np.float32)
            observation[f"state.{side}_gripper"] = np.array([0.4], dtype=np.float32)
        observation.update({f"video.{role}": [FRAME_B64] for role in YAM_CAMERAS})
        action = client.get_action(observation)

    sent = server.requests[0].body["observation"]
    assert set(sent) == {f"state.{part}" for part in YAM_EEF_PARTS} | {f"video.{r}" for r in YAM_CAMERAS} | {"prompt"}
    for side in SIDES:
        assert f"action.{side}_gripper" not in reply["per_key"], "the base model answers its grip only"
        assert np.allclose(action[SIDES[side]], smoothing.filter_from(np.asarray(reply[SIDES[side]]), 0), atol=1e-9)


## What a YAM station declares on the EEF head (`robots.adapter.YAM_POSE_EMBODIMENT`): the YAM's parts and the poses
## read, and the EEF models' action head answered (each arm's joints, tool pose and grip; no `<side>_gripper`).
YAM_POSE_CONFIG: dict[str, Any] = {
    "state": YAM_EEF_PARTS,
    "action": {
        "right_arm": [0, 6],
        "left_arm": [6, 12],
        "right_eef_xyz": [12, 15],
        "right_eef_rot6": [15, 21],
        "right_grip": [21, 22],
        "left_eef_xyz": [22, 25],
        "left_eef_rot6": [25, 31],
        "left_grip": [31, 32],
    },
    "video": YAM_EEF_CONFIG["video"],
}


def yam_observation(path: npt.NDArray[np.float64]) -> dict[str, Any]:
    """A YAM station's observation at the first row of `path`: its own joints and gripper, and its three cameras."""
    observation: dict[str, Any] = {"prompt": "put apple into the bowl"}
    for i, side in enumerate(SIDES):
        observation[f"state.{side}_arm"] = path[i, 0].astype(np.float32)
        observation[f"state.{side}_gripper"] = np.array([0.4], dtype=np.float32)
    observation.update({f"video.{role}": [FRAME_B64] for role in YAM_CAMERAS})
    return observation


def test_on_a_yam_the_eef_head_solves_a_flagships_tool_poses_in_strict_mode(
    server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """A YAM station's EEF head: strict on the pose body, the base model's answer (no `<side>_gripper`) passes, and the
    chunks are its tool poses solved to joints, not its joint head (held at zero here)."""
    with ik_client({**connection, "checkpoint": BASE, "embodiment": YAM_POSE_CONFIG, "strict": True}, "yam") as client:
        path = joint_path(client, "yam", seed=9)
        reply = eef_reply(client, path)
        serve_actions(server, lambda: reply)
        action = client.get_action(yam_observation(path))
        reached = tool_poses(client, joints_of(action))
        target = tool_poses(client, path)

    assert np.abs(joints_of(action)).max() > 0.1, "solved from the tool poses, not the zero joint head"
    assert np.linalg.norm(reached[..., :3, 3] - target[..., :3, 3], axis=-1).max() < 0.01


def test_on_a_yam_the_eef_head_runs_the_joint_answer_of_a_model_with_no_eef_head_and_says_so_once(
    server: FakeDreamAPI, connection: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """A model with no EEF head answers only the YAM's joints and grippers: on the EEF head the client runs those
    joint chunks as they come, as head `joint` would, and warns once per session."""
    caplog.set_level(logging.WARNING, logger="dream.client_ik")
    with ik_client({**connection, "checkpoint": BASE, "embodiment": YAM_POSE_CONFIG, "strict": True}, "yam") as client:
        path = joint_path(client, "yam", seed=11)
        chunks = {key: np.column_stack([path[i], np.full(ROWS, 0.5)]).tolist() for i, key in enumerate(SIDES.values())}
        per_key = {}
        for i, side in enumerate(SIDES):
            per_key[f"action.{side}_arm"] = path[i].tolist()
            per_key[f"action.{side}_gripper"] = [[0.5]] * ROWS
        serve_actions(server, lambda: {"per_key": per_key, **chunks})
        first = client.get_action(yam_observation(path))
        client.get_action(yam_observation(path))

    for side, key in SIDES.items():
        assert np.allclose(first[key], smoothing.filter_from(np.asarray(chunks[key]), 0), atol=1e-9), side
    warned = [record.getMessage() for record in caplog.records if "has no EEF head" in record.getMessage()]
    assert warned == [f"{BASE} has no EEF head (it answers no tool poses): running its joint head on this YAM instead"]


@pytest.mark.parametrize("checkpoint", [BASE, YAM_FINE_TUNE])
def test_a_model_trained_on_yam_arms_reads_zero_joints_and_the_metal_arms_own_hand_and_gripper(
    server: FakeDreamAPI, connection: dict[str, Any], checkpoint: str
) -> None:
    """The joint slots go out as zeros, the tool pose and the gripper are the arm's own."""
    with ik_client({**connection, "checkpoint": checkpoint, "embodiment": station_body()}) as client:
        path = joint_path(client, "metal", seed=7)
        reply = eef_reply(client, path)
        serve_actions(server, lambda: reply)
        observation = station_observation(path[:, 0], gripper_mm=40.0)
        client.get_action(observation)

    sent = server.requests[0].body["observation"]
    assert sorted(key for key in sent if key.startswith("video.")) == [
        "video.overhead",
        "video.wrist_l",
        "video.wrist_r",
    ]
    for side in SIDES:
        assert sent[f"state.{side}_arm"] == [0.0] * 6
        assert np.allclose(sent[f"state.{side}_gripper"], [0.4]) and np.allclose(sent[f"state.{side}_grip"], [0.4])
        assert np.allclose(sent[f"state.{side}_eef_xyz"], reply["per_key"][f"action.{side}_eef_xyz"][0], atol=1e-6)
    assert np.allclose(observation["state.right_gripper"], [40.0]), "the caller's dict is not changed"


@pytest.mark.parametrize("mode", client_ik.PROPRIO_TEST_MODES)
def test_the_proprio_test_switch_replaces_only_what_its_mode_names(  # TEMPORARY
    monkeypatch: pytest.MonkeyPatch, server: FakeDreamAPI, connection: dict[str, Any], mode: str
) -> None:
    monkeypatch.setenv(client_ik.PROPRIO_TEST_ENV, mode)
    with ik_client({**connection, "checkpoint": BASE, "embodiment": station_body()}) as client:
        path = joint_path(client, "metal", seed=9)
        reply = eef_reply(client, path)
        serve_actions(server, lambda: reply)
        client.get_action(station_observation(path[:, 0], gripper_mm=40.0))
    sent = server.requests[0].body["observation"]
    for i, side in enumerate(SIDES):
        joints = {
            "raw": path[i, 0],
            "zero": np.zeros(6),
            "avg": client_ik.PROPRIO_AVG[f"state.{side}_arm"],
            "avg_hand": np.zeros(6),
            "avg_all": client_ik.PROPRIO_AVG[f"state.{side}_arm"],
        }[mode]
        assert np.allclose(sent[f"state.{side}_arm"], joints, atol=1e-6)
        hand = mode in ("avg_hand", "avg_all")
        xyz = client_ik.PROPRIO_AVG[f"state.{side}_eef_xyz"] if hand else reply["per_key"][f"action.{side}_eef_xyz"][0]
        assert np.allclose(sent[f"state.{side}_eef_xyz"], xyz, atol=1e-6)
        assert np.allclose(sent[f"state.{side}_grip"], [0.4]), "the gripper stays real"


def test_the_proprio_test_switch_is_refused_where_it_cannot_apply(  # TEMPORARY
    monkeypatch: pytest.MonkeyPatch, connection: dict[str, Any]
) -> None:
    monkeypatch.setenv(client_ik.PROPRIO_TEST_ENV, "wrong")
    with pytest.raises(DreamConfigError, match="use one of"):
        ik_client({**connection, "checkpoint": BASE, "embodiment": station_body()})
    monkeypatch.setenv(client_ik.PROPRIO_TEST_ENV, "zero")
    with pytest.raises(DreamConfigError, match="trained on YAM arms"):
        ik_client({**connection, "checkpoint": FINE_TUNE})
    with pytest.raises(DreamConfigError, match="YAM station"):
        dream._open_policy({**connection, "checkpoint": BASE})


def test_a_model_trained_on_yam_arms_never_drives_the_arm_with_its_joint_answer(
    monkeypatch: pytest.MonkeyPatch, server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    calls = spy_on_solve(monkeypatch)
    with ik_client({**connection, "checkpoint": BASE, "embodiment": station_body()}) as client:
        path = joint_path(client, "metal", seed=8)
        yam = path + 1.0
        reply = eef_reply(client, path)
        for i, (side, chunk_key) in enumerate(SIDES.items()):
            reply["per_key"][f"action.{side}_arm"] = yam[i].tolist()
            reply[chunk_key] = np.column_stack([yam[i], reply["per_key"][f"action.{side}_grip"]]).tolist()
        reply["q_r"], reply["q_l"] = reply["q_r_chunk"][0], reply["q_l_chunk"][0]
        serve_actions(server, lambda: reply)
        first = client.get_action(station_observation(path[:, 0], gripper_mm=40.0))
        commanded = {key: np.asarray(first[key])[5] for key in SIDES.values()}
        second = client.get_action(station_observation(path[:, 5], gripper_mm=40.0), commanded=commanded)
        reached = tool_poses(client, joints_of(first))
        target = tool_poses(client, path)

    for action in (first, second):
        assert "q_r" not in action and "q_l" not in action
        assert np.linalg.norm(joints_of(action) - yam, axis=-1).min() > 1.0
    assert np.linalg.norm(reached[..., :3, 3] - target[..., :3, 3], axis=-1).max() < 5e-3
    for call in calls:
        for i in range(2):
            assert np.linalg.norm(yam[i] - call["q_from"][i], axis=-1).min() > 1.0


@pytest.mark.parametrize("checkpoint", [FINE_TUNE, PLAIN])
def test_a_model_trained_on_the_robot_reads_its_own_joints_gripper_and_cameras(
    server: FakeDreamAPI, connection: dict[str, Any], checkpoint: str
) -> None:
    cameras = ("head", "wrist_l", "wrist_r")
    with ik_client({**connection, "checkpoint": checkpoint, "embodiment": station_body(cameras)}) as client:
        path = joint_path(client, "metal", seed=8)
        serve_actions(server, lambda: eef_reply(client, path))
        client.get_action(station_observation(path[:, 0], gripper_mm=40.0, cameras=cameras))

    sent = server.requests[0].body["observation"]
    assert sorted(key for key in sent if key.startswith("video.")) == ["video.head", "video.wrist_l", "video.wrist_r"]
    for i, side in enumerate(SIDES):
        assert np.allclose(sent[f"state.{side}_arm"], path[i, 0], atol=1e-6)
        assert np.allclose(sent[f"state.{side}_gripper"], [40.0]) and np.allclose(sent[f"state.{side}_grip"], [0.4])


def test_a_fine_tune_gets_the_bodys_cameras_in_the_roles_it_was_trained_with(
    server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    with ik_client({**connection, "checkpoint": FINE_TUNE, "embodiment": station_body()}) as client:
        path = joint_path(client, "metal", seed=9)
        serve_actions(server, lambda: eef_reply(client, path))
        client.get_action(station_observation(path[:, 0], gripper_mm=40.0))

    sent = server.requests[0].body["observation"]
    assert sorted(key for key in sent if key.startswith("video.")) == ["video.head", "video.wrist_l", "video.wrist_r"]
    assert np.allclose(sent["state.right_arm"], path[0, 0], atol=1e-6)
    assert np.allclose(sent["state.right_gripper"], [40.0])


def test_cameras_that_cannot_fill_a_fine_tunes_roles_are_a_config_error_at_init(connection: dict[str, Any]) -> None:
    pytest.importorskip("dream_ik")
    cameras = ("scene", "left_wrist", "right_wrist")
    keys = {**connection, "checkpoint": FINE_TUNE, "embodiment": station_body(cameras)}

    with pytest.raises(
        DreamConfigError, match=re.escape(f"cannot be sent the way '{FINE_TUNE}' reads them")
    ) as raised:
        dream._open_policy({**keys, "ik": {"arm": "metal", "head": "eef"}})

    assert "missing ['head', 'wrist_l', 'wrist_r'], extra ['left_wrist', 'right_wrist', 'scene']" in str(raised.value)


@pytest.mark.parametrize(
    ("cameras", "missing", "extra"),
    [
        (("scene", "wrist_l"), ["wrist_r"], []),
        (("wrist_l", "wrist_r"), ["overhead"], []),
        (("scene", "top", "wrist_l", "wrist_r"), ["overhead"], ["scene", "top"]),
        (
            ("scene", "left_wrist", "right_wrist"),
            ["overhead", "wrist_l", "wrist_r"],
            ["left_wrist", "right_wrist", "scene"],
        ),
    ],
    ids=["no_right_wrist", "no_scene", "two_scenes", "wrists_misnamed"],
)
def test_cameras_that_cannot_fill_a_yam_models_slots_are_a_config_error_at_init(
    connection: dict[str, Any], cameras: tuple[str, ...], missing: list[str], extra: list[str]
) -> None:
    pytest.importorskip("dream_ik")
    keys = {
        **connection,
        "checkpoint": BASE,
        "embodiment": station_body(cameras),
        "ik": {"arm": "metal", "head": "eef"},
    }
    with pytest.raises(DreamConfigError, match=re.escape(f"cannot be sent the way '{BASE}' reads them")) as raised:
        dream._open_policy(keys)
    assert f"missing {missing}, extra {extra}" in str(raised.value)


def test_a_request_whose_cameras_are_not_the_bodys_is_refused_before_anything_is_sent(
    server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    with ik_client({**connection, "checkpoint": BASE, "embodiment": station_body()}, strict=False) as client:
        observation = station_observation(np.zeros((2, 6)), gripper_mm=40.0)
        with pytest.raises(DreamObservationError, match=r"missing \['wrist_r'\], extra \[\]"):
            client.get_action({key: value for key, value in observation.items() if key != "video.wrist_r"})
        with pytest.raises(DreamObservationError, match=r"missing \[\], extra \['top'\]"):
            client.get_action({**observation, "video.top": [FRAME_B64]})
    assert server.requests == []


def test_ik_without_dreams_ik_installed_is_a_config_error(
    monkeypatch: pytest.MonkeyPatch, connection: dict[str, Any]
) -> None:
    """An install without dream's IK (Python 3.10) refuses the key at init, saying what it needs and where the library
    comes from: `dreamlabs` from PyPI, a name Dream Labs holds."""
    monkeypatch.setitem(sys.modules, "dream_ik.solver", None)
    with pytest.raises(
        DreamConfigError,
        match=r"needs dream's IK, which the library ships for Python 3\.11 or newer \(.+\); in a Python 3\.11 or "
        r"newer environment, install the library: pip install --upgrade dreamlabs$",
    ):
        dream._open_policy({**connection, "ik": {"arm": "metal", "head": "eef"}})


def test_keep_out_keeps_that_arm_off_its_obstacles(server: FakeDreamAPI, connection: dict[str, Any]) -> None:
    with ik_client(connection) as plain:
        path = joint_path(plain, "metal", seed=1)
        hand = tool_poses(plain, path)[0, ROWS // 2, :3, 3]
    box = {"kind": "box", "center": hand.tolist(), "size": [0.04, 0.04, 0.04]}
    with ik_client(connection, keep_out={"right": [box]}) as client:
        reply = eef_reply(client, path)
        serve_actions(server, lambda: reply)
        reached = tool_poses(client, joints_of(client.get_action(body_observation(path[:, 0]))))

    to_box = np.linalg.norm(np.maximum(np.abs(reached[0, :, :3, 3] - hand) - 0.02, 0.0), axis=-1)
    assert to_box.min() > 0.04


def test_a_metal_arm_is_kept_off_a_pole_only_when_the_station_lists_it(
    server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    sf0 = {
        "right": [{"kind": "capsule", "center": [0.032, 0.318, 0.27], "radius": 0.02, "height": 0.6}],
        "left": [{"kind": "capsule", "center": [0.012, -0.323, 0.27], "radius": 0.04, "height": 0.6}],
    }
    turn = np.radians(np.linspace(50.0, 110.0, ROWS))
    path = np.zeros((2, ROWS, 6))
    path[..., 1], path[..., 2] = -1.15, 0.6
    path[..., 0] = np.stack([turn, -turn])
    axes = np.array([sf0[side][0]["center"][:2] for side in SIDES])
    radii = np.array([[sf0[side][0]["radius"]] for side in SIDES])

    def hand_to_pole(keep_out: Any) -> npt.NDArray[np.float64]:
        with ik_client(connection, keep_out=keep_out) as client:
            serve_actions(server, lambda: eef_reply(client, path))
            reached = tool_poses(client, joints_of(client.get_action(body_observation(path[:, 0]))))
        return np.linalg.norm(reached[..., :2, 3] - axes[:, None], axis=-1) - radii

    assert hand_to_pole(None).min() < 0.0
    assert hand_to_pole(sf0).min() > 0.03


@pytest.mark.parametrize(
    ("keep_out", "message"),
    [
        ([], r"it must be a dict with some of the keys \['left', 'right'\]"),
        ({"middle": []}, r"it must be a dict with some of the keys"),
        ({"right": {}, "left": []}, r"'keep_out.right' \{\}: it must be a list"),
        ({"right": [{"kind": "cone"}], "left": []}, r"its 'kind' must be one of \['box', 'capsule'\]"),
        (
            {"right": [{"kind": "box", "center": [0, 0, 0], "size": [0.1, 0.1, 0.1], "yaw": 0.3}], "left": []},
            r"a bad obstacle in 'keep_out.right'",
        ),
        ({"right": [{"kind": "capsule", "center": [0, 0, 0]}], "left": []}, r"a bad obstacle in 'keep_out.right'"),
    ],
    ids=["not_a_dict", "unknown_side", "side_not_a_list", "unknown_kind", "extra_field", "field_missing"],
)
def test_a_bad_keep_out_is_a_config_error_at_init(connection: dict[str, Any], keep_out: Any, message: str) -> None:
    pytest.importorskip("dream_ik")
    with pytest.raises(DreamConfigError, match=message):
        dream._open_policy({**connection, "ik": {"arm": "metal", "head": "eef", "keep_out": keep_out}})


def test_clamp_to_reach_moves_a_row_outside_straight_to_the_radius_of_its_nearest_point() -> None:
    """A row inside a disk, or exactly on a point, is untouched; a row outside lands exactly the radius from its
    nearest point, on the line to it; with one point every row outside lands on that point's circle."""
    radius = client_ik.REACH_RADIUS_M
    points = np.array([[0.3, 0.0], [0.3, 0.1]])
    xy = np.array([[0.31, 0.02], [0.3, 0.1], [0.5, 0.0], [0.3, 0.3], [0.1, -0.1], [0.3 + radius, 0.0]])
    out = client_ik.clamp_to_reach(xy, points)
    assert np.array_equal(out[[0, 1, 5]], xy[[0, 1, 5]]), "inside, on a point, and exactly on the edge stay"
    nearest = np.array([0, 1, 0])
    for row, point in zip([2, 3, 4], points[nearest], strict=True):
        assert np.isclose(np.linalg.norm(out[row] - point), radius, atol=1e-15)
        towards, moved = xy[row] - point, out[row] - point
        assert np.isclose(towards[0] * moved[1] - towards[1] * moved[0], 0.0, atol=1e-15), "on the line to it"
        assert towards @ moved > 0, "on the row's side of the point"
    one = client_ik.clamp_to_reach(xy, points[:1])
    assert np.allclose(np.linalg.norm(one - points[0], axis=1)[2:5], radius, atol=1e-15)
    assert np.array_equal(one[[0, 5]], xy[[0, 5]])
    assert np.array_equal(client_ik.clamp_to_reach(xy, points), out), "the input is not changed"


def test_the_reach_fence_holds_from_the_first_row_whose_hand_heads_out_past_the_edge() -> None:
    """A hand inside, or one moving back in from outside, is never held; one heading out, however slowly, is held at
    the last row within the radius plus the slack, and one starting outside and heading farther out at its start."""
    edge = client_ik.REACH_RADIUS_M + client_ik.REACH_SLACK_M
    point = np.array([[0.3, 0.0]])
    out = np.stack([0.3 + 0.01 * np.arange(11), np.zeros(11)], axis=1)  # the start, then 10 rows 1 cm apart
    assert client_ik.reach_fence(out[:5], point) is None, "inside the whole way"
    row = client_ik.reach_fence(out, point)
    assert row == 5 and np.linalg.norm(out[row] - point) <= edge < np.linalg.norm(out[row + 1] - point)
    assert client_ik.reach_fence(out[::-1], point) is None, "a hand outside moving back in"
    assert client_ik.reach_fence(out[6:], point) == 0, "a hand outside heading farther out holds at once"
    drift = np.stack([0.3 + 0.0005 * np.arange(200), np.zeros(200)], axis=1)  # 0.5 mm a row, out past the edge
    row = client_ik.reach_fence(drift, point)
    assert row is not None and np.linalg.norm(drift[row] - point) <= edge < np.linalg.norm(drift[row + 1] - point)
    wobble = np.array([[0.5, 0.0], [0.5 + 1e-12, 0.0], [0.5009, 0.0], [0.45, 0.0], [0.4505, 0.0]])
    assert client_ik.reach_fence(wobble, point) is None, "outside, a wobble under 1 mm on the way in is not held"


def test_the_reach_fence_stops_a_path_across_a_hole_in_the_area_on_its_near_side() -> None:
    """Painted around a mount (a 10 cm hole), a hand driven straight through it at 1 cm a row stops before the mount,
    within the slack of the area."""
    grid = np.arange(-0.24, 0.2401, 0.02)
    cells = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
    points = cells[(np.linalg.norm(cells, axis=1) >= 0.1) & (np.linalg.norm(cells, axis=1) <= 0.24)] + [0.4, 0.0]
    hands = np.stack([np.linspace(0.2, 0.6, 41), np.zeros(41)], axis=1)
    row = client_ik.reach_fence(hands, points)
    assert row is not None and hands[row, 0] < 0.4, "held on the near side of the mount"
    nearest = np.linalg.norm(hands[: row + 1, None] - points[None], axis=-1).min(axis=1)
    assert nearest.max() <= client_ik.REACH_RADIUS_M + client_ik.REACH_SLACK_M


def test_the_reach_fence_holds_a_solved_hand_that_heads_out_whatever_its_targets(
    monkeypatch: pytest.MonkeyPatch, server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """With the targets left where they are (no clamp), the right arm's solve follows them out of its one-point area;
    the fence holds it from the first row past the edge, so every solved hand stays within the edge. The left arm, not
    listed, is solved as without `reach`."""
    with ik_client(connection) as plain:
        path = joint_path(plain, "metal", seed=1)
        reply = eef_reply(plain, path)
        serve_actions(server, lambda: reply)
        free = plain.get_action(body_observation(path[:, 0]))
        start = tool_poses(plain, path[:, :1])[0, 0, :2, 3]
    monkeypatch.setattr(client_ik, "clamp_to_reach", lambda xy, points: xy)
    with ik_client(connection, reach={"right": [start.tolist()]}) as client:
        kept = client.get_action(body_observation(path[:, 0]))
        reached = tool_poses(client, joints_of(kept))

    edge = client_ik.REACH_RADIUS_M + client_ik.REACH_SLACK_M
    unfenced = np.linalg.norm(tool_poses(client, joints_of(free))[0, :, :2, 3] - start, axis=1)
    assert unfenced.max() > edge + 0.01, "the path must leave the area for this to test anything"
    hand = np.linalg.norm(reached[0, :, :2, 3] - start, axis=1)
    assert hand.max() <= edge + 1e-6, hand.max()
    right = np.asarray(kept["q_r_chunk"])[:, :-1]
    assert np.allclose(right[-10:], right[-1], atol=1e-9), "held to the end of the chunk"
    assert np.array_equal(np.asarray(kept["q_l_chunk"]), np.asarray(free["q_l_chunk"])), "the left arm: unchanged"


def test_a_region_keeps_the_arm_behind_a_wall_the_model_drives_it_through(
    server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """Hand poses driven 10 cm through a wall face in front of the arm: every row the client returns keeps the arm's
    centreline behind it, within dream's slack, though the solve alone takes the hand through; the other arm (no
    region) is untouched."""
    from dream_ik.solver import ChunkIK
    from dream_ik.specs import EMBODIMENTS, SERVE, Region

    emb = EMBODIMENTS["metal"]
    with ik_client(connection) as plain:
        path = joint_path(plain, "metal", seed=1)
        reply = eef_reply(plain, path)
        probe = ChunkIK(
            "metal", SERVE.tool, HZ, {s: emb.obstacles[s] + (Region(hi=(0.0, None, None)),) for s in emb.sides}
        )
        wall = float(probe.region_depth(path).max()) + 0.02  # 2 cm past the path's farthest point forward
        for side in SIDES:
            xyz = np.asarray(reply["per_key"][f"action.{side}_eef_xyz"])
            xyz[:, 0] = np.linspace(xyz[0, 0], wall + 0.10, len(xyz))  # out to 10 cm past the wall
            reply["per_key"][f"action.{side}_eef_xyz"] = xyz.tolist()
        serve_actions(server, lambda: reply)
        through = joints_of(plain.get_action(body_observation(path[:, 0])))
    region = {"right": {"hi": [wall, None, None]}}
    with ik_client(connection, region=region) as client:
        kept = joints_of(client.get_action(body_observation(path[:, 0])))
        solver = client._ik._solver
        depth = solver.region_depth(kept[0], "right")
        free = solver.region_depth(through[0], "right")

    assert free.max() > 0.03, "without the region the hand goes through the wall"
    assert depth.max() <= client_ik.SERVE_REGION_SLACK_M + 1e-5
    # the left arm has no region; it shares the right's vmapped solve, whose collision path the region turns on
    assert np.abs(kept[1] - through[1]).max() < 1e-3


@pytest.mark.parametrize(
    ("region", "message"),
    [
        ({"middle": {"lo": [0, 0, 0]}}, r"it must be a dict with some of the keys"),
        ({"right": {"low": [0, 0, 0]}}, r"'region.right' .*: it must be"),
        ({"right": {"lo": [0, 0]}}, r"'region.right' .*: it must be"),
        ({"right": {"lo": ["0", 0, 0]}}, r"'region.right' .*: it must be"),
        ({"right": {"lo": [0.3, None, None], "hi": [0.1, None, None]}}, r"'region.right' "),
    ],
    ids=["unknown_side", "unknown_face", "two_values", "text", "inside_out"],
)
def test_a_bad_region_is_a_config_error_at_init(connection: dict[str, Any], region: Any, message: str) -> None:
    pytest.importorskip("dream_ik")
    with pytest.raises(DreamConfigError, match=message):
        dream._open_policy({**connection, "ik": {"arm": "metal", "head": "eef", "region": region}})


@pytest.mark.parametrize(
    ("reach", "message"),
    [
        ([[0.3, 0.0]], r"'reach' \[\[0\.3, 0\.0\]\]: it must be a dict with some of the keys \['left', 'right'\]"),
        ({}, r"it must be a dict with some of the keys"),
        ({"middle": [[0.3, 0.0]]}, r"it must be a dict with some of the keys"),
        ({"right": []}, r"'reach.right' \[\]: it must be a list of \[x, y\] points, at least one"),
        ({"right": [0.3, 0.0]}, r"a bad point 0\.3 in 'reach.right'"),
        ({"right": [[0.3, 0.0, 0.1]]}, r"a bad point \[0\.3, 0\.0, 0\.1\] in 'reach.right'"),
        ({"right": [["0.3", 0.0]]}, r"a bad point \['0\.3', 0\.0\] in 'reach.right'"),
        ({"right": [[True, 0.0]]}, r"a bad point \[True, 0\.0\] in 'reach.right'"),
        ({"left": [[float("nan"), 0.0]]}, r"a bad point \[nan, 0\.0\] in 'reach.left'"),
        ({"left": [[0.3, float("inf")]]}, r"a bad point \[0\.3, inf\] in 'reach.left'"),
    ],
    ids=["not_a_dict", "empty", "unknown_side", "no_points", "not_pairs", "three", "text", "bool", "nan", "inf"],
)
def test_a_bad_reach_is_a_config_error_at_init(connection: dict[str, Any], reach: Any, message: str) -> None:
    pytest.importorskip("dream_ik")
    with pytest.raises(DreamConfigError, match=message):
        dream._open_policy({**connection, "ik": {"arm": "metal", "head": "eef", "reach": reach}})


def test_reach_keeps_a_listed_hand_inside_its_area_seen_from_above(
    monkeypatch: pytest.MonkeyPatch, server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """The right arm's area is one point, its first target: every target outside moves to that disk's edge, x and y
    only. Its z, every rotation and the left arm's targets (not listed) are as without `reach`, and the solved hand,
    by forward kinematics, stays inside within the solve's own miss."""
    calls = spy_on_solve(monkeypatch)
    with ik_client(connection) as plain:
        path = joint_path(plain, "metal", seed=1)
        reply = eef_reply(plain, path)
        serve_actions(server, lambda: reply)
        plain.get_action(body_observation(path[:, 0]))
    point = np.asarray(reply["per_key"]["action.right_eef_xyz"][0][:2])
    with ik_client(connection, reach={"right": [point.tolist()]}) as client:
        reached = tool_poses(client, joints_of(client.get_action(body_observation(path[:, 0]))))

    free, kept = calls
    radius = client_ik.REACH_RADIUS_M
    away = np.linalg.norm(free["xyz"][0, :, :2] - point, axis=1)
    outside = away > radius
    assert 10 < outside.sum() < ROWS, "the path must leave the area for this to test anything"
    assert np.array_equal(kept["xyz"][0, ~outside], free["xyz"][0, ~outside])
    expected = point + (free["xyz"][0, outside, :2] - point) * (radius / away[outside])[:, None]
    assert np.allclose(kept["xyz"][0, outside, :2], expected, atol=1e-15)
    assert np.array_equal(kept["xyz"][0, :, 2], free["xyz"][0, :, 2]), "z unchanged"
    assert np.array_equal(kept["xyz"][1], free["xyz"][1]), "the left arm has no reach: unchanged"
    assert np.array_equal(kept["rot6"], free["rot6"]), "every rotation unchanged"
    hand = np.linalg.norm(reached[0, :, :2, 3] - point, axis=1)
    assert hand.max() < radius + 3e-3, hand.max()


def test_without_reach_the_solve_is_bit_identical_and_an_area_around_every_target_changes_no_bit(
    monkeypatch: pytest.MonkeyPatch, server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """Without `reach` the clamp is never called; with an area holding every target it changes no bit of the reply."""
    clamp = client_ik.clamp_to_reach
    monkeypatch.setattr(client_ik, "clamp_to_reach", None)
    with ik_client(connection) as plain:
        path = joint_path(plain, "metal", seed=4)
        reply = eef_reply(plain, path)
        serve_actions(server, lambda: reply)
        free = plain.get_action(body_observation(path[:, 0]))
    monkeypatch.setattr(client_ik, "clamp_to_reach", clamp)
    every = {side: [row[:2] for row in reply["per_key"][f"action.{side}_eef_xyz"]] for side in SIDES}
    with ik_client(connection, reach=every) as client:
        kept = client.get_action(body_observation(path[:, 0]))

    for key in SIDES.values():
        assert np.array_equal(np.asarray(kept[key]), np.asarray(free[key]))


def test_a_bad_observation_is_refused_before_anything_is_sent(
    server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """The poses come from one source, the client; the joints they come from are six numbers per arm."""
    with ik_client(connection) as client:
        sent_pose = {**body_observation(np.zeros((2, 6))), "state.right_eef_xyz": np.zeros(3, dtype=np.float32)}
        with pytest.raises(DreamObservationError, match=r"state\.right_eef_xyz.*leave them out"):
            client.get_action(sent_pose)
        no_arm = body_observation(np.zeros((2, 6)))
        del no_arm["state.left_arm"]
        with pytest.raises(DreamObservationError, match=r"no 'state\.left_arm'"):
            client.get_action(no_arm)
        wide = {**body_observation(np.zeros((2, 6))), "state.right_arm": np.zeros(7, dtype=np.float32)}
        with pytest.raises(DreamObservationError, match=r"state\.right_arm.*\(7,\)"):
            client.get_action(wide)
        words = {**body_observation(np.zeros((2, 6))), "state.right_arm": ["a"] * 6}
        with pytest.raises(DreamObservationError, match=r"state\.right_arm.*not numbers"):
            client.get_action(words)
    assert server.requests == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda reply: reply.pop("per_key"), "no 'per_key' dict"),
        (lambda reply: reply["per_key"].pop("action.left_eef_rot6"), r"lacks 'action\.left_eef_rot6'"),
        (lambda reply: reply["per_key"].update({"action.right_eef_xyz": [[0.0] * 4] * ROWS}), r"\(48, 4\).*\(H, 3\)"),
        (lambda reply: reply["per_key"].update({"action.left_grip": [0.5] * (ROWS - 1)}), "disagree on the number"),
    ],
    ids=["no_per_key", "missing_part", "wrong_width", "rows_disagree"],
)
def test_a_reply_without_usable_tool_poses_is_an_action_error(
    server: FakeDreamAPI, connection: dict[str, Any], change: Callable[[dict[str, Any]], Any], message: str
) -> None:
    """Head `eef` needs every tool pose and grip, as rows of their widths; nothing is returned otherwise. `strict`
    is off so the client's own check speaks, not the embodiment's."""
    with ik_client(connection, strict=False) as client:
        path = np.zeros((2, ROWS, 6))
        reply = eef_reply(client, path)
        change(reply)
        serve_actions(server, lambda: reply)
        with pytest.raises(DreamActionError, match=message):
            client.get_action(body_observation(path[:, 0]))


## Per FaaS beta robot, its checkpoint's `<side>_arm` width and the state part of its gripper drive, as the training
## metadata names them (None: Trossen sends the drive as the last value of `<side>_arm`).
BETA_BODIES: dict[str, tuple[int, str | None]] = {
    "ai_worker_faas": (7, "gripper"),
    "trossen_wb_faas": (7, None),
    "mantis_faas": (7, "gripper"),
    "robo_t_faas": (7, "gripper"),
    "openarm2_faas": (7, "gripper"),
    "bracketbot_faas": (7, "gripper"),
    "so101_faas": (6, "gripper"),
    "galaxea_faas": (6, "gripper"),
    "seeed_faas": (6, "gripper"),
    "unitree_r1_faas": (7, "hand"),
}


def beta_body(arm: str, extra: tuple[str, ...] = ()) -> dict[str, Any]:
    """A beta robot's FaaS body, as its checkpoint's metadata lays it out, with `extra` state parts of width 1."""
    width, drive = BETA_BODIES[arm]
    parts = {"arm": width, **({drive: 1} if drive else {}), "eef_xyz": 3, "eef_rot6": 6, "grip": 1}
    body: dict[str, Any] = {"state": {}, "action": {}, "video": {"top": "observation.images.top"}}
    for group, names in (("state", [*parts, *extra]), ("action", ["eef_xyz", "eef_rot6", "grip", "arm"])):
        start = 0
        for side in SIDES:
            for name in names:
                end = start + parts.get(name, 1)
                body[group][f"{side}_{name}"] = [start, end]
                start = end
    return body


def beta_path(client: PolicyClient, arm: str, seed: int) -> npt.NDArray[np.float64]:
    """`[2, ROWS, dof]` smooth joint paths in the spec's order, near rest and inside the margined limits, each joint
    swinging by 4% of its range or 0.15 rad, at a third of the slowest speed cap or less; the held joints stay at 0."""
    from dream_ik.specs import EMBODIMENTS

    solver = client._ik._solver
    assert list(solver.sides) == list(SIDES)
    lo, hi = solver.side_lo, solver.side_hi
    speed = min(min(EMBODIMENTS[arm].side_arm(side).max_speed) for side in SIDES)
    rng = np.random.default_rng(seed)
    center = np.clip(solver.side_rest + rng.uniform(-0.2, 0.2, lo.shape), lo + 0.15 * (hi - lo), hi - 0.15 * (hi - lo))
    amplitude = np.minimum(0.04 * (hi - lo), min(0.15, speed / 3 / np.pi))
    t = np.arange(ROWS) / HZ
    phase = rng.uniform(0, 2 * np.pi, lo.shape)
    return center[:, None] + amplitude[:, None] * np.sin(np.pi * t[None, :, None] + phase[:, None])


def beta_poses(client: PolicyClient, joints: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """`[2, rows, 4, 4]` tool poses of `[2, rows, dof]` joints, one arm and one row at a time."""
    solver = client._ik._solver
    return np.stack(
        [np.stack([solver.fk(joints[i, row], side) for row in range(joints.shape[1])]) for i, side in enumerate(SIDES)]
    )


def recorded(arm: str, side: str, joints: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Spec-order radians `[..., dof]` in the units the robot recorded them in (dream's `ArmSpec.joint_units`)."""
    from dream_ik.specs import EMBODIMENTS

    units = EMBODIMENTS[arm].side_arm(side).joint_units
    return joints if units is None else (joints - np.asarray(units[1])) / np.asarray(units[0])


def drive_of(arm: str, side: str, grip: float) -> float:
    """The gripper drive the robot recorded for a grip (0 closed, 1 open): dream's `ArmSpec.jaw`."""
    from dream_ik.specs import EMBODIMENTS

    closed, opened = EMBODIMENTS[arm].side_arm(side).jaw
    return closed + grip * (opened - closed)


def beta_reply(client: PolicyClient, arm: str, path: npt.NDArray[np.float64]) -> dict[str, Any]:
    """A beta checkpoint's reply for `path`: its tool poses and a grip ramp, and its joint head in recorded units."""
    from dream_ik.decode import mat_to_rot6d
    from dream_ik.specs import EMBODIMENTS

    poses = beta_poses(client, path)
    grip = np.linspace(0.0, 1.0, path.shape[1])
    per_key: dict[str, Any] = {}
    for i, side in enumerate(SIDES):
        head = recorded(arm, side, path[i])
        if EMBODIMENTS[arm].jaw_in_arm:
            head = np.column_stack([head, [drive_of(arm, side, g) for g in grip]])
        per_key[f"action.{side}_eef_xyz"] = poses[i, :, :3, 3].tolist()
        per_key[f"action.{side}_eef_rot6"] = mat_to_rot6d(poses[i, :, :3, :3]).tolist()
        per_key[f"action.{side}_grip"] = grip.tolist()
        per_key[f"action.{side}_arm"] = head.tolist()
    return {"per_key": per_key}


def free_joints(arm: str, side: str) -> list[int]:
    """Spec indices of the joints the loop drives: all but the held ones."""
    from dream_ik.specs import EMBODIMENTS

    spec = EMBODIMENTS[arm].side_arm(side)
    return [i for i, name in enumerate(spec.joints) if name not in spec.held_joints]


def beta_observation(arm: str, joints: npt.NDArray[np.float64], grip: float = 0.25) -> dict[str, Any]:
    """The loop's observation of a beta robot at `joints` (`[2, dof]`, spec order): the driven joints in radians and
    the grip, no tool pose and no gripper drive."""
    observation: dict[str, Any] = {"video.top": [FRAME_B64], "prompt": "pick up the cup"}
    for i, side in enumerate(SIDES):
        observation[f"state.{side}_arm"] = joints[i, free_joints(arm, side)]
        observation[f"state.{side}_grip"] = np.array([grip])
    return observation


@pytest.mark.parametrize("arm", list(BETA_BODIES))
def test_a_beta_robot_gets_its_training_state_and_its_tool_poses_back_as_joints(
    server: FakeDreamAPI, connection: dict[str, Any], arm: str
) -> None:
    """Each FaaS beta robot: the checkpoint reads its joints and its gripper drive in the robot's recorded units,
    beside the tool poses, under exactly the body's state keys; the loop sends and gets radians and the grip. SO-101
    holds its dummy joint 6, Trossen sends the drive in `<side>_arm`, a Unitree R1 under `<side>_hand`."""
    body = beta_body(arm)
    with ik_client({**connection, "embodiment": body}, arm) as client:
        path = beta_path(client, arm, seed=4)
        reply = beta_reply(client, arm, path)
        serve_actions(server, lambda: reply)
        observation = beta_observation(arm, path[:, 0])
        action = client.get_action(observation)
        targets = beta_poses(client, path)
        reached = np.zeros_like(path)
        for i, side in enumerate(SIDES):
            reached[i][:, free_joints(arm, side)] = np.asarray(action[SIDES[side]])[:, :-1]
        reached_poses = beta_poses(client, reached)

    sent = server.requests[0].body["observation"]
    assert set(sent) == {f"state.{part}" for part in body["state"]} | {"video.top", "prompt"}
    for i, side in enumerate(SIDES):
        arm_units = recorded(arm, side, path[i, 0])
        drive = drive_of(arm, side, 0.25)
        _, drive_part = BETA_BODIES[arm]
        if drive_part is None:
            arm_units = np.append(arm_units, drive)
        else:
            assert np.allclose(sent[f"state.{side}_{drive_part}"], [drive], rtol=1e-12)
        assert np.allclose(sent[f"state.{side}_arm"], arm_units, rtol=1e-12, atol=1e-12)
        assert np.allclose(sent[f"state.{side}_eef_xyz"], targets[i, 0, :3, 3], atol=1e-9)
        assert np.allclose(sent[f"state.{side}_grip"], [0.25])
        chunk = np.asarray(action[SIDES[side]])
        assert chunk.shape == (ROWS, len(free_joints(arm, side)) + 1)
        assert np.allclose(chunk[:, -1], reply["per_key"][f"action.{side}_grip"], atol=1e-9)
    position_miss = np.linalg.norm(reached_poses[..., :3, 3] - targets[..., :3, 3], axis=-1)
    angle_miss = np.linalg.norm(reached_poses[..., :3, :3] - targets[..., :3, :3], axis=(-2, -1)) / np.sqrt(2)
    assert position_miss.max() < 3e-3  # m, as for Metal and Makers: dream's filter on the hand path, then the solve
    assert angle_miss.max() < np.radians(1.0)


@pytest.mark.parametrize("arm", ["so101_faas", "trossen_wb_faas"])
def test_a_beta_robots_joint_head_comes_back_in_radians(
    monkeypatch: pytest.MonkeyPatch, server: FakeDreamAPI, connection: dict[str, Any], arm: str
) -> None:
    """Head `joint`: the checkpoint's joint head, in the robot's units (SO-101's LeRobot range, Trossen's drive at the
    end), comes back as the driven joints in radians, then the grip; nothing is solved."""
    decode = pytest.importorskip("dream_ik.decode")
    keys = {**connection, "checkpoint": BETA, "embodiment": beta_body(arm)}
    with ik_client(keys, arm, head="joint") as client:
        path = beta_path(client, arm, seed=5)
        reply = beta_reply(client, arm, path)
        serve_actions(server, lambda: reply)
        monkeypatch.setattr(decode, "solve_chunk", None)
        action = client.get_action(beta_observation(arm, path[:, 0]))

    for i, side in enumerate(SIDES):
        expected = np.column_stack([path[i][:, free_joints(arm, side)], reply["per_key"][f"action.{side}_grip"]])
        assert np.allclose(action[SIDES[side]], smoothing.filter_from(expected, 0), atol=1e-9)


@pytest.mark.parametrize(
    ("embodiment", "message"),
    [
        (None, "needs the body's embodiment config"),
        (beta_body("so101_faas", extra=("tactile",)), r"one gripper part of the right arm.*right_gripper.*tactile"),
    ],
    ids=["no_body", "two_drive_parts"],
)
def test_a_beta_robot_whose_gripper_drive_key_the_body_does_not_name_is_a_config_error(
    connection: dict[str, Any], embodiment: Any, message: str
) -> None:
    """The drive's key comes from the body (most say `<side>_gripper`, a Unitree R1 `<side>_hand`); refused at init
    before the solver is built when the body does not name exactly one."""
    pytest.importorskip("dream_ik")
    keys = {**connection, "ik": {"arm": "so101_faas", "head": "eef"}, "strict": False}
    if embodiment is None:
        del keys["embodiment"]
    else:
        keys["embodiment"] = embodiment
    with pytest.raises(DreamConfigError, match=message):
        dream._open_policy(keys)


def test_a_model_trained_on_yam_arms_runs_on_the_makers_arm_with_zero_joints(
    server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """The Makers Arm has no map from its joints to YAM's, and needs none."""
    with ik_client({**connection, "checkpoint": BASE, "embodiment": station_body()}, "makers") as client:
        path = joint_path(client, "makers", seed=3)
        reply = eef_reply(client, path)
        serve_actions(server, lambda: reply)
        client.get_action(station_observation(path[:, 0], gripper_mm=40.0))

    sent = server.requests[0].body["observation"]
    for side in SIDES:
        assert sent[f"state.{side}_arm"] == [0.0] * 6
        assert np.allclose(sent[f"state.{side}_eef_xyz"], reply["per_key"][f"action.{side}_eef_xyz"][0], atol=1e-6)


def gripperless_so101_body() -> dict[str, Any]:
    """An SO-101's body as the setup wizard's New registered it before the gripper fix: each arm, then each hand's pose
    and grip, the station's cameras, and no `<side>_gripper` (an SO-101 station's)."""
    eef = [("right_eef_xyz", 3), ("right_eef_rot6", 6), ("right_grip", 1), ("left_eef_xyz", 3), ("left_eef_rot6", 6)]
    eef += [("left_grip", 1)]

    def spans(parts: list[tuple[str, int]]) -> dict[str, list[int]]:
        out, start = {}, 0
        for name, width in parts:
            out[name] = [start, start + width]
            start += width
        return out

    return {
        "state": spans([("left_arm", 6), ("right_arm", 6), *eef]),
        "action": spans([*eef, ("left_arm", 6), ("right_arm", 6)]),
        "video": {role: f"observation.images.{role}" for role in STATION_CAMERAS},
    }


def test_a_model_trained_on_yam_arms_runs_on_an_so101_whose_body_names_no_gripper_drive(
    server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """A station whose SO-101 body names no gripper drive (registered before the gripper fix) is refused at start
    with "must name one gripper part", though a model trained on YAM arms never reads the drive: it is sent the grip
    as `state.<side>_gripper`. The client now runs it; a fine-tune of the robot still needs the part."""
    keys = {**connection, "checkpoint": BASE, "embodiment": gripperless_so101_body()}
    with ik_client(keys, "so101_faas") as client:
        from dream_ik.decode import mat_to_rot6d

        path = beta_path(client, "so101_faas", seed=7)
        poses = beta_poses(client, path)  # each side on its own URDF
        grip = np.linspace(0.0, 1.0, path.shape[1])
        per_key: dict[str, Any] = {}
        reply: dict[str, Any] = {"per_key": per_key}
        for i, (side, chunk_key) in enumerate(SIDES.items()):
            per_key[f"action.{side}_eef_xyz"] = poses[i, :, :3, 3].tolist()
            per_key[f"action.{side}_eef_rot6"] = mat_to_rot6d(poses[i, :, :3, :3]).tolist()
            per_key[f"action.{side}_grip"] = grip.tolist()
            per_key[f"action.{side}_arm"] = np.zeros((path.shape[1], 6)).tolist()
            reply[chunk_key] = np.column_stack([np.zeros((path.shape[1], 6)), grip]).tolist()
        serve_actions(server, lambda: reply)
        observation = {
            k: v for k, v in beta_observation("so101_faas", path[:, 0], grip=0.4).items() if k != "video.top"
        }
        for role in STATION_CAMERAS:
            observation[f"video.{role}"] = [FRAME_B64]
        client.get_action(observation)

    sent = server.requests[0].body["observation"]
    for side in SIDES:
        assert sent[f"state.{side}_gripper"] == pytest.approx([0.4])
        assert sent[f"state.{side}_arm"] == [0.0] * 6


def test_a_model_trained_on_yam_arms_runs_on_an_arm_of_your_own_from_its_urdf(
    server: FakeDreamAPI, connection: dict[str, Any], tmp_path: Any
) -> None:
    """An arm file (`dream.arm_file`) of the Metal arm's URDF in place of the name: the model reads zero joints and the
    arm's own tool pose, and the reply's tool poses come back as the joints dream's own Metal arm returns."""
    specs = pytest.importorskip("dream_ik.specs")
    (tmp_path / "arm.urdf").write_text((specs.ASSETS / "metal.urdf").read_text())
    arm = {
        "urdf": "arm.urdf",
        "base": "base_link",
        "flange": "link6",
        "tcp": [0.135, 0.0, 0.0],
        "approach": [1.0, 0.0, 0.0],
        "jaw": [0.0, 1.0, 0.0],
        "rest": [0.0, -1.2, 1.5, 0.0, 0.0, 0.0],
    }
    (tmp_path / "arm.json").write_text(json.dumps(arm))

    def solve(ik: str) -> tuple[dict[str, Any], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The reply, the joints the client returns for it, and the tool's miss along the path (m)."""
        with ik_client({**connection, "checkpoint": BASE, "embodiment": station_body()}, ik) as client:
            path = joint_path(client, "metal", seed=1)
            reply = eef_reply(client, path)
            serve_actions(server, lambda: reply)
            joints = joints_of(client.get_action(station_observation(path[:, 0], gripper_mm=40.0)))
            miss = np.linalg.norm(
                tool_poses(client, joints)[..., :3, 3] - tool_poses(client, path)[..., :3, 3], axis=-1
            )
        return reply, joints, miss

    reply, joints, miss = solve(str(tmp_path / "arm.json"))
    sent = server.requests[0].body["observation"]
    for side in SIDES:
        assert sent[f"state.{side}_arm"] == [0.0] * 6
        assert np.allclose(sent[f"state.{side}_eef_xyz"], reply["per_key"][f"action.{side}_eef_xyz"][0], atol=1e-6)
    # The same solution as dream's own Metal arm: jaxls may build its cost groups in another order, which stops each
    # warm-started solve along the path at another point within its tolerance (2.3e-6 and then over 1e-4 rad apart
    # between runs); another IK solution would be radians apart.
    assert np.allclose(joints, solve("metal")[1], atol=1e-2)
    # As on dream's own Metal arm (test_the_tool_poses_go_out_and_come_back_as_joints_on_the_same_path).
    assert miss.max() < 3e-3  # m


@pytest.mark.parametrize("arm", ["so101_faas", "ai_worker_faas", "trossen_wb_faas", "unitree_r1_faas"])
def test_a_model_trained_on_yam_arms_runs_on_a_beta_robot_with_six_zero_joints_in_strict_mode(
    server: FakeDreamAPI, connection: dict[str, Any], arm: str
) -> None:
    """A beta robot's own body (seven joints, Trossen's drive in `<side>_arm`, a Unitree R1's `<side>_hand`), with
    the scene and wrist cameras: the model reads YAM's six joints as zeros, the arm's own tool pose and the grip as
    `state.<side>_gripper`, and no gripper drive; the strict checks read what is sent and answered."""
    body = {**beta_body(arm), "video": station_body()["video"]}
    with ik_client({**connection, "checkpoint": BASE, "embodiment": body}, arm) as client:
        path = beta_path(client, arm, seed=4)
        reply = beta_reply(client, arm, path)
        for side in SIDES:
            reply["per_key"][f"action.{side}_arm"] = np.zeros((ROWS, 6)).tolist()
        serve_actions(server, lambda: reply)
        observation = {key: value for key, value in beta_observation(arm, path[:, 0]).items() if key != "video.top"}
        for role in STATION_CAMERAS:
            observation[f"video.{role}"] = [FRAME_B64]
        action = client.get_action(observation)
        targets = beta_poses(client, path)

    sent = server.requests[0].body["observation"]
    parts = ("arm", "gripper", "eef_xyz", "eef_rot6", "grip")
    assert set(sent) == {f"state.{side}_{part}" for side in SIDES for part in parts} | {
        "video.overhead",
        "video.wrist_l",
        "video.wrist_r",
        "prompt",
        "embodiment",
    }
    assert sent["embodiment"] == "other_embodiment", "the serve withholds the state that is not this arm's"
    for i, side in enumerate(SIDES):
        assert sent[f"state.{side}_arm"] == [0.0] * 6
        assert np.allclose(sent[f"state.{side}_gripper"], [0.25]) and np.allclose(sent[f"state.{side}_grip"], [0.25])
        assert np.allclose(sent[f"state.{side}_eef_xyz"], targets[i, 0, :3, 3], atol=1e-9)
        assert np.asarray(action[SIDES[side]]).shape == (ROWS, len(free_joints(arm, side)) + 1)


def withheld(reply: dict[str, Any]) -> dict[str, Any]:
    """What the serve answers `reply` as on `other_embodiment`: the joint head null, and no joint rows or chunks."""
    joint_head = {f"action.{side}_arm" for side in SIDES}
    per_key = {key: None if key in joint_head else value for key, value in reply["per_key"].items()}
    dropped = (*client_ik.MODEL_JOINT_ROWS, *SIDES.values())
    return {**{key: value for key, value in reply.items() if key not in dropped}, "per_key": per_key}


@pytest.mark.parametrize("arm", ["metal", "makers"])
def test_on_another_arm_a_model_trained_on_yam_arms_is_sent_other_embodiment_and_its_null_joint_head_passes(
    server: FakeDreamAPI, connection: dict[str, Any], arm: str
) -> None:
    """Route A, unasked: with `ik` on an arm that is not a YAM, every request names the robot `other_embodiment`. The
    serve then answers the joint head as null and packs no chunks; strict mode takes that reply, and the client still
    solves `q_r_chunk` and `q_l_chunk` from the tool poses on the caller's side."""
    with ik_client({**connection, "checkpoint": BASE, "embodiment": station_body()}, arm) as client:
        assert client._checks is not None, "strict mode"
        path = joint_path(client, arm, seed=1)
        reply = withheld(eef_reply(client, path))
        serve_actions(server, lambda: reply)
        observation = station_observation(path[:, 0], gripper_mm=40.0)
        first = client.get_action(observation)
        client.get_action(observation)
        reached = tool_poses(client, joints_of(first))
        target = tool_poses(client, path)

    assert [request.body["observation"]["embodiment"] for request in server.requests] == ["other_embodiment"] * 2
    assert "embodiment" not in observation, "the caller's dict is not changed"
    for side in SIDES:
        assert server.requests[0].body["observation"][f"state.{side}_arm"] == [0.0] * 6
        assert first["per_key"][f"action.{side}_arm"] is None
        assert np.asarray(first[SIDES[side]]).shape == (ROWS, 7)
    # As in test_the_tool_poses_go_out_and_come_back_as_joints_on_the_same_path, on the same path.
    assert np.linalg.norm(reached[..., :3, 3] - target[..., :3, 3], axis=-1).max() < 3e-3  # m


def test_no_robot_is_named_on_a_yam_or_for_a_model_trained_on_the_arm(
    server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """Left out, `robot` is named only where the client sends zeros for the joints: on a YAM (`ik` arm `yam`) a base
    model reads the arm's own joints, and a model trained on the Metal arm reads the Metal's, so neither request
    carries `embodiment`."""
    with ik_client({**connection, "checkpoint": BASE, "embodiment": YAM_POSE_CONFIG}, "yam") as client:
        path = joint_path(client, "yam", seed=9)
        reply = eef_reply(client, path)
        serve_actions(server, lambda: reply)
        client.get_action(yam_observation(path))
    with ik_client(connection, "metal") as client:  # PLAIN, trained on the Metal arm
        path = joint_path(client, "metal", seed=1)
        reply = eef_reply(client, path)
        serve_actions(server, lambda: reply)
        client.get_action(body_observation(path[:, 0]))

    assert len(server.requests) == 2
    for request in server.requests:
        assert "embodiment" not in request.body["observation"]
        assert b'"embodiment"' not in request.raw


## Per case of an explicit `robot`: the arm, the checkpoint, its body, and the observation of the arm at `path`'s first
## row.
EXPLICIT_ROBOTS: dict[str, tuple[str, str, Callable[[], dict[str, Any]], Callable[[Any], dict[str, Any]], str]] = {
    "yam_on_the_metal": ("metal", BASE, station_body, lambda path: station_observation(path[:, 0], 40.0), "yam"),
    "other_embodiment_on_a_yam": ("yam", BASE, lambda: YAM_POSE_CONFIG, yam_observation, "other_embodiment"),
    "other_embodiment_for_a_metal_model": (
        "metal",
        PLAIN,
        lambda: copy.deepcopy(FAAS_CONFIG),
        lambda path: body_observation(path[:, 0]),
        "other_embodiment",
    ),
}


@pytest.mark.parametrize("case", list(EXPLICIT_ROBOTS))
def test_a_robot_given_wins_over_the_default_and_is_sent_as_given(
    server: FakeDreamAPI, connection: dict[str, Any], case: str
) -> None:
    """`robot` set beside `ik` is sent as given, whatever the default would be, in strict mode: `yam` on the Metal
    (the joints still go out as zeros), `other_embodiment` on a YAM and for a model trained on the Metal arm, each of
    which then gets the serve's null joint head and is still solved from its tool poses."""
    arm, checkpoint, body, observe, robot = EXPLICIT_ROBOTS[case]
    with ik_client({**connection, "checkpoint": checkpoint, "embodiment": body(), "robot": robot}, arm) as client:
        path = joint_path(client, arm, seed=1)
        reply = eef_reply(client, path)
        reply = withheld(reply) if robot == "other_embodiment" else reply
        serve_actions(server, lambda: reply)
        action = client.get_action(observe(path))

    sent = server.requests[0].body["observation"]
    assert sent["embodiment"] == robot
    if checkpoint == BASE and arm != "yam":
        assert sent["state.right_arm"] == [0.0] * 6, "the client still sends zeros for another arm's joints"
    for side in SIDES:
        assert np.asarray(action[SIDES[side]]).shape == (ROWS, 7)


def test_head_joint_with_robot_other_embodiment_is_a_config_error(connection: dict[str, Any]) -> None:
    """On `other_embodiment` the model answers no joints, so a joint head would have nothing to run."""
    with pytest.raises(DreamConfigError, match=r"'robot' is 'other_embodiment'.*head 'joint'"):
        ik_client(
            {**connection, "checkpoint": BASE, "embodiment": YAM_CONFIG, "robot": "other_embodiment"}, "yam", "joint"
        )


def test_a_beta_robots_observation_and_commanded_are_checked_in_the_loops_units(
    server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """SO-101's loop drives five joints and sends the grip; the client makes the drive. `commanded` is five joints,
    or the written row of six."""
    with ik_client({**connection, "embodiment": beta_body("so101_faas")}, "so101_faas") as client:
        path = beta_path(client, "so101_faas", seed=6)
        observation = beta_observation("so101_faas", path[:, 0])
        with pytest.raises(DreamObservationError, match=r"state\.right_gripper.*gripper drives"):
            client.get_action({**observation, "state.right_gripper": np.array([25.0])})
        with pytest.raises(DreamObservationError, match=r"no 'state\.left_grip'"):
            client.get_action({key: value for key, value in observation.items() if key != "state.left_grip"})
        with pytest.raises(DreamObservationError, match=r"state\.right_grip.*\(2,\).*\(1,\)"):
            client.get_action({**observation, "state.right_grip": np.array([0.5, 0.5])})
        with pytest.raises(DreamObservationError, match=r"state\.right_arm.*\(6,\).*5 joints"):
            client.get_action({**observation, "state.right_arm": path[0, 0]})
        with pytest.raises(DreamObservationError, match="5 joint angles, or its written row of 6"):
            client.get_action(observation, commanded={"q_r_chunk": np.zeros(7)})
        assert server.requests == []
        serve_actions(server, lambda: beta_reply(client, "so101_faas", path))
        joints = path[0, 0, :5]
        client.get_action(observation, commanded={"q_r_chunk": joints, "q_l_chunk": np.append(joints, 0.5)})
    assert len(server.requests) == 1


def test_ik_absent_leaves_get_action_as_it_was(
    monkeypatch: pytest.MonkeyPatch, server: FakeDreamAPI, connection: dict[str, Any]
) -> None:
    """Without the key nothing imports dream's IK, the body on the wire is the caller's observation, and the
    chunks are exactly the seam join's on the YAM wire's joints."""
    monkeypatch.setitem(sys.modules, "dream_ik", None)
    for name in [name for name in sys.modules if name.startswith("dream_ik.")]:
        monkeypatch.setitem(sys.modules, name, None)
    del connection["embodiment"]
    line = np.repeat(np.linspace(0.0, 1.0, ROWS)[:, None], 7, axis=1)
    serve_actions(server, lambda: {"q_r_chunk": line.tolist(), "q_l_chunk": (line + 0.1).tolist()})
    reference = smoothing.SeamSmoother(HZ)
    with dream._open_policy(connection) as client:
        for anchor in (0, 12, 30):
            observation = make_observation()
            observation["state.arm_r.joint_pos"] = line[anchor, :6].astype(np.float32)
            observation["state.arm_l.joint_pos"] = (line[anchor, :6] + 0.1).astype(np.float32)
            action = client.get_action(observation)
            reply = {"q_r_chunk": line.tolist(), "q_l_chunk": (line + 0.1).tolist()}
            expected = reference.smooth(observation, reply, 0.0, client._pacer._recent_rtt_s[-1])
            assert action == expected
            assert b'"observation":' + transport.encode(observation) in server.requests[-1].raw


def test_pad_holds_for_tick_and_continues_the_motion_for_traj() -> None:
    """A chunk cut at its clock row and padded back to its length: a hold at the last pose for `tick` (causal: the pad
    never reaches a kept row), the last motion continued for `traj`, which sees the whole chunk and would brake."""
    decode = pytest.importorskip("dream_ik.decode")
    n, skip, step = 10, 3, np.array([0.004, -0.001, 0.002])
    poses = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    poses[:, :3, 3] = np.arange(n)[:, None] * step
    held = client_ik._pad(decode, poses, skip, continue_motion=False)
    moved = client_ik._pad(decode, poses, skip, continue_motion=True)
    assert held.shape == moved.shape == poses.shape
    assert np.array_equal(held[: n - skip], poses[skip:]) and np.array_equal(
        held[n - skip :], np.repeat(poses[-1:], skip, 0)
    )
    assert np.array_equal(moved[: n - skip], poses[skip:])
    assert np.allclose(moved[n - skip :, :3, 3], np.arange(n, n + skip)[:, None] * step)

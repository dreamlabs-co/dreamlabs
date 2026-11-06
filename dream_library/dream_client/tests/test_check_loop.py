"""Tests of the loop check (`dream.check_loop`): the stand-in against a simulated YAM driven by the real client.

Each test runs the whole plan: a control loop like the one in the docs plays the stand-in's chunks on a simulated
two-arm YAM, 20 rows a second, through a real `dream.DreamClient` over HTTP. Time is a fake clock both sides share, so
the motion (every joint at once, then each part found at fault alone) runs in a second or two. The robot follows its
targets through a first-order lag (#SERVO_S); a fault is planted in the loop or the robot, and the check must print
PROBLEM on the line meant to catch it, and PASS on the rest.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import math
import pathlib
import re
import sys
import urllib.request
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest
from conftest import FRAME_B64, PLATFORM_KEY, FakePlatform

import dream
from dream import arm_file, check_arm, check_loop, client_ik

specs = pytest.importorskip("dream_ik.specs")

## The YAM's parts, as the docs' loop names them.
PARTS = {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]}
CAMERAS = ("overhead", "wrist_l", "wrist_r")
YAM = {"state": PARTS, "action": PARTS, "video": {role: f"observation.images.{role}" for role in CAMERAS}}
## Where the simulated arms stand: shoulder up, elbow bent, the grippers half open (radians, then 0 to 1).
POSE = np.array([0.0, 1.6, 1.3, 0.0, 0.3, 0.0, 0.5])
## The simulated servo's time constant, seconds, and the loop's rows a second.
SERVO_S, ROW_HZ = 0.05, 20.0


class Clock:
    """A fake clock both the stand-in and the loop read."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


@dataclasses.dataclass
class Faults:
    """What the loop or the robot gets wrong.

    Attributes:
    - swap: Two joints of each arm (0-based) swapped between the loop's order and the model's.
    - flip: A joint whose sign the loop flips.
    - lag_joint, lag_s: A joint that follows its target this many seconds late.
    - invert_grip: The loop sends the gripper the wrong way round.
    - play_hz: Rows a second the loop actually plays (it believes it plays ROW_HZ).
    - degrees: The loop sends degrees where the robot reads radians.
    """

    swap: tuple[int, int] | None = None
    flip: int | None = None
    lag_joint: int | None = None
    lag_s: float = 0.0
    invert_grip: bool = False
    play_hz: float = ROW_HZ
    degrees: bool = False


@dataclasses.dataclass
class SimArm:
    """A simulated arm, the same on both sides.

    Attributes:
    - pose: Where it stands, 6 joints (radians) and the gripper (0 to 1).
    - limits: `(lower, upper)` radians per joint; its servo stops each joint there.
    """

    pose: np.ndarray
    limits: Sequence[tuple[float, float]]


def _arm(name: str) -> check_arm.Arm:
    """dream's arm `name` (its right side), as the check reads it."""
    return check_arm.Arm(name, specs.EMBODIMENTS[name].side_arm("right"))


SIM_YAM = SimArm(POSE, [tuple(limit) for limit in _arm("yam").limits])


def run(
    faults: Faults,
    arm: SimArm,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    refill: int = 0,
    **options: Any,
) -> tuple[list[str], check_loop.StandIn]:
    """Runs the plan on the simulated robot through the real client and returns the lines the check prints.

    Each tick hands the client a new picture of every camera, timed in seconds, as a loop should.

    @param faults What is wrong.
    @param arm The simulated arm.
    @param tmp_path Where the trace goes.
    @param monkeypatch Points the client at the stand-in.
    @param refill Rows of a chunk left when the loop asks for the next one, which then replaces them (0: it plays
        every row first).
    @param options More #check_loop.StandIn options (`view`).
    @return Each line as printed, and the stand-in after the run.
    """
    clock = Clock()
    stand_in = check_loop.StandIn(rate=ROW_HZ, clock=clock, **options)
    monkeypatch.setenv("DREAM_GATEWAY_URL", stand_in.serve(0))
    arms = {side: arm.pose.copy() for side in ("right", "left")}
    history: list[tuple[float, dict[str, np.ndarray]]] = []
    rows: list[np.ndarray] = []
    target = {side: arm.pose.copy() for side in ("right", "left")}
    dt = 1 / faults.play_hz
    try:
        with dream.DreamClient() as client:
            client.policy({"checkpoint": "dreamlabs/dream-1-beta", "embodiment": YAM})
            for tick in range(int(80 * faults.play_hz)):
                observation: dict[str, Any] = {"prompt": "check the loop"}
                for side in ("right", "left"):
                    observation[f"state.{side}_arm"] = arms[side][:6].copy()
                    observation[f"state.{side}_gripper"] = arms[side][6:].copy()
                frames = {f"video.{role}": _picture(f"{role} {tick}") for role in CAMERAS}
                client.set_observation(frames, observation, timestamp=clock.t)
                if len(rows) <= refill:
                    try:
                        action = client.get_action()
                    except dream.DreamServerError:
                        break  # the stand-in has finished or stopped
                    rows = list(zip(np.asarray(action["q_r_chunk"]), np.asarray(action["q_l_chunk"]), strict=True))
                for side, row in zip(("right", "left"), rows.pop(0), strict=True):
                    target[side] = _through_loop(np.asarray(row, dtype=float), faults)
                history.append((clock.t, {side: value.copy() for side, value in target.items()}))
                for side in arms:
                    arms[side] = _servo(arms[side], _late(history, side, faults, clock.t), dt, arm.limits)
                clock.t += dt
    finally:
        stand_in.close()
    lines = [str(line) for line in check_loop.judge(stand_in)]
    trace = stand_in.trace()
    (tmp_path / "trace.json").write_text(json.dumps(trace))
    return lines, stand_in


def _picture(note: str) -> str:
    """#FRAME_B64 with `note` in a comment segment: a picture of its own that still reads as a JPEG."""
    jpeg, text = base64.b64decode(FRAME_B64), note.encode()
    return base64.b64encode(jpeg[:2] + b"\xff\xfe" + (len(text) + 2).to_bytes(2, "big") + text + jpeg[2:]).decode()


def _through_loop(row: np.ndarray, faults: Faults) -> np.ndarray:
    """A row as the loop sends it to the robot, with the loop's faults."""
    out = row.copy()
    if faults.swap is not None:
        i, j = faults.swap
        out[i], out[j] = row[j], row[i]
    if faults.flip is not None:
        out[faults.flip] = -row[faults.flip]
    if faults.invert_grip:
        out[6] = 1 - row[6]
    if faults.degrees:
        out[:6] = np.degrees(row[:6])
    return out


def _late(history: list[tuple[float, dict[str, np.ndarray]]], side: str, faults: Faults, now: float) -> np.ndarray:
    """The targets the robot acts on now: the newest, except the lagging joint's, which is `lag_s` old."""
    target = history[-1][1][side].copy()
    if faults.lag_joint is not None:
        old = [t for t in history if t[0] <= now - faults.lag_s]
        target[faults.lag_joint] = (old[-1] if old else history[0])[1][side][faults.lag_joint]
    return target


def _servo(position: np.ndarray, target: np.ndarray, dt: float, limits: Sequence[tuple[float, float]]) -> np.ndarray:
    """One step of the robot: each joint closes on its target with time constant #SERVO_S, inside its limits."""
    step = position + (target - position) * (1 - math.exp(-dt / SERVO_S))
    lower = np.array([lo for lo, _ in limits] + [0.0])
    upper = np.array([hi for _, hi in limits] + [1.0])
    return np.clip(step, lower, upper)


def _status(lines: list[str]) -> dict[str, str]:
    """Each check's status: `hold` to `PASS`, and so on."""
    return {check: line.split()[0] for line in lines for check in check_loop.CHECKS if line[9:].startswith(check)}


def test_healthy_loop_passes_every_check(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lines, stand_in = run(Faults(), SIM_YAM, tmp_path, monkeypatch)
    assert set(_status(lines).values()) == {"PASS"}, lines
    # The docs' loop sends the YAM's parts and no hand pose: a YAM pair, its joints circling.
    assert stand_in.arm == "yam" and stand_in.read_as() == "yam arms (joints only, no hand pose)"
    # The together-motion alone: nothing moves joint by joint, and the run lasts under 17 s.
    assert stand_in.alone == [] and [s.kind for s in stand_in.plan.segments] == ["hold", "together"]
    assert stand_in.calls[-1].t - stand_in.t0 < 17.0


def test_the_page_draws_the_arms_from_the_targets_and_each_call(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """StandIn.arms(), what the check's page draws the arms from: nothing before the first call; after the run, the
    targets of every row for each arm and gripper, each call's reported state, the clock, the tool point and each
    arm's chain of joints."""
    from dream.debug_view import DebugView

    assert check_loop.StandIn(rate=ROW_HZ).arms() is None
    _, stand_in = run(Faults(), SIM_YAM, tmp_path, monkeypatch)
    with DebugView(None, 0) as view:
        view.show_arms(stand_in.arms)
        with urllib.request.urlopen(view.url + "arms", timeout=5) as reply:
            arms = json.load(reply)
    assert stand_in.plan is not None and arms["done"] and arms["tcp_m"] == specs.TCP_OFFSET_M
    assert arms["arm"] == "yam" and sorted(arms["chains"]) == ["left", "right"]
    chain = arms["chains"]["right"]
    assert [link["col"] for link in chain["links"]] == [0, 1, 2, 3, 4, 5] and len(chain["tool"]) == 16
    rows = stand_in.plan.rows + check_loop.HORIZON
    sizes = {part: (len(table), len(table[0])) for part, table in arms["targets"].items()}
    assert sizes == {
        "right_arm": (rows, 6),
        "right_gripper": (rows, 1),
        "left_arm": (rows, 6),
        "left_gripper": (rows, 1),
    }
    assert [call["t"] for call in arms["calls"]] == [call.t - stand_in.t0 for call in stand_in.calls]
    assert arms["calls"][0]["state"]["left_arm"] == pytest.approx(POSE[:6].tolist())
    assert arms["now"] >= arms["calls"][-1]["t"]


def test_lagging_joint_is_a_lag_problem(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lines, stand_in = run(Faults(lag_joint=2, lag_s=0.4), SIM_YAM, tmp_path, monkeypatch)
    status = _status(lines)
    assert status["lag and tracking"] == "PROBLEM", lines
    late = next(line for line in lines if "lag and tracking" in line)
    assert "joint 3" in late and "the rest is on time, so the joints that move it follow" in late, late
    assert status["hold"] == status["right part moved"] == status["play rate"] == "PASS", lines
    # The together-motion caught it; then only the two joints 3 moved alone.
    assert (
        stand_in.first is not None and _status([str(line) for line in stand_in.first])["lag and tracking"] == "PROBLEM"
    )
    assert [c.name for c in stand_in.alone] == ["right_arm joint 3", "left_arm joint 3"]


def test_swapped_joints_are_a_part_problem(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lines, _ = run(Faults(swap=(1, 2)), SIM_YAM, tmp_path, monkeypatch)
    status = _status(lines)
    assert status["right part moved"] == "PROBLEM", lines
    assert status["play rate"] == "PASS", lines


def test_flipped_sign_is_a_direction_problem(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lines, _ = run(Faults(flip=3), SIM_YAM, tmp_path, monkeypatch)
    status = _status(lines)
    assert status["direction and size"] == "PROBLEM", lines
    assert "joint 4 went the other way" in next(line for line in lines if "direction and size" in line)
    assert status["hold"] == status["right part moved"] == status["play rate"] == "PASS", lines


def test_inverted_gripper_is_caught(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lines, _ = run(Faults(invert_grip=True), SIM_YAM, tmp_path, monkeypatch)
    status = _status(lines)
    assert status["direction and size"] == "PROBLEM", lines
    assert "gripper closed when told to open" in next(line for line in lines if "direction and size" in line)


def test_slow_play_is_a_rate_problem(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lines, stand_in = run(Faults(play_hz=ROW_HZ * 2 / 3), SIM_YAM, tmp_path, monkeypatch)
    assert _status(lines)["play rate"] == "PROBLEM", lines
    assert stand_in.alone == [], "a loop-wide fault moves nothing alone"


def test_degrees_stop_the_run_during_the_hold(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lines, _ = run(Faults(degrees=True), SIM_YAM, tmp_path, monkeypatch)
    status = _status(lines)
    assert status["hold"] == "PROBLEM" and "stopped" in lines[0], lines
    assert {status[c] for c in check_loop.CHECKS[1:]} == {"SKIP"}, lines


def run_direct(
    report: Callable[[], dict[str, np.ndarray]],
    play: Callable[[dict[str, np.ndarray]], None],
    robot: str | None,
    late_rows: int = 0,
) -> tuple[list[str], check_loop.StandIn]:
    """Plays the stand-in's chunks as a loop would, calling it directly (no client), 20 rows a second on a fake clock.

    @param report The state the next call carries, per part (without `state.`).
    @param play Applies one row of the answer (per part, without `action.`) to the simulated robot, one tick.
    @param robot The robot each call names (`embodiment`), or None to name none.
    @param late_rows The loop plays each row this many rows late (a queue), on top of the servo's lag.
    @return Each line as printed, and the stand-in after the run.
    """
    clock = Clock()
    stand_in = check_loop.StandIn(rate=ROW_HZ, clock=clock)
    rows: list[dict[str, np.ndarray]] = []
    queue: list[dict[str, np.ndarray]] = []
    session = None
    for _ in range(int(80 * ROW_HZ)):
        if not rows:
            observation: dict[str, Any] = {f"state.{k}": v for k, v in report().items()}
            if robot is not None:
                observation["embodiment"] = robot
            reply = stand_in.answer({"session_id": session, "observation": observation}, None, None)
            session = reply["session_id"]
            if reply["actions"] is None:
                break  # the stand-in has finished or stopped
            per_key = reply["actions"]["per_key"]
            rows = [{k[len("action.") :]: np.asarray(v[j]) for k, v in per_key.items()} for j in range(30)]
        queue.append(rows.pop(0))
        if len(queue) > late_rows:
            play(queue.pop(0))
        clock.t += 1 / ROW_HZ
    return [str(line) for line in check_loop.judge(stand_in)], stand_in


def _closer(value: np.ndarray, target: np.ndarray) -> np.ndarray:
    """One tick of a servo with time constant #SERVO_S."""
    return value + (target - value) * (1 - math.exp(-1 / ROW_HZ / SERVO_S))


def run_tool(flip_y: bool, late_rows: int) -> tuple[list[str], check_loop.StandIn]:
    """A model trained on YAM arms run on an arm the check does not know, as the client's `ik` runs it: each call
    sends the hands' poses and YAM's joints as zeros (`dream.client_ik`), and the station drives the tool poses.

    @param flip_y The loop sends each tool point's y the wrong way round.
    @param late_rows The loop plays each row this many rows late.
    @return Each line as printed, and the stand-in after the run.
    """
    pose: dict[str, np.ndarray] = {}
    for side in ("right", "left"):  # y at 0, so the flipped axis sends the right pose during the hold
        pose |= {f"{side}_eef_xyz": np.array([0.3, 0.0, 0.2]), f"{side}_eef_rot6": np.eye(3)[:2].reshape(6)}
        pose[f"{side}_grip"] = np.array([0.5])

    def report() -> dict[str, np.ndarray]:
        zeros = {f"{side}_arm": np.zeros(6) for side in ("right", "left")}
        return {**pose, **zeros, **{f"{side}_gripper": pose[f"{side}_grip"] for side in ("right", "left")}}

    def play(row: dict[str, np.ndarray]) -> None:
        for part, target in row.items():
            if part in pose:  # the station drives the tool poses and the grip
                flip = flip_y and part.endswith("eef_xyz")
                pose[part] = _closer(pose[part], target * np.array([1, -1, 1]) if flip else target)

    return run_direct(report, play, "other_embodiment", late_rows)


## Where the simulated Metal arms stand: its rest pose, the shoulder (joint 2) 6 degrees inside its upper limit, then
## the grip (0 to 1).
METAL_POSE = np.array([0.0, -0.1, 1.5, 0.0, 0.0, 0.0, 0.5])


def run_metal(faults: Faults) -> tuple[list[str], check_loop.StandIn]:
    """Two Metal arms with a fine-tune of their own, as the client's `ik` runs it (head `joint`): each call sends the
    measured joints, each hand's pose from them, the client's grip (0 to 1) and the arm's own gripper in millimetres
    beside it; the loop drives the joints and the grip.

    @param faults What the loop or the robot gets wrong.
    @return Each line as printed, and the stand-in after the run.
    """
    metal = _arm("metal")
    limits = [tuple(limit) for limit in metal.limits]
    arms = {side: METAL_POSE.copy() for side in ("right", "left")}

    def report() -> dict[str, np.ndarray]:
        out = {}
        for side, value in arms.items():
            hand = metal.hand(value[None, :6])[0]  # (9,)
            out |= {f"{side}_arm": value[:6].copy(), f"{side}_eef_xyz": hand[:3], f"{side}_eef_rot6": hand[3:]}
            out |= {f"{side}_grip": value[6:].copy(), f"{side}_gripper": 100 * value[6:]}
        return out

    def play(row: dict[str, np.ndarray]) -> None:
        for side in arms:
            target = _through_loop(np.concatenate([row[f"{side}_arm"], row[f"{side}_grip"]]), faults)
            arms[side] = _servo(arms[side], target, 1 / ROW_HZ, limits)

    return run_direct(report, play, None)


def test_on_an_arm_the_check_does_not_know_each_hand_circles_and_a_flipped_axis_is_caught() -> None:
    """A YAM model on another arm sends YAM's joints as zeros, so no arm's pose at them is the hands': each hand's pose
    circles and the tool points are judged; a loop that sends one axis the wrong way round is named."""
    lines, stand_in = run_tool(flip_y=False, late_rows=0)
    assert set(_status(lines).values()) == {"PASS"}, lines
    assert stand_in.arm is None and stand_in.read_as().startswith("an arm the check does not know")
    lines, _ = run_tool(flip_y=True, late_rows=0)
    assert _status(lines)["direction and size"] == "PROBLEM", lines
    assert "right_eef_xyz y went the other way" in next(line for line in lines if "direction and size" in line)


def test_a_tool_point_that_lags_inside_the_limit_passes() -> None:
    """A tool point 0.2 s behind (a real arm's servo, a slow computer's IK) is inside the lag limit."""
    lines, _ = run_tool(flip_y=False, late_rows=3)
    assert set(_status(lines).values()) == {"PASS"}, lines


def test_a_metal_arm_named_by_its_hand_poses_circles_in_its_joints_and_a_swapped_pair_is_named() -> None:
    """The hands' poses at the joints sent are a Metal arm's: its joints carry the circle, so a loop that drives them
    right passes every check (the client's grip judged, not the millimetre gripper beside it), and one that swaps two
    joints is a PROBLEM for them."""
    lines, stand_in = run_metal(Faults())
    assert set(_status(lines).values()) == {"PASS"}, lines
    assert stand_in.read_as() == "metal arms"
    lines, _ = run_metal(Faults(swap=(3, 5)))  # both wrist joints at 0
    assert _status(lines)["right part moved"] == "PROBLEM", lines


def test_the_arm_is_read_from_the_first_call() -> None:
    """The hands' poses name the client's arm whose pose at the joints sent they are; with no pose, the YAM's parts
    (or the robot `yam`) name a YAM; anything else names no arm."""
    yam_parts = {
        f"{side}_{part}": np.zeros(n) for side in ("right", "left") for part, n in (("arm", 6), ("gripper", 1))
    }
    metal = _arm("metal")
    hand = metal.hand(METAL_POSE[None, :6])[0]
    posed = {"right_arm": METAL_POSE[:6], "right_eef_xyz": hand[:3], "right_eef_rot6": hand[3:]}
    assert {side: arm.name for side, arm in check_arm.identify(yam_parts, None, None).items()} == {
        "right": "yam",
        "left": "yam",
    }
    assert check_arm.identify(yam_parts, "other_embodiment", None) is None
    assert check_arm.identify({"right_arm": np.zeros(7)}, "yam", None) is None  # no YAM has 7 joints
    assert check_arm.identify(posed, None, None)["right"].name == "metal"
    assert check_arm.identify({**posed, "right_arm": np.zeros(6)}, "other_embodiment", None) is None  # YAM's zeros


## An arm the library does not ship, of no vendor's: six revolute joints from `base` to the flange `l6`.
OWN_URDF = """<robot name="own_arm">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="l3"/>
  <link name="l4"/><link name="l5"/><link name="l6"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/><origin xyz="0 0 0.12"/>
    <axis xyz="0 0 1"/><limit lower="-2.8" upper="2.8" effort="20" velocity="2.5"/></joint>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><origin xyz="0 0 0.05"/>
    <axis xyz="0 1 0"/><limit lower="-1.6" upper="1.6" effort="20" velocity="2.5"/></joint>
  <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><origin xyz="0.25 0 0"/>
    <axis xyz="0 1 0"/><limit lower="-2.4" upper="0.2" effort="20" velocity="2.5"/></joint>
  <joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/><origin xyz="0.22 0 0"/>
    <axis xyz="1 0 0"/><limit lower="-2.0" upper="2.0" effort="5" velocity="3.0"/></joint>
  <joint name="j5" type="revolute"><parent link="l4"/><child link="l5"/><origin xyz="0.06 0 0"/>
    <axis xyz="0 1 0"/><limit lower="-1.7" upper="1.7" effort="5" velocity="3.0"/></joint>
  <joint name="j6" type="revolute"><parent link="l5"/><child link="l6"/><origin xyz="0.05 0 0"/>
    <axis xyz="1 0 0"/><limit lower="-2.6" upper="2.6" effort="5" velocity="3.0"/></joint>
</robot>"""
## Where the simulated arms of #OWN_URDF stand: six joints, the wrist bent clear of lining up joints 4 and 6, then the
## grip (0 to 1).
OWN_POSE = np.array([0.1, 0.5, -1.0, 0.1, 0.5, 0.1, 0.5])
## A fine-tune of the arm of #OWN_URDF, as the platform lists it.
OWN = "dream-1-ft:0c0d0e0f-own"
OWN_RECORD = {
    "id": OWN,
    "owner": "mine",
    "base_model": "dreamlabs/dream-1-beta",
    "embodiment": "emb_0c0d0e0f0a0b0c0d",
    "servable": True,
    "created_at": "2026-11-01T00:00:00Z",
    "cameras": None,
    "embodiment_config": None,
}


def own_arm(tmp_path: pathlib.Path) -> str:
    """An arm file of #OWN_URDF, its tool point set as `dreamcli tcp` records it.

    @return The arm file's absolute path.
    """
    arm: dict[str, Any] = {
        "urdf": "own_arm.urdf",
        "base": "base",
        "flange": "l6",
        "tcp": [0.09, 0.0, 0.0],
        "approach": [1.0, 0.0, 0.0],
        "jaw": [0.0, 1.0, 0.0],
        "rest": OWN_POSE[:6].tolist(),
    }
    (tmp_path / "own_arm.urdf").write_text(OWN_URDF)
    digest = arm_file.tool_digest(arm, OWN_URDF.encode())
    arm["tcp_calibration"] = {"status": "calibrated", "at": "2026-11-01T08:00:00+00:00", "digest": digest}
    path = (tmp_path / "own_arm.json").resolve()
    path.write_text(json.dumps(arm))
    return str(path)


def run_ik(
    host: str,
    arm: str,
    pose: np.ndarray,
    record: dict[str, Any],
    solve: str,
    faults: Faults,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[str], check_loop.StandIn]:
    """Two arms with a fine-tune of their own, through the real client with `ik` their arm (head `eef`), 20 rows a
    second on a fake clock: each call sends the joints and the grip, the client's IK solves each hand pose answered to
    joints, and the loop drives those joints (with its faults, #_through_loop).

    @param host The check's host name as the loop's `DREAM_GATEWAY_URL` gives it.
    @param arm The `ik` arm: a name dream ships, or an arm file's path.
    @param pose Where both arms stand: the joints, then the grip (0 to 1).
    @param record The fine-tune, as the platform lists it.
    @param solve The `ik` solve (`dream.client_ik.SOLVES`): `traj`, a station's, or `tick`.
    @param faults What the loop gets wrong between the client's joints and the robot.
    @param monkeypatch Points the client at the stand-in and the platform.
    @return Each line as printed, and the stand-in after the run.
    """
    key = arm if arm in client_ik.ARMS else arm_file.register(arm)
    limits = [tuple(limit) for limit in check_arm.Arm(arm, specs.EMBODIMENTS[key].side_arm("right")).limits]
    platform = FakePlatform([record])
    monkeypatch.setenv("DREAM_API_URL", platform.url)
    monkeypatch.setenv("DREAM_API_KEY", PLATFORM_KEY)
    clock = Clock()
    stand_in = check_loop.StandIn(rate=ROW_HZ, clock=clock)
    monkeypatch.setenv("DREAM_GATEWAY_URL", stand_in.serve(0).replace(client_ik.CHECK_HOST, host))
    arms = {side: pose.copy() for side in ("right", "left")}
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        with dream.DreamClient() as client:
            client.policy({"checkpoint": record["id"], "ik": {"arm": arm, "head": "eef", "solve": solve}})
            for tick in range(int(120 * ROW_HZ)):
                observation: dict[str, Any] = {"prompt": "check the loop"}
                for side, value in arms.items():
                    observation[f"state.{side}_arm"] = value[:-1].copy()
                    observation[f"state.{side}_grip"] = value[-1:].copy()
                client.set_observation({"video.top": _picture(f"top {tick}")}, observation, timestamp=clock.t)
                if not rows:
                    try:
                        action = client.get_action()
                    except dream.DreamServerError:
                        break  # the stand-in has finished or stopped
                    rows = list(zip(np.asarray(action["q_r_chunk"]), np.asarray(action["q_l_chunk"]), strict=True))
                for side, row in zip(("right", "left"), rows.pop(0), strict=True):
                    arms[side] = _servo(arms[side], _through_loop(np.asarray(row), faults), 1 / ROW_HZ, limits)
                clock.t += 1 / ROW_HZ
    finally:
        stand_in.close()
        platform.close()
    return [str(line) for line in check_loop.judge(stand_in)], stand_in


def test_an_arm_of_your_own_is_built_from_its_arm_file_and_turns_inside_its_urdf_limits(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fine-tune of an arm the library does not ship, run with `ik` its arm file: the calls to the check name the
    file, so the check builds the arm from it, and every joint turns inside its URDF's limits, the page draws its chain
    of joints, and every check passes. Calls to any other address name nothing, so the check does not know the arm."""
    path = own_arm(tmp_path)
    lines, stand_in = run_ik(client_ik.CHECK_HOST, path, OWN_POSE, OWN_RECORD, "tick", Faults(), monkeypatch)
    assert set(_status(lines).values()) == {"PASS"}, lines
    assert stand_in.read_as() == f"the arms of {path}, through the client's IK: the lines read the hands"
    assert stand_in.plan is not None and stand_in.plan.arms is not None
    limits = stand_in.plan.arms["right"].limits  # (6 joints, 2)
    targets = stand_in.plan.table()["right_arm"]  # (rows, 6 joints)
    assert np.all(targets >= limits[:, 0]) and np.all(targets <= limits[:, 1])
    assert np.all(np.ptp(targets, axis=0) > math.radians(5)), np.degrees(np.ptp(targets, axis=0))
    shown = stand_in.arms()
    assert shown is not None and shown["arm"] == path
    assert [link["col"] for link in shown["chains"]["right"]["links"] if link["col"] is not None] == list(range(6))
    _, elsewhere = run_ik("localhost", path, OWN_POSE, OWN_RECORD, "tick", Faults(), monkeypatch)
    assert elsewhere.arm is None
    assert elsewhere.read_as().startswith("an arm the check does not know")


def test_the_named_arm_is_read_only_where_its_hand_pose_is_the_calls(tmp_path: pathlib.Path) -> None:
    """The arm a call names is the arm only when its tool pose at the joints sent is the pose sent: a model trained on
    YAM arms is sent zeros for the joints, so the check then knows no arm, and a pose that is another arm's names that
    arm."""
    path = own_arm(tmp_path)
    own = check_arm.Arm("own", specs.EMBODIMENTS[arm_file.register(path)].side_arm("right"))
    hand = own.hand(OWN_POSE[None, :6])[0]  # (9,)
    posed = {"right_arm": OWN_POSE[:6], "right_eef_xyz": hand[:3], "right_eef_rot6": hand[3:]}
    assert check_arm.identify(posed, None, path)["right"].name == path
    assert check_arm.identify(posed, None, None) is None  # no arm the library ships
    assert check_arm.identify({**posed, "right_arm": np.zeros(6)}, "other_embodiment", path) is None
    metal = _arm("metal").hand(METAL_POSE[None, :6])[0]  # (9,)
    on_metal = {"right_arm": METAL_POSE[:6], "right_eef_xyz": metal[:3], "right_eef_rot6": metal[3:]}
    assert check_arm.identify(on_metal, None, path)["right"].name == "metal"


## The Metal's home, where a Metal station's arms start: every joint at 0, the shoulder (joint 2) at its upper limit
## and the elbow (joint 3) at its lower one, and joints 1 and 5 turning about one line, so an IK splits a turn between
## them as it likes; then the grip (0 to 1).
METAL_HOME = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
## A fine-tune of the Metal arms, as the platform lists it.
METAL_RECORD = {**OWN_RECORD, "id": "dream-1-ft:0d0e0f10-metal", "embodiment": "makermods_metal"}


def test_from_the_metal_home_the_clients_ik_is_judged_by_where_the_hands_go(monkeypatch: pytest.MonkeyPatch) -> None:
    """The client's IK reaches each hand pose answered with joints of its own, so the lines read the hands, which go
    where told: every check passes. Read by its joints, the same run is a PROBLEM."""
    lines, stand_in = run_ik(client_ik.CHECK_HOST, "metal", METAL_HOME, METAL_RECORD, "traj", Faults(), monkeypatch)
    assert set(_status(lines).values()) == {"PASS"}, lines
    assert stand_in.plan is not None and stand_in.plan.hands
    assert lines[1].endswith("each hand's move and turn followed its own command"), lines
    stand_in.plan.hands = False
    joints = [str(line) for line in check_loop.judge(stand_in)]
    assert "PROBLEM" in _status(joints).values(), joints


def test_a_joint_the_loop_flips_after_the_clients_ik_sends_the_hand_the_other_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loop that flips the sign of the client's joint 1 sends each hand short of its moves and turns (the client's
    IK, solving from the joints it reads back, makes up part of it with the others): a hand PROBLEM.

    Not yet: from this pose the circle every joint draws leaves every hand wrong, the play rate fit then reads a rate
    PROBLEM too, and a loop-wide PROBLEM moves no joint alone."""
    lines, stand_in = run_ik(
        client_ik.CHECK_HOST, "metal", METAL_POSE, METAL_RECORD, "traj", Faults(flip=0), monkeypatch
    )
    assert _status(lines)["direction and size"] == "PROBLEM", lines
    assert "right hand moved x0." in next(line for line in lines if "direction and size" in line), lines
    assert stand_in.plan is not None


def test_a_hands_turn_is_read_about_the_bases_axes() -> None:
    """A hand turned 10 degrees about the base's z from a tilted start reads 10 degrees about z, none about x or y."""

    def rot6(m: np.ndarray) -> np.ndarray:
        return np.concatenate([m[:, 0], m[:, 1]])

    c, s = math.cos(math.radians(10)), math.sin(math.radians(10))
    about_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    tilted = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # 90 degrees about x
    turned = check_arm.turned(np.stack([rot6(tilted), rot6(about_z @ tilted)]), rot6(tilted))
    assert np.allclose(turned, [[0.0, 0.0, 0.0], [0.0, 0.0, math.radians(10)]])


def test_a_first_call_that_names_no_arm_is_refused_before_anything_moves() -> None:
    """A loop on an arm the check does not know that sends no hand pose is refused at its first call: nothing moves,
    and the LOOP line says what to send."""
    from dream import check

    stand_in = check_loop.StandIn(rate=ROW_HZ, clock=Clock())
    observation = {"state.right_arm": [0.0] * 7, "state.right_gripper": [0.5]}
    reply = stand_in.answer({"session_id": None, "observation": observation}, None, None)
    assert reply["actions"] is None and stand_in.plan is None
    loop = check.loop_verdict(check_loop.judge(stand_in))
    assert loop.startswith("LOOP:   PROBLEM hold: stopped: the first call names no arm the check knows"), loop
    assert "the client's `ik` config key" in loop and "`yam`" in loop


def test_hand_poses_answered_are_the_arms_at_the_joints_answered() -> None:
    """On a known arm the answer's hand poses are where its joints take the hand, every row, so a loop that runs the
    tool poses (the client's `ik` head `eef`) moves the arm as one that runs the joints."""
    metal = _arm("metal")
    hand = metal.hand(METAL_POSE[None, :6])[0]
    state = {"right_arm": METAL_POSE[:6], "right_eef_xyz": hand[:3], "right_eef_rot6": hand[3:], "right_grip": [0.5]}
    plan = check_loop.Plan(state, arms={"right": metal}, hands=False)
    plan.extend([check_loop.Channel("right_arm", 4, "joint")], plan.rows)
    table = plan.table()
    poses = metal.hand(table["right_arm"])  # (rows, 9)
    assert np.allclose(poses[:, :3], table["right_eef_xyz"]) and np.allclose(poses[:, 3:], table["right_eef_rot6"])
    assert check_loop._across(table["right_eef_xyz"]) > check_loop.CIRCLE_PASS  # the hand draws its circle


@pytest.mark.parametrize(
    ("posed", "said"),
    [
        (False, "no joint moved: your loop does not send the robot the commands it gets"),
        (True, "no hand moved: your loop does not send the robot the commands it gets (on an arm the check does not"),
    ],
)
def test_arms_that_never_move_hear_their_loop_does_not_send_the_commands(posed: bool, said: str) -> None:
    """A loop that sends nothing on to the robot: on a known arm no joint moved, on another no hand did."""
    still = {
        f"{side}_{part}": value
        for side in ("right", "left")
        for part, value in (("arm", POSE[:6]), ("gripper", POSE[6:]))
    }
    if posed:
        still |= {f"{side}_eef_xyz": np.array([0.3, 0.0, 0.2]) for side in ("right", "left")}
        still |= {f"{side}_eef_rot6": np.eye(3)[:2].reshape(6) for side in ("right", "left")}
    lines, _ = run_direct(lambda: still, lambda row: None, "other_embodiment" if posed else None)
    assert _status(lines)["hold"] == "PASS" and _status(lines)["right part moved"] == "PROBLEM", lines
    assert said in lines[1], lines


def test_joint_near_a_limit_swings_away_from_it() -> None:
    plan = check_loop.Plan(
        {"right_arm": [0.0, 0.0, 1.3, 0.0, 0.3, 0.0], "right_gripper": [1.0]}, arms={"right": _arm("yam")}, hands=False
    )
    plan.extend([check_loop.Channel("right_arm", 1, "joint")], plan.rows)
    shoulder = next(s for s in plan.segments if s.kind == "joint" and s.cols == (1,))
    swing = [plan.offsets(shoulder.start + r)["right_arm"][1] for r in range(shoulder.rows)]
    assert min(swing) == 0.0 and math.isclose(max(swing), check_loop.SWING, rel_tol=1e-3)


def test_each_part_moved_alone_is_said_in_words() -> None:
    """Each joint of a 6-joint arm, each tool point axis and a gripper moved alone gets its line as its step starts."""
    joints = check_loop.Plan(
        {"right_arm": METAL_POSE[:6], "right_gripper": METAL_POSE[6:]}, arms={"right": _arm("metal")}, hands=False
    )
    alone = [check_loop.Channel("right_arm", col, "joint") for col in range(6)]
    joints.extend([*alone, check_loop.Channel("right_gripper", 0, "grip")], joints.rows)
    find = "moves alone, to find the part at fault:"
    assert [s.say() for s in joints.segments] == [
        "holding still: nothing should move",
        "each hand draws an 8 cm circle twice while every joint turns; each gripper closes and opens",
        *(f"right_arm joint {n} {find} one 10-degree swing out and back" for n in range(1, 7)),
        f"right_gripper {find} open, close, open",
    ]
    hands = check_loop.Plan({"right_eef_xyz": [0.3, 0.0, 0.2], "right_gripper": [1.0]}, arms=None, hands=False)
    hands.extend([check_loop.Channel("right_eef_xyz", col, "xyz") for col in range(3)], hands.rows)
    assert [s.say() for s in hands.segments[2:]] == [f"right_eef_xyz {a} {find} 4 cm out and back" for a in "xyz"]


## A Metal arm folded at home: the shoulder and the elbow at their limits, the rest near zero.
METAL_FOLDED = np.array([0.0321, -0.02137, 0.0116, 0.00489, 0.07216, -0.01945])


@pytest.mark.parametrize(
    ("arm", "pose"),
    [
        ("metal", METAL_FOLDED),
        ("metal", METAL_HOME[:6]),
        ("metal", METAL_POSE[:6]),
        ("yam", np.zeros(6)),
        ("yam", POSE[:6]),
    ],
)
def test_each_hand_draws_the_whole_circle_from_where_the_arm_stands(arm: str, pose: np.ndarray) -> None:
    """From a Metal arm's home, folded at its shoulder's and elbow's limits, the first three joints could draw only
    1 to 3 cm of the 8 cm circle (and every line passed while its hands hardly moved); every arm draws
    the whole circle now, at most 20 degrees a joint, inside its limits."""
    plan = check_loop.Plan({"right_arm": pose, "right_gripper": [0.5]}, arms={"right": _arm(arm)}, hands=False)
    move = plan.circle["right_arm"]  # (rows, joints)
    limits = _arm(arm).limits
    assert plan.across["right"] >= 0.075, plan.across
    assert np.max(np.abs(move)) <= check_loop.CIRCLE_MAX + 1e-4  # taking back a bump may add a hair
    q = pose + move
    assert np.all((q >= np.minimum(pose, limits[:, 0]) - 1e-9) & (q <= np.maximum(pose, limits[:, 1]) + 1e-9))


def _circle_calls(plan: check_loop.Plan, share: float, every: int) -> list[check_loop.Call]:
    """Calls every `every` rows through the circle, the arm `share` of the way to each target."""
    together = next(s for s in plan.segments if s.kind == "together")
    calls = []
    for row in range(together.start, together.end, every):
        u = min(row - together.start, len(plan.circle["right_arm"]) - 1)
        state = {"right_arm": plan.start["right_arm"] + share * plan.circle["right_arm"][u]}
        calls.append(check_loop.Call(t=row / 20, state=state, row=row, window=()))
    return calls


def test_hands_that_trace_under_6_cm_of_the_circle_fail_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The circle check reads each hand's tool point at the calls: the whole circle passes; a hand that goes a third
    of the way round its circle's size fails; too few calls cannot see it; a circle the check could not draw fails."""
    plan = check_loop.Plan(
        {"right_arm": METAL_FOLDED, "right_gripper": [0.5]}, arms={"right": _arm("metal")}, hands=False
    )
    whole = check_loop._circle_line(plan, _circle_calls(plan, 1.0, 30), 20.0)
    assert whole.status == "PASS" and whole.detail.startswith("each hand traced its 8 cm circle"), whole
    small = check_loop._circle_line(plan, _circle_calls(plan, 0.3, 30), 20.0)
    assert small.status == "PROBLEM" and "the right hand traced under 6 cm of the 8 cm circle" in small.detail, small
    # A call each lap, always where the hand starts its lap, cannot see the circle.
    assert check_loop._circle_line(plan, _circle_calls(plan, 1.0, 120), 20.0).status == "SKIP"
    monkeypatch.setattr(check_loop, "CIRCLE_MAX", math.radians(3))
    tight = check_loop.Plan(
        {"right_arm": METAL_FOLDED, "right_gripper": [0.5]}, arms={"right": _arm("metal")}, hands=False
    )
    line = check_loop._circle_line(tight, _circle_calls(tight, 1.0, 30), 20.0)
    assert line.status == "PROBLEM" and line.detail.startswith("the check could not draw the 8 cm circle"), line


def test_a_hand_that_traces_a_flat_band_fails_the_circle() -> None:
    """On a reBot whose shoulder and elbow stood at their limits, each hand went round only the circle's upper half,
    a 9 by 4 cm band whose points spanned 9 cm, and the circle passed. The circle now starts at its bottom, so a hand
    never goes below where it stood; a hand that still goes round only part of it is fit with an ellipse, and the line
    names the axis that fell short of its range or ran late."""
    plan = check_loop.Plan({"right_eef_xyz": [0.3, 0.0, 0.2], "right_gripper": [1.0]}, arms=None, hands=False)
    together = next(s for s in plan.segments if s.kind == "together")
    ring = plan.circle["right_eef_xyz"]  # (rows, 3 xyz)
    assert np.min(ring[:, 2]) >= 0.0 and math.isclose(np.max(ring[:, 2]), 2 * check_loop.CIRCLE_R, abs_tol=1e-3)

    def calls(highest: float, late: int) -> list[check_loop.Call]:
        """Calls every 30 rows, the hand on its circle but never more than `highest` (metres) above where it stood,
        its z `late` rows behind its x and y."""
        out = []
        for row in range(together.start, together.end, 30):
            at = row - together.start
            xy, z = ring[min(at, len(ring) - 1)], ring[int(np.clip(at - late, 0, len(ring) - 1))]  # (3 xyz,) each
            state = {"right_eef_xyz": [0.3, 0.0, 0.2] + np.array([xy[0], xy[1], min(z[2], highest)])}
            out.append(check_loop.Call(t=row / 20, state=state, row=row, window=()))
        return out

    whole = check_loop._circle_line(plan, calls(1.0, 0), 20.0)
    assert whole.status == "PASS" and "the right hand 8.0 by 8.0 cm" in whole.detail, whole
    band = check_loop._circle_line(plan, calls(check_loop.CIRCLE_R, 0), 20.0)
    assert band.status == "PROBLEM" and band.detail.startswith("the right hand traced a flat band"), band
    assert "the right hand's z moved x0.50 of its commands; the arm did not follow those commands" in band.detail, band
    assert "behind" not in band.detail, band
    # a reBot after the upward circle: z covered its 8 cm but ran behind x and y (joints on low gains)
    lagged = check_loop._circle_line(plan, calls(1.0, 12), 20.0)
    assert lagged.status == "PROBLEM" and lagged.detail.startswith("the right hand traced a flat band"), lagged
    assert "the right hand's z ran 0.60 s behind its x and y;" in lagged.detail, lagged
    assert "moved" not in lagged.detail, lagged


def test_every_run_saves_its_trace(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With no --trace, a run's trace goes to ~/.dreamlabs, its lines with it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    path = check_loop.trace_path()
    assert path.parent == tmp_path / ".dreamlabs" and path.name.startswith("check-"), path
    lines, stand_in = run(Faults(), SIM_YAM, tmp_path, monkeypatch)
    check_loop.save_trace(stand_in, check_loop.judge(stand_in), path)
    saved = json.loads(path.read_text())
    assert [line["check"] for line in saved["lines"]] == list(check_loop.CHECKS) and saved["calls"], saved.keys()


def test_together_motion_lasts_8_5_s_at_30_rows_a_second_on_any_arm() -> None:
    joints = check_loop.Plan(
        {"right_arm": [0.0] * 6, "right_gripper": [1.0]}, arms={"right": _arm("yam")}, hands=False
    )
    hands = check_loop.Plan({"right_eef_xyz": [0.3, 0.0, 0.2], "right_gripper": [1.0]}, arms=None, hands=False)
    for plan in (joints, hands):
        assert (plan.rows - check_loop.HORIZON) / check_loop.PLAN_HZ == 8.5
        assert [s.kind for s in plan.segments] == ["hold", "together"]


def test_together_motion_circles_each_gripper_and_moves_every_joint() -> None:
    arm, grip = POSE[:6], POSE[6:]
    yam = _arm("yam")
    plan = check_loop.Plan(
        {"right_arm": arm, "right_gripper": grip, "left_arm": arm, "left_gripper": grip},
        arms={"right": yam, "left": yam},
        hands=False,
    )
    n, fade = check_loop.TOGETHER_ROWS, check_loop.FADE_ROWS
    for side in ("right", "left"):
        move = plan.circle[f"{side}_arm"]  # (rows, 6 joints)
        # Every joint moves enough to judge: by the circle, or by one out-and-back on top (the wrist's three).
        assert np.min(np.max(np.abs(move), axis=0)) >= check_loop.CIRCLE_MIN
        tool = yam.hand(arm + move)[:, :3]  # (rows, 3 xyz)
        laps = tool[fade : n - fade]  # (rows at full speed, 3 xyz)
        radius = np.linalg.norm(laps - laps.mean(axis=0), axis=1)
        assert 0.033 < radius.min() and radius.max() < 0.046, (side, radius.min(), radius.max())  # a near-circle
        assert np.allclose(tool[0], tool[n])  # back where it started
        grips = plan.circle[f"{side}_gripper"]
        assert grips.min() < 0.2 and grips.max() > 0.9  # it closes and opens


def test_on_a_metal_arm_each_hand_circles_inside_its_urdf_limits() -> None:
    """On a Metal arm the circle is the Metal's: every joint of both arms moves enough to judge, the hand goes round a
    near-circle, and no joint comes within 2 degrees of a limit (the shoulder starts 6 degrees inside one)."""
    metal = _arm("metal")
    arm, grip = METAL_POSE[:6], METAL_POSE[6:]
    plan = check_loop.Plan(
        {"right_arm": arm, "right_gripper": grip, "left_arm": arm, "left_gripper": grip},
        arms={"right": metal, "left": metal},
        hands=False,
    )
    lower, upper = metal.limits[:, 0], metal.limits[:, 1]  # (6 joints,) each
    n, fade = check_loop.TOGETHER_ROWS, check_loop.FADE_ROWS
    for side in ("right", "left"):
        move = plan.circle[f"{side}_arm"]  # (rows, 6 joints)
        assert np.min(np.max(np.abs(move), axis=0)) >= check_loop.CIRCLE_MIN, np.degrees(np.max(np.abs(move), axis=0))
        assert np.all(arm + move >= lower + math.radians(2) - 1e-9)
        assert np.all(arm + move <= upper - math.radians(2) + 1e-9)
        tool = metal.hand(arm + move)[fade : n - fade, :3]  # (rows at full speed, 3 xyz)
        radius = np.linalg.norm(tool - tool.mean(axis=0), axis=1)
        assert 0.02 < radius.min() and radius.max() < 0.05, (side, radius.min(), radius.max())


def test_the_question_says_each_hand_circles_and_the_flags_are_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    """The person is told each hand circles while every joint turns, whatever the arm; `--arm` and `--eef` are gone."""
    from dream import check

    asked: list[str] = []

    def no(text: str) -> bool:
        asked.append(text)
        return False

    monkeypatch.setattr(check_loop, "confirm", no)
    assert check.main([]) == 2
    assert "each hand circles twice, 8 cm across, while every joint turns." in asked[0]
    for flag in (["--arm", "yam"], ["--eef"]):
        with pytest.raises(SystemExit):
            check.main(flag)
    assert len(asked) == 1


def test_without_a_recording_the_check_runs_and_skips_only_the_model(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Most people debugging a loop have recorded nothing: the check reaches its question without a recording, MODEL is
    a SKIP that says how to add one, FAULT clears the loop and the run exits 0; a folder given as `--recording` must
    hold a recording."""
    from dream import check

    asked: list[str] = []

    def no(text: str) -> bool:
        asked.append(text)
        return False

    monkeypatch.setattr(check_loop, "confirm", no)
    assert check.main([]) == 2 and len(asked) == 1
    assert check.UNRECORDED.startswith("MODEL:  SKIP ")
    assert not check.UNRECORDED.startswith((check.UNASKED, check.UNREAD)), "an exit-1 SKIP"
    assert check.fault("LOOP:   PASS    lag", "INPUTS: PASS    ok", check.UNRECORDED) == (
        "FAULT:  not your loop; the model was not checked"
    )
    assert check.main(["--recording", str(tmp_path)]) == 2 and len(asked) == 1
    assert f"{tmp_path}: no meta/info.json; give one session of your recordings" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("typed", "starts"),
    [
        ("the workspace is clear", True),
        (" The workspace is clear ", True),
        ("yes", False),
        ("the workspace", False),
        ("", False),
        (None, False),
    ],
)
def test_nothing_starts_until_the_workspace_is_clear(
    typed: str | None, starts: bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def answer(prompt: str) -> str:
        if typed is None:
            raise EOFError  # no terminal to type in
        return typed

    monkeypatch.setattr("builtins.input", answer)
    assert check_loop.confirm("The arms move.") is starts
    assert ("Not started: nothing moved." in capsys.readouterr().out) is not starts


def test_check_serves_nothing_until_the_words_are_typed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing is served on a yes; the question tells an agent it types the words only after the person has."""
    from dream import check

    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    monkeypatch.setattr(check_loop.StandIn, "serve", lambda *args: pytest.fail("served before the words"))
    assert check.main([]) == 2
    asked = capsys.readouterr().out
    assert "An agent types these words only after the person at the robot has typed them to it." in asked
    assert check_loop.main([]) == 2


def test_the_page_shows_each_call_and_yes_records_the_inputs_the_client_names(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every call the loop makes shows on the check's page; the person's yes records the id of the client's reminder,
    so the loop's next run has none."""
    from dream import check, input_check
    from dream.debug_view import DebugView

    with DebugView(None, 0) as view, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, stand_in = run(Faults(), SIM_YAM, tmp_path, monkeypatch, view=view)
        shown = view._list()
    [reminder] = [str(w.message) for w in caught if "what the model sees" in str(w.message)]
    assert stand_in.inputs is not None and reminder.endswith(f"{input_check.COMMAND} {stand_in.inputs}")
    assert len(shown) > 1 and all(trip["error"] is None for trip in shown)
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")
    assert check.ask_inputs(stand_in.inputs, view.url, check_loop.camera_rate(stand_in)).startswith("INPUTS: PASS")
    assert input_check.checked(stand_in.inputs)


@pytest.mark.parametrize(
    ("typed", "status"), [("no", "INPUTS: PROBLEM"), ("", "INPUTS: SKIP"), (None, "INPUTS: SKIP")]
)
def test_only_yes_records_the_inputs(
    typed: str | None, status: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from dream import check, input_check

    def answer(prompt: str) -> str:
        if typed is None:
            raise EOFError  # no terminal to type in
        return typed

    monkeypatch.setattr("builtins.input", answer)
    assert check.ask_inputs("0123456789ab", "http://127.0.0.1:8643/", _rate(5, 1.5)).startswith(status)
    assert "The person who looked answers, never an agent." in capsys.readouterr().out
    assert not input_check.checked("0123456789ab")
    assert check.ask_inputs(None, "http://127.0.0.1:8643/", _rate(5, 1.5)) == "INPUTS: SKIP    your loop made no call"


def test_a_camera_history_at_the_wrong_rate_is_an_inputs_problem_without_asking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dream import check, input_check

    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("asked although the rate decided it"))
    line = check.ask_inputs("0123456789ab", "http://127.0.0.1:8643/", _rate(30, 1.5))
    assert line.startswith("INPUTS: PROBLEM at least 6.0 frames a second per camera"), line
    assert not input_check.checked("0123456789ab")
    assert check.fault("LOOP:   PASS    lag", line, "MODEL:  PASS    ok").startswith("FAULT:  your control loop")


def _rate(history_hz: float, every_s: float, still: bool = False) -> check_loop.Line:
    """camera_rate over 15 s of calls `every_s` apart, each sending the newest 9 frames of a history that keeps
    `history_hz` frames a second (one picture on every frame when `still`)."""
    stand_in = check_loop.StandIn(clock=Clock())
    for t in np.arange(0, 15, every_s):
        kept = int(t * history_hz) + 1
        window = tuple(0 if still else n for n in range(max(0, kept - 9), kept))
        stand_in.calls.append(check_loop.Call(float(t), {}, 0, window))
    return check_loop.camera_rate(stand_in)


@pytest.mark.parametrize(
    ("history_hz", "every_s", "still", "said"),
    [
        (5, 1.5, False, "PASS     camera rate         5.0 frames a second per camera, the 5"),
        (5.4, 1.5, False, "PASS     camera rate"),  # within 10%
        (4, 1.5, False, "PROBLEM  camera rate         4.0 frames a second per camera, not the 5"),
        (1, 1.5, False, "PROBLEM  camera rate         1.0 frames a second per camera"),  # at each call only
        (30, 1.5, False, "PROBLEM  camera rate         at least 6.0 frames a second per camera"),  # in ms
        (5, 2.4, False, "SKIP     camera rate         not counted: calls came 2.4 s apart, more than the 1.6 s"),
        (5, 1.5, True, "SKIP     camera rate         not counted: the frames repeat (a still scene"),
    ],
)
def test_the_camera_rate_counts_the_frames_new_to_each_call(
    history_hz: float, every_s: float, still: bool, said: str
) -> None:
    """Frames new to each call per second against the 5 the model reads; a history kept at every call, or of every
    tick (a timestamp in milliseconds), is a PROBLEM; still pictures leave a reminder, and calls too far apart say how
    far, since the reminder would ask for what the loop already does."""
    line = _rate(history_hz, every_s, still)
    assert str(line).startswith(said), line
    assert line.status == "PASS" or line.detail.endswith(check_loop.FRAME_HINT) == ("apart" not in line.detail)


def test_a_loop_paced_by_the_client_fills_the_camera_history_at_5_frames_a_second(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through the real client: ticks timed in seconds keep 5 frames a second, counted when the loop asks after 30 of
    48 rows (the client's pacing); a loop that plays every row first asks too seldom to count them."""
    _, paced = run(Faults(), SIM_YAM, tmp_path, monkeypatch, refill=18)
    assert str(check_loop.camera_rate(paced)).startswith("PASS     camera rate         5.0 frames a second per camera")
    _, whole = run(Faults(), SIM_YAM, tmp_path, monkeypatch)
    assert check_loop.camera_rate(whole).detail.startswith("not counted: calls came 2.4 s apart")


def test_inputs_wrong_on_the_page_is_the_loops_fault() -> None:
    from dream import check

    passed = "LOOP:   PASS    lag", "MODEL:  PROBLEM far"
    assert check.fault(passed[0], "INPUTS: PROBLEM wrong", passed[1]).startswith("FAULT:  your control loop")
    assert check.fault(passed[0], "INPUTS: PASS    ok", passed[1]).startswith("FAULT:  the model")


def test_a_key_the_site_refuses_is_no_verdict_on_the_model(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused key, before the model's first call (here the platform's) or at it (the model's `DA-1000`), is a MODEL
    SKIP that says why, never a model PROBLEM, so FAULT does not blame the model."""
    from dream import check
    from dream.errors import DreamAuthError

    def refused(*args: object) -> None:
        raise DreamAuthError("This api key is not valid.", code="invalid_api_key")

    monkeypatch.setattr(check.check_model, "readable", lambda *args: ({}, {}))
    monkeypatch.setattr(check, "ask_model", refused)
    line = check.model_line(tmp_path, "dreamlabs/dream-1-beta", ["overhead"], False, True, 0)
    assert line == f"{check.UNASKED} This api key is not valid."
    verdict = check.fault("LOOP:   PASS    lag", "INPUTS: PASS    ok", line)
    assert verdict == "FAULT:  not your loop; the model was not checked"
    # The model refusing the key at its first call (`DA-1000`) is no verdict on the model either.
    unknown = DreamAuthError("the key was refused", code="DA-1000", reason="unknown_key")
    line = check.model_verdict([], unknown)
    assert line == f"{check.UNASKED} the key was refused", line
    assert check.fault("LOOP:   PASS    lag", "INPUTS: PASS    ok", line) == verdict


def test_a_recording_the_check_cannot_read_is_said_before_the_arms_move_and_skips_only_the_model(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A recording in another layout (an `info.json` with no `chunks_size`, as a recorder of its own may write) is
    named before the question, LOOP and INPUTS still run, and MODEL is a SKIP saying why, never a traceback after the
    motion."""
    from dream import check

    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps({"fps": 30}))
    monkeypatch.setattr(check_loop, "confirm", lambda text: False)
    assert check.main(["--recording", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "meta/info.json has no codebase_version, chunks_size, " in err
    assert "MODEL will read SKIP; LOOP and INPUTS still run." in err
    monkeypatch.setattr(check, "ask_model", lambda *args: pytest.fail("asked the model about an unread recording"))
    line = check.model_line(tmp_path, "dreamlabs/dream-1-beta", ["overhead"], False, False, 0)
    assert line.startswith(f"{check.UNREAD} meta/info.json has no codebase_version, chunks_size, "), line
    assert check.fault("LOOP:   PASS    lag", "INPUTS: PASS    ok", line) == (
        "FAULT:  not your loop; the model was not checked"
    )
    missing = check.check_model.Unreadable("episode_000000.mp4: no such video")  # found once the body is known
    monkeypatch.setattr(check.check_model, "readable", lambda *args: ({}, {}))
    monkeypatch.setattr(check, "ask_model", lambda *args: (_ for _ in ()).throw(missing))
    line = check.model_line(tmp_path, "dreamlabs/dream-1-beta", ["overhead"], False, False, 0)
    assert line == f"{check.UNREAD} episode_000000.mp4: no such video"


def test_a_request_the_model_refuses_is_the_loops_inputs_not_the_model(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model that refuses the request (`DA-4000`, here a YAM loop that sends no hand poses) judged nothing: the
    MODEL line quotes it and FAULT names the loop's inputs, never the model."""
    from dream import check
    from dream.errors import DreamBadRequest

    missing = DreamBadRequest("observation is missing state.right_eef_xyz", code="DA-4000", reason="malformed")
    monkeypatch.setattr(check.check_model, "readable", lambda *args: ({}, {}))
    monkeypatch.setattr(check, "ask_model", lambda *args: ([], missing))
    line = check.model_line(tmp_path, "dreamlabs/dream-1-beta", ["overhead"], False, True, 0)
    assert line == f"{check.REFUSED} observation is missing state.right_eef_xyz"
    verdict = check.fault("LOOP:   PASS    lag", "INPUTS: PASS    ok", line)
    assert verdict.startswith("FAULT:  your loop's inputs: the model refused them")


def test_the_model_check_sends_hand_poses_only_when_the_loop_did() -> None:
    """The stand-in notes whether the loop's first call carried each hand's pose (what the client adds with `ik`), so
    the model check asks the model with what the loop sends."""
    parts = (("arm", 6), ("gripper", 1))
    joints = {f"state.{side}_{part}": np.zeros(width) for side in ("right", "left") for part, width in parts}
    for poses, observation in ((False, joints), (True, {**joints, "state.right_eef_xyz": np.zeros(3)})):
        stand_in = check_loop.StandIn(clock=Clock())
        stand_in.answer({"session_id": None, "observation": observation}, None, None)
        assert stand_in.poses is poses


@pytest.mark.parametrize("command", [[], ["no-such-program-of-yours"]])
def test_a_loop_command_it_cannot_run_is_refused_before_the_question(
    command: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from dream import check

    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("asked before the command was checked"))
    assert check.main(["--", *command]) == 2


def test_the_loop_after_dashes_runs_on_the_address_and_ending_first_is_the_loops_fault(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """The command after `--` runs once the words are typed, with DREAM_GATEWAY_URL the stand-in's address and its
    output passed through; a loop that ends before the motion does is a LOOP PROBLEM."""
    from dream import check, input_check

    monkeypatch.setattr("builtins.input", lambda prompt: check_loop.START)
    reminder = f"import warnings; warnings.warn('{input_check.REMINDER} here')"
    loop = [sys.executable, "-c", f"import os; {reminder}; print('my loop at', os.environ['DREAM_GATEWAY_URL'])"]
    assert check.main(["--port", "0", "--view-port", "0", "--", *loop]) == 1
    out, err = capfd.readouterr()
    url = re.search(r"Running your loop with DREAM_GATEWAY_URL=(\S+):", out)[1]
    assert f"my loop at {url}\n" in out
    assert input_check.REMINDER not in err, "the reminder to run the check printed inside the check"
    assert "LOOP:   PROBLEM your loop exited (0) before the motion ended" in out
    assert "FAULT:  your control loop" in out

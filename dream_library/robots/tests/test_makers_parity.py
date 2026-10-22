"""The Makers driver against the driver it is ported from: the same CAN frames, byte for byte, and the same joints.

`data/makers_parity.json` is a snapshot of the source Makers driver run through one scenario on its own fake bus
with a fake clock:
an arm with a joint a whole turn off, the startup ramp, tracking, the soft-limit clamp, the gripper closing on an
object and letting go, missed replies, a stream that stops, the return to rest and torque off. This test replays the
scenario through the shared thread (#robots.motor_arm) on the Makers table (#robots.makers) over the ported fake
(`makers_fakes`), stepping time the way the source's run does, and requires every step's frames and every tick's
published joints to be the source's. The source calls its park "home"; the thread calls it rest. The source sends the
gripper all the way to its open end, which the thread keeps `grip_margin_mm` short of on purpose, so the
replay runs the table with no margin.
"""

import dataclasses
import hashlib
import json
import pathlib
from typing import Any

import makers_fakes
import numpy as np
import pytest

from robots import config, makers, motor_arm

SNAPSHOT = pathlib.Path(__file__).parent / "data" / "makers_parity.json"

## The arm's channel; it appears in no frame.
CHANNEL = "can_right"

## The source's gripper ends and holding torque: its FollowerConfig defaults, which are this package's ArmConfig's.
GRIPPER_DEG = (-96.0, -2.5)
HOLD_TORQUE_NM = 0.5

## Most ticks a return to rest is given, as in the snapshot script.
HOME_MAX_TICKS = 3000


def ramp(start: list[float], end: list[float], ticks: int) -> list[list[float]]:
    """`ticks` targets from `start` to `end`, one per tick: the snapshot script's ramp(), the same arithmetic."""
    return [[a + (b - a) * (i + 1) / ticks for a, b in zip(start, end, strict=True)] for i in range(ticks)]


def replay(snapshot: dict[str, Any]) -> tuple[list[dict[str, Any]], list[list[float]], motor_arm.MotorArm]:
    """Runs the snapshot's scenario through `robots.motor_arm` on the Makers table.

    @return Per step the frames sent and their digest, per tick the published joints, and the arm.
    """
    clock = makers_fakes.FakeClock()
    factory = makers_fakes.FakeCanFactory(clock)
    cfg = config.ArmConfig(
        can_channel=CHANNEL,
        gripper_open_deg=GRIPPER_DEG[0],
        gripper_closed_deg=GRIPPER_DEG[1],
        gripper_hold_torque_nm=HOLD_TORQUE_NM,
    )
    spec = dataclasses.replace(makers.ARM, grip_margin_mm=0.0)
    driver = motor_arm.MotorArm(spec, "right", cfg, opener=factory, clock=clock, sleep=clock.sleep)
    follower = driver.chain
    joints: list[list[float]] = []

    def frames() -> list[tuple[int, bytes]]:
        return factory.buses[CHANNEL].frames if CHANNEL in factory.buses else []

    def tick() -> None:
        t0 = clock()
        driver.tick()
        joints.append([float(x) for x in driver.latest()[0]])
        clock.sleep(max(0.0, driver.period - (clock() - t0)))

    steps: list[dict[str, Any]] = []
    for step in snapshot["scenario"]:
        sent = len(frames())
        op = step["op"]
        if op == "pose":
            for can_id, deg in step["deg"].items():
                factory.bus(CHANNEL).motors[int(can_id)].position_deg = deg
        elif op == "open":
            follower.open()
        elif op == "enable":
            follower.enable()
        elif op == "tick":
            for _ in range(step["n"]):
                tick()
        elif op == "track":
            for _ in range(step["ticks"]):
                driver.set_target(np.asarray(step["q"]))
                tick()
        elif op == "ramp":
            for target in ramp(step["from"], step["to"], step["ticks"]):
                driver.set_target(np.asarray(target))
                tick()
        elif op == "obstacle":
            factory.buses[CHANNEL].motors[step["id"]].obstacle_deg = step["deg"]
        elif op == "drop_every":
            factory.buses[CHANNEL].drop_every = step["n"]
        elif op == "home":
            driver.request_rest()
            for _ in range(HOME_MAX_TICKS):
                tick()
                if driver.rested.is_set():
                    break
        elif op == "close":
            follower.close(disable_torque=True)
        else:
            raise AssertionError(f"unknown step {op!r}")
        lines = [f"{can_id}:{data.hex()}" for can_id, data in frames()[sent:]]
        steps.append(
            {
                "frames": len(lines),
                "sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
                "lines": lines,
            }
        )
    return steps, joints, driver


@pytest.fixture(scope="module")
def snapshot() -> dict[str, Any]:
    """The source's run."""
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ours(snapshot: dict[str, Any]) -> tuple[list[dict[str, Any]], list[list[float]], motor_arm.MotorArm]:
    """This package's run of the same scenario."""
    return replay(snapshot)


def test_the_snapshot_is_of_the_pinned_robots_commit(snapshot: dict[str, Any]) -> None:
    """The fixture says which commit of the source it holds, the one the port's docstrings cite."""
    assert snapshot["robots_commit"] == "a7da8d99539d46bcf8e8ae1406433841ce172897"
    assert snapshot["robots_commit"][:8] in str(makers.__doc__)


def test_every_step_sends_the_sources_frames_byte_for_byte(
    snapshot: dict[str, Any], ours: tuple[list[dict[str, Any]], list[list[float]], motor_arm.MotorArm]
) -> None:
    """Handshake, wrap read, enable, every tick's seven MIT frames (gains and the gripper hold's included), the
    queries a missed reply costs, the return to rest and the disables: the same frames in the same order."""
    steps, _, _ = ours
    assert len(steps) == len(snapshot["steps"])
    for i, (mine, theirs) in enumerate(zip(steps, snapshot["steps"], strict=True)):
        step = snapshot["scenario"][i]
        assert (mine["frames"], mine["sha256"]) == (theirs["frames"], theirs["sha256"]), (
            f"step {i} ({step['op']}): {mine['frames']} frames here, {theirs['frames']} in the source; "
            f"first frames here: {mine['lines'][:8]}"
        )
    assert sum(step["frames"] for step in steps) > 4000


def test_every_tick_publishes_the_sources_joints(
    snapshot: dict[str, Any], ours: tuple[list[dict[str, Any]], list[list[float]], motor_arm.MotorArm]
) -> None:
    """The pose each tick publishes, `[6 joints rad, gripper mm]` float32, is the source's to the bit."""
    _, joints, _ = ours
    theirs = np.asarray(snapshot["joints"], dtype=np.float32)
    mine = np.asarray(joints, dtype=np.float32)
    assert mine.shape == theirs.shape
    differs = np.argwhere(mine != theirs)
    assert differs.size == 0, (
        f"first tick that differs: {differs[0][0]}, {mine[differs[0][0]]} vs {theirs[differs[0][0]]}"
    )


def test_the_run_ends_where_the_sources_does(
    snapshot: dict[str, Any], ours: tuple[list[dict[str, Any]], list[list[float]], motor_arm.MotorArm]
) -> None:
    """The park reached rest (the source's "at home"), the replies missed are the source's count, and every motor is
    off."""
    _, _, driver = ours
    assert snapshot["home_result"] == [True, "at home"]
    assert driver.rest_result == (True, "at rest")
    assert driver.chain.drops() == snapshot["drops"] > 0
    assert not driver.chain.bus.torque_enabled

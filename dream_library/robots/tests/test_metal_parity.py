"""The Metal driver against MakerMods' own: the same CAN frames, byte for byte, the same readings and the same goals.

`data/metal_parity.json` is MakerMods' Metal driver (their console's `MetalRobot` around their LeRobot fork's
`MetalFollower` and MakerLab's holding bus, from MakerMods' control loop) run through one scenario
on the Metal fake motors (`metal_fakes`) with a fake clock, written by `data/metal_parity_snapshot.py`: the gripper's
setup from force-position mode, the startup sync, tracking with the velocity feedforward, the soft-limit clamp, the
gripper closing on an object and letting go, missed joint replies, a pause, a joint fault, and the disable of every
motor. This test replays the scenario through the Metal arm's chain (#robots.damiao.DamiaoChain on the #robots.metal
table) on the same fake motors, ticking as their console's servo loop does (one read and one command per tick), and
requires every step's frames, every tick's readings and every tick's goals to be theirs.

What the snapshot leaves out, and why, is in its docstring: MakerLab's holding thread (here the driver's tick writes
the gripper) and the console's release (the scenario ends with MakerLab's own disconnect). The thread above the
chain (#robots.motor_arm.MotorArm: the speed cap, the stale target, the park) has no counterpart in their code and is
tested in `test_metal.py`.
"""

import hashlib
import json
import pathlib
from typing import Any

import metal_fakes
import pytest

from robots import config, errors, metal

SNAPSHOT = pathlib.Path(__file__).parent / "data" / "metal_parity.json"

## The arm's channel; it appears in no frame.
CHANNEL = "/dev/serial/by-id/metal-right"

## One servo tick, seconds, as in the snapshot script.
PERIOD_S = 1.0 / metal.ARM.tick_hz

## The hold torque the snapshot's arm runs, N.m: MakerMods' Metal setting.
HOLD_TORQUE_NM = 0.5


def ramp(start: list[float], end: list[float], ticks: int) -> list[list[float]]:
    """`ticks` goals from `start` to `end`, one per tick: the snapshot script's ramp(), the same arithmetic."""
    return [[a + (b - a) * (i + 1) / ticks for a, b in zip(start, end, strict=True)] for i in range(ticks)]


class Run:
    """The scenario's run here: per step the frames sent and their digest, per tick the readings and goals."""

    def __init__(self) -> None:
        self.clock = metal_fakes.FakeClock()
        self.factory = metal_fakes.FakeCanFactory(self.clock)
        cfg = config.ArmConfig(can_channel=CHANNEL, **metal_fakes.GRIPPER_ENDS, gripper_hold_torque_nm=HOLD_TORQUE_NM)
        self.follower = metal.ARM.chain("right", cfg, opener=self.factory, clock=self.clock, sleep=self.clock.sleep)
        self.steps: list[dict[str, Any]] = []
        self.readings: list[list[float]] = []
        self.goals: list[list[float]] = []
        self.gripper_torques: list[float] = []
        self.raised: list[str] = []

    def frames(self) -> list[tuple[int, bytes]]:
        """Every frame sent on the live bus."""
        return self.factory.buses[CHANNEL].frames if CHANNEL in self.factory.buses else []

    def tick(self, goal: list[float]) -> None:
        """One servo tick: read every motor, send the goal, sleep out the period."""
        t0 = self.clock()
        self.readings.append([float(x) for x in self.follower.read().q_deg])
        applied = self.follower.send_action(dict(zip(self.follower.names, goal, strict=True)))
        self.goals.append([float(applied[joint]) for joint in self.follower.names])
        self.gripper_torques.append(self.factory.buses[CHANNEL].motors[7].torque_nm)
        self.clock.sleep(max(0.0, PERIOD_S - (self.clock() - t0)))

    def step(self, step: dict[str, Any]) -> None:
        """Plays one scenario step."""
        op = step["op"]
        if op == "pose":
            for can_id, deg in step["deg"].items():
                self.factory.bus(CHANNEL).motors[int(can_id)].position_deg = deg
        elif op == "register":
            self.factory.bus(CHANNEL).motors[step["id"]].registers[step["rid"]] = step["u32"].to_bytes(4, "little")
        elif op == "connect":
            self.follower.open()
            self.follower.enable()
        elif op == "track":
            for _ in range(step["ticks"]):
                self.tick(step["q"])
        elif op == "ramp":
            for goal in ramp(step["from"], step["to"], step["ticks"]):
                self.tick(goal)
        elif op == "hold":
            for _ in range(step["ticks"]):
                self.tick(self.goals[-1])
        elif op == "obstacle":
            self.factory.buses[CHANNEL].motors[step["id"]].obstacle_deg = step["deg"]
        elif op == "drop_every":
            self.factory.buses[CHANNEL].drop_every = step["n"]
            self.factory.buses[CHANNEL].drop_ids = None if step["ids"] is None else set(step["ids"])
        elif op == "pause":
            self.clock.sleep(step["s"])
        elif op == "fault":
            self.factory.buses[CHANNEL].motors[step["id"]].fault = step["status"]
        elif op == "failing_tick":
            with pytest.raises(errors.ArmError) as raised:
                self.tick(step["q"])
            self.raised.append(str(raised.value))
        elif op == "close":
            self.follower.close(disable_torque=True)
        else:
            raise AssertionError(f"unknown step {op!r}")

    def play(self, scenario: list[dict[str, Any]]) -> "Run":
        """Plays every step, recording its frames."""
        for step in scenario:
            sent = len(self.frames())
            self.step(step)
            lines = [f"{can_id}:{data.hex()}" for can_id, data in self.frames()[sent:]]
            self.steps.append(
                {
                    "frames": len(lines),
                    "sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(),
                    "lines": lines,
                }
            )
        return self


@pytest.fixture(scope="module")
def snapshot() -> dict[str, Any]:
    """MakerMods' run."""
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ours(snapshot: dict[str, Any]) -> Run:
    """This package's run of the same scenario, with the gravity torque off: their follower sends none (the one place
    the Metal table departs from it, #robots.metal.JOINTS)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(metal, "gravity_nm", lambda q_deg: (0.0,) * len(q_deg))
        return Run().play(snapshot["scenario"])


def test_the_snapshot_is_of_the_source_the_port_cites(snapshot: dict[str, Any]) -> None:
    """The fixture names the versions of MakerMods' code it ran, the ones the port's docstrings cite."""
    assert snapshot["source_versions"] == {"lerobot": "5ce2fe4", "makermodslab": "f3ee37c"}
    assert "5ce2fe4" in str(metal.__doc__)
    assert len(snapshot["source_sha256"]) == 9


def test_every_step_sends_their_frames_byte_for_byte(snapshot: dict[str, Any], ours: Run) -> None:
    """The gripper's setup, the reads, the preloads, the enables, every tick's reads, sync queries and MIT frames
    (gains, feedforward and the gripper's holding law included), the re-queries a missed reply costs, and the
    disables and the gripper's restore: the same frames in the same order."""
    assert len(ours.steps) == len(snapshot["steps"])
    for i, (mine, theirs) in enumerate(zip(ours.steps, snapshot["steps"], strict=True)):
        step = snapshot["scenario"][i]
        assert (mine["frames"], mine["sha256"]) == (theirs["frames"], theirs["sha256"]), (
            f"step {i} ({step['op']}): {mine['frames']} frames here, {theirs['frames']} in theirs; "
            f"first frames here: {mine['lines'][:8]}"
        )
    assert sum(step["frames"] for step in ours.steps) > 4000


def test_every_tick_reads_and_sends_what_theirs_does(snapshot: dict[str, Any], ours: Run) -> None:
    """Each tick's seven readings and seven goals, degrees, and the gripper's push are theirs to the bit."""
    assert len(ours.readings) == len(snapshot["readings"]) > 250
    for tick, (mine, theirs) in enumerate(zip(ours.readings, snapshot["readings"], strict=True)):
        assert mine == theirs, f"tick {tick} reads {mine}, theirs {theirs}"
    for tick, (mine, theirs) in enumerate(zip(ours.goals, snapshot["goals"], strict=True)):
        assert mine == theirs, f"tick {tick} sends {mine}, theirs {theirs}"
    assert ours.gripper_torques == snapshot["gripper_torques"]


def test_the_gripper_holds_the_object_at_the_hold_torque(ours: Run) -> None:
    """Closing on the object at 40 degrees, the first frame pushes at the full position gain (MakerMods do not cap a
    free close), and the holding law then settles the push at about 0.5 N.m until the open releases it."""
    pushes = [-torque for torque in ours.gripper_torques]
    assert max(pushes) > 5.0
    assert pushes[pushes.index(max(pushes)) + 40] == pytest.approx(HOLD_TORQUE_NM, abs=0.05)


def test_the_run_ends_as_theirs_does(snapshot: dict[str, Any], ours: Run) -> None:
    """The fault stops both at the same tick, and no motor is left on."""
    assert len(ours.raised) == len(snapshot["raised"]) == 1
    assert "wrist_yaw" in ours.raised[0] and "wrist_yaw" in snapshot["raised"][0]
    bus = ours.factory.buses[CHANNEL]
    assert [motor.can_id for motor in bus.motors.values() if motor.enabled] == snapshot["enabled_after_close"] == []
    assert bus.closed

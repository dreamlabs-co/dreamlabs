"""Unit tests of the Metal arm (`robots.metal`) on its chain (`robots.damiao`, `robots.metal_gripper`) and the shared
bus-arm thread (`robots.motor_arm`), on the Metal fakes.

Most tests step the chain or the thread's tick by hand on a fake clock (`metal_fakes.FakeClock`), so rates are exact;
the ones about the thread itself and the station's arm wrapper run it over the same fake motors. The parity with
MakerMods' own driver is `test_metal_parity.py`; the body's config, wire and embodiment are `test_metal_body.py`.
"""

import logging
import math
import struct
import time
from collections.abc import Callable
from typing import Any

import can
import metal_fakes
import numpy as np
import pytest

from robots import config, damiao, errors, followers, home, metal, metal_gripper, motor_arm, motor_guard, types

CHANNEL = "can_right"

## A pose inside every soft limit, near zero so a return to rest is short: degrees by CAN id.
POSE_DEG = {1: 5.0, 2: -12.0, 3: 15.0, 4: -5.0, 5: 4.0, 6: -6.0, 7: 30.0}

## Per-tick steps, degrees: the chain's startup sync and the thread's tracking cap.
SYNC_STEP_DEG = damiao.SYNC_STEP_DEG
MAX_STEP_DEG = motor_arm.MAX_SPEED_DEG_S / metal.ARM.tick_hz

## The motors' names, base to gripper.
NAMES = [joint.name for joint in metal.JOINTS]

## One count of a 16-bit MIT position, degrees (every Metal model spans the same +-6.28 rad).
Q_STEP_DEG = math.degrees(2 * damiao.MOTOR_LIMITS["metal_jlo"][0] / 65535)
## One count of a 12-bit MIT velocity on the joints' model, degrees per second.
V_STEP_DEG_S = math.degrees(2 * damiao.MOTOR_LIMITS["metal_jlo"][1] / 4095)


def wait_for(predicate: Callable[[], bool], what: str, timeout_s: float = 5.0) -> None:
    """Polls `predicate` on the wall clock, failing the test after `timeout_s`."""
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail(f"timed out waiting for {what}")
        time.sleep(0.005)


def thread_sleep(clock: Any) -> Callable[[float], None]:
    """The driver's sleep on `clock`, plus a moment of real time once a tick.

    A driver thread on a fake clock never sleeps for real, and the GIL releases it makes on the way (numpy's
    np.interp, a few microseconds each tick) are too short for the test's thread to take the GIL back: it waited 10 s
    and more to return from `start()`. A real station sleeps out every tick.
    """

    def sleep(seconds: float) -> None:
        clock.sleep(seconds)
        time.sleep(0.0001)

    return sleep


class Rig:
    """One arm on a fake bus: the factory and the arm's thread over its chain, on one clock."""

    def __init__(self, clock: Any, pose_deg: dict[int, float] | None = None) -> None:
        self.clock = clock
        self.factory = metal_fakes.FakeCanFactory(clock)
        for can_id, deg in (POSE_DEG if pose_deg is None else pose_deg).items():
            self.factory.bus(CHANNEL).motors[can_id].position_deg = deg
        cfg = config.ArmConfig(can_channel=CHANNEL, **metal_fakes.GRIPPER_ENDS, gripper_hold_torque_nm=0.5)
        self.arm = motor_arm.MotorArm(
            metal.ARM, "right", cfg, opener=self.factory, clock=clock, sleep=thread_sleep(clock)
        )
        self.chain = self.arm.chain

    @property
    def bus(self) -> metal_fakes.FakeCanBus:
        """The live bus (the one built before the first open, until then)."""
        return self.factory.bus(CHANNEL)

    def motor(self, name: str) -> metal_fakes.FakeMotor:
        """The fake motor of a joint."""
        return self.bus.motors[self.chain.bus.motors[name].can_id]

    def open(self) -> "Rig":
        """Opens and energizes the arm."""
        self.chain.open()
        self.chain.enable()
        return self

    def send(self, goal_deg: dict[str, float]) -> dict[str, float]:
        """One servo tick on the chain alone: read every motor, send the goal, sleep out the period."""
        t0 = self.clock()
        self.chain.read()
        applied = self.chain.send_action(goal_deg)
        self.clock.sleep(max(0.0, self.arm.period - (self.clock() - t0)))
        return applied

    def tick(self, n: int = 1, target: list[float] | None = None) -> None:
        """`n` ticks, each with `target` set first when given, timed as the thread's own run times them."""
        for _ in range(n):
            if target is not None:
                self.arm.set_target(np.asarray(target, dtype=np.float64))
            t0 = self.clock()
            self.arm.tick()
            self.clock.sleep(max(0.0, self.arm.period - (self.clock() - t0)))

    def velocities(self, name: str) -> list[float]:
        """The velocity feedforward of every MIT frame a motor got, degrees per second."""
        motor = self.motor(name)
        return [
            metal_fakes.decode_mit(data, motor.model)[1]
            for can_id, data in self.bus.frames
            if can_id == motor.can_id and data[:7] != b"\xff" * 7
        ]


def fake_rig(**kwargs: Any) -> Rig:
    """An opened arm on a fake clock."""
    return Rig(metal_fakes.FakeClock(), **kwargs).open()


def pose_goal(deg: dict[int, float]) -> dict[str, float]:
    """A chain goal, degrees per motor, from degrees by CAN id."""
    return {joint.name: deg[joint.motor_id] for joint in metal.JOINTS}


def pose_target(deg: dict[int, float], gripper_mm: float) -> list[float]:
    """A thread target, `[6 joints rad, gripper stroke mm]`, from joint degrees by CAN id."""
    return [math.radians(deg[i]) for i in range(1, 7)] + [gripper_mm]


# The wire


def test_mit_frames_and_state_frames_round_trip_through_the_fake_motor() -> None:
    """A position frame decodes back to its angle, velocity and kp; a state frame back to the motor's state."""
    data = damiao.encode_mit("metal_jlo", 100.0, 5.0, -40.0, 12.0, 0.0)
    position, velocity, kp = metal_fakes.decode_mit(data, "metal_jlo")
    assert position == pytest.approx(-40.0, abs=2 * Q_STEP_DEG)
    assert velocity == pytest.approx(12.0, abs=V_STEP_DEG_S)
    assert kp == pytest.approx(100.0, abs=0.13)

    motor = metal_fakes.FakeMotor(2, position_deg=-37.5)
    motor.enabled, motor.torque_nm, motor.temp_rotor_c = True, -3.0, 41
    state = damiao.decode_state(metal_fakes.encode_state(motor), "metal_j2", t=7.0)
    assert (state.status, state.temp_mos_c, state.temp_rotor_c, state.t) == (damiao.STATUS_ENABLED, 35.0, 41.0, 7.0)
    assert state.position_deg == pytest.approx(-37.5, abs=2 * Q_STEP_DEG)
    assert state.torque_nm == pytest.approx(-3.0, abs=0.06)
    assert damiao.is_state_frame(metal_fakes.encode_state(motor), 2)
    assert not damiao.is_state_frame(struct.pack("<HBB", 2, damiao.OP_READ, damiao.REG_TIMEOUT) + bytes(4), 2)


# Open


def test_open_sets_the_gripper_up_preloads_every_joint_and_energizes_nothing() -> None:
    """After open every motor is still off, each joint's stored target is its own angle at zero gain, and the gripper
    runs MIT mode with its 500 ms CAN timeout; enable() then energizes all seven where they are."""
    rig = Rig(metal_fakes.FakeClock())
    rig.chain.open()

    assert not any(motor.enabled for motor in rig.bus.motors.values())
    for can_id, deg in POSE_DEG.items():
        motor = rig.bus.motors[can_id]
        assert motor.commands[-1] == pytest.approx(deg, abs=2 * Q_STEP_DEG), can_id
        assert motor.kps[-1] == 0.0, can_id
    gripper = rig.motor("gripper")
    assert gripper.mode == damiao.MODE_MIT
    assert gripper.registers[metal_fakes.REG_TIMEOUT] == struct.pack("<I", metal_gripper.WATCHDOG_TICKS)

    rig.chain.enable()
    assert all(motor.enabled for motor in rig.bus.motors.values())
    assert rig.chain.read().q_deg == pytest.approx([POSE_DEG[i] for i in range(1, 8)], abs=2 * Q_STEP_DEG)


def test_open_names_a_motor_that_does_not_answer_and_closes_the_adapter() -> None:
    """A silent joint fails the open with its name; the adapter is closed and the gripper restored."""
    rig = Rig(metal_fakes.FakeClock())
    rig.factory.bus(CHANNEL).motors[4].silent = True

    with pytest.raises(damiao.MotorSilent, match="wrist_flex"):
        rig.chain.open()

    assert rig.bus.closed
    assert not any(motor.enabled for motor in rig.bus.motors.values())
    assert rig.motor("gripper").registers[metal_fakes.REG_TIMEOUT] == struct.pack("<I", 0)


def test_a_pose_past_the_tolerance_refuses_to_start_and_one_within_it_opens() -> None:
    """More than 5 degrees past a soft limit is a lost zero: refused, nothing energized. Within 5 is how the arm
    rests with torque off, and opens."""
    rig = Rig(metal_fakes.FakeClock(), pose_deg={**POSE_DEG, 2: 20.0})
    with pytest.raises(damiao.StartupPose, match=r"shoulder_lift reads \+20\.0 deg, outside its range -180 to 0"):
        rig.chain.open()
    assert rig.bus.closed
    assert not any(motor.enabled for motor in rig.bus.motors.values())

    rig = Rig(metal_fakes.FakeClock(), pose_deg={**POSE_DEG, 2: 4.0, 3: -4.0}).open()
    assert rig.chain.enabled


@pytest.mark.parametrize(
    ("can_id", "state", "message"),
    [
        (3, {"enabled": True}, "elbow_flex is still powered from an earlier session"),
        (5, {"fault": 9}, "wrist_yaw is in fault 0x9, undervoltage"),
        (2, {"fault": 12}, "shoulder_lift is in fault 0xC, coil over-temperature .*let it cool"),
    ],
)
def test_an_energized_or_faulted_joint_refuses_to_start(can_id: int, state: dict[str, Any], message: str) -> None:
    """A joint left energized would go limp under the zero-gain preload; a fault is named in plain words with what to
    do (0xC, the coil over-temperature, says to let it cool, not to re-zero). The
    refusal switches nothing off: the powered joint still holds."""
    rig = Rig(metal_fakes.FakeClock())
    for key, value in state.items():
        setattr(rig.factory.bus(CHANNEL).motors[can_id], key, value)

    with pytest.raises(damiao.StartupPose, match=message):
        rig.chain.open()

    assert rig.bus.closed
    powered = {i for i, motor in rig.bus.motors.items() if motor.enabled}
    assert powered == ({can_id} if state.get("enabled") else set())


@pytest.mark.parametrize("gripper_powered", [False, True])
def test_an_arm_left_powered_by_an_earlier_session_is_refused_and_left_holding(gripper_powered: bool) -> None:
    rig = Rig(metal_fakes.FakeClock())
    motors = rig.factory.bus(CHANNEL).motors
    for can_id in range(1, 7):
        motors[can_id].enabled = True
    motors[7].enabled = gripper_powered
    refusal: Any = metal_gripper.GripperSafetyError if gripper_powered else damiao.StartupPose
    words = "support the arm, then switch its power off and on"

    with pytest.raises(refusal, match=words):
        rig.chain.open()

    assert rig.bus.closed
    assert all(motors[can_id].enabled for can_id in range(1, 7)), "a refused arm still holds"
    assert not any(can_id < 7 and data == b"\xff" * 7 + bytes([damiao.CMD_DISABLE]) for can_id, data in rig.bus.frames)
    assert not motors[7].enabled and motors[7].registers[metal_fakes.REG_TIMEOUT] == struct.pack("<I", 0)


def _mode(motor: metal_fakes.FakeMotor, mode: int) -> None:
    motor.registers[metal_fakes.REG_MODE] = struct.pack("<I", mode)


def _warm(motor: metal_fakes.FakeMotor, _: int) -> None:
    motor.temp_mos_c = 46


def _scaling(motor: metal_fakes.FakeMotor, _: int) -> None:
    motor.registers[metal_fakes.REG_PMAX] = struct.pack("<f", 12.5)


def _energized(motor: metal_fakes.FakeMotor, _: int) -> None:
    motor.enabled = True


@pytest.mark.parametrize(
    ("setup", "message"),
    [
        ((_mode, 2), "must start in MIT or force-position mode"),
        ((_warm, 0), "cool below 45 C"),
        ((_scaling, 0), "MIT scaling \\(register 21\\)"),
        ((_energized, 0), "is not disabled"),
    ],
)
def test_the_gripper_setup_refuses_and_leaves_the_gripper_as_it_was(
    setup: tuple[Callable[[metal_fakes.FakeMotor, int], None], int], message: str
) -> None:
    """MakerLab's setup checks: the mode, both temperatures, the MIT scaling, a disabled motor; nothing is kept."""
    rig = Rig(metal_fakes.FakeClock())
    gripper = rig.factory.bus(CHANNEL).motors[7]
    setup[0](gripper, setup[1])
    mode_before = gripper.mode

    with pytest.raises(metal_gripper.GripperSafetyError, match=message):
        rig.chain.open()

    assert rig.bus.closed
    assert not any(motor.enabled for motor in rig.bus.motors.values())
    assert gripper.mode == mode_before
    assert gripper.registers[metal_fakes.REG_TIMEOUT] == struct.pack("<I", 0)


def test_a_force_position_gripper_runs_mit_and_gets_its_mode_back_at_close() -> None:
    """MakerMods' other gripper mode (4) is switched to MIT for the session, and restored with the timeout at close."""
    rig = Rig(metal_fakes.FakeClock())
    _mode(rig.factory.bus(CHANNEL).motors[7], damiao.MODE_FORCE_POSITION)
    rig.open()
    assert rig.motor("gripper").mode == damiao.MODE_MIT
    rig.send(pose_goal(POSE_DEG))

    rig.chain.close(disable_torque=True)

    gripper = rig.motor("gripper")
    assert gripper.mode == damiao.MODE_FORCE_POSITION
    assert gripper.registers[metal_fakes.REG_TIMEOUT] == struct.pack("<I", 0)
    assert not any(motor.enabled for motor in rig.bus.motors.values())
    assert rig.bus.closed


# Reads


def test_a_missed_reply_is_asked_once_more_and_never_served_from_the_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """One missed reply costs one more query; a motor that misses twice stops the read with its name."""
    rig = fake_rig()
    bus = rig.bus
    queue = bus._queue
    missed: list[bytes] = []

    def drop_one(motor: metal_fakes.FakeMotor, data: bytes) -> None:
        if motor.can_id == 2 and not missed:
            missed.append(data)
            return
        queue(motor, data)

    monkeypatch.setattr(bus, "_queue", drop_one)
    sent = len(bus.frames)
    reading = rig.chain.read()

    queries = [data for can_id, data in bus.frames[sent:] if can_id == metal_fakes.PARAM_ID and data[0] == 2]
    assert len(queries) == 2 and missed
    assert reading.q_deg[1] == pytest.approx(POSE_DEG[2], abs=2 * Q_STEP_DEG)
    assert rig.chain.bus.drops() == 1

    rig.motor("shoulder_lift").silent = True
    with pytest.raises(damiao.MotorSilent, match="shoulder_lift"):
        rig.chain.read()


@pytest.mark.parametrize(
    ("can_id", "state", "message"),
    [
        (5, {"fault": 9}, "wrist_yaw: motor in fault 0x9, undervoltage"),
        (3, {"enabled": False}, "elbow_flex: motor switched itself off"),
    ],
)
def test_an_energized_arm_requires_every_motor_energized(can_id: int, state: dict[str, Any], message: str) -> None:
    """A motor that faults or drops out while the arm runs fails the read, saying what the fault is and what to do."""
    rig = fake_rig()
    for key, value in state.items():
        setattr(rig.bus.motors[can_id], key, value)

    with pytest.raises(errors.ArmError, match=message):
        rig.chain.read()


def test_a_gripper_that_holds_its_replies_is_flushed_and_read_fresh() -> None:
    """The Metal gripper answers when the next frame appears: a state query of another motor releases its reply."""
    rig = Rig(metal_fakes.FakeClock())
    rig.factory.bus(CHANNEL).motors[7].holds_replies = True
    rig.open()
    rig.motor("gripper").position_deg = 42.0

    reading = rig.chain.read()
    rig.tick(10, target=pose_target(POSE_DEG, 30.0))

    assert reading.q_deg[6] == pytest.approx(42.0, abs=2 * Q_STEP_DEG)
    flush = (metal_fakes.PARAM_ID, bytes([1, 0, damiao.CMD_REFRESH, 0, 0, 0, 0, 0]))
    assert flush in rig.bus.frames
    assert not rig.arm.fault and rig.chain.gripper.error == ""


# Commands


def test_the_startup_sync_moves_a_degree_a_tick_until_within_three() -> None:
    """A joint far from its goal moves at most a degree a tick from where it reads, until it is within 3 degrees."""
    rig = fake_rig()
    goal = pose_goal({**POSE_DEG, 1: POSE_DEG[1] + 10.0})

    sent = [rig.send(goal)["shoulder_pan"] for _ in range(10)]

    assert np.diff([POSE_DEG[1], *sent[:6]]) == pytest.approx([SYNC_STEP_DEG] * 6, abs=2 * Q_STEP_DEG)
    assert sent[-1] == goal["shoulder_pan"]
    assert rig.motor("shoulder_pan").commands[-1] == pytest.approx(goal["shoulder_pan"], abs=2 * Q_STEP_DEG)
    assert rig.chain._synced


def test_a_joint_that_cannot_close_the_gap_is_let_go_after_a_second() -> None:
    """A joint that makes no progress for #SYNC_STALL_S (blocked, or parked past a limit) is let go to its goal."""
    rig = fake_rig()
    rig.motor("shoulder_pan").obstacle_deg = POSE_DEG[1]
    goal = pose_goal({**POSE_DEG, 1: POSE_DEG[1] - 20.0})

    sent = [rig.send(goal)["shoulder_pan"] for _ in range(int(damiao.SYNC_STALL_S * metal.ARM.tick_hz) + 5)]

    assert sent[0] == pytest.approx(POSE_DEG[1] - SYNC_STEP_DEG, abs=2 * Q_STEP_DEG)
    assert sent[-1] == goal["shoulder_pan"]
    assert rig.chain._synced


def test_goals_are_clamped_to_the_soft_limits_the_gripper_included() -> None:
    """A goal past a limit is sent at the limit; the gripper stops at 137.5 degrees, fully open."""
    rig = fake_rig()
    rig.send(pose_goal(POSE_DEG))
    beyond = {**pose_goal(POSE_DEG), "shoulder_lift": 30.0, "wrist_yaw": -120.0, "gripper": 200.0}

    applied = rig.send(beyond)

    assert (applied["shoulder_lift"], applied["wrist_yaw"], applied["gripper"]) == (0.0, -85.0, 137.5)
    assert rig.motor("shoulder_lift").commands[-1] == pytest.approx(0.0, abs=2 * Q_STEP_DEG)
    assert rig.motor("gripper").commands[-1] == pytest.approx(137.5, abs=2 * Q_STEP_DEG)


def test_the_feedforward_is_the_filtered_command_velocity() -> None:
    """A degree a tick at 30 Hz is 30 deg/s, filtered at 0.04 a command; a late command's time step is clamped to
    0.1 s and its velocity to 120 deg/s; rearm() starts it from zero."""
    rig = fake_rig()
    rig.send(pose_goal(POSE_DEG))
    for k in range(1, 11):
        rig.send(pose_goal({**POSE_DEG, 1: POSE_DEG[1] + k}))

    expected = [30.0 * (1.0 - (1.0 - damiao.FF_ALPHA) ** k) for k in range(1, 11)]
    assert rig.velocities("shoulder_pan")[-10:] == pytest.approx(expected, abs=V_STEP_DEG_S)

    rig.clock.sleep(0.2)
    rig.send(pose_goal({**POSE_DEG, 1: POSE_DEG[1] + 30.0}))
    late = (1.0 - damiao.FF_ALPHA) * expected[-1] + damiao.FF_ALPHA * damiao.FF_MAX_DEG_S
    assert rig.velocities("shoulder_pan")[-1] == pytest.approx(late, abs=V_STEP_DEG_S)

    rig.chain.rearm()
    rig.send(pose_goal({**POSE_DEG, 1: POSE_DEG[1] + 30.0}))
    assert rig.velocities("shoulder_pan")[-1] == pytest.approx(0.0, abs=V_STEP_DEG_S)


# The gripper's hold


def test_closing_on_an_object_holds_it_at_the_hold_torque_and_an_open_lets_go() -> None:
    """MakerLab's holding law: on contact the gripper squeezes at 0.5 N.m, not at its full position gain; an open
    command releases the object."""
    rig = fake_rig()
    rig.motor("gripper").obstacle_deg = 20.0

    rig.tick(240, target=pose_target(POSE_DEG, 0.0))

    assert rig.chain.gripper.controller.holding
    assert -rig.motor("gripper").torque_nm == pytest.approx(0.5, abs=0.05)
    rig.tick(90, target=pose_target(POSE_DEG, 80.0))
    assert not rig.chain.gripper.controller.holding
    assert rig.motor("gripper").commands[-1] == pytest.approx(rig.arm.stroke.to_deg(80.0), abs=2 * Q_STEP_DEG)


def test_a_hot_gripper_is_switched_off_and_raises_the_arms_alarm_while_the_joints_hold() -> None:
    rig = fake_rig()
    rig.send(pose_goal(POSE_DEG))
    rig.motor("gripper").temp_rotor_c = 55
    gripper_frames = len(rig.motor("gripper").commands)

    rig.send(pose_goal(POSE_DEG))
    rig.send(pose_goal(POSE_DEG))

    assert not rig.motor("gripper").enabled
    assert len(rig.motor("gripper").commands) == gripper_frames
    assert "the gripper stopped" in rig.chain.alarm and "coil reads 55 C (driver board 35 C)" in rig.chain.alarm
    assert all(rig.bus.motors[i].enabled for i in range(1, 7))
    assert rig.chain.read().temperatures_c is not None
    assert rig.arm.alarm() == rig.chain.alarm


def test_gripper_feedback_older_than_its_stale_time_stops_the_gripper_but_a_late_tick_does_not() -> None:
    rig = fake_rig()
    rig.send(pose_goal(POSE_DEG))
    rig.clock.sleep(metal_gripper.STALE_S + 0.05)
    rig.send(pose_goal(POSE_DEG))

    rig.chain.read()
    rig.clock.sleep(metal_gripper.STALE_S + 0.05)
    rig.chain.send_action(pose_goal(POSE_DEG))
    assert "feedback stale" in rig.chain.alarm
    assert not rig.motor("gripper").enabled


def test_a_gripper_blocked_opening_is_held_where_it_stops_and_the_log_says_where_to_set_its_open_end(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rig = fake_rig()
    rig.tick(30, target=pose_target(POSE_DEG, 25.0))
    rig.motor("gripper").stop_deg = 90.0

    with caplog.at_level(logging.WARNING, logger="robots.motor_guard"):
        rig.tick(90, target=pose_target(POSE_DEG, 100.0))

    assert rig.motor("gripper").commands[-1] == pytest.approx(rig.arm.stroke.to_deg(76.0), abs=0.2)
    told = [record.getMessage() for record in caplog.records if "stops opening" in record.getMessage()]
    assert len(told) == 1 and "set gripper_open_deg to about 90" in told[0]
    assert rig.arm.alarm() is None
    rig.tick(30, target=pose_target(POSE_DEG, 40.0))
    assert rig.motor("gripper").commands[-1] == pytest.approx(rig.arm.stroke.to_deg(40.0), abs=0.2)


def test_a_blocked_joint_is_held_just_past_where_it_stopped_then_raises_the_arms_alarm() -> None:
    rig = fake_rig()
    rig.tick(30, target=pose_target(POSE_DEG, 25.0))
    rig.motor("elbow_flex").stop_deg = POSE_DEG[3] + 2.0

    rig.tick(int(motor_guard.BLOCK_S * metal.ARM.tick_hz) + 20, target=pose_target({**POSE_DEG, 3: 60.0}, 25.0))

    held = POSE_DEG[3] + 2.0 + motor_guard.JOINT_HOLD_DEG
    assert rig.motor("elbow_flex").commands[-1] == pytest.approx(held, abs=0.2)
    assert rig.arm.alarm() is None
    rig.tick(int(motor_guard.BLOCK_STOP_S * metal.ARM.tick_hz), target=pose_target({**POSE_DEG, 3: 60.0}, 25.0))
    alarm = rig.arm.alarm()
    assert alarm is not None and alarm.startswith("elbow_flex has been blocked for 5 s")
    assert rig.motor("elbow_flex").commands[-1] == pytest.approx(held, abs=0.2)


def test_a_joint_blocked_while_the_arm_holds_with_no_target_is_eased_off_then_raises_the_alarm() -> None:
    rig = fake_rig()
    rig.tick(30, target=pose_target(POSE_DEG, 25.0))
    rig.motor("elbow_flex").stop_deg = POSE_DEG[3] + 2.0
    rig.tick(1, target=pose_target({**POSE_DEG, 3: 60.0}, 25.0))
    rig.tick(int(motor_arm.TARGET_STALE_S * metal.ARM.tick_hz) + 2)
    assert rig.arm._target is None

    rig.tick(int((motor_guard.BLOCK_S + 2.0) * metal.ARM.tick_hz))

    held = POSE_DEG[3] + 2.0 + motor_guard.JOINT_HOLD_DEG
    assert rig.motor("elbow_flex").commands[-1] == pytest.approx(held, abs=0.2)
    assert rig.arm.alarm() is None
    rig.tick(int(motor_guard.BLOCK_STOP_S * metal.ARM.tick_hz))
    alarm = rig.arm.alarm()
    assert alarm is not None and alarm.startswith("elbow_flex has been blocked for")
    assert rig.motor("elbow_flex").commands[-1] == pytest.approx(held, abs=0.2)


def test_an_overheating_joint_raises_the_arms_alarm_naming_the_motor_and_the_sensor(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rig = fake_rig()
    rig.motor("shoulder_lift").temp_mos_c = 56
    rig.motor("elbow_flex").temp_rotor_c = 66

    with caplog.at_level(logging.WARNING, logger="robots.motor_guard"):
        rig.tick(30, target=pose_target(POSE_DEG, 25.0))
        assert rig.arm.alarm() is None
        rig.tick(40, target=pose_target(POSE_DEG, 25.0))

    alarm = rig.arm.alarm()
    assert alarm is not None and alarm.startswith("elbow_flex coil reached 66 C (the run stops at 65 C)")
    assert caplog.text.count("shoulder_lift driver board at 56 C") == 1
    assert rig.arm.read().motor_temps_c == (35.0, 56.0, 66.0, 35.0, 35.0, 35.0, 35.0)
    rig.motor("shoulder_lift").temp_mos_c = 50
    rig.motor("elbow_flex").temp_rotor_c = 54
    rig.tick(2)
    assert rig.arm.alarm() is None


def test_a_normal_run_never_trips_the_guard(caplog: pytest.LogCaptureFixture) -> None:
    rig = fake_rig()
    rig.motor("gripper").obstacle_deg = 20.0
    poses = [
        ({**POSE_DEG, 1: 40.0, 2: -40.0, 3: 50.0}, 100.0),
        ({**POSE_DEG, 4: 40.0, 5: -30.0, 6: 50.0}, 0.0),
        (POSE_DEG, 60.0),
    ]
    with caplog.at_level(logging.INFO, logger="robots.motor_guard"):
        for pose, grip in poses:
            rig.tick(120, target=pose_target(pose, grip))

    assert not caplog.records
    assert rig.arm.alarm() is None
    assert rig.motor("shoulder_pan").commands[-1] == pytest.approx(POSE_DEG[1], abs=2 * Q_STEP_DEG)


# Close


def test_a_joint_that_does_not_confirm_its_disable_is_asked_again_and_reported() -> None:
    """Torque off must not be claimed while a motor may be energized: the joint is asked #DISABLE_RETRIES more times,
    named in the error, and every other motor is still switched off and the adapter closed."""
    rig = fake_rig()
    rig.send(pose_goal(POSE_DEG))
    rig.motor("elbow_flex").silent = True
    sent = len(rig.bus.frames)

    with pytest.raises(errors.ArmError, match="elbow_flex did not confirm its disable"):
        rig.chain.close(disable_torque=True)

    disables = [frame for frame in rig.bus.frames[sent:] if frame == (3, b"\xff" * 7 + bytes([damiao.CMD_DISABLE]))]
    assert len(disables) == damiao.DISABLE_RETRIES + 1
    assert not any(motor.enabled for motor in rig.bus.motors.values() if motor.can_id != 3)
    assert rig.bus.closed


def test_a_switch_off_after_a_close_under_power_opens_the_adapter_again_and_switches_every_joint_off() -> None:
    rig = fake_rig()
    rig.send(pose_goal(POSE_DEG))
    rig.chain.close(disable_torque=False)
    joints = [rig.bus.motors[can_id] for can_id in range(1, 7)]
    assert rig.bus.closed and all(motor.enabled for motor in joints)

    rig.factory.fail_opens.add(CHANNEL)
    with pytest.raises(errors.ArmError, match="still powered .* support the arm, then switch its power off"):
        rig.chain.close(disable_torque=True)
    assert all(motor.enabled for motor in joints)

    rig.factory.fail_opens.clear()
    opens = rig.factory.opens[CHANNEL]
    rig.chain.close(disable_torque=True)
    assert rig.bus.closed and not any(motor.enabled for motor in rig.bus.motors.values())
    assert rig.factory.opens[CHANNEL] == opens + 1
    rig.chain.close(disable_torque=True)
    assert rig.factory.opens[CHANNEL] == opens + 1, "nothing left powered, nothing to open again for"


# The thread


def test_the_first_tick_holds_the_measured_pose_and_takes_it_as_rest() -> None:
    """Seeded from the arm's own pose: nothing jumps before a target arrives (a joint past its limit is held at the
    limit). Rest is that pose, the gripper at its most open goal."""
    rig = fake_rig(pose_deg={**POSE_DEG, 2: 2.0})
    assert rig.arm.rest is None
    rig.tick(3)

    for can_id, deg in {**POSE_DEG, 2: 0.0}.items():
        assert rig.bus.motors[can_id].commands[-1] == pytest.approx(deg, abs=2 * Q_STEP_DEG), can_id
    assert rig.arm.rest is not None
    np.testing.assert_allclose(
        np.degrees(rig.arm.rest[:6]), [POSE_DEG[1], 2.0, *(POSE_DEG[i] for i in range(3, 7))], atol=2 * Q_STEP_DEG
    )
    assert rig.arm.rest[6] == metal.GRIP_SCALE_MM - metal.ARM.grip_margin_mm


def test_a_new_target_eases_in_then_tracks_at_the_cap() -> None:
    """A degree a tick until synced, then #robots.motor_arm.MAX_SPEED_DEG_S (four degrees a tick), never more."""
    rig = fake_rig()
    near = pose_target({**POSE_DEG, 1: POSE_DEG[1] + 10.0}, 20.0)
    rig.tick(4, target=near)
    pan = rig.motor("shoulder_pan").commands
    assert np.diff(pan[-3:]) == pytest.approx([SYNC_STEP_DEG] * 2, abs=2 * Q_STEP_DEG)

    rig.tick(20, target=near)
    assert rig.arm.stepper.synced and rig.chain._synced
    rig.tick(5, target=pose_target({**POSE_DEG, 1: POSE_DEG[1] + 60.0}, 20.0))

    steps = np.diff(rig.motor("shoulder_pan").commands[-5:])
    assert steps == pytest.approx([MAX_STEP_DEG] * 4, abs=2 * Q_STEP_DEG)


def test_a_target_that_goes_stale_rearms_the_sync() -> None:
    """No target for #TARGET_STALE_S: the arm holds, and the next target eases in at a degree a tick again."""
    rig = fake_rig()
    rig.tick(30, target=pose_target(POSE_DEG, 20.0))
    assert rig.arm.stepper.synced and rig.chain._synced

    rig.tick(int(motor_arm.TARGET_STALE_S * metal.ARM.tick_hz) + 5)
    assert not rig.arm.stepper.synced
    held = rig.motor("shoulder_pan").commands[-1]
    rig.tick(3, target=pose_target({**POSE_DEG, 1: 40.0}, 20.0))

    pan = rig.motor("shoulder_pan").commands
    assert pan[-3] == pytest.approx(held + SYNC_STEP_DEG, abs=2 * Q_STEP_DEG)
    assert np.diff(pan[-3:]) == pytest.approx([SYNC_STEP_DEG] * 2, abs=2 * Q_STEP_DEG)


def test_targets_that_stop_again_hold_where_the_last_stream_left_the_arm() -> None:
    rig = fake_rig()
    rig.tick(30, target=pose_target(POSE_DEG, 20.0))
    rig.tick(int(motor_arm.TARGET_STALE_S * metal.ARM.tick_hz) + 5)
    rig.tick(60, target=pose_target({**POSE_DEG, 1: 25.0}, 20.0))
    assert rig.motor("shoulder_pan").commands[-1] == pytest.approx(25.0, abs=0.5)

    rig.tick(int(motor_arm.TARGET_STALE_S * metal.ARM.tick_hz) + 30)

    assert rig.arm._target is None
    assert rig.motor("shoulder_pan").commands[-1] == pytest.approx(25.0, abs=0.5)


def test_the_gripper_is_published_as_its_stroke_and_the_loop_reads_the_metal_grip() -> None:
    """The thread publishes the stroke in mm, linear between the arm's gripper ends; its driver side hands the loop
    mm / 100, and takes the loop's grip back the same way."""
    rig = fake_rig()
    rig.tick(2)
    q, velocity = rig.arm.latest()

    assert q.dtype == np.float32 and velocity.shape == (6,)
    assert float(q[6]) == pytest.approx(25.0, abs=0.03)  # 30 of the test ends' 120 degrees
    arm = rig.arm
    reading = arm.read()
    assert reading.gripper == pytest.approx(0.25, abs=3e-4)
    np.testing.assert_allclose(reading.positions_rad, np.radians([POSE_DEG[i] for i in range(1, 7)]), atol=1e-3)

    for _ in range(90):
        arm.write(np.radians([POSE_DEG[i] for i in range(1, 7)]), 0.7)
        rig.tick()
    assert rig.motor("gripper").position_deg == pytest.approx(84.0, abs=2 * Q_STEP_DEG)  # 70 mm on the test ends


def test_a_command_that_is_not_six_joints_is_refused() -> None:
    """The arm takes six joints and a gripper only."""
    rig = fake_rig()
    arm = followers.FollowerArm("right", rig.arm, home_rad=np.zeros(6))
    with pytest.raises(errors.ArmError, match="expected \\(6,\\)"):
        arm.write(types.ArmCommand(positions_rad=np.zeros(7), gripper=0.5))


# The station's arm


def open_follower(clock: Any) -> tuple[followers.FollowerArm, Rig]:
    """A FollowerArm over a Metal arm whose thread runs on `clock`."""
    rig = Rig(clock).open()
    rig.arm.start()
    wait_for(rig.arm.first_pose.is_set, "the first pose")
    return followers.FollowerArm("right", rig.arm, home_rad=np.zeros(6)), rig


def test_the_follower_arm_reads_and_writes_a_metal_arm() -> None:
    """The loop's wrapper drives the Metal arm unchanged: radians and a `[0, 1]` grip both ways."""
    arm, rig = open_follower(metal_fakes.RealClock())
    try:
        assert arm.read().gripper == pytest.approx(0.25, abs=1e-3)
        target = np.radians([POSE_DEG[i] for i in range(1, 7)]) + np.array([0.05, 0.0, 0.0, 0.0, 0.0, -0.05])

        def there() -> bool:
            # The loop writes every tick; a target left alone goes stale after half a second.
            arm.write(types.ArmCommand(positions_rad=target, gripper=0.7))
            reading = arm.read()
            return bool(np.allclose(reading.positions_rad, target, atol=1e-3)) and abs(reading.gripper - 0.7) < 1e-3

        wait_for(there, "the arm at the target")
        assert arm.is_alive()
    finally:
        arm.close()
    assert not arm.is_alive()
    joints = [rig.bus.motors[i].enabled for i in range(1, 7)]
    assert all(joints) and not rig.motor("gripper").enabled, "close() leaves the joints holding, the gripper released"
    assert rig.bus.closed


def test_a_tick_that_fails_latches_the_arm_and_torque_off_leaves_it_holding_and_says_so() -> None:
    arm, rig = open_follower(metal_fakes.FakeClock())
    rig.motor("wrist_yaw").fault = 9

    wait_for(lambda: not rig.arm.is_alive(), "the thread to latch")

    assert "wrist_yaw: motor in fault 0x9" in rig.arm.fault
    assert not arm.is_alive()
    rig.arm.stop()
    before = {name: rig.motor(name).position_deg for name in NAMES}
    with pytest.raises(
        errors.ArmError,
        match=r"^arm 'right' stopped answering while it held \(.*wrist_yaw: motor in fault 0x9.*\); it still holds "
        r"under power so it does not drop: support it, then switch its power off$",
    ):
        arm.torque_off()
    assert rig.arm.rest_result is None
    assert {name: rig.motor(name).position_deg for name in NAMES} == before
    assert all(rig.bus.motors[can_id].enabled for can_id in (1, 2, 3, 4, 6)), "a joint was switched off: it drops"
    assert rig.bus.closed


def test_torque_off_after_a_close_under_power_leaves_the_held_arm_powered_and_says_so() -> None:
    arm, rig = open_follower(metal_fakes.RealClock())
    arm.close()
    opens = rig.factory.opens[rig.bus.channel]
    assert all(rig.bus.motors[can_id].enabled for can_id in range(1, 7))

    with pytest.raises(errors.ArmError, match="it still holds under power so it does not drop: support it"):
        arm.torque_off()

    assert rig.bus.closed and all(rig.bus.motors[can_id].enabled for can_id in range(1, 7))
    assert rig.factory.opens[rig.bus.channel] == opens


def test_letting_the_arm_go_limp_ends_a_park_under_way_and_one_asked_for_after() -> None:
    arm, rig = open_follower(metal_fakes.RealClock())
    try:
        away = np.radians([POSE_DEG[i] for i in range(1, 7)]) + np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])

        def there() -> bool:
            arm.write(types.ArmCommand(positions_rad=away, gripper=0.5))
            return abs(arm.read().positions_rad[2] - away[2]) < 0.01

        wait_for(there, "the arm away from rest")
        rig.arm.request_rest()
        time.sleep(0.2)
        assert not rig.arm.rested.is_set(), "a 29 degree park at 15 deg/s is still under way"
        arm.disengage()
        assert rig.arm.rested.wait(1.0)
        assert rig.arm.rest_result == (False, "let go limp before it reached rest")

        rig.arm.request_rest()
        assert rig.arm.rested.wait(1.0)
        assert rig.arm.rest_result == (False, "let go limp before it reached rest")
    finally:
        arm.close()


def test_a_switch_off_that_did_not_confirm_is_tried_again_by_the_next_torque_off(monkeypatch: Any) -> None:
    arm, rig = open_follower(metal_fakes.RealClock())
    original = damiao.DamiaoChain._disable
    lost = {"elbow_flex"}
    monkeypatch.setattr(
        damiao.DamiaoChain, "_disable", lambda self, joint: joint not in lost and original(self, joint)
    )
    with pytest.raises(errors.ArmError, match="elbow_flex did not confirm its disable"):
        arm.torque_off()
    assert rig.bus.closed and rig.bus.motors[3].enabled

    lost.clear()
    opens = rig.factory.opens[rig.bus.channel]
    arm.torque_off()
    assert rig.bus.closed and not any(motor.enabled for motor in rig.bus.motors.values())
    assert rig.factory.opens[rig.bus.channel] == opens + 1
    arm.torque_off()
    assert rig.factory.opens[rig.bus.channel] == opens + 1


def test_no_gripper_goal_comes_within_its_margin_of_the_open_end() -> None:
    rig = Rig(metal_fakes.FakeClock(), pose_deg={**POSE_DEG, 7: metal_fakes.GRIPPER_OPEN_DEG}).open()
    sent = len(rig.motor("gripper").commands)
    joints = np.radians([POSE_DEG[i] for i in range(1, 7)])
    rig.tick()
    for _ in range(120):
        rig.arm.write(joints, home.HOME_GRIPPER)
        rig.tick()
    rig.arm.request_rest()
    while not rig.arm.rested.is_set():
        rig.tick()

    open_goal_deg = rig.arm.stroke.to_deg(metal.GRIP_SCALE_MM - metal.ARM.grip_margin_mm)
    assert open_goal_deg < metal_fakes.GRIPPER_OPEN_DEG - 2.0
    assert max(rig.motor("gripper").commands[sent:]) == pytest.approx(open_goal_deg, abs=2 * Q_STEP_DEG)
    assert rig.arm.rest_result == (True, "at rest")


def test_torque_off_returns_to_the_open_pose_then_switches_every_motor_off() -> None:
    """The arm has no brakes: torque off parks it back at the pose it was opened in, the gripper open, then disables
    every motor."""
    rig = Rig(metal_fakes.FakeClock()).open()
    rig.tick(90, target=pose_target({**POSE_DEG, 1: 40.0, 3: 50.0}, 60.0))
    assert rig.motor("shoulder_pan").position_deg == pytest.approx(40.0, abs=2 * Q_STEP_DEG)
    rig.arm.start()
    arm = followers.FollowerArm("right", rig.arm, home_rad=np.zeros(6))

    arm.torque_off()

    assert rig.arm.rest_result == (True, "at rest")
    assert not rig.arm.is_alive()
    tolerance_deg = math.degrees(motor_arm.REST_TOL_RAD)
    for can_id in range(1, 7):
        assert rig.bus.motors[can_id].position_deg == pytest.approx(POSE_DEG[can_id], abs=tolerance_deg), can_id
    stroke_mm = rig.arm.stroke.to_mm(rig.motor("gripper").position_deg)
    assert stroke_mm >= rig.arm.open_goal_mm - motor_arm.REST_TOL_MM
    assert not any(motor.enabled for motor in rig.bus.motors.values())
    assert rig.bus.closed
    assert not arm.is_alive()


# open_arm()


def test_open_arm_needs_a_hold_torque() -> None:
    """No hold torque, no arm: a Metal gripper always holds; refused before any adapter is opened."""
    factory = metal_fakes.FakeCanFactory()

    with pytest.raises(errors.ConfigError, match="always holds"):
        motor_arm.open_arm(
            metal.ARM,
            config.ArmConfig(can_channel=CHANNEL, **metal_fakes.GRIPPER_ENDS, gripper_hold_torque_nm=None),
            "right",
            opener=factory,
        )
    assert factory.opens == {}


def test_open_arm_opens_a_metal_arm_and_refuses_a_lost_zero() -> None:
    """open_arm() hands back a live FollowerArm holding where it was; a lost zero leaves the adapter closed."""
    cfg = config.ArmConfig(can_channel=CHANNEL, **metal_fakes.GRIPPER_ENDS)
    factory = metal_fakes.FakeCanFactory()
    for can_id, deg in POSE_DEG.items():
        factory.bus(CHANNEL).motors[can_id].position_deg = deg

    arm = motor_arm.open_arm(metal.ARM, cfg, "right", opener=factory)
    try:
        state = arm.read()
        np.testing.assert_allclose(state.positions_rad, np.radians([POSE_DEG[i] for i in range(1, 7)]), atol=1e-3)
        assert state.gripper == pytest.approx(0.25, abs=1e-3)
    finally:
        arm.close()

    factory = metal_fakes.FakeCanFactory()
    factory.bus(CHANNEL).motors[3].position_deg = -90.0
    with pytest.raises(damiao.StartupPose):
        motor_arm.open_arm(metal.ARM, cfg, "right", opener=factory)
    assert factory.buses[CHANNEL].closed
    assert not any(motor.enabled for motor in factory.buses[CHANNEL].motors.values())


def test_open_arm_refuses_a_motor_still_warm_energizing_nothing_and_opens_it_once_cooled() -> None:
    cfg = config.ArmConfig(can_channel=CHANNEL, **metal_fakes.GRIPPER_ENDS)
    factory = metal_fakes.FakeCanFactory()
    for can_id, deg in POSE_DEG.items():
        factory.bus(CHANNEL).motors[can_id].position_deg = deg
    elbow = factory.bus(CHANNEL).motors[3]
    elbow.temp_rotor_c = int(metal.ARM.temp_warn_c) + 3

    with pytest.raises(
        errors.ArmError,
        match=r"^the right arm's elbow_flex is at 58 C \(coil\): let it cool below 55 C, then try again \(on a "
        r"station, press Reset\)$",
    ):
        motor_arm.open_arm(metal.ARM, cfg, "right", opener=factory)
    assert factory.buses[CHANNEL].closed
    assert not any(motor.enabled for motor in factory.buses[CHANNEL].motors.values())

    elbow.temp_rotor_c = int(metal.ARM.temp_warn_c) - 1
    arm = motor_arm.open_arm(metal.ARM, cfg, "right", opener=factory)
    try:
        assert arm.is_alive()
    finally:
        arm.close()


@pytest.mark.parametrize("fault", ["open", "send"])
def test_an_adapter_that_fails_at_open_names_the_arms_adapter_and_what_to_do(
    fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(metal_fakes.FakeClock())
    if fault == "open":
        rig.factory.fail_opens.add(CHANNEL)
        error = "CanInitializationError: could not open can_right"
    else:

        def dead(self: Any, message: Any, timeout: float | None = None) -> None:
            raise can.CanOperationError("Could not write to serial device")

        monkeypatch.setattr(metal_fakes.FakeCanBus, "send", dead)
        error = "CanOperationError: Could not write to serial device"
    with pytest.raises(errors.ArmError) as failed:
        rig.chain.open()
    assert str(failed.value) == (
        f"the right arm's CAN adapter can_right failed ({error}): unplug the adapter and plug it back in, or switch "
        "the arm's power off and on, then try again (on a station, press Reset)"
    )
    assert not rig.chain.bus.is_open


def mit_torque(data: bytes, model: str) -> float:
    """The torque, N.m, out of an MIT command frame (its last 12 bits)."""
    tmax = damiao.MOTOR_LIMITS[model][2]
    return metal_fakes.uint_to_float(((data[6] & 0xF) << 8) | data[7], -tmax, tmax, 12)


def test_each_joint_frame_carries_the_gravity_torque_of_the_pose_it_read() -> None:
    """MakerMods' SDK follower's feedforward: each joint's MIT torque is the arm's gravity torque at the read pose."""
    rig = fake_rig(pose_deg={**POSE_DEG, 2: -100.0, 3: 60.0})
    rig.send(pose_goal({**POSE_DEG, 2: -100.0, 3: 60.0}))
    want = metal.gravity_nm(tuple(rig.chain.read().q_deg[:6]))
    assert abs(want[1]) > 3.0
    for joint, torque in zip(metal.JOINTS[:6], want, strict=True):
        motor = rig.motor(joint.name)
        frames = [data for can_id, data in rig.bus.frames if can_id == motor.can_id and data[:7] != b"\xff" * 7]
        step = 2 * damiao.MOTOR_LIMITS[motor.model][2] / 4095
        assert mit_torque(frames[-1], motor.model) == pytest.approx(torque, abs=step), joint.name


def _potential_energy(q_deg: list[float]) -> float:
    """The links' potential energy, J, at `q_deg`, by homogeneous transforms written out apart from metal.py's."""
    t = np.eye(4)
    energy = 0.0
    for (xyz, (r, p, y), axis, mass, com), angle in zip(metal.LINKS, q_deg, strict=True):
        cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
        origin = np.eye(4)
        origin[:3, :3] = [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
        origin[:3, 3] = xyz
        a, c, s = np.asarray(axis), math.cos(math.radians(angle)), math.sin(math.radians(angle))
        turn = np.eye(4)
        turn[:3, :3] = c * np.eye(3) + s * np.cross(np.eye(3), a) + (1 - c) * np.outer(a, a)
        t = t @ origin @ turn
        energy += mass * metal.GRAVITY_M_S2 * (t @ np.array([*com, 1.0]))[2]
    return energy


@pytest.mark.parametrize(
    "q_deg",
    [
        [0.0] * 6,
        [5.0, -12.0, 15.0, -5.0, 4.0, -6.0],
        [43.5, -124.3, 61.3, 5.2, -10.9, -5.7],
        [-30, -60, 120, -90, 45, 90],
    ],
)
def test_the_gravity_torque_is_the_slope_of_the_arms_potential_energy(q_deg: list[float]) -> None:
    """The torque that holds a joint against gravity is the slope of the potential energy along it (per radian), times
    the SDK's factor; shoulder, elbow and wrist torques measured on a Metal arm match this model (slopes
    1.00, 1.03, 0.98)."""
    d = 1e-3
    for i, (torque, factor) in enumerate(zip(metal.gravity_nm(tuple(q_deg)), metal.GRAVITY_FACTORS, strict=True)):
        up, down = list(q_deg), list(q_deg)
        up[i] += d
        down[i] -= d
        slope = (_potential_energy(up) - _potential_energy(down)) / math.radians(2 * d)
        assert torque == pytest.approx(factor * slope, abs=1e-4), i

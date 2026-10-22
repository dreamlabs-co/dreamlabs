"""Unit tests of the shared bus-arm thread (`robots.motor_arm`) on the Makers table (`robots.makers`) and its chain
(`robots.robstride`), on the ported fake bus.

Most tests step the thread's tick by hand on a fake clock (`makers_fakes.FakeClock`), so rates are exact; the ones
about the thread itself and the station's arm wrapper run it on the wall clock over the same fake motors. The
parity with the source Makers driver (a JSON snapshot of its frames and joints, `data/makers_parity.json`) is
`test_makers_parity.py`; the station end to end is
`test_makers_station.py`.
"""

import logging
import math
import pathlib
import time
import types as pytypes
from collections.abc import Callable
from typing import Any

import can
import makers_fakes
import numpy as np
import pytest
import setup_fakes

from robots import config, errors, followers, home, makers, makers_hold, motor_arm, robstride, types

CHANNEL = "can_right"

## The source's gripper ends, this package's ArmConfig defaults.
GRIPPER_DEG = (-96.0, -2.5)

## A pose inside every soft limit, near zero so a return to rest is short: degrees by CAN id.
POSE_DEG = {1: 5.0, 2: -12.0, 3: 15.0, 4: -5.0, 5: 4.0, 6: -6.0, 7: -50.0}

## Per-tick steps of the thread, degrees.
SYNC_STEP_DEG = motor_arm.SYNC_SPEED_DEG_S / makers.ARM.tick_hz
MAX_STEP_DEG = motor_arm.MAX_SPEED_DEG_S / makers.ARM.tick_hz

## The gripper's stroke at the default ends.
STROKE = motor_arm.LinearStroke(*GRIPPER_DEG, makers.GRIPPER_MAX_MM)
OPEN_GOAL_MM = makers.GRIPPER_MAX_MM - makers.ARM.grip_margin_mm

## The six joints' names, base to wrist.
JOINT_NAMES = [joint.name for joint in makers.JOINTS]

## One count of a 16-bit MIT position, degrees: what a commanded angle is quantized to.
Q_STEP_DEG = math.degrees(2 * robstride.MOTOR_LIMITS["O0"][0] / 65535)


def wait_for(predicate: Callable[[], bool], what: str, timeout_s: float = 5.0) -> None:
    """Polls `predicate` on the wall clock, failing the test after `timeout_s`."""
    deadline = time.monotonic() + timeout_s
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail(f"timed out waiting for {what}")
        time.sleep(0.005)


class Rig:
    """One arm on a fake bus: the factory and the arm's thread over its chain, on one clock."""

    def __init__(
        self,
        clock: Any,
        pose_deg: dict[int, float] | None = None,
        hold_torque_nm: float | None = makers_hold.DEFAULT_HOLD_TORQUE_NM,
    ) -> None:
        self.clock = clock
        self.factory = makers_fakes.FakeCanFactory(clock)
        for can_id, deg in (POSE_DEG if pose_deg is None else pose_deg).items():
            self.factory.bus(CHANNEL).motors[can_id].position_deg = deg
        cfg = config.ArmConfig(can_channel=CHANNEL, gripper_hold_torque_nm=hold_torque_nm)
        self.arm = motor_arm.MotorArm(makers.ARM, "right", cfg, opener=self.factory, clock=clock, sleep=clock.sleep)
        self.chain = self.arm.chain

    @property
    def bus(self) -> makers_fakes.FakeCanBus:
        """The live bus."""
        return self.factory.buses[CHANNEL]

    def motor(self, name: str) -> makers_fakes.FakeMotor:
        """The fake motor of a joint."""
        return self.bus.motors[self.chain.bus.motors[name].can_id]

    def open(self) -> "Rig":
        """Opens and energizes the arm."""
        self.chain.open()
        self.chain.enable()
        return self

    def tick(self, n: int = 1, target: list[float] | None = None) -> None:
        """`n` ticks, each with `target` set first when given, timed as the thread's own run times them."""
        for _ in range(n):
            if target is not None:
                self.arm.set_target(np.asarray(target, dtype=np.float64))
            t0 = self.clock()
            self.arm.tick()
            self.clock.sleep(max(0.0, self.arm.period - (self.clock() - t0)))


def fake_rig(**kwargs: Any) -> Rig:
    """An opened arm on a fake clock."""
    return Rig(makers_fakes.FakeClock(), **kwargs).open()


def pose_target(deg: dict[int, float], gripper_mm: float) -> list[float]:
    """`[6 joints rad, gripper mm]` from joint degrees by CAN id."""
    return [math.radians(deg[i]) for i in range(1, 7)] + [gripper_mm]


# The wire


def test_mit_frames_round_trip_through_the_fake_motor() -> None:
    """A position frame decodes back to its angle within one count, and its kp rides in it."""
    data = robstride.encode_mit("O1", 150.0, 4.5, -40.0)
    assert abs(makers_fakes.decode_mit_position(data, "O1") - -40.0) < Q_STEP_DEG
    assert makers_fakes.decode_mit_kp(data) == pytest.approx(150.0, abs=0.13)
    state = robstride.decode_feedback(makers_fakes.encode_feedback(3, "O1", 12.5, torque=1.5), "O1", t=7.0)
    assert (state.motor_id, state.t) == (3, 7.0)
    assert state.position_deg == pytest.approx(12.5, abs=Q_STEP_DEG)
    assert state.torque_nm == pytest.approx(1.5, abs=0.01)
    assert robstride.is_fault_frame(bytes([3]) + (0x10).to_bytes(4, "little") + b"\x00\x00\x00")
    assert not robstride.is_fault_frame(makers_fakes.encode_feedback(3, "O1", 0.0))


def test_open_names_a_motor_that_does_not_answer_and_closes_the_adapter() -> None:
    """The handshake is a state query per motor; a silent one fails the open with its name and the arm's, not
    just the adapter's, nothing energized."""
    rig = Rig(makers_fakes.FakeClock())
    del rig.factory.bus(CHANNEL).motors[4]

    with pytest.raises(errors.ArmError, match=r"^arm 'right' did not open: can_right: .*wrist_flex did not answer"):
        rig.chain.open()

    assert rig.bus.closed
    assert not any(motor.enabled for motor in rig.bus.motors.values())


def test_a_canable_opens_directly_unless_an_slcand_holds_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[dict[str, Any]] = []
    fake_can = pytypes.SimpleNamespace(
        interface=pytypes.SimpleNamespace(Bus=lambda **kwargs: opened.append(kwargs) or kwargs)
    )
    monkeypatch.setattr(robstride, "python_can", lambda: fake_can)
    proc, net, dev = tmp_path / "proc", tmp_path / "net", tmp_path / "dev"
    dev.mkdir()
    for name in ("ttyACM0", "ttyACM1", "ttyACM2"):
        (dev / name).write_text("")
    setup_fakes.net_link(net, "can1", up=True, operstate="unknown")
    setup_fakes.slcand(proc, 30, "-o", "-f", "-s8", str(dev / "ttyACM1"), "can1")
    setup_fakes.slcand(proc, 31, "-o", "-s8", str(dev / "ttyACM2"))
    monkeypatch.setattr(robstride, "PROC_DIR", proc)
    monkeypatch.setattr(robstride, "NET_DIR", net)
    monkeypatch.setattr(robstride, "SERIAL_DEVICE_PREFIX", f"{dev}/")

    robstride.open_can_bus(str(dev / "ttyACM0"))
    robstride.open_can_bus(str(dev / "ttyACM1"))
    robstride.open_can_bus("can_follower_l")
    assert opened == [
        {"channel": str(dev / "ttyACM0"), "interface": "slcan", "bitrate": robstride.BITRATE},
        {"channel": "can1", "interface": "socketcan"},
        {"channel": "can_follower_l", "interface": "socketcan"},
    ]
    setup_fakes.net_link(net, "can1", up=False)
    with pytest.raises(errors.ArmError, match=r"as CAN link can1, which is down: run sudo ip link set can1 up"):
        robstride.open_can_bus(str(dev / "ttyACM1"))
    with pytest.raises(errors.ArmError, match=r"under a link it did not name: stop that slcand \(sudo kill 31\)"):
        robstride.open_can_bus(str(dev / "ttyACM2"))


def test_a_canable_without_pyserial_says_to_install_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    class Missing(Exception):
        pass

    def refuse(**kwargs: Any) -> None:
        raise Missing("The serial module is not installed")

    fake_can = pytypes.SimpleNamespace(
        CanInterfaceNotImplementedError=Missing, interface=pytypes.SimpleNamespace(Bus=refuse)
    )
    monkeypatch.setattr(robstride, "python_can", lambda: fake_can)
    monkeypatch.setattr(robstride, "PROC_DIR", tmp_path / "proc")
    with pytest.raises(errors.ConfigError, match=r"needs pyserial .*pip install pyserial, then start dreamcli setup"):
        robstride.open_can_bus("/dev/serial/by-id/usb-Protofusion_Labs_CANable_AAA-if00")


# Safety checks at open


def test_a_joint_a_whole_turn_off_is_corrected_for_the_session() -> None:
    """A motor that reads 360 degrees off its pose after a power cycle is read and commanded through the offset."""
    rig = fake_rig(pose_deg={**POSE_DEG, 6: 354.0})

    reading = rig.chain.read()
    rig.tick(3)

    assert reading.q_deg[5] == pytest.approx(-6.0, abs=Q_STEP_DEG)
    assert rig.motor("wrist_roll").commands[-1] == pytest.approx(354.0, abs=Q_STEP_DEG)


def test_a_stale_zero_refuses_to_energize_and_to_command() -> None:
    """A joint outside its limits by more than the grace and not by a whole turn: the arm opens but never moves."""
    rig = Rig(makers_fakes.FakeClock(), pose_deg={**POSE_DEG, 2: 60.0})
    rig.chain.open()

    with pytest.raises(robstride.StaleZero, match="shoulder_lift at \\+60.0 deg"):
        rig.chain.enable()
    with pytest.raises(robstride.StaleZero):
        rig.chain.command([0.0, -20.0, 20.0, 0.0, 0.0, 0.0, -50.0])
    assert not any(motor.enabled for motor in rig.bus.motors.values())
    assert rig.motor("shoulder_lift").commands == []


def test_open_arm_refuses_a_stale_zero_and_leaves_the_adapter_closed() -> None:
    """open_arm() closes what it opened when the arm must not move."""
    factory = makers_fakes.FakeCanFactory()
    factory.bus(CHANNEL).motors[3].position_deg = -90.0

    with pytest.raises(robstride.StaleZero):
        motor_arm.open_arm(makers.ARM, config.ArmConfig(can_channel=CHANNEL), "right", opener=factory)

    assert factory.buses[CHANNEL].closed
    assert not any(motor.enabled for motor in factory.buses[CHANNEL].motors.values())


# The tick


def test_the_first_tick_holds_the_measured_pose() -> None:
    """Seeded from the arm's own pose: nothing jumps before a target arrives."""
    rig = fake_rig()
    rig.tick(3)

    for can_id, deg in POSE_DEG.items():
        assert rig.bus.motors[can_id].commands[-1] == pytest.approx(deg, abs=0.05), can_id


def test_a_new_target_eases_in_at_the_ramp_speed_then_tracks_at_the_cap() -> None:
    """30 deg/s (half a degree a tick) until synced, then 120 deg/s (two degrees a tick), never more."""
    rig = fake_rig()
    near = pose_target({**POSE_DEG, 1: POSE_DEG[1] + 6.0}, 40.0)
    rig.tick(4, target=near)
    pan = rig.motor("shoulder_pan").commands
    assert np.diff(pan[-3:]) == pytest.approx([SYNC_STEP_DEG] * 2, abs=2 * Q_STEP_DEG)

    rig.tick(40, target=near)
    assert rig.arm.stepper.synced
    far = pose_target({**POSE_DEG, 1: POSE_DEG[1] + 40.0}, 40.0)
    rig.tick(5, target=far)

    steps = np.diff(rig.motor("shoulder_pan").commands[-5:])
    assert steps == pytest.approx([MAX_STEP_DEG] * 4, abs=2 * Q_STEP_DEG)


def test_targets_are_clamped_to_the_soft_limits() -> None:
    """A target past a joint's limit stops at the limit; the gripper stops its margin short of its open end."""
    rig = fake_rig()
    beyond = pose_target({**POSE_DEG, 2: 30.0, 4: 150.0}, 200.0)
    rig.tick(400, target=beyond)

    assert max(rig.motor("shoulder_lift").commands) == pytest.approx(-3.2, abs=Q_STEP_DEG)
    assert max(rig.motor("wrist_flex").commands) == pytest.approx(104.0, abs=Q_STEP_DEG)
    assert min(rig.motor("gripper").commands) == pytest.approx(STROKE.to_deg(OPEN_GOAL_MM), abs=Q_STEP_DEG)


def test_no_gripper_goal_comes_within_its_margin_of_the_open_end() -> None:
    rig = fake_rig(pose_deg={**POSE_DEG, 7: GRIPPER_DEG[0]})
    sent = len(rig.motor("gripper").commands)
    rig.tick()
    for _ in range(120):
        rig.arm.write(np.zeros(6), home.HOME_GRIPPER)
        rig.tick()
    rig.arm.request_rest()
    while not rig.arm.rested.is_set():
        rig.tick()

    open_goal_deg = STROKE.to_deg(OPEN_GOAL_MM)
    assert open_goal_deg > GRIPPER_DEG[0] + 2.0
    assert min(rig.motor("gripper").commands[sent:]) == pytest.approx(open_goal_deg, abs=Q_STEP_DEG)
    assert rig.arm.rest_result == (True, "at rest")


def test_a_target_that_goes_stale_rearms_the_ramp() -> None:
    """No target for #TARGET_STALE_S: the arm holds, and the next target eases in at the ramp speed again."""
    rig = fake_rig()
    rig.tick(60, target=pose_target(POSE_DEG, 40.0))
    assert rig.arm.stepper.synced

    rig.tick(int(motor_arm.TARGET_STALE_S * makers.ARM.tick_hz) + 5)
    held = rig.motor("shoulder_pan").commands[-1]
    assert not rig.arm.stepper.synced
    rig.tick(3, target=pose_target({**POSE_DEG, 1: 40.0}, 40.0))

    pan = rig.motor("shoulder_pan").commands
    assert pan[-3] == pytest.approx(held + SYNC_STEP_DEG, abs=2 * Q_STEP_DEG)
    assert np.diff(pan[-3:]) == pytest.approx([SYNC_STEP_DEG] * 2, abs=2 * Q_STEP_DEG)


def test_published_gripper_is_the_stroke_and_the_loop_shape_is_its_fraction() -> None:
    """The thread publishes the stroke in mm (80 open); its driver side hands the loop that over 80."""
    rig = fake_rig()
    rig.tick(2)
    q, velocity = rig.arm.latest()
    expected_mm = STROKE.to_mm(POSE_DEG[7])

    assert q.dtype == np.float32 and velocity.shape == (6,)
    assert float(q[6]) == pytest.approx(expected_mm, abs=0.05)
    reading = rig.arm.read()
    assert reading.gripper == pytest.approx(expected_mm / makers.GRIPPER_MAX_MM, abs=1e-3)
    np.testing.assert_allclose(reading.positions_rad, np.radians([POSE_DEG[i] for i in range(1, 7)]), atol=1e-3)
    assert STROKE.to_deg(makers.GRIPPER_MAX_MM) == GRIPPER_DEG[0]
    assert STROKE.to_deg(0.0) == GRIPPER_DEG[1]


# The gripper's hold


@pytest.mark.parametrize("hold_torque_nm", [makers_hold.DEFAULT_HOLD_TORQUE_NM, None])
def test_closing_on_an_object_is_held_at_the_hold_torque(hold_torque_nm: float | None) -> None:
    """With the hold the gripper squeezes an object at about its torque; without it, at the full position gain."""
    rig = fake_rig(hold_torque_nm=hold_torque_nm)
    rig.motor("gripper").obstacle_deg = -30.0
    closed = pose_target(POSE_DEG, 0.0)

    rig.tick(240, target=closed)

    squeeze = rig.motor("gripper").torque_nm
    if hold_torque_nm is None:
        assert squeeze == pytest.approx(20.0 * math.radians(27.5), rel=0.02)
    else:
        assert rig.chain._hold is not None and rig.chain._hold.holding
        assert squeeze == pytest.approx(hold_torque_nm, abs=0.05)
        rig.tick(120, target=pose_target(POSE_DEG, makers.GRIPPER_MAX_MM))
        assert not rig.chain._hold.holding
        assert rig.motor("gripper").commands[-1] == pytest.approx(STROKE.to_deg(OPEN_GOAL_MM), abs=Q_STEP_DEG)


# Silence and faults


def test_an_arm_that_stops_answering_is_latched_down() -> None:
    """Every reply missed for #ARM_SILENT_S: the thread latches a fault and publishes nothing more."""
    rig = fake_rig()
    rig.tick(3)
    published = rig.arm.latest()[0].copy()
    rig.bus.drop_every = 1

    rig.tick(40)

    assert "no motor has answered" in rig.arm.fault
    np.testing.assert_array_equal(rig.arm.latest()[0], published)


def test_one_silent_motor_is_named_and_raises_the_arms_alarm_without_latching_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rig = fake_rig()
    rig.tick(2)
    motor = rig.bus.motors[rig.chain.bus.motors["wrist_yaw"].can_id]

    with caplog.at_level(logging.WARNING, logger="robots.robstride"):
        del rig.bus.motors[motor.can_id]
        rig.tick(int(robstride.JOINT_SILENT_S * makers.ARM.tick_hz) + 10)

    assert not rig.arm.fault
    assert any("wrist_yaw has not answered" in record.getMessage() for record in caplog.records)
    alarm = rig.arm.alarm()
    assert alarm is not None and alarm.startswith("wrist_yaw has not answered for 1.")
    assert alarm.endswith(
        "so the run stops and the arm parks, which cannot drive that motor: support the arm at the wrist_yaw joint "
        "while it parks, then check that motor before the next run"
    ), alarm
    rig.bus.motors[motor.can_id] = motor
    rig.tick(2)
    assert rig.arm.alarm() is None


def test_a_motor_fault_holds_the_arm_and_the_thread_keeps_ticking(caplog: pytest.LogCaptureFixture) -> None:
    """A fault word from one motor is logged once and the tick goes on; a cleared fault resumes the arm."""
    rig = Rig(makers_fakes.RealClock()).open()
    rig.arm.start()
    try:
        wait_for(rig.arm.first_pose.is_set, "the first pose")
        with caplog.at_level(logging.INFO, logger="robots.motor_arm"):
            rig.motor("gripper").fault_word = 0x08
            wait_for(lambda: "motor fault, holding" in caplog.text, "the fault line")
            time.sleep(0.1)
            assert rig.arm.is_alive()
            rig.motor("gripper").fault_word = 0
            wait_for(lambda: "ticking again" in caplog.text, "the recovery line")
        assert caplog.text.count("motor fault, holding") == 1
        assert rig.arm.is_alive()
    finally:
        rig.arm.stop()


def test_a_fault_that_does_not_clear_leaves_no_fresh_pose_and_a_read_is_refused() -> None:
    rig = Rig(makers_fakes.RealClock()).open()
    rig.arm.start()

    def fresh() -> bool:
        try:
            rig.arm.latest()
        except errors.ArmError:
            return False
        return True

    try:
        wait_for(rig.arm.first_pose.is_set, "the first pose")
        rig.motor("gripper").fault_word = 0x08
        time.sleep(motor_arm.POSE_STALE_S + 0.3)
        with pytest.raises(errors.ArmError, match=r"arm 'right' has published no fresh pose for 1\.\d s \(.*fault"):
            rig.arm.latest()
        assert rig.arm.is_alive()
        rig.motor("gripper").fault_word = 0
        wait_for(fresh, "a fresh pose")
    finally:
        rig.arm.stop()


# The station's arm


def open_follower(clock: Any, pose_deg: dict[int, float] | None = None) -> tuple[followers.FollowerArm, Rig]:
    """A FollowerArm over a Makers Arm whose thread runs on `clock`."""
    rig = Rig(clock, pose_deg=pose_deg).open()
    rig.arm.start()
    wait_for(rig.arm.first_pose.is_set, "the first pose")
    return followers.FollowerArm("right", rig.arm, home_rad=np.zeros(6)), rig


def test_the_follower_arm_reads_and_writes_a_makers_arm() -> None:
    """The loop's wrapper drives the Makers Arm unchanged: radians and a `[0, 1]` gripper both ways."""
    arm, rig = open_follower(makers_fakes.RealClock())
    try:
        state = arm.read()
        assert state.gripper == pytest.approx(STROKE.to_mm(-50.0) / 80.0, abs=1e-3)
        target = np.radians([POSE_DEG[i] for i in range(1, 7)]) + np.array([0.05, 0.0, 0.0, 0.0, 0.0, -0.05])
        arm.write(types.ArmCommand(positions_rad=target, gripper=0.75))

        def there() -> bool:
            reading = arm.read()
            return bool(np.allclose(reading.positions_rad, target, atol=1e-3)) and abs(reading.gripper - 0.75) < 1e-3

        wait_for(there, "the arm at the target")
        assert rig.motor("gripper").position_deg == pytest.approx(STROKE.to_deg(60.0), abs=Q_STEP_DEG)
        assert arm.is_alive()
    finally:
        arm.close()
    assert not arm.is_alive()
    assert all(motor.enabled for motor in rig.bus.motors.values()), "close() must leave the motors holding"


def test_torque_off_returns_to_rest_then_switches_every_motor_off() -> None:
    """The arm has no brakes: torque off parks it at rest (joints at zero, gripper open) first, then disables."""
    arm, rig = open_follower(makers_fakes.FakeClock())

    arm.torque_off()

    assert rig.arm.rest_result == (True, "at rest")
    assert not rig.arm.is_alive()
    assert not any(motor.enabled for motor in rig.bus.motors.values())
    assert rig.bus.closed
    for name in JOINT_NAMES:
        assert abs(rig.motor(name).position_deg) <= math.degrees(motor_arm.REST_TOL_RAD), name
    gripper_mm = STROKE.to_mm(rig.motor("gripper").position_deg)
    assert gripper_mm >= OPEN_GOAL_MM - motor_arm.REST_TOL_MM
    assert not arm.is_alive()


def test_torque_off_on_a_latched_arm_leaves_it_holding_and_says_so() -> None:
    arm, rig = open_follower(makers_fakes.FakeClock())
    rig.arm.fault = "test: latched"
    assert not arm.is_alive()
    names = [*JOINT_NAMES, "gripper"]
    rig.arm.stop()
    before = {name: rig.motor(name).position_deg for name in names}

    with pytest.raises(
        errors.ArmError,
        match=r"^arm 'right' stopped answering while it held \(test: latched\); it still holds under power so it does "
        r"not drop: support it, then switch its power off$",
    ):
        arm.torque_off()

    assert rig.arm.rest_result is None
    assert {name: rig.motor(name).position_deg for name in names} == before
    assert all(motor.enabled for motor in rig.bus.motors.values())
    assert rig.bus.closed


def test_a_command_that_is_not_six_joints_is_refused() -> None:
    """The arm takes six joints and a gripper only."""
    rig = fake_rig()
    arm = followers.FollowerArm("right", rig.arm, home_rad=np.zeros(6))
    with pytest.raises(errors.ArmError, match="expected \\(6,\\)"):
        arm.write(types.ArmCommand(positions_rad=np.zeros(7), gripper=0.5))


@pytest.mark.parametrize("fault", ["open", "send"])
def test_an_adapter_that_fails_at_open_names_the_arms_adapter_and_what_to_do(
    fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(makers_fakes.FakeClock())
    if fault == "open":
        rig.factory.fail_opens.add(CHANNEL)
        error = "CanInitializationError: could not open can_right"
    else:

        def dead(self: Any, message: Any, timeout: float | None = None) -> None:
            raise can.CanOperationError("Could not write to serial device")

        monkeypatch.setattr(makers_fakes.FakeCanBus, "send", dead)
        error = "CanOperationError: Could not write to serial device"
    with pytest.raises(errors.ArmError) as failed:
        rig.chain.open()
    assert str(failed.value) == (
        f"the right arm's CAN adapter can_right failed ({error}): unplug the adapter and plug it back in, or switch "
        "the arm's power off and on, then try again (on a station, press Reset)"
    )
    assert not rig.chain.bus.is_open

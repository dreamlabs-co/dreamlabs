"""Unit tests of `robots.motor_guard.MotorGuard` on hand-made ticks."""

import logging
import math

import metal_fakes
import pytest

from robots import motor_guard

NAMES = ("j1", "j2", "j3", "j4", "j5", "j6", "gripper")
HZ = 60.0


def make_guard(clock: metal_fakes.FakeClock, *, holds_grip: bool = True) -> motor_guard.MotorGuard:
    """A guard on `clock`, warning at 55 C and stopping at 65 C, the gripper's motor at 2 degrees a mm."""
    return motor_guard.MotorGuard(
        "left",
        NAMES,
        6,
        temp_warn_c=55.0,
        temp_stop_c=65.0,
        holds_grip=holds_grip,
        to_deg=lambda mm: 2.0 * mm,
        clock=clock,
    )


def pose(j2_deg: float = 0.0, grip_mm: float = 40.0) -> list[float]:
    """`[6 joints rad, gripper mm]` with the second joint at `j2_deg`."""
    return [0.0, math.radians(j2_deg), 0.0, 0.0, 0.0, 0.0, grip_mm]


def push(
    guard: motor_guard.MotorGuard, clock: metal_fakes.FakeClock, goal: list[float], measured: list[float], ticks: int
) -> list[float]:
    """`ticks` ticks of an arm pinned at `measured` while asked for `goal`."""
    out = list(goal)
    for _ in range(ticks):
        out = guard.adjust(goal, measured, out)
        clock.sleep(1.0 / HZ)
    return out


def temps(**hot: dict[str, float]) -> tuple[dict[str, float], ...]:
    """Every motor's two sensors at 40 C, the ones named in `hot` replaced."""
    return tuple(hot.get(name, {"coil": 40.0, "driver board": 38.0}) for name in NAMES)


def test_a_blocked_joint_is_held_just_past_where_it_stopped_then_raises_the_alarm() -> None:
    clock = metal_fakes.FakeClock()
    guard = make_guard(clock)
    stuck = pose(10.0)

    out = push(guard, clock, pose(40.0), stuck, int(motor_guard.BLOCK_S * HZ) + 2)

    assert out[1] == pytest.approx(math.radians(10.0 + motor_guard.JOINT_HOLD_DEG))
    assert guard.alarm() is None
    out = push(guard, clock, pose(40.0), stuck, int(motor_guard.BLOCK_STOP_S * HZ) + 5)
    alarm = guard.alarm()
    assert alarm is not None and alarm.startswith("j2 has been blocked for 5 s (asked to go 30 deg further)")
    assert "Clear the way before the next run" in alarm

    out = guard.adjust(pose(5.0), stuck, out)
    assert out[1] == pytest.approx(math.radians(5.0)) and guard.alarm() is None


def test_a_gripper_blocked_opening_is_held_and_the_log_says_where_to_set_its_open_end(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = metal_fakes.FakeClock()
    guard = make_guard(clock)

    with caplog.at_level(logging.WARNING, logger="robots.motor_guard"):
        out = push(guard, clock, pose(grip_mm=80.0), pose(grip_mm=65.0), int(3 * HZ))

    assert out[6] == pytest.approx(65.0 + motor_guard.GRIP_HOLD_MM)
    assert len(caplog.records) == 1
    assert "stops opening at 130 deg, short of its open end 160 deg (15 mm short)" in caplog.text
    assert "set gripper_open_deg to about 130" in caplog.text
    assert guard.alarm() is None


@pytest.mark.parametrize(("holds_grip", "held_mm"), [(False, 30.0 - motor_guard.GRIP_SQUEEZE_MM), (True, 0.0)])
def test_a_gripper_closed_on_an_object_is_squeezed_gently_unless_the_chain_holds_it(
    holds_grip: bool, held_mm: float
) -> None:
    clock = metal_fakes.FakeClock()
    guard = make_guard(clock, holds_grip=holds_grip)

    out = push(guard, clock, pose(grip_mm=0.0), pose(grip_mm=30.0), int(1 * HZ))

    assert out[6] == pytest.approx(held_mm)


def test_a_hot_motor_is_logged_once_and_one_past_the_stop_raises_the_alarm_naming_it_and_its_sensor(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = metal_fakes.FakeClock()
    guard = make_guard(clock)

    with caplog.at_level(logging.WARNING, logger="robots.motor_guard"):
        for _ in range(3):
            guard.temperatures(temps(j3={"coil": 56.0, "driver board": 41.0}))
            clock.sleep(0.5)
        assert guard.alarm() is None
        for _ in range(3):
            guard.temperatures(temps(j3={"coil": 66.0, "driver board": 41.0}))
            clock.sleep(0.5)

    assert caplog.text.count("j3 coil at 56 C") == 1
    alarm = guard.alarm()
    assert alarm is not None and alarm.startswith("j3 coil reached 66 C (the run stops at 65 C)")
    assert "Let it cool below 55 C before the next run" in alarm
    assert guard.motor_temps_c == (40.0, 40.0, 66.0, 40.0, 40.0, 40.0, 40.0)
    guard.temperatures(temps(j3={"coil": 58.0, "driver board": 41.0}))
    assert guard.alarm() is not None
    guard.temperatures(temps())
    assert guard.alarm() is None


def test_one_frame_past_the_stop_temperature_is_never_acted_on() -> None:
    clock = metal_fakes.FakeClock()
    guard = make_guard(clock)

    guard.temperatures(temps(j1={"coil": 70.0, "driver board": 40.0}))
    clock.sleep(0.5)
    guard.temperatures(temps())
    clock.sleep(2.0)
    guard.temperatures(temps())

    assert guard.alarm() is None


def test_tracking_as_the_metal_arms_did_on_2026_10_12_never_trips_the_guard(caplog: pytest.LogCaptureFixture) -> None:
    clock = metal_fakes.FakeClock()
    guard = make_guard(clock)
    step_deg = 120.0 / HZ
    commands: list[list[float]] = []
    for target_deg, grip_mm in [(60.0, 80.0), (-30.0, 10.0), (20.0, 50.0)]:
        start_deg = math.degrees(commands[-1][1]) if commands else 0.0
        moves = int(abs(target_deg - start_deg) / step_deg)
        sign = math.copysign(1.0, target_deg - start_deg)
        commands += [pose(start_deg + sign * step_deg * (k + 1), grip_mm) for k in range(moves)]
        commands += [pose(target_deg, grip_mm)] * int(2 * HZ)

    with caplog.at_level(logging.INFO, logger="robots.motor_guard"):
        for k, command in enumerate(commands):
            lagged = commands[max(0, k - 4)]
            measured = [*lagged[:6], lagged[6] - 0.3]
            measured[1] -= math.radians(6.8)
            out = guard.adjust(command, measured, commands[max(0, k - 1)])
            assert out == command, k
            guard.temperatures(temps())
            clock.sleep(1.0 / HZ)

    assert not caplog.records
    assert guard.alarm() is None

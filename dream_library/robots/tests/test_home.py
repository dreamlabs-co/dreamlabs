"""Behaviour tests for `robots.home.return_to_home` and its CLI `main` against the fake robots.

The move is paced on an injected clock whose `sleep` advances it, so no test waits out the
three seconds; the fake robots record every 7-vector written, which is what the assertions
read: one command per step, joints shrinking monotonically to zero, the gripper opening to 1.0.
"""

import logging
from collections.abc import Callable

import conftest
import numpy as np
import pytest

from robots import followers, home

## Joint pose the arms rest at before the move: mixed signs, so "monotone toward zero" is a real check.
REST_RAD = np.array([1.0, -0.5, 0.3, 0.2, -0.1, 0.4])

## Gripper opening before the move; home opens it fully.
REST_GRIPPER = 0.3

## The home command: zeros and the gripper open.
HOME_COMMAND = [0.0] * conftest.ARM_DOF + [1.0]


class PacedClock:
    """A clock whose sleep advances it, so the move's pacing is checked without wall time.

    Attributes:
    - now_s: Current time, seconds.
    - sleeps: Every duration handed to sleep(), in order.
    - on_sleep: Called with the sleep index before each sleep; a test injects a fault mid-move here.
    """

    def __init__(self) -> None:
        self.now_s = 100.0
        self.sleeps: list[float] = []
        self.on_sleep: Callable[[int], None] | None = None

    def __call__(self) -> float:
        return self.now_s

    def sleep(self, dt_s: float) -> None:
        if self.on_sleep is not None:
            self.on_sleep(len(self.sleeps))
        self.sleeps.append(dt_s)
        self.now_s += max(0.0, dt_s)


def make_arms() -> tuple[dict[str, followers.FollowerArm], dict[str, conftest.FakeRobot]]:
    """Two fake robots at #REST_RAD / #REST_GRIPPER wrapped as FollowerArms."""
    robots = {name: conftest.FakeRobot(joint_pos=REST_RAD, gripper_pos=REST_GRIPPER) for name in ("left", "right")}
    return {name: conftest.yam_arm(name, robot) for name, robot in robots.items()}, robots


def test_home_interpolates_every_step_to_zeros_with_the_gripper_open(caplog: pytest.LogCaptureFixture) -> None:
    """One command per step, each joint's magnitude never growing, the gripper never closing, the last step
    exactly the home pose, then a hold that re-sends it; the move is announced once at INFO."""
    arms, robots = make_arms()
    clock = PacedClock()

    with caplog.at_level(logging.INFO, logger="robots.home"):
        ok = home.return_to_home(arms, duration_s=3.0, steps=120, clock=clock, sleep=clock.sleep)

    assert ok
    for robot in robots.values():
        # 120 interpolation steps and the final hold, which re-sends the last command.
        assert len(robot.commands) == 121
        commands = np.array(robot.commands)
        np.testing.assert_allclose(commands[-1], HOME_COMMAND)
        np.testing.assert_allclose(commands[-2], HOME_COMMAND)
        # The first step is one 120th of the way.
        np.testing.assert_allclose(commands[0][:6], REST_RAD * (119 / 120))
        np.testing.assert_allclose(commands[0][6], REST_GRIPPER + (1.0 - REST_GRIPPER) / 120)
        assert (np.diff(np.abs(commands[:120, :6]), axis=0) <= 1e-12).all()
        assert (np.diff(commands[:120, 6]) >= -1e-12).all()
    # Paced at 40 Hz: one sleep per step of the whole period, since the fake writes take no clock time.
    assert len(clock.sleeps) == 120
    np.testing.assert_allclose(clock.sleeps, 3.0 / 120)
    started = [r for r in caplog.records if "moving ['left', 'right'] home" in r.getMessage()]
    assert len(started) == 1 and started[0].levelno == logging.INFO
    assert "largest joint delta 1.000 rad" in started[0].getMessage()


def test_home_goes_to_each_arms_own_home() -> None:
    """An arm whose config names a home other than zeros ends there, the gripper open; the other arm still at zeros."""
    robots = {name: conftest.FakeRobot(joint_pos=REST_RAD, gripper_pos=REST_GRIPPER) for name in ("left", "right")}
    arms = {name: conftest.yam_arm(name, robot) for name, robot in robots.items()}
    left_home = np.array([0.1, -1.2, 1.0, 0.0, 0.2, -0.3])
    arms["left"] = followers.FollowerArm("left", arms["left"]._driver, home_rad=left_home)
    clock = PacedClock()

    assert home.return_to_home(arms, duration_s=1.0, steps=10, clock=clock, sleep=clock.sleep)

    np.testing.assert_array_equal(robots["left"].commands[-1], [*left_home, 1.0])
    np.testing.assert_array_equal(robots["right"].commands[-1], HOME_COMMAND)
    np.testing.assert_allclose(robots["left"].commands[0][:6], REST_RAD + (left_home - REST_RAD) / 10)


def test_a_write_fault_mid_move_drops_that_arm_and_finishes_the_other(caplog: pytest.LogCaptureFixture) -> None:
    """A driver exception on one arm at step 11 is logged, that arm holds what it last accepted (no more writes),
    the other arm still reaches home, and the result names that arm and why."""
    arms, robots = make_arms()
    clock = PacedClock()

    def fault_after_ten_steps(index: int) -> None:
        if index == 9:
            robots["left"].write_error = RuntimeError("CAN bus down")

    clock.on_sleep = fault_after_ten_steps

    with caplog.at_level(logging.ERROR, logger="robots.home"):
        missed = home.move_home(arms, duration_s=3.0, steps=120, clock=clock, sleep=clock.sleep)

    assert list(missed) == ["left"] and missed["left"].startswith("its write failed at step 11/120: ")
    assert "CAN bus down" in missed["left"]
    # Ten accepted steps; the eleventh raised and nothing was written to the left arm after it.
    assert len(robots["left"].commands) == 10
    assert len(robots["right"].commands) == 121
    np.testing.assert_allclose(robots["right"].commands[-1], HOME_COMMAND)
    failed = [r.getMessage() for r in caplog.records if "write failed at step 11/120" in r.getMessage()]
    assert len(failed) == 1 and "arm 'left'" in failed[0]
    assert any("arms ['left'] did not reach home" in r.getMessage() for r in caplog.records)


def test_an_arm_that_cannot_be_read_is_not_moved(caplog: pytest.LogCaptureFixture) -> None:
    """A read fault before the move leaves that arm untouched, homes the other, and returns False."""
    arms, robots = make_arms()
    robots["right"].read_error = RuntimeError("no observation")
    clock = PacedClock()

    with caplog.at_level(logging.ERROR, logger="robots.home"):
        ok = home.return_to_home(arms, duration_s=1.0, steps=10, clock=clock, sleep=clock.sleep)

    assert not ok
    assert robots["right"].commands == []
    assert len(robots["left"].commands) == 11
    np.testing.assert_allclose(robots["left"].commands[-1], HOME_COMMAND)
    assert any("arm 'right' read failed" in r.getMessage() for r in caplog.records)


def test_no_readable_arm_moves_nothing_and_returns_false() -> None:
    """Both reads failing is reported as False without a single write and without raising."""
    arms, robots = make_arms()
    for robot in robots.values():
        robot.read_error = RuntimeError("no observation")
    clock = PacedClock()

    assert not home.return_to_home(arms, duration_s=1.0, steps=10, clock=clock, sleep=clock.sleep)
    assert all(robot.commands == [] for robot in robots.values())
    assert clock.sleeps == []


def test_a_measured_gripper_outside_the_unit_range_is_clipped_before_the_blend() -> None:
    """Sensor noise can report 1.02; the origin is clipped so every step validates and the last is exactly home."""
    arms, robots = make_arms()
    robots["left"].gripper_pos = np.array([1.02])
    clock = PacedClock()

    assert home.return_to_home(arms, duration_s=1.0, steps=4, clock=clock, sleep=clock.sleep)
    grippers = [command[6] for command in robots["left"].commands]
    assert grippers == [1.0] * 5


def test_main_builds_both_arms_from_the_config_and_returns_the_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """`robots.home.main` opens `follower_l` as left and `follower_r` as right through the injected builder,
    homes them and returns 0, or 1 when an arm failed."""
    arms, robots = make_arms()
    built: list[tuple[str, str]] = []

    def build_arm(arm_cfg: object, name: str) -> followers.FollowerArm:
        built.append((arm_cfg.can_channel, name))
        return arms[name]

    monkeypatch.setattr(home.time, "sleep", lambda dt_s: None)
    cfg = conftest.make_app_config()

    assert home.main(cfg, duration_s=0.1, build_arm=build_arm) == 0
    assert built == [("can_left", "left"), ("can_right", "right")]
    for robot in robots.values():
        np.testing.assert_allclose(robot.commands[-1], HOME_COMMAND)

    robots["left"].read_error = RuntimeError("no observation")
    assert home.main(cfg, duration_s=0.1, build_arm=build_arm) == 1

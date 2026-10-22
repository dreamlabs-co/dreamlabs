"""Pins `robots.followers.FollowerArm`: the read and write contract, hold and disengage, and the i2rt seam.

Every test drives the wrapper through `conftest.FakeRobot`, which has the i2rt `Robot` shape
and records the 7-vectors it was commanded; nothing here touches i2rt or hardware.
"""

import math
import pathlib
import sys
import types as pytypes
from typing import Any

import conftest
import numpy as np
import pytest
import test_config

from robots import config, errors, followers, types, yam

## A distinctive joint target vector, radians, so a test can tell it from the rest position.
TARGET_RAD = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])


def _arm(robot: conftest.FakeRobot, name: str = "left") -> followers.FollowerArm:
    return conftest.yam_arm(name, robot)


# --- read -------------------------------------------------------------------------------------


def test_read_assembles_arm_state_from_one_driver_observation() -> None:
    """The read path reports joint positions, velocities and gripper in radians / [0, 1]."""
    robot = conftest.FakeRobot(joint_pos=np.arange(6), gripper_pos=0.25)
    robot.joint_vel = np.full(6, 0.5)

    state = _arm(robot).read()

    assert isinstance(state, types.ArmState)
    np.testing.assert_array_equal(state.positions_rad, np.arange(6))
    np.testing.assert_array_equal(state.velocities_rad_s, np.full(6, 0.5))
    assert state.gripper == 0.25
    assert state.positions_rad.dtype == np.float64
    assert state.velocities_rad_s.dtype == np.float64


def test_commanded_fields_are_none_before_the_first_write(fake_robot: conftest.FakeRobot) -> None:
    """The two `commanded_*` fields are None before the first write."""
    state = _arm(fake_robot).read()

    assert state.commanded_rad is None
    assert state.commanded_gripper is None


def test_commanded_fields_are_populated_after_a_write(fake_robot: conftest.FakeRobot) -> None:
    """State carries the commanded targets (joints and gripper) once the arm has been written."""
    arm = _arm(fake_robot)
    arm.write(types.ArmCommand(positions_rad=TARGET_RAD, gripper=0.75))

    state = arm.read()

    assert state.commanded_rad is not None
    np.testing.assert_array_equal(state.commanded_rad, TARGET_RAD)
    assert state.commanded_gripper == 0.75
    # The measured side is still what the driver reported, not the command.
    np.testing.assert_array_equal(state.positions_rad, np.zeros(6))


def test_commanded_fields_track_the_latest_write(fake_robot: conftest.FakeRobot) -> None:
    """`commanded_*` is the most recent command, not the first."""
    arm = _arm(fake_robot)
    arm.write(types.ArmCommand(positions_rad=TARGET_RAD, gripper=0.1))
    arm.write(types.ArmCommand(positions_rad=-TARGET_RAD, gripper=0.9))

    state = arm.read()

    assert state.commanded_rad is not None
    np.testing.assert_array_equal(state.commanded_rad, -TARGET_RAD)
    assert state.commanded_gripper == 0.9


@pytest.mark.parametrize("missing_key", ["joint_pos", "joint_vel", "gripper_pos"])
def test_read_raises_arm_error_when_the_observation_lacks_a_key(missing_key: str) -> None:
    """Read validates six joints and a gripper present, else ArmError naming the arm."""
    robot = conftest.FakeRobot()
    full = robot.get_observations()
    robot.get_observations = lambda: {k: v for k, v in full.items() if k != missing_key}  # type: ignore[method-assign]

    with pytest.raises(errors.ArmError, match=f"arm 'right'.*{missing_key}"):
        _arm(robot, "right").read()


def test_read_raises_arm_error_on_wrong_joint_count() -> None:
    """An observation with five joints is malformed, not silently padded."""
    robot = conftest.FakeRobot()
    robot.joint_pos = np.zeros(5)

    with pytest.raises(errors.ArmError, match="joint_pos"):
        _arm(robot).read()


def test_read_raises_arm_error_on_multi_valued_gripper() -> None:
    """`gripper_pos` must hold exactly one value."""
    robot = conftest.FakeRobot()
    robot.gripper_pos = np.array([0.1, 0.2])

    with pytest.raises(errors.ArmError, match="gripper_pos"):
        _arm(robot).read()


def test_read_raises_arm_error_when_the_observation_is_not_a_dict() -> None:
    """A driver that returns something other than a dict is caught by the wrapper."""
    robot = conftest.FakeRobot()
    robot.get_observations = lambda: None  # type: ignore[method-assign]

    with pytest.raises(errors.ArmError):
        _arm(robot).read()


# --- write ------------------------------------------------------------------------------------


def test_write_sends_the_seven_vector_positions_then_gripper(fake_robot: conftest.FakeRobot) -> None:
    """Write sends `[*positions_rad, gripper]` as one 7-vector."""
    _arm(fake_robot).write(types.ArmCommand(positions_rad=TARGET_RAD, gripper=0.3))

    assert len(fake_robot.commands) == 1
    assert fake_robot.commands[0].shape == (7,)
    np.testing.assert_array_equal(fake_robot.commands[0][:6], TARGET_RAD)
    assert fake_robot.commands[0][6] == 0.3


def test_write_accepts_the_gripper_range_boundaries(fake_robot: conftest.FakeRobot) -> None:
    """The gripper is a float in [0, 1], boundaries included (0 closed, 1 open)."""
    arm = _arm(fake_robot)

    arm.write(types.ArmCommand(positions_rad=np.zeros(6), gripper=0.0))
    arm.write(types.ArmCommand(positions_rad=np.zeros(6), gripper=1.0))

    assert [command[6] for command in fake_robot.commands] == [0.0, 1.0]


@pytest.mark.parametrize("shape", [(5,), (7,), (6, 1), (1, 6), ()])
def test_write_raises_arm_error_on_wrong_shape(fake_robot: conftest.FakeRobot, shape: tuple[int, ...]) -> None:
    """Write validates shape [6]; anything else is ArmError and nothing is sent."""
    with pytest.raises(errors.ArmError, match="shape"):
        _arm(fake_robot).write(types.ArmCommand(positions_rad=np.zeros(shape), gripper=0.5))

    assert fake_robot.commands == []


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_write_raises_arm_error_on_non_finite_joint_target(fake_robot: conftest.FakeRobot, bad: float) -> None:
    """Write validates finite values; a NaN or inf target never reaches the driver."""
    positions_rad = np.zeros(6)
    positions_rad[2] = bad

    with pytest.raises(errors.ArmError, match="non-finite"):
        _arm(fake_robot).write(types.ArmCommand(positions_rad=positions_rad, gripper=0.5))

    assert fake_robot.commands == []


@pytest.mark.parametrize("gripper", [-0.01, 1.01, 2.0, np.nan])
def test_write_raises_arm_error_on_gripper_outside_unit_range(fake_robot: conftest.FakeRobot, gripper: float) -> None:
    """Gripper outside [0, 1] (NaN included) is ArmError and nothing is sent."""
    with pytest.raises(errors.ArmError, match="gripper"):
        _arm(fake_robot).write(types.ArmCommand(positions_rad=np.zeros(6), gripper=gripper))

    assert fake_robot.commands == []


def test_rejected_write_does_not_update_commanded_fields(fake_robot: conftest.FakeRobot) -> None:
    """`commanded_*` reflects what reached the arm, so a rejected command leaves it untouched."""
    arm = _arm(fake_robot)
    with pytest.raises(errors.ArmError):
        arm.write(types.ArmCommand(positions_rad=np.zeros(6), gripper=1.5))

    state = arm.read()

    assert state.commanded_rad is None
    assert state.commanded_gripper is None


def test_arm_error_message_names_the_arm(fake_robot: conftest.FakeRobot) -> None:
    """Messages name the component and the fact."""
    with pytest.raises(errors.ArmError, match="arm 'right'"):
        _arm(fake_robot, "right").write(types.ArmCommand(positions_rad=np.zeros(3), gripper=0.5))


# --- the gripper command is not shaped here --------------------------------------------------


@pytest.mark.parametrize("gripper", [0.0, 0.05, 0.19, 0.2, 0.5, 1.0])
def test_write_sends_the_gripper_target_untouched(fake_robot: conftest.FakeRobot, gripper: float) -> None:
    """What the loop hands down is what reaches the wire, a near-closed target included: there is no snap to
    closed, because grip force is regulated on the measured effort (`robots.gripper_force`)."""
    arm = _arm(fake_robot)

    arm.write(types.ArmCommand(positions_rad=TARGET_RAD, gripper=gripper))

    assert fake_robot.commands[0][6] == gripper
    assert arm.read().commanded_gripper == gripper


def test_hold_resends_the_gripper_target_as_written(fake_robot: conftest.FakeRobot) -> None:
    """What was written is what hold() repeats, with nothing rounded to closed in between."""
    arm = _arm(fake_robot)
    arm.write(types.ArmCommand(positions_rad=TARGET_RAD, gripper=0.1))

    arm.hold()

    assert [command[6] for command in fake_robot.commands] == [0.1, 0.1]


# --- gripper effort ----------------------------------------------------------------------------


def test_read_reports_the_gripper_effort_when_the_driver_has_it(fake_robot: conftest.FakeRobot) -> None:
    """i2rt's `gripper_eff` (motor torque units) is passed through raw as `ArmState.gripper_effort`."""
    observation = fake_robot.get_observations()
    observation["gripper_eff"] = np.array([-1.25])
    fake_robot.get_observations = lambda: observation  # type: ignore[method-assign]

    assert _arm(fake_robot).read().gripper_effort == -1.25


def test_read_reports_no_gripper_effort_without_the_key(fake_robot: conftest.FakeRobot) -> None:
    """A driver that reports no effort (the fakes) reads fine, with `gripper_effort` None."""
    assert _arm(fake_robot).read().gripper_effort is None


def test_read_raises_arm_error_on_multi_valued_gripper_effort(fake_robot: conftest.FakeRobot) -> None:
    """A present but malformed `gripper_eff` is a driver fault, reported like a malformed `gripper_pos`."""
    observation = fake_robot.get_observations()
    observation["gripper_eff"] = np.array([1.0, 2.0])
    fake_robot.get_observations = lambda: observation  # type: ignore[method-assign]

    with pytest.raises(errors.ArmError, match="'gripper_eff' has 2 values"):
        _arm(fake_robot).read()


# --- hold -------------------------------------------------------------------------------------


def test_hold_recommands_the_last_written_command(fake_robot: conftest.FakeRobot) -> None:
    """Hold re-commands the last written targets (the "hold last command" of the stall rows)."""
    arm = _arm(fake_robot)
    arm.write(types.ArmCommand(positions_rad=TARGET_RAD, gripper=0.4))

    arm.hold()

    assert len(fake_robot.commands) == 2
    np.testing.assert_array_equal(fake_robot.commands[1], fake_robot.commands[0])
    # No read was needed: the hold came from memory, not from the driver.
    state = arm.read()
    assert state.commanded_rad is not None
    np.testing.assert_array_equal(state.commanded_rad, TARGET_RAD)


def test_hold_before_any_write_commands_the_current_position() -> None:
    """With nothing written yet, hold commands the measured position (safe hold)."""
    robot = conftest.FakeRobot(joint_pos=TARGET_RAD, gripper_pos=0.6)
    arm = _arm(robot)

    arm.hold()

    assert len(robot.commands) == 1
    np.testing.assert_array_equal(robot.commands[0][:6], TARGET_RAD)
    assert robot.commands[0][6] == pytest.approx(0.6)
    # The hold counts as a command from now on.
    state = arm.read()
    assert state.commanded_rad is not None
    np.testing.assert_array_equal(state.commanded_rad, TARGET_RAD)
    assert state.commanded_gripper == pytest.approx(0.6)


@pytest.mark.parametrize(("measured", "expected"), [(1.2, 1.0), (-0.05, 0.0)])
def test_hold_clips_a_measured_gripper_slightly_outside_unit_range(measured: float, expected: float) -> None:
    """A safe hold must never fail validation on sensor noise, so the gripper is clipped."""
    robot = conftest.FakeRobot(gripper_pos=measured)

    _arm(robot).hold()

    assert robot.commands[0][6] == expected


# --- disengage --------------------------------------------------------------------------------


def test_disengage_prefers_gravity_comp_idle() -> None:
    """Disengage uses `enter_gravity_comp_idle()` when the driver has it."""
    robot = conftest.FakeRobot(has_gravity_comp=True, has_zero_torque=True)

    _arm(robot).disengage()

    assert robot.gravity_comp_calls == 1
    assert robot.zero_torque_calls == 0
    assert robot.commands == []


def test_disengage_falls_back_to_zero_torque() -> None:
    """Without gravity comp, disengage uses `zero_torque_mode()`."""
    robot = conftest.FakeRobot(has_zero_torque=True)

    _arm(robot).disengage()

    assert robot.zero_torque_calls == 1
    assert robot.commands == []


def test_disengage_falls_back_to_hold_when_the_driver_has_neither() -> None:
    """A driver with neither passive mode is held instead."""
    robot = conftest.FakeRobot(joint_pos=TARGET_RAD)

    _arm(robot).disengage()

    assert len(robot.commands) == 1
    np.testing.assert_array_equal(robot.commands[0][:6], TARGET_RAD)


def test_disengage_hold_fallback_reuses_the_last_command(fake_robot: conftest.FakeRobot) -> None:
    """The hold fallback of disengage is the same hold, so it re-sends what was last written."""
    arm = _arm(fake_robot)
    arm.write(types.ArmCommand(positions_rad=TARGET_RAD, gripper=0.2))

    arm.disengage()

    assert len(fake_robot.commands) == 2
    np.testing.assert_array_equal(fake_robot.commands[1], fake_robot.commands[0])


# --- driver exceptions ------------------------------------------------------------------------


def test_driver_read_exception_propagates_unchanged() -> None:
    """Driver exceptions travel as-is; the wrapper never wraps them."""
    robot = conftest.FakeRobot()
    original = OSError("CAN bus down")
    robot.read_error = original

    with pytest.raises(OSError) as info:
        _arm(robot).read()

    assert info.value is original


def test_driver_write_exception_propagates_unchanged(fake_robot: conftest.FakeRobot) -> None:
    """A raising `command_joint_pos` surfaces the driver's own exception."""
    original = RuntimeError("motor fault")
    fake_robot.write_error = original

    with pytest.raises(RuntimeError) as info:
        _arm(fake_robot).write(types.ArmCommand(positions_rad=np.zeros(6), gripper=0.5))

    assert info.value is original


def test_driver_exception_during_hold_propagates_unchanged() -> None:
    """Hold's read of the driver is not shielded either."""
    robot = conftest.FakeRobot()
    original = OSError("CAN bus down")
    robot.read_error = original

    with pytest.raises(OSError) as info:
        _arm(robot).hold()

    assert info.value is original


def test_failed_driver_write_is_not_remembered_as_commanded(fake_robot: conftest.FakeRobot) -> None:
    """A command the driver rejected never reached the arm, so `commanded_*` stays as it was."""
    fake_robot.write_error = RuntimeError("motor fault")
    arm = _arm(fake_robot)
    with pytest.raises(RuntimeError):
        arm.write(types.ArmCommand(positions_rad=TARGET_RAD, gripper=0.5))

    assert arm.read().commanded_rad is None


# --- from_config ------------------------------------------------------------------------------


def _hide_i2rt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Makes every `i2rt` import fail, even when the extra happens to be installed."""
    for name in ("i2rt", "i2rt.robots", "i2rt.robots.get_robot", "i2rt.robots.utils"):
        monkeypatch.setitem(sys.modules, name, None)


def _install_fake_i2rt(monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]]) -> None:
    """Installs a minimal `i2rt` package tree whose get_yam_robot records its kwargs."""
    get_robot = pytypes.ModuleType("i2rt.robots.get_robot")
    utils = pytypes.ModuleType("i2rt.robots.utils")

    class GripperType(str):
        pass

    def get_yam_robot(channel: str, gripper_type: Any, ee_mass: float | None) -> conftest.FakeRobot:
        calls.append({"channel": channel, "gripper_type": gripper_type, "ee_mass": ee_mass})
        robot = conftest.FakeRobot()
        # The attributes i2rt's MotorChainRobot carries and from_config overwrites, at i2rt's own defaults.
        robot.temp_record_flag = False  # type: ignore[attr-defined]
        robot.gravity_comp_factor = np.ones(7)  # type: ignore[attr-defined]
        robot._limit_gripper_force = 50.0  # type: ignore[attr-defined]
        return robot

    get_robot.get_yam_robot = get_yam_robot  # type: ignore[attr-defined]
    utils.GripperType = GripperType  # type: ignore[attr-defined]
    robots_pkg = pytypes.ModuleType("i2rt.robots")
    robots_pkg.get_robot = get_robot  # type: ignore[attr-defined]
    robots_pkg.utils = utils  # type: ignore[attr-defined]
    i2rt_pkg = pytypes.ModuleType("i2rt")
    i2rt_pkg.robots = robots_pkg  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "i2rt", i2rt_pkg)
    monkeypatch.setitem(sys.modules, "i2rt.robots", robots_pkg)
    monkeypatch.setitem(sys.modules, "i2rt.robots.get_robot", get_robot)
    monkeypatch.setitem(sys.modules, "i2rt.robots.utils", utils)


def test_from_config_raises_config_error_when_i2rt_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """I2rt is an optional extra; without it from_config fails with ConfigError naming the install."""
    _hide_i2rt(monkeypatch)

    with pytest.raises(errors.ConfigError, match=r"pip install --upgrade dreamlabs") as info:
        yam.open_arm(config.ArmConfig(can_channel="can_left"), "left")

    assert "left" in str(info.value)
    assert isinstance(info.value.__cause__, ImportError)


def test_from_config_opens_the_configured_channel_and_gripper(monkeypatch: pytest.MonkeyPatch) -> None:
    """From_config builds the i2rt robot from the arm config's channel and gripper type."""
    calls: list[dict[str, Any]] = []
    _install_fake_i2rt(monkeypatch, calls)

    arm = yam.open_arm(config.ArmConfig(can_channel="can_right", gripper="linear_4310"), "right")

    assert isinstance(arm, followers.FollowerArm)
    assert arm.name == "right"
    assert calls == [{"channel": "can_right", "gripper_type": "linear_4310", "ee_mass": 0.62}]
    assert arm.read().commanded_rad is None


def test_from_config_opens_an_anvil_gripper_as_the_linear_4310_that_drives_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """`anvil_gripper` is not an i2rt name: the arm opens with the i2rt type `config.GRIPPER_DRIVERS` maps it to."""
    calls: list[dict[str, Any]] = []
    _install_fake_i2rt(monkeypatch, calls)

    yam.open_arm(config.ArmConfig(can_channel="can_left", gripper="anvil_gripper"), "left")

    assert calls[0]["gripper_type"] == "linear_4310"


def test_from_config_applies_the_fleet_driver_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """ee_mass reaches get_yam_robot; temp export, gravity comp factor and the SDK limiter are set on the
    driver."""
    calls: list[dict[str, Any]] = []
    _install_fake_i2rt(monkeypatch, calls)
    cfg = config.ArmConfig(can_channel="can_left", ee_mass_kg=0.5, gravity_comp_factor=0.9)

    arm = yam.open_arm(cfg, "left")

    robot = arm._driver.robot  # type: ignore[attr-defined]
    assert calls[0]["ee_mass"] == 0.5
    assert robot.temp_record_flag is True
    assert robot.gravity_comp_factor == 0.9
    assert robot._limit_gripper_force == -1.0


def test_from_config_leaves_driver_settings_alone_when_the_config_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """Null / false leave i2rt's own values: no ee_mass, its per-joint gravity factors, its 50 N limiter, no temps."""
    calls: list[dict[str, Any]] = []
    _install_fake_i2rt(monkeypatch, calls)
    cfg = config.ArmConfig(
        can_channel="can_left",
        ee_mass_kg=None,
        gravity_comp_factor=None,
        sdk_gripper_force_limit=True,
        temp_export=False,
    )

    arm = yam.open_arm(cfg, "left")

    robot = arm._driver.robot  # type: ignore[attr-defined]
    assert calls[0]["ee_mass"] is None
    assert robot.temp_record_flag is False
    np.testing.assert_array_equal(robot.gravity_comp_factor, np.ones(7))
    assert robot._limit_gripper_force == 50.0


def test_from_config_warns_when_the_driver_has_no_sdk_limiter(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A driver without `_limit_gripper_force` is reported, never a silent no-op."""
    calls: list[dict[str, Any]] = []
    _install_fake_i2rt(monkeypatch, calls)
    original = sys.modules["i2rt.robots.get_robot"].get_yam_robot  # type: ignore[attr-defined]

    def bare_get_yam_robot(**kwargs: Any) -> conftest.FakeRobot:
        robot = original(**kwargs)
        del robot._limit_gripper_force
        return robot

    monkeypatch.setattr(sys.modules["i2rt.robots.get_robot"], "get_yam_robot", bare_get_yam_robot)

    with caplog.at_level("WARNING", logger="robots.yam"):
        yam.open_arm(config.ArmConfig(can_channel="can_left"), "left")

    assert any("_limit_gripper_force" in record.message and "left" in record.message for record in caplog.records)


def test_module_does_not_import_i2rt_at_load() -> None:
    """The core never needs i2rt; the modules' namespaces carry no i2rt binding."""
    i2rt_names = ("get_yam_robot", "GripperType", "DMSingleMotorCanInterface")
    for module in (followers, yam):
        assert not any(name.startswith("i2rt") or name in i2rt_names for name in vars(module))


def test_arm_error_is_a_robots_error() -> None:
    """Every package exception derives from RobotsError."""
    assert issubclass(errors.ArmError, errors.RobotsError)
    assert issubclass(errors.ConfigError, errors.RobotsError)


# --- driver liveness ---------------------------------------------------------------------------


class _DeadChain:
    """The shape of i2rt's `MotorChainRobot.motor_chain` once its CAN thread has exited."""

    running = False


def test_read_raises_when_the_driver_chain_has_stopped(fake_robot: conftest.FakeRobot) -> None:
    """A stopped i2rt motor chain fails the read instead of serving the last cached state forever."""
    fake_robot.motor_chain = _DeadChain()  # type: ignore[attr-defined]
    with pytest.raises(errors.ArmError, match="motor chain is no longer running"):
        _arm(fake_robot).read()


def test_write_raises_when_the_driver_chain_has_stopped(fake_robot: conftest.FakeRobot) -> None:
    """A stopped chain refuses the command, so the loop stops rather than commanding a dead arm."""
    fake_robot.motor_chain = _DeadChain()  # type: ignore[attr-defined]
    with pytest.raises(errors.ArmError, match="motor chain is no longer running"):
        _arm(fake_robot).write(types.ArmCommand(positions_rad=TARGET_RAD, gripper=0.5))
    assert fake_robot.commands == []


# --- torque off ---------------------------------------------------------------------------------


class _LiveChain:
    """The shape of i2rt's `DMChainCanInterface` while it runs: its channel and its `(id, type)` motor list."""

    running = True

    def __init__(self, channel: str, motor_ids: list[int]) -> None:
        self.channel = channel
        self.motor_list = [(motor_id, "DM4310") for motor_id in motor_ids]


def _install_fake_dm_driver(
    monkeypatch: pytest.MonkeyPatch, events: list[Any], failing_ids: set[int] = frozenset()
) -> None:
    """Installs a fake `i2rt.motor_drivers.dm_driver` whose single-motor interface records what it is asked."""
    dm_driver = pytypes.ModuleType("i2rt.motor_drivers.dm_driver")

    class DMSingleMotorCanInterface:
        def __init__(self, channel: str) -> None:
            events.append(("open", channel))

        def motor_off(self, motor_id: int) -> None:
            if motor_id in failing_ids:
                raise RuntimeError(f"motor {motor_id} did not answer")
            events.append(("off", motor_id))

        def close(self) -> None:
            events.append(("close",))

    dm_driver.DMSingleMotorCanInterface = DMSingleMotorCanInterface  # type: ignore[attr-defined]
    motor_drivers = pytypes.ModuleType("i2rt.motor_drivers")
    motor_drivers.dm_driver = dm_driver  # type: ignore[attr-defined]
    i2rt_pkg = pytypes.ModuleType("i2rt")
    i2rt_pkg.motor_drivers = motor_drivers  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "i2rt", i2rt_pkg)
    monkeypatch.setitem(sys.modules, "i2rt.motor_drivers", motor_drivers)
    monkeypatch.setitem(sys.modules, "i2rt.motor_drivers.dm_driver", dm_driver)


def test_torque_off_closes_the_driver_then_disables_every_motor_on_its_chain(
    monkeypatch: pytest.MonkeyPatch, fake_robot: conftest.FakeRobot
) -> None:
    """close() sends nothing and the motors hold, so after it the DM disable
    command goes to each motor id of the chain, in chain order, over an interface on the arm's channel."""
    events: list[Any] = []
    _install_fake_dm_driver(monkeypatch, events)
    fake_robot.motor_chain = _LiveChain("can_left", [1, 2, 3, 4, 5, 6, 7])  # type: ignore[attr-defined]
    fake_robot.close = lambda: events.append(("driver closed",))  # type: ignore[method-assign]
    arm = _arm(fake_robot)

    arm.torque_off()

    assert events == [
        ("driver closed",),
        ("open", "can_left"),
        *(("off", motor_id) for motor_id in range(1, 8)),
        ("close",),
    ]
    assert not arm.is_alive()


def test_torque_off_on_a_driver_without_a_chain_only_closes(fake_robot: conftest.FakeRobot) -> None:
    """A sim robot or a fake has no motor chain, so there is nothing to switch off and i2rt is never imported."""
    arm = _arm(fake_robot)

    arm.torque_off()

    assert fake_robot.closed
    assert not arm.is_alive()
    assert not any(name.startswith("i2rt") for name in vars(yam))


def test_torque_off_skips_a_motor_that_does_not_answer_and_names_it(
    monkeypatch: pytest.MonkeyPatch, fake_robot: conftest.FakeRobot, caplog: pytest.LogCaptureFixture
) -> None:
    """One dead motor must not leave the others energized: the sweep goes on past it, the interface is still
    closed, and then the switch-off raises naming the ids that may still be live, so the station does not report
    the arm switched off."""
    events: list[Any] = []
    _install_fake_dm_driver(monkeypatch, events, failing_ids={3})
    fake_robot.motor_chain = _LiveChain("can_right", [1, 3, 7])  # type: ignore[attr-defined]

    with caplog.at_level("WARNING", logger="robots.yam"):
        with pytest.raises(errors.ArmError, match=r"motors \[3\] on can_right may still be energized"):
            _arm(fake_robot, "right").torque_off()

    assert events == [("open", "can_right"), ("off", 1), ("off", 7), ("close",)]
    assert any("motor 3" in record.message and "right" in record.message for record in caplog.records)


def test_torque_off_raises_when_the_channel_cannot_be_opened_after_closing_the_driver(
    monkeypatch: pytest.MonkeyPatch, fake_robot: conftest.FakeRobot
) -> None:
    """The driver is closed either way; a channel that will not open propagates, for the station's teardown to
    log as the failed step it is."""
    dm_driver = pytypes.ModuleType("i2rt.motor_drivers.dm_driver")

    def refuse(channel: str) -> None:
        raise OSError(f"no such device: {channel}")

    dm_driver.DMSingleMotorCanInterface = refuse  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "i2rt.motor_drivers.dm_driver", dm_driver)
    fake_robot.motor_chain = _LiveChain("can_gone", [1])  # type: ignore[attr-defined]

    with pytest.raises(OSError, match="can_gone"):
        _arm(fake_robot).torque_off()

    assert fake_robot.closed


# --- config: the driver settings ----------------------------------------------------------------


def test_arm_config_driver_settings_default_to_the_fleet_values(tmp_path: pathlib.Path) -> None:
    """The defaults: ee_mass 0.62 kg, gravity comp factor 1.0, the SDK gripper force limit off, temps on."""
    cfg = test_config.load_variant(tmp_path, follower_r={"can_channel": "can_right"})

    assert cfg.follower_r == config.ArmConfig(
        can_channel="can_right",
        gripper="linear_4310",
        ee_mass_kg=0.62,
        gravity_comp_factor=1.0,
        sdk_gripper_force_limit=False,
        temp_export=True,
    )


def test_arm_config_reads_every_driver_setting(tmp_path: pathlib.Path) -> None:
    """Every key lands typed; ints are accepted where a float is read; null means "leave i2rt's default"."""
    section = {
        "can_channel": "can_left",
        "ee_mass_kg": 1,
        "gravity_comp_factor": None,
        "sdk_gripper_force_limit": True,
        "temp_export": False,
    }

    arm = test_config.load_variant(tmp_path, follower_l=section).follower_l

    assert arm.ee_mass_kg == 1.0 and isinstance(arm.ee_mass_kg, float)
    assert arm.gravity_comp_factor is None
    assert arm.sdk_gripper_force_limit is True
    assert arm.temp_export is False


def test_arm_config_home_defaults_to_zeros_and_reads_six_angles(tmp_path: pathlib.Path) -> None:
    """A YAM's home is all zeros unless its section names six angles in radians."""
    assert test_config.load_variant(tmp_path, follower_r={"can_channel": "can_right"}).follower_r.home == (0.0,) * 6
    home = [0.1, -0.2, 0.3, 0, 0, 1]
    cfg = test_config.load_variant(tmp_path, follower_r={"can_channel": "can_right", "home": home})
    assert cfg.follower_r.home == (0.1, -0.2, 0.3, 0.0, 0.0, 1.0)


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("home", [0.0] * 17),
        ("home", "zeros"),
        ("home", [0, 0, 0, 0, 0, None]),
        ("ee_mass_kg", 0),
        ("ee_mass_kg", -0.5),
        ("ee_mass_kg", "0.62"),
        ("ee_mass_kg", True),
        ("ee_mass_kg", math.inf),
        ("gravity_comp_factor", 0),
        ("gravity_comp_factor", "1.0"),
        ("gravity_comp_factor", [1.0] * 7),
        ("sdk_gripper_force_limit", 1),
        ("sdk_gripper_force_limit", "false"),
        ("sdk_gripper_force_limit", None),
        ("temp_export", 0),
    ],
)
def test_arm_config_rejects_a_driver_setting_of_the_wrong_type_or_range(
    tmp_path: pathlib.Path, key: str, bad: Any
) -> None:
    """A wrong type or range fails at load with the dotted key in the message."""
    with pytest.raises(errors.ConfigError, match=f"follower_l.{key}"):
        test_config.load_variant(tmp_path, follower_l={"can_channel": "can_left", key: bad})


class _StoppedDriver(followers.Driver):
    """A driver whose arm stopped answering, saying why or not."""

    def __init__(self, why: str) -> None:
        self.why = why

    def read(self) -> followers.Reading:
        raise AssertionError("read after the driver stopped")

    def write(self, positions_rad: Any, gripper: float) -> None:
        raise AssertionError("write after the driver stopped")

    def close(self) -> None:
        pass

    def torque_off(self) -> None:
        pass

    def is_alive(self) -> bool:
        return False

    def why_stopped(self) -> str:
        return self.why


@pytest.mark.parametrize(
    ("why", "words"),
    [
        ("ArmError: servo 3 (elbow_flex) did not answer", "arm 'left' stopped: ArmError: servo 3 (elbow_flex) did"),
        ("", "arm 'left' stopped: its driver stopped answering. Check the arm's power"),
    ],
)
def test_a_stopped_driver_is_reported_with_its_reason_in_words_for_any_bus(why: str, words: str) -> None:
    arm = followers.FollowerArm("left", _StoppedDriver(why), home_rad=np.zeros(6))
    with pytest.raises(errors.ArmError) as raised:
        arm.read()
    assert words in str(raised.value) and "CAN" not in str(raised.value)


class _Recording(followers.Driver):
    """A driver that records how it was let go."""

    def __init__(self, drops_when_limp: bool, alive: bool) -> None:
        self.drops_when_limp = drops_when_limp
        self.alive = alive
        self.calls: list[str] = []

    def read(self) -> followers.Reading:
        raise AssertionError("not read")

    def write(self, positions_rad: Any, gripper: float) -> None:
        raise AssertionError("not written")

    def close(self) -> None:
        self.calls.append("close")

    def torque_off(self) -> None:
        self.calls.append("torque_off")

    def is_alive(self) -> bool:
        return self.alive


@pytest.mark.parametrize(
    ("drops_when_limp", "alive", "call"), [(True, True, "torque_off"), (True, False, "close"), (False, True, "close")]
)
def test_close_for_reopen_switches_off_only_a_brakeless_arm_that_still_answers(
    drops_when_limp: bool, alive: bool, call: str
) -> None:
    driver = _Recording(drops_when_limp, alive)
    arm = followers.FollowerArm("left", driver, home_rad=np.zeros(6))

    arm.close_for_reopen()

    assert driver.calls == [call] and not arm.is_alive()

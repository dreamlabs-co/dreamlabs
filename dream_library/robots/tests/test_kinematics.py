"""Tests for `robots.kinematics`: the import guard and, when mujoco and i2rt are installed, the real YAM model.

The loop's clamp and plan tests use linear stand-ins (`test_control.py`); the one test here that
touches MuJoCo pins the facts the clamp relies on: `grasp_site` exists, the zero pose puts it in
front of the base, the base joint swings it about the vertical axis, and the finger slides do not
move it. Skipped without the `hardware` extra.
"""

import sys

import numpy as np
import pytest

from robots import config, errors, kinematics, robstride, yam


def test_the_module_imports_without_mujoco_and_the_constructor_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`import robots.kinematics` never needs mujoco; building an `ArmKinematics` without it is a ConfigError
    naming `pip install --upgrade dreamlabs`, and so is `from_config` without i2rt."""
    monkeypatch.setitem(sys.modules, "mujoco", None)
    with pytest.raises(errors.ConfigError, match=r"mujoco.*pip install --upgrade dreamlabs"):
        kinematics.ArmKinematics("unused.xml")

    monkeypatch.setitem(sys.modules, "i2rt", None)
    with pytest.raises(errors.ConfigError, match=r"i2rt.*pip install --upgrade dreamlabs"):
        kinematics.ArmKinematics.from_config(config.ArmConfig(can_channel="can_left"))


def test_a_missing_arm_package_names_pip_install_dreamlabs(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Makers or Metal arm without python-can, and a YAM follower without i2rt, each say to install the library
    from PyPI, which carries both: never a wheel from Setup API or a `dream-robots` extra, a name not on PyPI."""
    monkeypatch.setitem(sys.modules, "can", None)
    with pytest.raises(errors.ConfigError, match=r"python-can is not installed.*run pip install --upgrade dreamlabs$"):
        robstride.python_can()

    for module in ("i2rt", "i2rt.robots", "i2rt.robots.get_robot", "i2rt.robots.utils"):
        monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(errors.ConfigError, match=r"follower 'left' needs i2rt.*: pip install --upgrade dreamlabs$"):
        yam.open_arm(config.ArmConfig(can_channel="can0", home=(0.0,) * 6), "left")


def test_the_real_yam_model_places_the_grasp_site_and_ignores_the_fingers() -> None:
    """On the i2rt YAM + linear_4310 model: the zero pose puts `grasp_site` about a quarter metre in front of the
    base and above it, a base yaw of 1 rad rotates it about z (same radius, same height), the finger slides
    leave it where it is, and every call returns a fresh array."""
    pytest.importorskip("mujoco")
    pytest.importorskip("i2rt")
    arm = kinematics.ArmKinematics.from_config(config.ArmConfig(can_channel="can_left"))

    at_zero = arm.ee_position(np.zeros(6))
    assert at_zero.shape == (3,) and at_zero.dtype == np.float64
    assert 0.15 < at_zero[0] < 0.4 and abs(at_zero[1]) < 0.02 and 0.1 < at_zero[2] < 0.3

    yawed = arm.ee_position(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert np.hypot(*yawed[:2]) == pytest.approx(np.hypot(*at_zero[:2]), abs=1e-9)
    assert yawed[2] == pytest.approx(at_zero[2], abs=1e-9)
    assert np.arctan2(yawed[1], yawed[0]) - np.arctan2(at_zero[1], at_zero[0]) == pytest.approx(1.0, abs=1e-9)

    # The finger slides (qpos[6:8]) are never written: set them by hand and the site must not move.
    arm._data.qpos[6:8] = 0.03
    np.testing.assert_allclose(arm.ee_position(np.zeros(6)), at_zero, atol=1e-12)
    assert arm.ee_position(np.zeros(6)) is not at_zero


def test_an_anvil_gripper_builds_the_arm_on_the_linear_4310_model_that_drives_it() -> None:
    """`anvil_gripper` is not an i2rt name: the model is built with the i2rt type `config.GRIPPER_DRIVERS` maps it
    to, as `FollowerArm.from_config` opens the driver, so a station set to it starts instead of failing with
    `'anvil_gripper' is not a valid GripperType`."""
    pytest.importorskip("mujoco")
    pytest.importorskip("i2rt")
    anvil = kinematics.ArmKinematics.from_config(config.ArmConfig(can_channel="can_left", gripper="anvil_gripper"))
    yam = kinematics.ArmKinematics.from_config(config.ArmConfig(can_channel="can_left", gripper="linear_4310"))

    joints = np.array([0.3, -0.2, 0.1, 0.4, -0.5, 0.2])
    np.testing.assert_allclose(anvil.ee_position(joints), yam.ee_position(joints), atol=1e-12)


def test_a_gripper_name_nobody_knows_still_fails_in_from_config() -> None:
    """A name that is neither i2rt's nor in `config.GRIPPER_DRIVERS` is handed to i2rt as it is and refused there."""
    pytest.importorskip("mujoco")
    pytest.importorskip("i2rt")
    with pytest.raises(ValueError, match="no_such_gripper"):
        kinematics.ArmKinematics.from_config(config.ArmConfig(can_channel="can_left", gripper="no_such_gripper"))

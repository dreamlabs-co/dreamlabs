"""The i2rt YAM's driver: #I2rtDriver, a #robots.followers.Driver over an i2rt robot; the only module that knows i2rt.

open_arm() opens one arm from its #robots.config.ArmConfig and carries the follower-process settings
(end-effector mass, gravity compensation factor, SDK gripper force limiter, temperature export) onto the i2rt driver.
The i2rt robot's joints are the YAM URDF's and its gripper is already 0 closed to 1 open, so the driver converts
nothing: it reads the observation dict and sends the 7-vector command. Grip force is regulated a level up, by
#robots.gripper_force.

System 0 (#robots.sys0) writes the arm a level below that: write_motor_command() swaps in a whole MIT-mode block the
way i2rt's own command methods do, and set_motor_loop_hook() runs a check on i2rt's motor loop before every update,
which is where a stalled tracker is caught. open_arm(fast_motor_loop=True) opens the arm behind
#robots.sys0.fast_loop, the faster i2rt transport the tracker was trained behind.

@see open_arm(), I2rtDriver
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from robots import config, errors, followers
from robots.sys0 import fast_loop

_log = logging.getLogger(__name__)

## Observation keys i2rt's `Robot.get_observations()` must carry for one read.
_JOINT_POS_KEY = "joint_pos"
_JOINT_VEL_KEY = "joint_vel"
_GRIPPER_POS_KEY = "gripper_pos"
## Observation key i2rt's `MotorChainRobot` adds: the gripper motor's effort, in its torque units. Optional,
## because a driver without it (the test fakes) still reads.
_GRIPPER_EFF_KEY = "gripper_eff"
## A YAM arm's joints; i2rt's command vector is these followed by one gripper value.
JOINTS = 6
## What `MotorChainRobot.update()` reads as "no gripper force limiter": it skips the limiter when the stored
## limit is `<= 0`, and `get_yam_robot` hardcodes 50 N otherwise.
_SDK_GRIPPER_FORCE_LIMIT_OFF = -1.0


class I2rtDriver(followers.Driver):
    """A #robots.followers.Driver over an object with the i2rt `Robot` shape.

    The robot has `get_observations()` returning a dict with `joint_pos` `[6]`, `joint_vel` `[6]` and `gripper_pos`
    `[1]` (plus `gripper_eff` `[1]` when the driver has it), `command_joint_pos()` taking a `[7]` vector, and
    optionally `enter_gravity_comp_idle()` / `zero_torque_mode()`, a `motor_chain` with `running`, `channel` and
    `motor_list`, and `close()`.

    Attributes:
    - name: `"left"` or `"right"`, for messages.
    - robot: The i2rt robot.

    @see read(), write(), close(), torque_off(), is_alive(), disengage(), write_motor_command(), set_motor_loop_hook()
    """

    def __init__(self, name: str, robot: Any) -> None:
        """Wraps an opened i2rt robot; nothing is commanded here.

        @param name `"left"` or `"right"`, for messages.
        @param robot The robot; see the class docstring.
        """
        self.name = name
        self.robot = robot

    def read(self) -> followers.Reading:
        """One `get_observations()`.

        @return The reading; `gripper_effort` is i2rt's raw `gripper_eff` when it reports one, else None.
        @throws ArmError The observation lacks six joints or a gripper value.
        """
        observation = self.robot.get_observations()
        return followers.Reading(
            positions_rad=_joint_vector(self.name, observation, _JOINT_POS_KEY),
            velocities_rad_s=_joint_vector(self.name, observation, _JOINT_VEL_KEY),
            gripper=_scalar_value(self.name, observation, _GRIPPER_POS_KEY),
            gripper_effort=(
                None
                if _GRIPPER_EFF_KEY not in observation
                else _scalar_value(self.name, observation, _GRIPPER_EFF_KEY)
            ),
        )

    def write(self, positions_rad: npt.NDArray[np.float64], gripper: float) -> None:
        """Sends `[*positions_rad, gripper]` as one `command_joint_pos()`.

        @param positions_rad Six joint targets, radians.
        @param gripper The gripper target, 0 closed to 1 open.
        """
        self.robot.command_joint_pos(np.append(positions_rad, gripper))

    def is_alive(self) -> bool:
        """False when i2rt's motor chain has stopped, which i2rt otherwise hides.

        i2rt's CAN thread clears `robot.motor_chain.running` and exits when a motor stops answering, but
        `get_observations()` keeps returning the last state and `command_joint_pos()` keeps accepting targets, so a
        loop would run blind on a dead arm (seen on a YAM station: 36 s of ticks after the bus died). A robot without
        the flag (the test fakes) is taken as alive.

        @return False once i2rt's motor chain stopped.
        """
        chain = getattr(self.robot, "motor_chain", None)
        return chain is None or getattr(chain, "running", True) is not False

    def why_stopped(self) -> str:
        """Why is_alive() is False: i2rt's motor chain stopped, which it does when a motor stops answering.

        @return The reason.
        """
        return "its motor chain is no longer running (a motor stopped answering on the CAN bus)"

    def close(self) -> None:
        """Stops i2rt's threads and closes the CAN channel; it sends nothing, so the motors keep their last target.

        A robot without `close()` (the test fakes, a sim robot) is left alone.
        """
        close = getattr(self.robot, "close", None)
        if close is not None:
            close()

    def torque_off(self) -> None:
        """Closes the robot, then switches every motor on its chain off, so the arm goes limp.

        What close() leaves out: i2rt's `MotorChainRobot.close()` stops its thread and shuts the CAN socket without
        sending a frame, and a DM motor keeps the last MIT target it was given when the bus goes quiet (the firmware
        CAN timeout is 0 on a YAM's motors), so a closed arm stays stiff until its power is cut. This is the
        same sweep a YAM follower process runs on shutdown: the chain's channel and motor ids are read before close()
        shuts the bus, then _switch_off_motors() sends the DM disable command to each id over an interface of its own.
        A robot without a motor chain (a sim robot, the test fakes) is only closed. The arm drops under gravity from
        wherever it was.

        @throws Exception Whatever i2rt raises when the channel cannot be opened for the sweep; the robot is closed by
            then. A motor that does not answer is logged and skipped, and once every other motor is off an ArmError
            names it.
        """
        chain = getattr(self.robot, "motor_chain", None)
        channel = getattr(chain, "channel", None)
        motor_ids = [int(entry[0]) for entry in getattr(chain, "motor_list", [])]
        self.close()
        if channel is None or not motor_ids:
            return
        _switch_off_motors(self.name, channel, motor_ids)

    def write_motor_command(
        self,
        target: npt.NDArray[np.float64],
        kp: npt.NDArray[np.float64],
        kd: npt.NDArray[np.float64],
        torque: npt.NDArray[np.float64],
    ) -> None:
        """Swaps in one whole MIT-mode block: the six joints' targets, gains and torques, the gripper's position.

        The block is what i2rt's motor loop sends every cycle, with gravity compensation added to the torque; it is
        replaced under the robot's command lock as `command_joint_pos()` replaces it. The gripper keeps the robot's
        own gains.

        @param target Position targets, `[7]`: six joints in radians, then the gripper in `[0, 1]`.
        @param kp Stiffness per joint, `[6]`.
        @param kd Damping per joint, `[6]`.
        @param torque Torque per joint on top of gravity compensation, `[6]`, newton metres.
        @throws ArmError The robot has no MIT command block (not an i2rt `MotorChainRobot`).
        """
        robot = self.robot
        if not _has_command_block(robot):
            raise errors.ArmError(f"arm '{self.name}': the driver has no MIT command block to write")
        count = len(robot.motor_chain)
        # i2rt's JointCommands, taken from the block in place rather than imported on every 4 ms write.
        block = type(robot._commands).init_all_zero(count)
        block.pos = robot.remapper.to_robot_joint_pos_space(robot._clip_robot_joint_pos_command(np.array(target)))
        block.kp = np.concatenate([kp, np.asarray(robot._kp, dtype=np.float64)[JOINTS:]])
        block.kd = np.concatenate([kd, np.asarray(robot._kd, dtype=np.float64)[JOINTS:]])
        block.torques = np.concatenate([torque, np.zeros(count - JOINTS)])
        with robot._command_lock:
            robot._commands = block

    def set_motor_loop_hook(self, hook: Callable[[], None] | None) -> None:
        """Runs `hook` on i2rt's motor loop before every update, or removes it.

        i2rt's loop calls the robot's `update()` every cycle; the hook wraps it on the instance, so removing it leaves
        the robot's own method. A robot without the loop (the test fakes, a sim robot) takes no hook.

        @param hook A fast check to run before each update, or None to remove the one installed.
        """
        robot = self.robot
        if hook is None:
            robot.__dict__.pop("update", None)
            return
        update = getattr(type(robot), "update", None)
        if update is None:
            return

        def hooked_update() -> None:
            hook()
            update(robot)

        robot.update = hooked_update

    def disengage(self) -> bool:
        """Lets the arm go: gravity compensation, else zero torque.

        Under gravity compensation the arm floats where it is without fighting the operator; under zero torque it sags.

        @return False when the robot offers neither: the wrapper holds it.
        """
        for method in ("enter_gravity_comp_idle", "zero_torque_mode"):
            release = getattr(self.robot, method, None)
            if release is not None:
                release()
                return True
        return False


def open_arm(cfg: config.ArmConfig, name: str, *, fast_motor_loop: bool = False) -> followers.FollowerArm:
    """Opens the CAN channel named by the config, applies its driver settings and wraps the robot.

    i2rt is imported here and nowhere else, so the rest of the package (and every test) runs without the hardware
    extra installed. `ee_mass_kg` goes into `get_yam_robot` (its gravity model needs it before the first torque); the
    other settings are attributes set on the built robot, see _configure_driver().

    @param cfg Arm config: CAN channel, gripper type, the driver settings and the home pose.
    @param name `"left"` or `"right"`, for messages.
    @param fast_motor_loop Run i2rt's motor loops faster (#robots.sys0.fast_loop), which System 0 was trained behind;
        patched in before the robot starts its threads.
    @return A #robots.followers.FollowerArm over an #I2rtDriver.
    @throws ConfigError Its home does not list #JOINTS angles, or i2rt is not importable.
    """
    if len(cfg.home) != JOINTS:
        raise errors.ConfigError(
            f"arm '{name}': its home in the config lists {len(cfg.home)} angles, but a YAM arm has {JOINTS} joints"
        )
    try:
        # i2rt is not on PyPI, so it is an optional extra and mypy cannot see it either.
        from i2rt.robots.get_robot import get_yam_robot  # type: ignore[import-not-found]
        from i2rt.robots.utils import GripperType  # type: ignore[import-not-found]
    except ImportError as exc:
        raise errors.ConfigError(
            f"follower '{name}' needs i2rt, which is not installed; the dream library brings it: "
            "pip install --upgrade dreamlabs"
        ) from exc
    if fast_motor_loop:
        fast_loop.install_before_robot()
    robot = get_yam_robot(
        channel=cfg.can_channel, gripper_type=GripperType(config.i2rt_gripper(cfg.gripper)), ee_mass=cfg.ee_mass_kg
    )
    _configure_driver(name, robot, cfg)
    if fast_motor_loop:
        fast_loop.install_pipelined_bus(robot)
        _log.info("arm '%s': fast motor loop on %s", name, cfg.can_channel)
    return followers.FollowerArm(name, I2rtDriver(name, robot), home_rad=cfg.home)


def _configure_driver(name: str, robot: Any, cfg: config.ArmConfig) -> None:
    """Sets the driver attributes a YAM follower process sets, from the arm config.

    Each is guarded with `hasattr`, because `get_yam_robot` can return a
    driver class without it (a sim robot, another i2rt version) and a missing attribute must
    not stop the arm from opening; the one that matters for safety is logged when missing.

    - `temp_record_flag` (`temp_export`): i2rt reads every motor's temperatures anyway; the flag
      only gates their export in `get_observations()`.
    - `gravity_comp_factor`: i2rt stores a per-joint array (one entry per motor, gripper
      included) and multiplies it into the gravity torques on every update; this overwrites
      it with one scalar, which numpy broadcasts over every joint. The config value is that
      scalar; None leaves i2rt's per-joint defaults.
    - `_limit_gripper_force` (`sdk_gripper_force_limit`): `get_yam_robot` hardcodes a 50 N
      `GripperForceLimiter` (clog detect, then hold), and `update()` skips it when the stored
      limit is `<= 0`. It is off here, and the control loop regulates grip force itself
      instead, because the two fight (the SDK's clog threshold sits right at a low cap and its
      hold level above it), which shows up as a jittery grasp. The loop-side cap is
      #robots.gripper_force.GripperForceCap here, on by default; `sdk_gripper_force_limit: true`
      puts the SDK's limiter back, for a station running with `grip_force_cap: 0`.

    @param name `"left"` or `"right"`, for messages.
    @param robot The driver `get_yam_robot` returned.
    @param cfg The arm config carrying the settings.
    """
    if cfg.temp_export and hasattr(robot, "temp_record_flag"):
        robot.temp_record_flag = True
    if cfg.gravity_comp_factor is not None and hasattr(robot, "gravity_comp_factor"):
        robot.gravity_comp_factor = cfg.gravity_comp_factor
    if cfg.sdk_gripper_force_limit:
        return
    if hasattr(robot, "_limit_gripper_force"):
        robot._limit_gripper_force = _SDK_GRIPPER_FORCE_LIMIT_OFF
    else:
        _log.warning("arm '%s': driver has no _limit_gripper_force; the SDK gripper force limiter stays on", name)


def _switch_off_motors(name: str, channel: str, motor_ids: list[int]) -> None:
    """Sends the DM disable command to each motor on a channel, over an interface opened and closed here.

    Its own interface, after the driver has closed its bus, because the window inside i2rt's close (thread
    joined, socket still open) is not reachable from outside it; the few milliseconds between the two cost
    nothing, since the motors were holding indefinitely anyway. Best effort per motor: one that does not answer
    (i2rt gives up after its five retries) is logged and skipped so it cannot leave the others energized, and
    the error raised at the end names the ids that may still be live.

    @param name `"left"` or `"right"`, for the log lines.
    @param channel The arm's CAN link.
    @param motor_ids The chain's motor ids, in chain order.
    """
    # i2rt is imported here as in open_arm(): the extra is optional and mypy cannot see it.
    from i2rt.motor_drivers.dm_driver import DMSingleMotorCanInterface  # type: ignore[import-not-found]

    interface = DMSingleMotorCanInterface(channel=channel)
    failed: list[int] = []
    try:
        for motor_id in motor_ids:
            try:
                interface.motor_off(motor_id)
            except Exception:
                # The one broad except: the sweep must reach every motor whatever one of them does.
                _log.exception("arm '%s': motor %d on %s did not switch off", name, motor_id, channel)
                failed.append(motor_id)
    finally:
        interface.close()
    if failed:
        raise errors.ArmError(
            f"arm '{name}': motors {failed} on {channel} may still be energized; cut the arm's power"
        )
    _log.info("arm '%s': motors %s on %s switched off", name, motor_ids, channel)


def _has_command_block(robot: Any) -> bool:
    """Whether the robot is an i2rt `MotorChainRobot`, with the command block write_motor_command() swaps."""
    return all(
        hasattr(robot, attr) for attr in ("_commands", "_command_lock", "remapper", "motor_chain", "_kp", "_kd")
    )


def _joint_vector(name: str, observation: Any, key: str) -> npt.NDArray[np.float64]:
    """Pulls one `[6]` float64 vector out of a driver observation.

    @param name Arm name, for the message.
    @param observation The dict returned by `get_observations()`.
    @param key Which entry to read.
    @return The vector as float64.
    @throws ArmError If the entry is missing or not six values long.
    """
    if not isinstance(observation, dict) or key not in observation:
        raise errors.ArmError(f"arm '{name}' observation lacks '{key}'")
    vector = np.asarray(observation[key], dtype=np.float64).reshape(-1)
    if vector.shape != (JOINTS,):
        raise errors.ArmError(f"arm '{name}' observation '{key}' has {vector.size} values, expected {JOINTS}")
    return vector


def _scalar_value(name: str, observation: Any, key: str) -> float:
    """Pulls a single-valued entry (the gripper position or effort) out of a driver observation.

    @param name Arm name, for the message.
    @param observation The dict returned by `get_observations()`.
    @param key Which entry to read.
    @return The value as a float.
    @throws ArmError If the entry is missing or does not hold exactly one value.
    """
    if not isinstance(observation, dict) or key not in observation:
        raise errors.ArmError(f"arm '{name}' observation lacks '{key}'")
    value = np.asarray(observation[key], dtype=np.float64).reshape(-1)
    if value.size != 1:
        raise errors.ArmError(f"arm '{name}' observation '{key}' has {value.size} values, expected 1")
    return float(value[0])

"""The Seeed reBot B601-DM as a robot of your own (`body: custom`, seeed_rebot.yaml beside this file), on Seeed's
LeRobot driver (`pip install lerobot-robot-seeed-b601`).

The calls follow the driver Dream Labs runs its own reBot on: motor degrees in, logical degrees out (send_action
divides by joint_directions), the jaw mapped through its measured travel.

Neither class calls the driver's connect() or disconnect(). In its version 1.0.5, connect(calibrate=False) opens the
bus by `can_adapter` (SocketCAN unless set), then configure() switches every motor on (enable_all()) with no target
sent first, so where each motor goes then is up to its firmware; connect() with calibrate=True may also rezero every
motor (set_zero_position(), with no calibration file) and drives the jaw closed (detect_gripper_zero()). disconnect()
drives every joint to 0, blind, unless `_skip_safe_zero_on_disconnect` is set. So both classes open the bus as
a torque-off probe does: #Arm then switches each motor on holding where it is, #ReadOnlyArm (the dry run's
driver) switches none on. Left out: MIT gains, gravity feed-forward, the jaw's grasp hold, and tracking a jaw that
wrapped a turn at power-up (read() refuses such a reading instead).

@see Arm, ReadOnlyArm
"""

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from robots import config, errors, followers

## Per side, its LeRobot calibration id (ours; use your own).
CALIBRATION_IDS = {"right": "follower1", "left": "follower2"}
## The arm's motors in joint order, CAN ids 1-6; the jaw is `gripper`, CAN id 7.
MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll")
## Per side, the jaw motor's degrees shut and open, measured on our reBot: measure your own.
JAW_DEG = {"right": (15.0, -326.7), "left": (7.3, -327.2)}
## The jaw's force ceiling; the driver's 0.1 cannot move it.
JAW_TORQUE = 0.3
## A reading this far past a limit, degrees, is a turn wrapped at power-up: commanding it turns a joint.
LIMIT_SLACK_DEG = 10.0
## The speed cap of each motor's first target, rad/s (the driver's jaw speed, 150 deg/s); that target is where it is.
HOLD_SPEED_RAD_S = math.radians(150.0)


class Arm(followers.Driver):
    """One B601-DM arm, switched on and holding where it is."""

    ## No gravity compensation: disengaged, the arm falls.
    drops_when_limp = True

    def __init__(self, cfg: config.ArmConfig, side: str) -> None:
        """Opens the arm on its serial port and switches each motor on holding where it is, with no connect().

        @param cfg `can_channel` is the arm's serial port.
        @param side `"left"` or `"right"`.
        @throws ArmError A motor does not answer, or a joint reads past its limit; no motor is switched on then.
        """
        self._open(cfg.can_channel, side)
        self.read()  # refuses a turn wrapped at power-up before any motor is on
        self._switch_on()
        self.dirs = dict(self.dev.config.joint_directions)

    def _open(self, port: str, side: str) -> None:
        from lerobot_robot_seeed_b601.config_seeed_b601_dm_follower import SeeedB601DMFollowerConfig
        from lerobot_robot_seeed_b601.seeed_b601_dm_follower import SeeedB601DMFollower
        from lerobot_robot_seeed_b601.seeed_b601_follower import MotorBridgeController

        self.side, self.jaw = side, JAW_DEG[side]
        dm = SeeedB601DMFollowerConfig(port=port, id=CALIBRATION_IDS[side])
        dm.joint_limits = {**dm.joint_limits, "gripper": (self.jaw[1], self.jaw[0])}  # the jaw's measured travel
        dm.force_pos_torque_ration = JAW_TORQUE
        self.dev, self.limits = SeeedB601DMFollower(dm), dict(dm.joint_limits)
        # lerobot's __del__ disconnects whenever bus is set, and disconnect() would drive every joint to 0, blind
        self.dev._skip_safe_zero_on_disconnect = True
        self.dev.bus = MotorBridgeController.from_dm_serial(serial_port=port, baud=dm.dm_serial_baud)
        self.dev._add_motors_to_bus()

    def _states(self) -> dict[str, Any]:
        for m in self.dev.motors.values():
            m.request_feedback()
        self.dev.bus.poll_feedback_once()
        states = {n: m.get_state() for n, m in self.dev.motors.items()}
        silent = [n for n, st in states.items() if st is None]
        if silent:
            raise errors.ArmError(f"arm {self.side}: {silent} did not answer (24 V off, or a broken link?)")
        return states

    def _degrees(self) -> tuple[dict[str, float], dict[str, float]]:
        states = self._states()
        return {n: math.degrees(st.pos) for n, st in states.items()}, {n: math.degrees(states[n].vel) for n in MOTORS}

    def _switch_on(self) -> None:
        # One motor at a time: off while its mode is set, then on and sent where it is at once, so the first target
        # each motor gets is its own position.
        from lerobot_robot_seeed_b601.seeed_b601_follower import MotorBridgeMode

        for name, st in self._states().items():
            motor = self.dev.motors[name]
            motor.disable()
            motor.ensure_mode(MotorBridgeMode.FORCE_POS if name == "gripper" else MotorBridgeMode.POS_VEL)
            motor.enable()
            if name == "gripper":
                motor.send_force_pos(st.pos, HOLD_SPEED_RAD_S, JAW_TORQUE)
            else:
                motor.send_pos_vel(st.pos, HOLD_SPEED_RAD_S)

    def read(self) -> followers.Reading:
        """The joints in radians, the jaw from 0 shut to 1 open.

        @return The #robots.followers.Reading.
        @throws ArmError A joint reads past its limit, a turn wrapped at power-up.
        """
        deg, vel = self._degrees()
        slack = LIMIT_SLACK_DEG
        bad = [n for n, d in deg.items() if not self.limits[n][0] - slack <= d <= self.limits[n][1] + slack]
        if bad:
            raise errors.ArmError(
                f"arm {self.side}: {bad} read past their limits, a turn wrapped at power-up: power the arm off, lift "
                "that joint ~30 deg off its stop, power it on"
            )
        shut, opened = self.jaw
        grip = (deg["gripper"] - shut) / (opened - shut)
        return followers.Reading(np.radians([deg[n] for n in MOTORS]), np.radians([vel[n] for n in MOTORS]), grip)

    def write(self, positions_rad: npt.NDArray[np.float64], gripper: float) -> None:
        """Sends the joints and the jaw.

        @param positions_rad Six joint targets, radians.
        @param gripper 0 shut to 1 open.
        """
        shut, opened = self.jaw
        deg = {n: math.degrees(q) for n, q in zip(MOTORS, positions_rad, strict=True)}
        deg["gripper"] = shut + gripper * (opened - shut)
        self.dev.send_action({f"{n}.pos": d / self.dirs[n] for n, d in deg.items()})

    def disengage(self) -> bool:
        """Switches every motor off where it is; with no gravity compensation the arm falls, so support it.

        @return True.
        """
        for m in self.dev.motors.values():
            m.disable()
        return True

    def close(self) -> None:
        """Closes the bus without a move; the motors keep what they hold."""
        for m in self.dev.motors.values():
            m.close()
        self.dev.bus.close()
        self.dev.bus = None  # lerobot's __del__ disconnects whenever bus is set

    def torque_off(self) -> None:
        """Switches every motor off where it is and closes the bus, without the driver's disconnect()."""
        self.disengage()
        self.close()


class ReadOnlyArm(Arm):
    """One B601-DM arm opened as #Arm opens it, but with no motor switched on: the motors are only asked for feedback.
    The dry run's driver; write() refuses."""

    def __init__(self, cfg: config.ArmConfig, side: str) -> None:
        """Opens the arm's bus with no motor switched on.

        @param cfg `can_channel` is the arm's serial port.
        @param side `"left"` or `"right"`.
        """
        self._open(cfg.can_channel, side)

    def write(self, positions_rad: npt.NDArray[np.float64], gripper: float) -> None:
        """Refuses: this arm was opened read only.

        @param positions_rad Unused.
        @param gripper Unused.
        @throws ArmError Always.
        """
        raise errors.ArmError(f"arm {self.side} was opened read only (ReadOnlyArm): it takes no target")

    def disengage(self) -> bool:
        """Nothing to switch off.

        @return True.
        """
        return True

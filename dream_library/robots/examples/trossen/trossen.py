"""The Trossen Stationary AI's WXAI follower arms as a robot of your own (`body: custom`, trossen.yaml beside this
file), on Trossen's driver (`pip install trossen-arm`).

The calls follow the trossen_arm documentation and were not run against an arm; check them against the version you
install. get_all_positions() reports 6 joints in radians, then the jaw's opening in metres (#JAW_M).

@see Arm, ReadOnlyArm
"""

import numpy as np
import numpy.typing as npt

from robots import config, errors, followers

## The jaw's opening, metres, shut and open: the fine-tune's data records it so, and the dream client turns the grip
## into it (dream's trossen_wb_faas spec).
JAW_M = (0.0, 0.044)
## The WXAI arm's joints: get_all_positions() reports them, then the jaw.
JOINTS = 6
## Each 250 Hz command is reached over one period, seconds.
GOAL_TIME_S = 0.004


class Arm(followers.Driver):
    """One WXAI follower arm, in position mode and holding where it is."""

    def __init__(self, cfg: config.ArmConfig, side: str) -> None:
        """Connects to the arm (it comes up idle) and puts it in position mode.

        @param cfg `can_channel` is the arm's IP address.
        @param side `"left"` or `"right"`.
        """
        self._open(cfg.can_channel)
        self.dev.set_all_modes(self.ta.Mode.position)

    def _open(self, address: str) -> None:
        import trossen_arm

        self.ta, self.dev = trossen_arm, trossen_arm.TrossenArmDriver()
        model, jaw = trossen_arm.Model.wxai_v0, trossen_arm.StandardEndEffector.wxai_v0_follower
        self.dev.configure(model, jaw, address, False)  # leaves the arm idle

    def read(self) -> followers.Reading:
        """The joints in radians, the jaw from 0 shut to 1 open.

        @return The #robots.followers.Reading.
        """
        p, v = list(self.dev.get_all_positions()), list(self.dev.get_all_velocities())
        n, (shut, opened) = JOINTS, JAW_M
        return followers.Reading(np.array(p[:n]), np.array(v[:n]), (p[n] - shut) / (opened - shut))

    def write(self, positions_rad: npt.NDArray[np.float64], gripper: float) -> None:
        """Sends the joints and the jaw's opening.

        @param positions_rad Six joint targets, radians.
        @param gripper 0 shut to 1 open.
        """
        shut, opened = JAW_M
        self.dev.set_all_positions([*positions_rad, shut + gripper * (opened - shut)], GOAL_TIME_S, False)

    def disengage(self) -> bool:
        """Zero external effort: the arm holds itself up under gravity compensation.

        @return True.
        """
        self.dev.set_all_modes(self.ta.Mode.external_effort)
        self.dev.set_all_external_efforts([0.0] * (JOINTS + 1), 0.0, False)
        return True

    def close(self) -> None:
        """Releases the driver without a move."""
        self.dev.cleanup()

    def torque_off(self) -> None:
        """Idles the arm, then releases the driver."""
        self.dev.set_all_modes(self.ta.Mode.idle)
        self.dev.cleanup()


class ReadOnlyArm(Arm):
    """One WXAI follower arm left idle, as configure() leaves it: the dry run's driver; write() refuses."""

    def __init__(self, cfg: config.ArmConfig, side: str) -> None:
        """Connects to the arm and leaves it idle.

        @param cfg `can_channel` is the arm's IP address.
        @param side `"left"` or `"right"`.
        """
        self.side = side
        self._open(cfg.can_channel)

    def write(self, positions_rad: npt.NDArray[np.float64], gripper: float) -> None:
        """Refuses: this arm was opened read only.

        @param positions_rad Unused.
        @param gripper Unused.
        @throws ArmError Always.
        """
        raise errors.ArmError(f"arm {self.side} was opened read only (ReadOnlyArm): it takes no target")

    def disengage(self) -> bool:
        """Nothing to switch off: the arm is idle.

        @return True.
        """
        return True

    def torque_off(self) -> None:
        """Releases the driver; the arm stayed idle."""
        self.dev.cleanup()

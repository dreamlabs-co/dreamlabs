"""The starting point for a robot of your own: copy this file, fill in the four methods, and point the config at it.

Everything else is shared with our robots: the control loop, the station, the safety caps, the dream client's IK and
the wire to your fine-tune. Your driver only reads and writes one arm in the units of your arm's URDF (the one dream's
IK solves with, the config's `custom.ik`): joints in radians with the URDF's zero and direction, the gripper from 0
closed to 1 open. If your motors count differently (degrees, ticks, another zero or direction), convert here.

```yaml
body: custom
custom:
  driver: my_arm.py:MyArm          # this file, copied; or package.module:MyArm
  ik: makers                       # dream's name for your arm's URDF
  embodiment: <username>/<name>    # the embodiment your fine-tune was trained on
follower_l: {can_channel: /dev/ttyACM0, home: [0, 0, 0, 0, 0, 0]}
follower_r: {can_channel: /dev/ttyACM1, home: [0, 0, 0, 0, 0, 0]}
```

`configs/custom.example.yaml` is a whole file. `dreamcli robot run --config <file> --task "..." --dry-run` reads your
arms and logs their joints without writing a target, the first thing to try.

Your gripper's range can be measured rather than coded: fill in TemplateArm.gripper_deg(), and `dreamcli setup` records
each gripper closed and open by hand into its arm's `gripper_closed_deg` and `gripper_open_deg`, which `cfg` then
hands your driver to map the gripper with.

As shipped, #TemplateArm is an arm in memory: it reads back what it was last told. That is enough to run the whole
stack with no robot attached.

Two more things are yours to set. disengage() lets the arm go limp for the setup wizard's steps by hand; without it the
arm is held stiff there. `drops_when_limp` says the arm has no brakes, which changes how a reset and a fault let it go
(the attribute's comment below says how).

close() and every path that disconnects must never command motion: the only move a driver makes on its own is
torque_off() parking a brakeless arm, at a pose you chose. Mind this on a LeRobot robot (an SO-101 on its Feetech
STS3215 bus, for one), where read() is `get_observation()`, whose `"<motor>.pos"` values you turn into radians, and
write() is `send_action()` with targets turned back. A vendor's LeRobot plugin may home, rezero, power or move the arm
in `connect()` and `disconnect()`: Seeed's `lerobot_robot_seeed_b601`, for one, may rezero every motor in `connect()`
(`set_zero_position()`, when it finds no calibration file), enable them all and drive the gripper closed, and its
`disconnect()` runs `safe_zero()`, moving the whole arm to its zero pose. Read the plugin's source
before calling either; better, drive the motor SDK directly, or the plugin's lower-level bus without `connect()` and
`disconnect()`. An arm reads and takes as many joints as its `home` in the config lists, 1 to
`robots.followers.MAX_JOINTS` (`robots.followers.FollowerArm.dof`), and a read of another count fails naming the arm;
the wire sends only the joints dream's spec of the arm drives. (The SO-101 ships as `body: so101`.)

@see TemplateArm, robots.followers.Driver
"""

import numpy as np
import numpy.typing as npt

from robots import config, followers


class TemplateArm(followers.Driver):
    """One arm, in memory: read() answers the last target, so the loop sees the arm reach every command at once.

    Replace each method's body with your robot's I/O; keep the signatures.
    """

    ## True for an arm without brakes, which falls when its motors switch off. A reset then lets the arm go with
    ## torque_off() rather than close() (`FollowerArm.close_for_reopen()`), so torque_off() must park it first or it
    ## drops on every reset; an arm fault in a run holds the arms rather than disengaging them; and an arm that stops
    ## answering while it holds is reported as still holding under power, to be supported before its power goes off
    ## (`FollowerArm.holds_dead()`), where False reports it only as not answering.
    drops_when_limp = False

    def __init__(self, cfg: config.ArmConfig, side: str) -> None:
        """Opens one arm; the station calls this once per arm, and again after a reset.

        Open your bus or your LeRobot robot here, and leave the arm powered and holding where it is: the loop starts
        from what read() reports.

        @param cfg The arm's config section: `can_channel` is the device to open, `home` the arm's home joints.
        @param side `"left"` or `"right"`; while `dreamcli setup` is still finding which arm is which ("Find your
            arms"), the adapter's id instead, so per-side calibration cannot be looked up yet: open the arm to be read
            and moved by hand, and key anything per arm by `cfg.can_channel` or leave it until a side is known.
        """
        # Setup's arm finding passes the adapter id, not a side.
        self._side = side if side in ("left", "right") else None
        self._positions_rad = np.array(cfg.home, dtype=np.float64)
        self._gripper = 1.0
        self._is_limp = False

    def read(self) -> followers.Reading:
        """The arm's latest state; called every tick, so it should cost little.

        @return Joint positions and velocities `[dof]` (one per angle of the home) in radians and rad/s, and the
            gripper from 0 closed to 1 open.
        """
        return followers.Reading(self._positions_rad.copy(), np.zeros_like(self._positions_rad), self._gripper)

    def write(self, positions_rad: npt.NDArray[np.float64], gripper: float) -> None:
        """Takes a new target, checked (finite, one per joint; a gripper in `[0, 1]`) and rate-capped by the loop.

        @param positions_rad A target per joint, `[dof]`, radians.
        @param gripper The gripper target, 0 closed to 1 open.
        """
        if self._is_limp:
            # The first target after disengage() re-engages the arm holding where the hand left it, and is dropped.
            self._is_limp = False
            return
        self._positions_rad = positions_rad.copy()
        self._gripper = gripper

    def disengage(self) -> bool:
        """Lets the arm go limp where it is and keeps reading it; delete this if your arm has no passive mode.

        The setup wizard calls it so a hand can move the arm (finding the arms, home); without it (the default returns
        False) the arm is held stiff there instead. On your arm: switch the motors off, or to zero torque, and leave
        read() answering. The next write() must not jump to its target from wherever the hand left the arm: switch
        the motors on holding the pose they read now, and drop that first target.

        @return True once the arm is limp.
        """
        self._is_limp = True
        return True

    def close(self) -> None:
        """Releases the device without moving the arm; the motors keep holding their last target if they can.

        Never through a call that moves the arm, as some vendors' `disconnect()` does.
        """

    def torque_off(self) -> None:
        """Switches the motors off and releases the device; park the arm first if it would fall."""

    def gripper_deg(self) -> float | None:
        """The gripper motor's angle as of the last read(), degrees, before read() maps it to 0 closed to 1 open.

        The setup wizard records it with each gripper closed, then opened, fully by hand, and saves the two angles as
        the arm's `gripper_closed_deg` and `gripper_open_deg`; `cfg` hands them to your driver, NaN until measured.
        Keep the angle read() saw (the wizard asks from its own thread) and map with the two: the gripper is
        `(deg - closed) / (open - closed)` in read() and `closed + gripper * (open - closed)` in write(); refuse to
        write while they are NaN. Return None if your driver keeps its gripper's range itself: the wizard then skips
        that step and says why.

        @return Degrees. The arm in memory has no gripper motor: None.
        """
        return None

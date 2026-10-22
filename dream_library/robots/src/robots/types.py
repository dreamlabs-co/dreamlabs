"""The control loop's own data types: what goes in as an observation and out as an action.

All seven are frozen dataclasses with numpy arrays inside; none of them has behaviour. They
are the loop's types, chosen for what consumers of `ControlLoop.state` and `ControlLoop.video`
need, and #robots.adapter converts them to and from the dicts `dream` speaks.

Units: joint quantities are radians (`_rad`, `_rad_s`), grippers are a float in `[0, 1]`
(YAM convention: 0 closed, 1 open), timestamps are `time.monotonic()` seconds (`_s`).
"""

import dataclasses

import numpy as np
import numpy.typing as npt


@dataclasses.dataclass(frozen=True)
class Frame:
    """One captured camera image plus the newest frame encoded for the policy as of that capture; never mutated.

    The encoded frame rides on the capture so the loop reads both with one attribute read and no lock
    (fields that change together are bundled into one new `Frame`). The policy's frame window is the
    dream client's: the loop hands it `jpeg` and the client keeps the frames it needs.

    Attributes:
    - data: RGB pixels, uint8, shape `[H, W, 3]`, at the capture size.
    - camera: Camera name from the config, a serve role such as `"top"`.
    - timestamp_s: Monotonic capture time, seconds.
    - jpeg: The newest capture the reader encoded for the policy (one every `obs_period_s` of capture time, faces
      blurred, downscaled to the policy size), as a base64 JPEG string; this capture's own when it was the one
      encoded. Empty on a frame no reader encoded (a test's or the live view's own).
    - jpeg_timestamp_s: Monotonic capture time of `jpeg`, seconds; 0.0 when `jpeg` is empty.
    """

    data: npt.NDArray[np.uint8]
    camera: str
    timestamp_s: float
    jpeg: str = ""
    jpeg_timestamp_s: float = 0.0


@dataclasses.dataclass(frozen=True)
class ArmState:
    """What one follower arm reported on the last read, plus what it was last told.

    Attributes:
    - positions_rad: Joint positions, shape `[dof]`: the arm's joints (`robots.followers.FollowerArm.dof`).
    - velocities_rad_s: Joint velocities, shape `[dof]`.
    - gripper: Gripper opening in `[0, 1]`.
    - commanded_rad: Last commanded joint targets, shape `[dof]`; `None` before the first write.
    - commanded_gripper: Last commanded gripper target; `None` before the first write.
    - gripper_effort: The gripper motor's effort as the driver reports it (i2rt's `gripper_eff`, motor
      torque units, not newtons); `None` when the driver reports none.
    - motor_temps_c: The hottest sensor of each motor, base to gripper, degrees C; `None` when the driver reports
      none.
    """

    positions_rad: npt.NDArray[np.float64]
    velocities_rad_s: npt.NDArray[np.float64]
    gripper: float
    commanded_rad: npt.NDArray[np.float64] | None
    commanded_gripper: float | None
    ## Absent on drivers that do not report it; `robots.gripper_force` regulates the grasp on it.
    gripper_effort: float | None = None
    motor_temps_c: tuple[float, ...] | None = None


@dataclasses.dataclass(frozen=True)
class RobotState:
    """Both arms as read on one tick.

    Attributes:
    - left: The left follower's #ArmState.
    - right: The right follower's #ArmState.
    - timestamp_s: Monotonic tick time, seconds.
    """

    left: ArmState
    right: ArmState
    timestamp_s: float


@dataclasses.dataclass(frozen=True)
class Observation:
    """Everything the policy sees on one tick: robot state and the latest frame per camera.

    Attributes:
    - state: The #RobotState read this tick.
    - images: Latest #Frame per camera, keyed by camera name; the same objects the capture
      threads published, handed on rather than copied.
    - timestamp_s: Monotonic tick time, seconds.
    """

    state: RobotState
    images: dict[str, Frame]
    timestamp_s: float


@dataclasses.dataclass(frozen=True)
class ArmCommand:
    """Joint targets for one arm.

    Attributes:
    - positions_rad: Target joint positions, shape `[dof]`: one per joint of the arm.
    - gripper: Target gripper opening in `[0, 1]`.
    """

    positions_rad: npt.NDArray[np.float64]
    gripper: float


@dataclasses.dataclass(frozen=True)
class Action:
    """One tick's command for both arms. Always bimanual: there is no single-arm path.

    Attributes:
    - left: The left arm's #ArmCommand.
    - right: The right arm's #ArmCommand.
    """

    left: ArmCommand
    right: ArmCommand


@dataclasses.dataclass(frozen=True)
class ActionChunk:
    """One `get_action` reply: a fixed horizon of consecutive actions.

    Attributes:
    - actions: The actions in execution order; `len(actions)` is the checkpoint's horizon.
    - observed_at_s: Monotonic time of the observation this chunk answers, seconds.
    """

    actions: list[Action]
    observed_at_s: float

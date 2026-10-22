"""The application config: one YAML file that builds the arms, the cameras and the loop.

load() reads the file with `yaml.safe_load` and checks every key before anything is
constructed: a missing required key, an unknown key, or a value of the wrong type raises
#robots.errors.ConfigError naming the key. There are no silent defaults beyond the ones the
dataclasses below declare.

The file holds the hardware and the loop, and nothing about the account: where DreamAPI and the
platform are and the key for both are the credentials file (`dream.credentials`,
`~/.dreamlabs/credentials`), and the model is the file's top-level `checkpoint` (written by
`dreamcli setup`), else the body's default, unless a run names another. A file that still
carries the retired keys `policy`, `api_key` or `live.url` fails at load naming
where each went (#_MOVED_KEYS), never as an unknown key.

The file layout mirrors #AppConfig:

```yaml
follower_l: {can_channel: can_left, gripper: linear_4310}
follower_r: {can_channel: can_right}
cameras:
  - {name: top, device: 0, width: 640, height: 480, fps: 30, rotate: 0}
  - {name: left, serial: "260322275394", width: 640, height: 360}
task: "pick up the cup and place it on the plate"
control_hz: 20
action_horizon: 48
action_timeout_s: 3.0
stale_after_s: 0.2
max_stall_ticks: 240
first_chunk_timeout_s: 120.0
ramp_s: 2.0
ramp_max_rad_s: 0.5
max_ee_speed_m_s: 0.6
max_gripper_step: 0.05
grip_force_cap: 0.65
trace_path: null
inference_log_path: null
policy_image_size: 640x360
obs_window: 9
obs_fps: 5.0
command_hz: 250
command_interp_max_s: 0.1
live: {robot_id: rbt_0123456789abcdef}
rollouts: {enabled: true, dir: ~/.dreamlabs/rollouts, robot_name: my-station}
```

The `live` section is optional: absent, the station is not on Live View; present, it shows itself there as that robot
through `dream.LiveViewClient` on the credentials' platform. The `rollouts` section is optional too and off by
default, since every run is recorded on the platform by its own dream client: `enabled: true` also saves every run as
one episode on this disk (`robots.rollouts`).

`body` says which arms these are (#BODY_NAMES, `robots.bodies`): `yam` (the default, the layout above), `makers`,
a MakerMods Makers Arm (`robots.makers`), whose arm sections take `can_channel` and the gripper keys of
#_MAKERS_ARM_KEYS instead of the YAM's, whose cameras are `scene`, `wrist_l` and `wrist_r`, and whose `control_hz`
defaults to #MAKERS_CONTROL_HZ (`configs/makers.example.yaml`), or `metal`, a MakerMods Metal arm (`robots.metal`),
whose arm sections take #_METAL_ARM_KEYS, the gripper's open and closed angles among them, required
(gripper_ends_help()), with the Makers cameras and #METAL_CONTROL_HZ (`configs/metal.example.yaml`), or `so101`, an
SO-101 (`robots.so101`), whose arm sections take #_SO101_ARM_KEYS (the serial port by id or USB port, the arm's LeRobot
calibration file and its home, all required), whose top-level `embodiment` names the embodiment its fine-tune was
trained on, and whose `control_hz` defaults to #SO101_CONTROL_HZ (`configs/so101.example.yaml`).

`body: custom` is a robot of your own (#CUSTOM_BODY, `configs/custom.example.yaml`): the `custom` section names your
driver, dream's IK of your arm (one dream ships, or an arm file of your URDF) and the embodiment your fine-tune was
trained on (#CustomConfig), and each arm section takes `can_channel` (the device your driver opens) and `home`,
required.

Every body but the YAM runs through the dream client's IK and takes `action_space` (#ACTION_SPACES): `eef`, the
default, executes the model's tool poses, and `joint` its own joints, for a fine-tune of that very robot only.

@see load()
"""

import dataclasses
import logging
import math
import os
import pathlib
import socket
import typing
from typing import Any

import yaml
from shared.contracts import live as wire

from robots import errors, followers, gripper_force, makers_hold, metal_gripper, so101, user_class

_log = logging.getLogger(__name__)

APP_KEYS = frozenset(
    {
        "body",
        "follower_l",
        "follower_r",
        "cameras",
        "task",
        "control_hz",
        "action_horizon",
        "action_timeout_s",
        "stale_after_s",
        "max_stall_ticks",
        "first_chunk_timeout_s",
        "ramp_s",
        "ramp_max_rad_s",
        "max_ee_speed_m_s",
        "max_gripper_step",
        "grip_force_cap",
        "trace_path",
        "inference_log_path",
        "policy_image_size",
        "obs_window",
        "obs_fps",
        "command_hz",
        "command_interp_max_s",
        "sync_above_ms",
        "sys0_sync_above_ms",
        "face_blur",
        "live",
        "exposure",
        "custom",
        "embodiment",
        "rollouts",
        "checkpoint",
        "action_space",
        "ik_solve",
        "home_before_motors_off",
        "home_on_key_refused",
        "motor_temp_warn_c",
        "motor_temp_stop_c",
    }
)
## Keys of the optional `rollouts` mapping.
_ROLLOUTS_KEYS = frozenset({"enabled", "dir", "robot_name"})
## Keys of the optional `exposure` mapping.
_EXPOSURE_KEYS = frozenset({"managed", "target_median", "limit_us", "seed_us", "smooth_idle"})
## Keys of one `follower_*` mapping.
_ARM_KEYS = frozenset(
    {
        "can_channel",
        "gripper",
        "ee_mass_kg",
        "gravity_comp_factor",
        "sdk_gripper_force_limit",
        "temp_export",
        "home",
    }
)
## Keys of one `follower_*` mapping when `body` is `makers`: the adapter and the gripper's stroke and hold.
_MAKERS_ARM_KEYS = frozenset(
    {
        "can_channel",
        "gripper_open_deg",
        "gripper_closed_deg",
        "gripper_hold_torque_nm",
        "home",
        "keep_out",
        "reach",
        "region",
    }
)
## Keys of one `follower_*` mapping when `body` is `metal`: the adapter, the gripper's ends and its hold.
_METAL_ARM_KEYS = frozenset(
    {
        "can_channel",
        "gripper_open_deg",
        "gripper_closed_deg",
        "gripper_hold_torque_nm",
        "home",
        "keep_out",
        "reach",
        "region",
    }
)
## Keys of one `follower_*` mapping when `body` is `so101`: the serial port, the LeRobot calibration file and the home.
_SO101_ARM_KEYS = frozenset({"can_channel", "calibration", "home", "keep_out", "reach", "region"})
## An SO-101's joints: its servos but the gripper; the angles its `home` lists past them are 0.
_SO101_JOINTS = len(so101.JOINTS) - 1
## Keys of one `follower_*` mapping when `body` is `custom`: the device the driver opens, the home, and the gripper's
## ends when the setup wizard measured them.
_CUSTOM_ARM_KEYS = frozenset(
    {"can_channel", "home", "keep_out", "gripper_open_deg", "gripper_closed_deg", "reach", "region"}
)
_KEEP_OUT_KEYS = {
    "box": frozenset({"kind", "center", "size"}),
    "capsule": frozenset({"kind", "center", "radius", "height"}),
}
## Keys of the `custom` mapping (#CustomConfig).
_CUSTOM_KEYS = frozenset({"driver", "ik", "embodiment", "grip_scale"})
## Our bodies (`robots.bodies.BODIES`); the first is the default.
BODY_NAMES = ("yam", "makers", "metal", "so101")
## The bodies whose motor temperatures dream watches (`robots.motor_guard`), so the only ones the station file's
## `motor_temp_warn_c` and `motor_temp_stop_c` apply to.
MOTOR_TEMP_BODIES = ("makers", "metal", "so101")
## The `body` of a robot of your own, run by your driver (#CustomConfig).
CUSTOM_BODY = "custom"
## What a robot on the dream client's IK (every body but the YAM) may execute, `action_space` (#AppConfig): `eef`, the
## model's tool poses solved to the arm's joints with dream's IK (the default), or `joint`, the model's own joints,
## which only a fine-tune of that robot may run (`robots.bodies.check_model`).
ACTION_SPACES = typing.get_args(wire.ActionSpace)
## How the dream client turns an `eef` run's tool poses into joints, `ik_solve` (#AppConfig): `traj` (the
## default), the whole chunk in one solve that continues the arm's motion, or `tick`, row by row (the dream
## client's `ik` key `solve`).
IK_SOLVES = ("tick", "traj")
## What an `eef` run of every body solves with when neither the config's `ik_solve` nor `dreamcli robot up --ik-solve`
## names one: `traj`, the trajectory solve at the model code's velocity 5 / acceleration 50 (on a YAM, Unseen Task
## v3.2 at the same checkpoint: 0.62 against tick's 0.61 and the 20 / 100 weights' 0.53 on the same scenarios). The
## client compiles it on a thread and solves the first chunks with `tick` until it is ready (dream.client_ik), so a
## slow machine still meets the station's start-up deadlines.
DEFAULT_IK_SOLVE: str = "traj"

## What a YAM runs unless its config or `dreamcli robot up joint` says otherwise: the model's EEF head, its tool poses
## solved to the YAM's joints by the dream client's IK, as on every other body (`action_space` of #AppConfig).
YAM_ACTION_SPACE: str = "eef"
## The Makers Arm's tick rate unless the file names another: 20, the action rate every body plays Dream-1's chunks at.
## The dream client's IK holds every solved chunk under the arm's own joint speed caps at that rate (120 deg/s), as the
## arm's driver does (`robots.motor_arm.MAX_SPEED_DEG_S`).
MAKERS_CONTROL_HZ = 20.0
## The Metal arm's tick rate unless the file names another. The Metal datasets are recorded at 30 fps, but at 30 Hz
## the zero-shot base model swings the arm near home; at 20 Hz it runs smooth. 20 is also the YAM's default and
## MakerMods' console's rate. The arm's driver still ticks at 30 (`robots.metal.ARM`).
METAL_CONTROL_HZ = 20.0
## The SO-101's tick rate unless the file names another: 20, the action rate every body plays Dream-1's chunks at.
SO101_CONTROL_HZ = 20.0
## A `max_stall_ticks` under one late reply reads as this: at 10, one late reply ends an SO-101 run.
MAX_STALL_TICKS = 240
## Where an SO-101's serial port must be named from: a `/dev/ttyACM*` name changes when a cable is replugged, so the
## two arms could swap.
SERIAL_BY_ID_PREFIX = "/dev/serial/by-id/"
## Where else: by the USB port, for two adapters that report one serial and so share one by-id link
## (`robots.setup_devices`).
SERIAL_BY_PATH_PREFIX = "/dev/serial/by-path/"
## How a candleLight adapter driven over raw USB is named, a transport not supported here
## (`robots.robstride`): the same adapter is a SocketCAN link under the kernel's gs_usb driver.
_RAW_USB_PREFIX = "gs_usb:"
## Keys of one `cameras[i]` mapping; exactly one of `device` and `serial` is set.
_CAMERA_KEYS = frozenset({"name", "device", "serial", "source", "width", "height", "fps", "rotate"})
LIVE_KEYS = frozenset({"robot_id", "cameras", *("width_px", "fps", "bitrate_kbps", "encoder", "record")})
## `live` keys a station file may still carry that set nothing: the encoding is the platform's
## (`dream.LiveViewClient`), and recording is every run's own dream client's. Read and ignored, with one log line.
LIVE_IGNORED_KEYS = frozenset({"width_px", "fps", "bitrate_kbps", "encoder", "record"})
## The rotations `CameraReader` applies; anything else would change the frame shape non-trivially.
_ROTATIONS = (0, 90, 180, 270)
## A retired per-joint step cap (the end-effector speed cap is the replacement): a config still carrying it fails
## by name, with the replacement, rather than as "unknown key".
_RETIRED_STEP_KEY = "max_joint_step_rad"
## A retired prefetch threshold (the loop sizes it itself, `robots.control`): a config still
## carrying it fails by name, with the reason, rather than as "unknown key".
_RETIRED_REFILL_KEY = "refill_at_remaining"
## The arm key the loop-side grip force cap replaced (`robots.gripper_force`): an arm section
## still carrying it fails by name, with the replacement, rather than as "unknown key".
_RETIRED_ARM_KEY = "gripper_close_snap"

## The gripper types a config may name that i2rt does not know by that name, each with the i2rt `GripperType` that
## drives it. Anvil's gripper (the one an OpenYAM is sold with, 110 mm of stroke on a DM-J4310) is a linear DM4310
## gripper like i2rt's own, so i2rt's `linear_4310` motor settings and calibration drive it; only its stroke differs,
## which `robots.setup_wizard` checks and i2rt's force limiter (off by default) would need. A name not here is
## handed to i2rt as it is.
GRIPPER_DRIVERS: dict[str, str] = {"anvil_gripper": "linear_4310"}

## The keys that held the account until the credentials file did, each with where it went: a file still carrying one
## fails by name, with the way out, rather than as "unknown key". `checkpoint` under `policy` is the top-level
## `checkpoint` now.
_MOVED_KEYS: dict[str, str] = {
    "policy": (
        "the gateway is DREAM_GATEWAY_URL in the credentials file (~/.dreamlabs/credentials) and the model is the "
        "top-level 'checkpoint' or the one a start names, else the body's default"
    ),
    "api_key": "it is DREAM_API_KEY in the credentials file (~/.dreamlabs/credentials)",
    "live.url": "the platform is DREAM_API_URL in the credentials file (~/.dreamlabs/credentials)",
}


@dataclasses.dataclass(frozen=True)
class ArmConfig:
    """How to open one follower arm: an i2rt YAM, a Makers Arm (`robots.makers`) when `body` is `makers`, a Metal arm
    (`robots.metal`) when it is `metal`, or an SO-101 (`robots.so101`) when it is `so101`.

    The YAM reads `can_channel` through `temp_export`; the Makers Arm and the Metal arm read `can_channel` and the
    three `gripper_*` keys. Each body's file refuses the others' keys, so a field the body does not read keeps its
    default.

    Attributes:
    - can_channel: SocketCAN channel name, e.g. `"can_left"`; on a Makers or Metal arm also a CANable's serial
      device (`/dev/...`, slcan firmware), see `robots.robstride.open_can_bus`; on an SO-101 its serial port, by id
      (#SERIAL_BY_ID_PREFIX) or by USB port (#SERIAL_BY_PATH_PREFIX).
    - gripper: i2rt `GripperType` value name, or a name in #GRIPPER_DRIVERS (i2rt_gripper() gives the i2rt one).
    - ee_mass_kg: End-effector mass handed to i2rt's gravity model, kg; null keeps i2rt's own default.
    - gravity_comp_factor: One scalar i2rt multiplies into its gravity torques on every joint; null keeps
      i2rt's per-joint defaults (`robots.yam` documents what it replaces).
    - sdk_gripper_force_limit: Keep i2rt's motor-level 50 N gripper force limiter; off by default, see
      `robots.yam.open_arm`.
    - temp_export: Have i2rt include per-motor temperatures in its observations.
    - gripper_open_deg: The gripper motor's angle with the jaws fully open, degrees: the loop's gripper 1.0. The grip
      runs linearly in the motor angle between the two ends, on every bus arm. Makers Arm: also the
      gripper's soft limit on that side, below `gripper_closed_deg` (it closes toward increasing angles). Metal arm:
      required, above `gripper_closed_deg` (gripper_ends_help()). SO-101: the high end of the gripper's calibrated
      range, read from `calibration`, not the file.
    - gripper_closed_deg: The gripper motor's angle with the jaws closed, degrees: the loop's gripper 0 (an SO-101's
      calibrated low end).
    - gripper_hold_torque_nm: Makers Arm: the gripper's holding torque on an object, N.m at the motor, within
      `robots.makers_hold.MIN_HOLD_TORQUE_NM` to `MAX_HOLD_TORQUE_NM`; null turns holding and its closing cap off
      (plain position control at full gain). Metal arm: the same, never null (its gripper always holds).
    - home: The arm's home pose, one angle per joint in radians in its URDF's frame: where a run starts from and where
      Home and Stop move it, the gripper open (`robots.home`). How many it lists, 1 to `robots.followers.MAX_JOINTS`,
      is how many joints the arm has (`robots.followers.FollowerArm.dof`); the arm's driver refuses another count when
      it opens. Zeros on a YAM and a Makers Arm (the YAM's home pose, the
      Makers Arm's folded calibration pose); required on a Metal arm, whose zero pose is the arm straight up
      (home_help()), on an SO-101, and on a robot of your own.
    - calibration: An SO-101's LeRobot calibration file, an absolute path (`robots.feetech.read_calibration`); None
      on every other arm.
    - keep_out: What dream's IK keeps this arm off (a camera mount, a pole), in the arm's base frame, metres: each
      `{"kind": "box", "center": [x, y, z], "size": [x, y, z]}` (axis-aligned, full extent) or `{"kind": "capsule",
      "center": [x, y, z], "radius": r, "height": h}` (standing upright). Every collision sphere of the arm (4 cm)
      stays 2 cm off each. None when the arm's section has no `keep_out`: the arm keeps those of dream's model of it
      (none on Metal and Makers); an empty list: none. None on a YAM, which runs joints.
    - reach: The area this arm's hand targets are kept inside, seen from above, at any height: `[x, y]` points in the
      arm's base frame, metres, which the setup wizard paints by moving the hand over it. dream's IK moves each hand
      target farther than `dream.client_ik.REACH_RADIUS_M` (4 cm) from every point to that distance from the nearest
      one (dream docs: the `ik` key's `reach`). None when the arm's section has no `reach`: the hand can go anywhere.
      None on a YAM, which runs joints.
    - region: The box this arm must stay inside, in the arm's base frame, metres: `{"lo": [x, y, z], "hi": [x, y,
      z]}`, any value null (that face open), either key left out (all its faces open). It bounds the arm's whole
      centreline, every link and the grasp point, so a `lo` z at the table lets the hand reach the table while no link
      goes through it. The dream client clips the hand targets inside it, its IK keeps the arm 2 cm in, and a row that
      leaves it by more than 1 cm holds the arm (the dream client's `ik` key `region`). None when the arm's section has
      no `region`: no box. None on a YAM, which runs joints.
    - temp_warn_c: The station file's top-level `motor_temp_warn_c`, degrees C, in place of the body's own
      (`robots.motor_arm.ArmSpec.temp_warn_c`); None keeps the body's.
    - temp_stop_c: The station file's top-level `motor_temp_stop_c`, degrees C, in place of the body's own; None
      keeps the body's.

    On a robot of your own (`body: custom`) `can_channel` is the device your driver opens, whatever it is (a CAN
    link, a serial port), and `home` is required: one angle per joint your driver reads, exactly as many as dream's
    model of the arm (`custom.ik`) has; a run refuses another count before an arm opens (`robots.adapter.Wire`), so
    a home padded with 0s up to six angles must drop the padding. `gripper_open_deg` and `gripper_closed_deg` are
    optional, both or neither, NaN without: the setup wizard measures them when your driver reports its gripper
    motor's angle (`robots.followers.Driver.gripper_deg()`), and your driver maps its gripper with them. The other
    fields keep their defaults, which nothing reads.
    """

    can_channel: str
    ## Default gripper type shipped with the YAM.
    gripper: str = "linear_4310"
    ## 0.62 kg: the measured mass of the YAM's end effector (the linear_4310 gripper).
    ee_mass_kg: float | None = 0.62
    ## Full gravity compensation on every joint, what a YAM follower runs with.
    gravity_comp_factor: float | None = 1.0
    ## Off: the SDK limiter fights the loop-side `grip_force_cap`, which is the regulator here.
    sdk_gripper_force_limit: bool = False
    ## On: i2rt reads the temperatures anyway, and a hot motor is the first thing to check after a fault.
    temp_export: bool = True
    ## -96: the vendor's tested open limit, about 3 degrees short of the narrower jaw's
    ## stop on one Maker Arm, so fully open never leans on a stop. Another Maker Arm opens to -100.5. Re-measure after
    ## a
    ## gripper swap.
    gripper_open_deg: float = -96.0
    ## -2.5: the jaws closed, near the calibration pose's zero.
    gripper_closed_deg: float = -2.5
    ## 0.5 N.m: the vendor's saved value on a Makers Arm.
    gripper_hold_torque_nm: float | None = makers_hold.DEFAULT_HOLD_TORQUE_NM
    ## Zeros: the YAM's home pose and the Makers Arm's folded calibration pose. A Metal file must give its own.
    home: tuple[float, ...] = (0.0,) * 6
    ## Only an SO-101 has one.
    calibration: str | None = None
    ## None: those of dream's model of the arm.
    keep_out: tuple[dict[str, Any], ...] | None = None
    ## None: no area, the hand can go anywhere.
    reach: tuple[tuple[float, float], ...] | None = None
    ## None: no box. Else `{"lo": [x, y, z], "hi": [x, y, z]}`, as the dream client's `ik` key `region` takes it.
    region: dict[str, list[float | None]] | None = None
    ## None: the body's own, `motor_arm.ArmSpec.temp_warn_c`.
    temp_warn_c: float | None = None
    ## None: the body's own, `motor_arm.ArmSpec.temp_stop_c`.
    temp_stop_c: float | None = None


@dataclasses.dataclass(frozen=True)
class CustomConfig:
    """A robot of your own (`body: custom`): the `custom` section.

    Attributes:
    - driver: Your driver, `package.module:Class` (importable in the Python that runs dreamcli) or `file.py:Class` (a
      relative path is read from the config file's directory; stored absolute). `Class(arm_config, side)` opens one
      arm and returns a `robots.followers.Driver` (`side` is `"left"` or `"right"`, or the adapter's id while
      `dreamcli setup` finds the arms); `robots.driver_template` is the commented starting point.
    - ik: dream's name for your arm's URDF (`dream_ik.specs`), or the path of an arm file of your own URDF
      (`dream.arm_file`, a `.json`; a relative path is read from the config file's directory; stored absolute), which
      the dream client solves the model's tool poses with and the end-effector speed cap measures with.
    - embodiment: The embodiment your fine-tune was trained on, `<username>/<name>` as `dream.DreamClient`'s
      get_embodiments() lists it: its config is the wire (`robots.adapter.EefWire`). The setup wizard registers one
      for an arm whose URDF it takes (`robots.adapter.zero_shot_embodiment`), which runs Dream Labs' base models.
    - grip_scale: What the embodiment's `<side>_gripper` part reads with the gripper fully open (the loop's 1.0), in
      the datasets' own unit; None when it records no `<side>_gripper`.
    """

    driver: str
    ik: str
    embodiment: str
    grip_scale: float | None = None


def gripper_ends_help(where: str) -> str:
    """What an arm without default gripper ends must supply, for the error when it does not.

    @param where Which arm, for the message (`"'follower_r'"`).
    @return The message.
    """
    return (
        f"{where} needs 'gripper_open_deg' and 'gripper_closed_deg': the gripper motor's angle, degrees, with the "
        "jaws fully open and closed. The grip runs linearly between them; the setup wizard (dreamcli setup) measures "
        "them on its home step: open the gripper fully by hand, then close it"
    )


def home_help(where: str) -> str:
    """What an arm without a safe default home must supply, for the error when it does not.

    @param where Which arm, for the message (`"'follower_r'"`).
    @return The message.
    """
    return (
        f"{where} needs 'home': the arm's home pose, one joint angle per joint in radians, where every run starts and "
        "where Home and Stop move it. This arm's zero pose is not a safe home (a Metal arm at zero stands straight "
        "up), so there is no default: the setup wizard (dreamcli setup) captures it, or write the angles the arm "
        "reads at its rest pose ('dreamcli robot run --dry-run' logs them)"
    )


def i2rt_gripper(gripper: str) -> str:
    """The i2rt `GripperType` value that drives a configured gripper type.

    @param gripper The `gripper` of an #ArmConfig.
    @return The same name, or what #GRIPPER_DRIVERS maps it to.
    """
    return GRIPPER_DRIVERS.get(gripper, gripper)


@dataclasses.dataclass(frozen=True)
class CameraConfig:
    """How to open one camera and what frame to expect from it.

    A camera is an OpenCV device (`device`), an Intel RealSense (`serial`) or a camera class of your own (`source`),
    exactly one: load() rejects an entry that sets two or none, and `CameraReader.from_config` picks the capture
    source by which one is set.

    Attributes:
    - name: Camera role (`top`, `left` or `right`); the adapter sends this camera's frames under it.
    - device: OpenCV device index or a `/dev/video*` path; None for a RealSense camera.
    - serial: RealSense serial number, e.g. `"260322275394"`; None for an OpenCV device.
    - source: A camera class of your own, `package.module:Class` or `file.py:Class` (made absolute from the config
      file's directory): `Class(camera_config)` opens the camera, `read()` returns `(ok, frame)` with a BGR
      `[height, width, 3]` uint8 frame, as `cv2.VideoCapture.read()` does, and `release()` lets it go. For an SDK
      camera OpenCV cannot open (GigE, ZED, a ROS topic). None for an OpenCV device or a RealSense.
    - width: Frame width in pixels.
    - height: Frame height in pixels.
    - fps: Capture rate requested from the driver.
    - rotate: Rotation applied at capture, degrees clockwise: 0, 90, 180 or 270.
    """

    name: str
    width: int
    height: int
    ## Not an OpenCV device unless set.
    device: int | str | None = None
    ## Not a RealSense camera unless set.
    serial: str | None = None
    ## Not a camera class of your own unless set.
    source: str | None = None
    ## Capture rate requested from the driver.
    fps: int = 30
    ## No rotation.
    rotate: int = 0


@dataclasses.dataclass(frozen=True)
class LiveConfig:
    """The `live` section: which robot the station shows itself as on Live View, and which of its cameras.

    The station runs `dream.LiveViewClient`, the client any control loop uses, on the credentials' platform; the
    encoding is the platform's, and every run is recorded by its own dream client, so neither is set here.

    Attributes:
    - robot_id: The robot this station is registered as (`POST /v1/robots`, or Robot Setup's Save).
    - cameras: The config's camera names a viewer may watch, in this order; every camera when the file leaves it out.
    """

    robot_id: str
    cameras: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class AppConfig:
    """Everything `robots.run.main` needs, parsed and checked.

    Attributes:
    - follower_l: The left arm's #ArmConfig.
    - follower_r: The right arm's #ArmConfig.
    - cameras: One #CameraConfig per camera; names are unique (`robots.run` checks they are serve roles).
    - task: The language prompt the policy receives with every observation; `""` when the file has none, and the
      operator sends it with the Start (`robots.station` refuses to start without one).
    - control_hz: Tick rate. 20 by default on a YAM: the checkpoint's rows are 30 Hz actions, so 20 Hz plays every
      plan at two thirds speed, the safer pace on a wide-area link and around people. #MAKERS_CONTROL_HZ on a
      Makers Arm, #METAL_CONTROL_HZ on a Metal arm and #SO101_CONTROL_HZ on an SO-101, 20 as well.
    - action_horizon: The checkpoint's chunk length; declared in the embodiment before the first call.
    - action_timeout_s: A refill older than this is late and its result is discarded; 3 s by default, room for a
      wide-area round trip and its jitter.
    - stale_after_s: A frame older than this fails the tick.
    - max_stall_ticks: Consecutive empty-buffer ticks tolerated before the loop stops, once the first chunk
      has landed; a file's value shorter than `action_timeout_s` reads as #MAX_STALL_TICKS (more when
      that is still shorter), with a warning.
    - first_chunk_timeout_s: Seconds after the first tick by which the first chunk must have landed; until
      then a warming pool is waited out at the pace DreamAPI asks for and no stall is counted.
    - ramp_s: Seconds over which the arms blend from their measured pose into the first target
      after start and after every new session.
    - ramp_max_rad_s: Joint speed cap during that blend; the ramp lasts
      `max(ramp_s, max_delta_rad / ramp_max_rad_s)` seconds.
    - max_ee_speed_m_s: Cap on each arm's end-effector speed from one command to the next, metres per second
      (`robots.kinematics` turns joints into metres). 0.6 on the station.
    - max_gripper_step: Largest change one tick may command on the gripper, in `[0, 1]` units.
    - grip_force_cap: Gripper effort the loop holds a grasp at, in i2rt's `gripper_eff` units (motor torque,
      not newtons); the loop backs the commanded close off until the measured effort sits there
      (`robots.gripper_force`). 0 disables the cap and hands grip force back to the motor
      PD, which is when `sdk_gripper_force_limit` is worth turning on.
    - trace_path: CSV file the loop appends one row per commanded tick to, a leading `~` expanded; None writes no
      trace.
    - inference_log_path: JSON Lines file the loop appends one record per inference, error, stall and run
      boundary to (`robots.inference_log`); None writes none (`robots.station` then uses `~/robots-inference.jsonl`).
    - policy_image_size: `(width, height)` every frame the policy sees is downscaled to before JPEG
      encoding (`WIDTHxHEIGHT` in the file); capture stays at each camera's `width` x `height`.
    - obs_window: Frames per camera in every observation (`image_window.<role>`), oldest first; the
      checkpoint's `video_context_frames`. The dream client keeps the window (its policy config's `obs_window`).
    - obs_fps: Rate those frames are kept at, frames per second: the training video fps
      (`dataset_fps / video_stride`). A camera encodes one capture for the policy every `1 / obs_fps` seconds, and
      the dream client keeps them at this rate (its policy config's `obs_fps`).
    - command_hz: Rate the loop's command thread writes the arms at, interpolating between the rows.
    - command_interp_max_s: Longest run from one written command to the next row, seconds; bounds the
      interpolation after a stall.
    - sync_above_ms: The dream client's sync threshold (`dream.pacing`) without System 0, milliseconds: a round trip
      above it runs the whole chunk and holds the arms through the next request. None leaves it to
      `run.policy_config()`: `run.SYNC_ABOVE_MS`, 900 ms on every body.
    - sys0_sync_above_ms: The same threshold with System 0 (a YAM's default); None leaves it to
      `run.policy_config()`, 1000 ms (`run.SYS0_SYNC_EXTRA_MS` past `run.SYNC_ABOVE_MS`).
    - face_blur: Blur faces out of every frame before it leaves the box for the API (`shared.faceblur`, applied
      at capture in `robots.cameras`). On by default; turning it off sends raw camera frames to DreamAPI. The live
      agent blurs every frame it encodes under the same flag.
    - live: The #LiveConfig of the live-view agent, or None when the file has no `live` section (the station then
      runs no agent).
    - rollouts: The #RolloutsConfig: whether every run is saved on this disk, and where.
    - body: Which arms these are, one of #BODY_NAMES (`robots.bodies`), or #CUSTOM_BODY.
    - custom: A robot of your own's #CustomConfig; None on our bodies.
    - embodiment: The embodiment an SO-101's fine-tune was trained on, `<username>/<name>` as `dream.DreamClient`'s
      get_embodiments() lists it: its config is the wire (`robots.adapter.EefWire`); None on every other body.
    - checkpoint: The model the station runs unless a run names another (`dreamcli robot run --checkpoint`, a Start
      from the website); None runs the body's default (`robots.bodies.Body.checkpoint`, the YAM's). The setup wizard
      writes the model picked for every body but the YAM, so a Start without one is never refused.
    - action_space: What the run executes, one of #ACTION_SPACES: `eef`, the model's tool poses solved to the arm's
      joints by the dream client (the default), or `joint`, the model's own joints, for a fine-tune of this robot
      only (`robots.bodies.check_model`). A run may name another (`dreamcli robot run --action-space`, a Start from
      the Live view). Not a key of a YAM's file: a YAM runs its model's joints, on the client's joint head.
    - ik_solve: How an `eef` run's tool poses become joints, one of #IK_SOLVES: `tick`, row by row, or `traj`, the
      whole chunk in one solve that continues the arm's motion. None (the key absent): the dream client's default,
      `tick`.
    """

    follower_l: ArmConfig
    follower_r: ArmConfig
    cameras: list[CameraConfig]
    task: str
    action_horizon: int
    stale_after_s: float
    max_stall_ticks: int
    max_ee_speed_m_s: float
    ## 20 Hz plays the checkpoint's 30 Hz rows at two thirds speed: the safer pace on a wide-area link and around
    ## people; a station file may name another rate.
    control_hz: float = 20.0
    ## Three seconds: a wide-area round trip (about 0.6 s) and its jitter, with room for a slow reply.
    action_timeout_s: float = 3.0
    ## Two minutes: a serve takes about a minute to boot, and the arms rest until it answers.
    first_chunk_timeout_s: float = 120.0
    ## Two seconds from rest into the first target, what a YAM station ramps over.
    ramp_s: float = 2.0
    ## Slow enough that an arm resting far from the policy's home pose is never yanked.
    ramp_max_rad_s: float = 0.5
    ## Gripper units per tick, the whole stroke in two thirds of a second.
    max_gripper_step: float = 0.05
    ## 0.65: a cap at 1.5 crushes cups, 2.0 grips harder than any task needs (`robots.gripper_force.DEFAULT_CAP`).
    grip_force_cap: float = gripper_force.DEFAULT_CAP
    ## A CSV appended with one row per commanded tick (`robots.control.trace_columns()`) for post-mortems; None = off.
    trace_path: str | None = None
    ## Off here; the station agent supplies its own default so a run is never without one.
    inference_log_path: str | None = None
    ## 250 Hz: the motor PD gets a fresh target every 4 ms whatever the row rate.
    command_hz: float = 250.0
    ## 100 ms: a row after a stall is reached within 100 ms instead of over the stall.
    command_interp_max_s: float = 0.10
    ## None: `run.policy_config()` sets `run.SYNC_ABOVE_MS` (900 ms). A station on another link may set its
    ## own, lower to hold through slow replies sooner.
    sync_above_ms: float | None = None
    ## None: `run.policy_config()` sets it (1000 ms); System 0's, as `sync_above_ms` is the plain loop's.
    sys0_sync_above_ms: float | None = None
    ## 640x360: the serve resizes to the model's input anyway, and 720p would
    ## cost 3-4x the upload for nothing.
    policy_image_size: tuple[int, int] = (640, 360)
    ## Nine frames: the base model checkpoint's `video_context_frames`.
    obs_window: int = 9
    ## 5 fps: the base model checkpoint's training video rate (30 fps dataset, video_stride 6).
    obs_fps: float = 5.0
    ## Faces are scrubbed at capture before any frame is sent to the API; off sends raw frames (debugging only).
    face_blur: bool = True
    ## No live agent unless the file names the robot to link as.
    live: LiveConfig | None = None
    ## Managed colour exposure on every RealSense camera; see #ExposureConfig.
    exposure: "ExposureConfig" = dataclasses.field(default_factory=lambda: ExposureConfig())
    ## Every run is saved on this disk unless the file turns it off; see #RolloutsConfig.
    rollouts: "RolloutsConfig" = dataclasses.field(default_factory=lambda: RolloutsConfig())
    ## The YAM, every station's body before the Makers Arm.
    body: str = BODY_NAMES[0]
    ## Only a robot of your own has one.
    custom: CustomConfig | None = None
    ## Only an SO-101 has one at the top level (a robot of your own names it in `custom`).
    embodiment: str | None = None
    ## None runs the body's default model.
    checkpoint: str | None = None
    ## The model's tool poses, solved to joints, on every body but the YAM, whose default
    ## is #YAM_ACTION_SPACE instead, read in load().
    action_space: str = "eef"
    ## None: the dream client's default solve, `tick`.
    ik_solve: str | None = None
    ## `dreamcli setup` only: Switch off the motors after the powered checks drives the held arms home
    ## first, then switches them off (`setup_session.Session.release`). Off, the arms go limp where they hold, as
    ## before; the wizard reads it from the file there when it starts, and the station agent ignores it.
    home_before_motors_off: bool = False
    ## The station agent only: a run DreamAPI ends by refusing the key as `disabled` (out of credit, over the
    ## spending limit, or revoked) is followed by the Home move (`robots.station.StationController`). On unless the
    ## file sets `false`, which holds the arms where the run left them, as for any other fault.
    home_on_key_refused: bool = True


@dataclasses.dataclass(frozen=True)
class ExposureConfig:
    """Managed colour exposure for the RealSense cameras: the exposure controller a YAM station runs.

    The firmware's auto-exposure is switched off and the capture thread steers exposure itself, holding each
    frame's median grey level at `target_median`: the median is the scene (the bench and what is on it), so a
    backlit panel in the top of the frame cannot expose the bench a quarter as bright as the scenes the model
    was trained on. Two knobs on a ladder: exposure first, up to 85% of the frame period or `limit_us`, then gain; and
    gain back down before exposure, because gain is amplified noise and exposure is real light. The controller
    moves only between runs: the station holds it for the length of a run, so a brightness change is never
    written into the middle of an episode. An OpenCV device ignores the section (a webcam's exposure is its
    driver's business).

    Attributes:
    - managed: Steer exposure from the capture thread. False hands the cameras back to the firmware's own
      unbounded auto-exposure, the stock look, for comparing against the controller.
    - target_median: Grey level, 0 to 255, the frame's median is held at.
    - limit_us: Ceiling in microseconds on what the controller may choose, or None for 85% of the frame period
      alone (`robots.cameras.MANAGED_CEILING_FRACTION`: the whole period halves the sensor's frame rate).
    - seed_us: Where the controller starts, in microseconds; None seeds at half the ceiling.
    - smooth_idle: On by default. Between runs, settle at full speed only after the camera opens, then follow
      drift slowly: a correction waits until the frame has been off target for 3 s and moves at most 5% a step
      (`robots.cameras.MANAGED_SLOW_STEP`). `false` brings back the old stepping, where every step between runs
      may scale exposure up to 1.6x or down to 0.7x every half second, which shows as flicker in the live view
      of an idle station.
    """

    managed: bool = True
    target_median: float = 90.0
    limit_us: float | None = None
    seed_us: float | None = None
    smooth_idle: bool = True


@dataclasses.dataclass(frozen=True)
class RolloutsConfig:
    """Whether the loop saves every run as a LeRobot v2.1 episode on this disk, and where (`robots.rollouts`).

    Attributes:
    - enabled: Save each run on this disk. Off by default: every run's dream client records it on the platform, so a
      second copy here is opt-in.
    - dir: The folder one subfolder per episode is written into; a leading `~` is expanded when the recorder opens
      it. The default sits in the station directory (`robots.cli.DEFAULT_HOME`).
    - robot_name: The name written into every episode's `rollout.json`; this machine's hostname unless set.
    """

    enabled: bool = False
    dir: str = "~/.dreamlabs/rollouts"
    robot_name: str = dataclasses.field(default_factory=socket.gethostname)


def load(path: str | os.PathLike[str]) -> AppConfig:
    """Reads and checks the YAML config file.

    @param path The file to read.
    @return The parsed #AppConfig.
    @throws ConfigError The file is not a mapping, a required key is missing, a key is unknown or one of
        #_MOVED_KEYS, or a value has the wrong type or range; the message names the key.
    """
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise errors.ConfigError(f"config file '{os.fspath(path)}' must be a mapping, got {_describe(raw)}")
    if _RETIRED_STEP_KEY in raw:
        raise errors.ConfigError(
            f"config key '{_RETIRED_STEP_KEY}' is gone: the rate clamp caps the end effector's speed now; "
            "set 'max_ee_speed_m_s' (metres per second, 0.6 on the station) instead"
        )
    for key in ("policy", "api_key"):
        if key in raw:
            raise errors.ConfigError(f"config key '{key}' is gone: {_MOVED_KEYS[key]}; remove it from the file")
    if _RETIRED_REFILL_KEY in raw:
        raise errors.ConfigError(
            f"config key '{_RETIRED_REFILL_KEY}' is gone: the loop sizes its refill from its measured round trip; "
            "remove it from the file"
        )
    _reject_unknown(raw, APP_KEYS, "")
    body = _read_str(raw, "body", "", default=AppConfig.body)
    if body not in (*BODY_NAMES, CUSTOM_BODY):
        raise errors.ConfigError(f"config key 'body' must be one of {[*BODY_NAMES, CUSTOM_BODY]}, got {body!r}")
    base = pathlib.Path(path).parent
    custom = _read_custom(raw, base) if body == CUSTOM_BODY else None
    if custom is None and "custom" in raw:
        raise errors.ConfigError(
            f"config key 'custom' is for 'body: {CUSTOM_BODY}' only; this file's body is {body!r}"
        )
    if body != "so101" and "embodiment" in raw:
        raise errors.ConfigError(f"config key 'embodiment' is for 'body: so101' only; this file's body is {body!r}")
    # A YAM takes `action_space` too, because its Dream-1 base models have an EEF head: `eef` is its
    # default, the model's tool poses through the dream client's IK, the same path every other body takes,
    # and `joint` runs the model's joint head.
    default_space = YAM_ACTION_SPACE if body == "yam" else AppConfig.action_space
    action_space = _read_str(raw, "action_space", "", default=default_space)
    if action_space not in ACTION_SPACES:
        raise errors.ConfigError(
            f"config key 'action_space' must be one of {list(ACTION_SPACES)}, got {action_space!r}"
        )
    ik_solve = _read_optional_str(raw, "ik_solve")
    if ik_solve is not None and ik_solve not in IK_SOLVES:
        raise errors.ConfigError(f"config key 'ik_solve' must be one of {list(IK_SOLVES)}, got {ik_solve!r}")
    cameras = _read_cameras(raw, base)
    default_hz = {"makers": MAKERS_CONTROL_HZ, "metal": METAL_CONTROL_HZ, "so101": SO101_CONTROL_HZ}
    control_hz = _read_positive_float(raw, "control_hz", "", default=default_hz.get(body, AppConfig.control_hz))
    action_timeout_s = _read_positive_float(raw, "action_timeout_s", "", default=AppConfig.action_timeout_s)
    warn_c, stop_c = _read_motor_temps(raw, body)
    return AppConfig(
        body=body,
        custom=custom,
        embodiment=_read_str(raw, "embodiment", "") if body == "so101" else None,
        follower_l=dataclasses.replace(
            _read_arm(raw, "follower_l", body, base), temp_warn_c=warn_c, temp_stop_c=stop_c
        ),
        follower_r=dataclasses.replace(
            _read_arm(raw, "follower_r", body, base), temp_warn_c=warn_c, temp_stop_c=stop_c
        ),
        cameras=cameras,
        task=_read_optional_task(raw),
        control_hz=control_hz,
        action_horizon=_read_int(raw, "action_horizon", "", minimum=1),
        action_timeout_s=action_timeout_s,
        stale_after_s=_read_positive_float(raw, "stale_after_s", ""),
        max_stall_ticks=_read_stall_ticks(raw, os.fspath(path), control_hz, action_timeout_s),
        max_ee_speed_m_s=_read_positive_float(raw, "max_ee_speed_m_s", ""),
        first_chunk_timeout_s=_read_positive_float(
            raw, "first_chunk_timeout_s", "", default=AppConfig.first_chunk_timeout_s
        ),
        ramp_s=_read_positive_float(raw, "ramp_s", "", default=AppConfig.ramp_s),
        ramp_max_rad_s=_read_positive_float(raw, "ramp_max_rad_s", "", default=AppConfig.ramp_max_rad_s),
        max_gripper_step=_read_positive_float(raw, "max_gripper_step", "", default=AppConfig.max_gripper_step),
        grip_force_cap=_read_non_negative_float(raw, "grip_force_cap", "", default=AppConfig.grip_force_cap),
        trace_path=_read_optional_path(raw, "trace_path"),
        inference_log_path=_read_optional_path(raw, "inference_log_path"),
        policy_image_size=_read_image_size(raw, "policy_image_size", default=AppConfig.policy_image_size),
        obs_window=_read_int(raw, "obs_window", "", minimum=1, default=AppConfig.obs_window),
        obs_fps=_read_positive_float(raw, "obs_fps", "", default=AppConfig.obs_fps),
        command_hz=_read_positive_float(raw, "command_hz", "", default=AppConfig.command_hz),
        command_interp_max_s=_read_positive_float(
            raw, "command_interp_max_s", "", default=AppConfig.command_interp_max_s
        ),
        sync_above_ms=_read_optional_positive_float(raw, "sync_above_ms", "", default=AppConfig.sync_above_ms),
        sys0_sync_above_ms=_read_optional_positive_float(
            raw, "sys0_sync_above_ms", "", default=AppConfig.sys0_sync_above_ms
        ),
        face_blur=_read_bool(raw, "face_blur", "", default=AppConfig.face_blur),
        live=_read_live(raw, [camera.name for camera in cameras]),
        exposure=_read_exposure(raw),
        rollouts=_read_rollouts(raw),
        checkpoint=_read_optional_str(raw, "checkpoint"),
        action_space=action_space,
        ik_solve=ik_solve,
        home_before_motors_off=_read_bool(raw, "home_before_motors_off", "", default=AppConfig.home_before_motors_off),
        home_on_key_refused=_read_bool(raw, "home_on_key_refused", "", default=AppConfig.home_on_key_refused),
    )


def _read_motor_temps(raw: dict[str, Any], body: str) -> tuple[float | None, float | None]:
    """Reads the optional top-level `motor_temp_warn_c` and `motor_temp_stop_c`, the owner's own motor temperature
    limits (`dreamcli robot temp` writes them): #ArmConfig's `temp_warn_c` and `temp_stop_c`.

    @param raw The top-level config mapping.
    @param body The file's body.
    @return `temp_warn_c` and `temp_stop_c`, each None when the file leaves it to the body.
    @throws ConfigError A value is not a positive number, the body is not one of #MOTOR_TEMP_BODIES, or the warning
        temperature is not below the stop temperature.
    """
    warn = _read_optional_positive_float(raw, "motor_temp_warn_c", "", default=None)
    stop = _read_optional_positive_float(raw, "motor_temp_stop_c", "", default=None)
    if (warn is not None or stop is not None) and body not in MOTOR_TEMP_BODIES:
        raise errors.ConfigError(
            f"config keys 'motor_temp_warn_c' and 'motor_temp_stop_c' are for bodies {list(MOTOR_TEMP_BODIES)}: "
            f"dream does not watch the motor temperatures of 'body: {body}'"
        )
    if warn is not None and stop is not None and warn >= stop:
        raise errors.ConfigError(
            f"config key 'motor_temp_warn_c' ({warn:g}) must be below 'motor_temp_stop_c' ({stop:g})"
        )
    return warn, stop


def _read_rollouts(raw: dict[str, Any]) -> RolloutsConfig:
    """Parses the optional `rollouts` mapping; absent means off.

    @param raw The top-level config mapping.
    @return The parsed #RolloutsConfig.
    @throws ConfigError The section is not a mapping, has an unknown key, or a value has the wrong type.
    """
    section = raw.get("rollouts")
    if section is None:
        return RolloutsConfig()
    if not isinstance(section, dict):
        raise errors.ConfigError(f"config key 'rollouts' must be a mapping, got {_describe(section)}")
    _reject_unknown(section, _ROLLOUTS_KEYS, "rollouts")
    defaults = RolloutsConfig()
    return RolloutsConfig(
        enabled=_read_bool(section, "enabled", "rollouts", default=defaults.enabled),
        dir=_read_str(section, "dir", "rollouts", default=defaults.dir),
        robot_name=_read_str(section, "robot_name", "rollouts", default=defaults.robot_name),
    )


def _read_exposure(raw: dict[str, Any]) -> ExposureConfig:
    """Parses the optional `exposure` mapping; absent means managed with the defaults.

    @param raw The top-level config mapping.
    @return The parsed #ExposureConfig.
    @throws ConfigError The section is not a mapping, has an unknown key, a `target_median` outside 1 to 254, a
        `seed_us` above `limit_us`, or a value of the wrong type or range.
    """
    section = raw.get("exposure")
    if section is None:
        return ExposureConfig()
    if not isinstance(section, dict):
        raise errors.ConfigError(f"config key 'exposure' must be a mapping, got {_describe(section)}")
    _reject_unknown(section, _EXPOSURE_KEYS, "exposure")
    target_median = _read_positive_float(section, "target_median", "exposure", default=ExposureConfig.target_median)
    if not 1.0 <= target_median <= 254.0:
        raise errors.ConfigError(f"config key 'exposure.target_median' must be within 1 to 254, got {target_median:g}")
    limit_us = _read_optional_positive_float(section, "limit_us", "exposure", default=None)
    seed_us = _read_optional_positive_float(section, "seed_us", "exposure", default=None)
    if limit_us is not None and seed_us is not None and seed_us > limit_us:
        raise errors.ConfigError(
            f"config key 'exposure.seed_us' ({seed_us:g}) must not exceed 'exposure.limit_us' ({limit_us:g})"
        )
    return ExposureConfig(
        managed=_read_bool(section, "managed", "exposure", default=ExposureConfig.managed),
        target_median=target_median,
        limit_us=limit_us,
        seed_us=seed_us,
        smooth_idle=_read_bool(section, "smooth_idle", "exposure", default=ExposureConfig.smooth_idle),
    )


def read_arm(section: dict[str, Any], key: str, body: str) -> ArmConfig:
    """Parses one arm's mapping on its own, as load() parses `follower_l` and `follower_r`.

    `robots.setup_wizard` checks the arm settings the operator gives with it.

    @param section The arm's mapping.
    @param key The section key, `follower_l` or `follower_r`, for messages.
    @param body The config's `body`.
    @return The parsed #ArmConfig; a relative calibration file is read from the working directory.
    @throws ConfigError As load() would for that section.
    """
    return _read_arm({key: section}, key, body, pathlib.Path.cwd())


def _read_arm(raw: dict[str, Any], key: str, body: str, base: pathlib.Path) -> ArmConfig:
    """Parses one `follower_*` mapping, with the keys of the config's body.

    @param raw The top-level config mapping.
    @param key The section key, `follower_l` or `follower_r`; also the dotted prefix in messages.
    @param body The config's `body`: a Makers Arm's section is read by _read_makers_arm(), a Metal arm's by
        _read_metal_arm(), an SO-101's by _read_so101_arm(), a custom robot's takes #_CUSTOM_ARM_KEYS.
    @param base The config file's directory, which a relative calibration file is read from.
    @return The parsed #ArmConfig.
    @throws ConfigError The section is missing or not a mapping, has an unknown key (one of the other body's
        included), a custom arm's has one gripper end without the other, or a value has the wrong type or range.
    """
    section = _require(raw, key, "")
    if not isinstance(section, dict):
        raise errors.ConfigError(f"config key '{key}' must be a mapping, got {_describe(section)}")
    if _RETIRED_ARM_KEY in section:
        raise errors.ConfigError(
            f"config key '{_dotted(key, _RETIRED_ARM_KEY)}' is gone: the gripper command reaches the wire as the "
            "policy sent it, and grip force is capped on the measured effort now; set the top-level "
            "'grip_force_cap' instead (i2rt gripper_eff units, 0.65 on the station) and remove it from the file"
        )
    if body == "makers":
        return _read_makers_arm(section, key)
    if body == "metal":
        return _read_metal_arm(section, key)
    if body == "so101":
        return _read_so101_arm(section, key, base)
    if body == CUSTOM_BODY:
        _reject_unknown(section, _CUSTOM_ARM_KEYS, key)
        open_deg, closed_deg = _read_custom_gripper_ends(section, key)
        return ArmConfig(
            can_channel=_read_str(section, "can_channel", key),
            gripper_open_deg=open_deg,
            gripper_closed_deg=closed_deg,
            home=_read_home(section, key, required=True),
            keep_out=_read_keep_out(section, key),
            reach=_read_reach(section, key),
            region=_read_region(section, key),
        )
    _reject_unknown(section, _ARM_KEYS, key)
    return ArmConfig(
        can_channel=_read_str(section, "can_channel", key),
        gripper=_read_str(section, "gripper", key, default=ArmConfig.gripper),
        ee_mass_kg=_read_optional_positive_float(section, "ee_mass_kg", key, default=ArmConfig.ee_mass_kg),
        gravity_comp_factor=_read_optional_positive_float(
            section, "gravity_comp_factor", key, default=ArmConfig.gravity_comp_factor
        ),
        sdk_gripper_force_limit=_read_bool(
            section, "sdk_gripper_force_limit", key, default=ArmConfig.sdk_gripper_force_limit
        ),
        temp_export=_read_bool(section, "temp_export", key, default=ArmConfig.temp_export),
        home=_read_home(section, key, required=False),
    )


def _read_makers_arm(section: dict[str, Any], key: str) -> ArmConfig:
    """Parses one `follower_*` mapping of a Makers Arm: #_MAKERS_ARM_KEYS.

    @param section The arm's mapping.
    @param key The section key, for messages.
    @return The parsed #ArmConfig; the YAM's fields keep their defaults, which nothing reads.
    @throws ConfigError The section has an unknown key, `can_channel` names a raw-USB adapter (#_RAW_USB_PREFIX), the
        open end of the gripper's stroke is not below the closed end, the holding torque is outside the vendor's
        range, or a value has the wrong type.
    """
    _reject_unknown(section, _MAKERS_ARM_KEYS, key)
    channel = _read_can_channel(section, key)
    open_deg = _read_float(section, "gripper_open_deg", key, default=ArmConfig.gripper_open_deg)
    closed_deg = _read_float(section, "gripper_closed_deg", key, default=ArmConfig.gripper_closed_deg)
    if not open_deg < closed_deg:
        raise errors.ConfigError(
            f"config key '{key}.gripper_open_deg' ({open_deg:g}) must be below '{key}.gripper_closed_deg' "
            f"({closed_deg:g}): the Makers gripper closes toward increasing angles"
        )
    hold = _read_optional_positive_float(
        section, "gripper_hold_torque_nm", key, default=ArmConfig.gripper_hold_torque_nm
    )
    if hold is not None and not makers_hold.MIN_HOLD_TORQUE_NM <= hold <= makers_hold.MAX_HOLD_TORQUE_NM:
        raise errors.ConfigError(
            f"config key '{key}.gripper_hold_torque_nm' must be within {makers_hold.MIN_HOLD_TORQUE_NM:g} to "
            f"{makers_hold.MAX_HOLD_TORQUE_NM:g} N.m, or null to turn holding off, got {hold:g}"
        )
    return ArmConfig(
        can_channel=channel,
        gripper_open_deg=open_deg,
        gripper_closed_deg=closed_deg,
        gripper_hold_torque_nm=hold,
        home=_read_home(section, key, required=False),
        keep_out=_read_keep_out(section, key),
        reach=_read_reach(section, key),
        region=_read_region(section, key),
    )


def _read_metal_arm(section: dict[str, Any], key: str) -> ArmConfig:
    """Parses one `follower_*` mapping of a Metal arm: #_METAL_ARM_KEYS.

    @param section The arm's mapping.
    @param key The section key, for messages.
    @return The parsed #ArmConfig; the other bodies' fields keep their defaults, which nothing reads.
    @throws ConfigError The section has an unknown key, `can_channel` names a raw-USB adapter (#_RAW_USB_PREFIX), the
        gripper's ends are missing, outside the gripper's soft limits or not closed below open, the holding torque is
        null or outside the vendor's range, or a value has the wrong type.
    """
    _reject_unknown(section, _METAL_ARM_KEYS, key)
    channel = _read_can_channel(section, key)
    open_deg, closed_deg = _read_gripper_ends(section, key)
    low, high = metal_gripper.LIMITS_DEG
    if not low <= closed_deg < open_deg <= high:
        raise errors.ConfigError(
            f"config keys '{key}.gripper_closed_deg' ({closed_deg:g}) and '{key}.gripper_open_deg' ({open_deg:g}) "
            f"must be within the gripper's soft limits {low:g} to {high:g}, closed below open: the Metal gripper "
            "opens toward increasing angles"
        )
    hold = _read_positive_float(section, "gripper_hold_torque_nm", key, default=makers_hold.DEFAULT_HOLD_TORQUE_NM)
    if not makers_hold.MIN_HOLD_TORQUE_NM <= hold <= makers_hold.MAX_HOLD_TORQUE_NM:
        raise errors.ConfigError(
            f"config key '{key}.gripper_hold_torque_nm' must be within {makers_hold.MIN_HOLD_TORQUE_NM:g} to "
            f"{makers_hold.MAX_HOLD_TORQUE_NM:g} N.m, got {hold:g}"
        )
    return ArmConfig(
        can_channel=channel,
        gripper_open_deg=open_deg,
        gripper_closed_deg=closed_deg,
        gripper_hold_torque_nm=hold,
        home=_read_home(section, key, required=True),
        keep_out=_read_keep_out(section, key),
        reach=_read_reach(section, key),
        region=_read_region(section, key),
    )


def _read_so101_arm(section: dict[str, Any], key: str, base: pathlib.Path) -> ArmConfig:
    """Parses one `follower_*` mapping of an SO-101: #_SO101_ARM_KEYS, every one required.

    @param section The arm's mapping.
    @param key The section key, for messages.
    @param base The config file's directory, which a relative calibration file is read from.
    @return The parsed #ArmConfig, the calibration file's path absolute and the gripper's ends its calibrated range.
    @throws ConfigError The section has an unknown key or lacks one, `can_channel` is not a port by id or by USB port
        (#SERIAL_BY_ID_PREFIX, #SERIAL_BY_PATH_PREFIX), the calibration file does not fit an SO-101, the home's sixth
        angle is not 0, or a value has the wrong type.
    """
    _reject_unknown(section, _SO101_ARM_KEYS, key)
    channel = _read_str(section, "can_channel", key)
    if not channel.startswith((SERIAL_BY_ID_PREFIX, SERIAL_BY_PATH_PREFIX)):
        raise errors.ConfigError(
            f"config key '{key}.can_channel' is {channel!r}: name the arm's serial port by id ({SERIAL_BY_ID_PREFIX}"
            f"..., 'ls -l /dev/serial/by-id/'), or by its USB port ({SERIAL_BY_PATH_PREFIX}...) when two adapters "
            "share a serial, since a /dev/ttyACM name changes when a cable is replugged"
        )
    calibration = str((base / pathlib.Path(_read_str(section, "calibration", key)).expanduser()).resolve())
    open_deg, closed_deg = so101.gripper_ends(calibration)
    home = _read_home(section, key, required=True)
    held = home[_SO101_JOINTS:]
    if any(held):
        raise errors.ConfigError(
            f"config key '{key}.home' ends in {', '.join(f'{x:g}' for x in held)}: an SO-101 has five joints, so the "
            "angles past them are 0"
        )
    return ArmConfig(
        can_channel=channel,
        gripper_open_deg=open_deg,
        gripper_closed_deg=closed_deg,
        calibration=calibration,
        home=home,
        keep_out=_read_keep_out(section, key),
        reach=_read_reach(section, key),
        region=_read_region(section, key),
    )


def _read_custom(raw: dict[str, Any], base: pathlib.Path) -> CustomConfig:
    """Parses the `custom` mapping of a robot of your own.

    @param raw The top-level config mapping.
    @param base The config file's directory, which a relative driver or arm file is read from.
    @return The #CustomConfig, a driver file's and an arm file's paths made absolute.
    @throws ConfigError The section is missing or not a mapping, has an unknown key, the driver is not
        `module:Class` or `file.py:Class`, or a value has the wrong type.
    """
    section = _require(raw, "custom", "")
    if not isinstance(section, dict):
        raise errors.ConfigError(f"config key 'custom' must be a mapping, got {_describe(section)}")
    _reject_unknown(section, _CUSTOM_KEYS, "custom")
    driver = user_class.path(_read_str(section, "driver", "custom"), "custom.driver", base)
    ik = _read_str(section, "ik", "custom")
    if ik.endswith(".json"):
        ik = str((base / pathlib.Path(ik).expanduser()).resolve())
    return CustomConfig(
        driver=driver,
        ik=ik,
        embodiment=_read_str(section, "embodiment", "custom"),
        grip_scale=_read_optional_positive_float(section, "grip_scale", "custom", default=None),
    )


def _read_home(section: dict[str, Any], key: str, *, required: bool) -> tuple[float, ...]:
    """Reads an arm's `home`: a finite joint angle per joint, radians, which says how many joints the arm has.

    @param section The arm's mapping.
    @param key The section key, for messages.
    @param required The arm has no safe default home (home_help()).
    @return The angles, or #ArmConfig's zeros when the key is absent and not required.
    @throws ConfigError The key is required and absent, or is not 1 to `robots.followers.MAX_JOINTS` finite numbers.
    """
    if "home" not in section:
        if required:
            raise errors.ConfigError(home_help(f"config section '{key}'"))
        return ArmConfig.home
    value = section["home"]
    numbers = isinstance(value, list) and all(_is_int(x) or isinstance(x, float) for x in value)
    if not numbers or not 1 <= len(value) <= followers.MAX_JOINTS or not all(math.isfinite(x) for x in value):
        raise errors.ConfigError(
            f"config key '{key}.home' must be 1 to {followers.MAX_JOINTS} finite joint angles in radians, one per "
            f"joint of the arm, got {_describe(value)}"
        )
    return tuple(float(x) for x in value)


def _read_keep_out(section: dict[str, Any], key: str) -> tuple[dict[str, Any], ...] | None:
    if "keep_out" not in section:
        return None
    value = section["keep_out"]
    if not isinstance(value, list):
        raise errors.ConfigError(f"config key '{key}.keep_out' must be a list of obstacles, got {_describe(value)}")
    out = []
    for i, entry in enumerate(value):
        where = f"{key}.keep_out[{i}]"
        kind = entry.get("kind") if isinstance(entry, dict) else None
        if kind not in _KEEP_OUT_KEYS:
            raise errors.ConfigError(
                f"config key '{where}' must be a mapping whose 'kind' is 'box' or 'capsule', got {_describe(entry)}"
            )
        if set(entry) != _KEEP_OUT_KEYS[kind]:
            raise errors.ConfigError(
                f"config key '{where}' is a {kind}: it takes exactly the keys {sorted(_KEEP_OUT_KEYS[kind])}, got "
                f"{sorted(entry)}"
            )
        obstacle: dict[str, Any] = {"kind": kind}
        for name in sorted(_KEEP_OUT_KEYS[kind] - {"kind"}):
            v = entry[name]
            three = name in ("center", "size")
            numbers = v if three and isinstance(v, list) else [v]
            ok = all((_is_int(x) or isinstance(x, float)) and math.isfinite(x) for x in numbers)
            if not ok or len(numbers) != (3 if three else 1) or (name != "center" and min(numbers) <= 0):
                what = (
                    {"center": "three finite numbers", "size": "three numbers above 0"}[name]
                    if three
                    else "a number above 0"
                )
                raise errors.ConfigError(f"config key '{where}.{name}' must be {what}, metres, got {_describe(v)}")
            obstacle[name] = [float(x) for x in numbers] if three else float(v)
        out.append(obstacle)
    return tuple(out)


def _read_reach(section: dict[str, Any], key: str) -> tuple[tuple[float, float], ...] | None:
    if "reach" not in section:
        return None
    value = section["reach"]
    if not isinstance(value, list) or not value:
        raise errors.ConfigError(
            f"config key '{key}.reach' must be a list of [x, y] points in metres, at least one, got {_describe(value)}"
        )
    for i, point in enumerate(value):
        pair = isinstance(point, list) and len(point) == 2
        if not pair or not all((_is_int(x) or isinstance(x, float)) and math.isfinite(x) for x in point):
            raise errors.ConfigError(
                f"config key '{key}.reach[{i}]' must be [x, y], two finite numbers in metres, got {_describe(point)}"
            )
    return tuple((float(x), float(y)) for x, y in value)


def _read_region(section: dict[str, Any], key: str) -> dict[str, list[float | None]] | None:
    """Reads an arm's `region`, the box it stays inside (#ArmConfig).

    @param section The arm's mapping.
    @param key The section key, for messages.
    @return `{"lo": [...], "hi": [...]}` with the faces the file gives, three values each, None for an open face; None
        when the section has no `region`.
    @throws ConfigError It is not a mapping of `lo` and `hi`, a face is not three finite numbers or nulls, or a `lo`
        is not below its `hi`.
    """
    if "region" not in section:
        return None
    value = section["region"]
    if not isinstance(value, dict) or not value or not set(value) <= {"lo", "hi"}:
        raise errors.ConfigError(
            f"config key '{key}.region' must be a mapping with 'lo' and or 'hi', each [x, y, z] in metres (null "
            f"for an open face), got {_describe(value)}"
        )
    out: dict[str, list[float | None]] = {}
    for face, values in value.items():
        three = isinstance(values, list) and len(values) == 3
        if not three or not all(
            v is None or ((_is_int(v) or isinstance(v, float)) and math.isfinite(v)) for v in values
        ):
            raise errors.ConfigError(
                f"config key '{key}.region.{face}' must be [x, y, z], three finite numbers in metres or null, got "
                f"{_describe(values)}"
            )
        out[face] = [None if v is None else float(v) for v in values]
    for axis, (lo, hi) in enumerate(zip(out.get("lo", [None] * 3), out.get("hi", [None] * 3), strict=True)):
        if lo is not None and hi is not None and lo >= hi:
            raise errors.ConfigError(
                f"config key '{key}.region' has lo {lo} not below hi {hi} on {'xyz'[axis]}: the box is empty"
            )
    return out


def _read_can_channel(section: dict[str, Any], key: str) -> str:
    """Reads a Makers or Metal arm's `can_channel`: a SocketCAN link or a serial device, never a raw-USB adapter.

    @param section The arm's mapping.
    @param key The section key, for messages.
    @return The channel.
    @throws ConfigError The key is missing or not a string, or names a raw-USB adapter (#_RAW_USB_PREFIX).
    """
    channel = _read_str(section, "can_channel", key)
    if channel.startswith(_RAW_USB_PREFIX):
        raise errors.ConfigError(
            f"config key '{key}.can_channel' is {channel!r}, a raw-USB candleLight adapter, which this driver does "
            "not open; leave the kernel's gs_usb driver on it and name the SocketCAN link it makes "
            "('ip -br link show type can')"
        )
    return channel


def _read_gripper_ends(section: dict[str, Any], key: str) -> tuple[float, float]:
    """Reads an arm's required gripper ends, `gripper_open_deg` and `gripper_closed_deg` (gripper_ends_help()).

    @param section The arm's mapping.
    @param key The section key, for messages.
    @return `(open_deg, closed_deg)`.
    @throws ConfigError Either is absent or not a finite number.
    """
    if "gripper_open_deg" not in section or "gripper_closed_deg" not in section:
        raise errors.ConfigError(gripper_ends_help(f"config section '{key}'"))
    return (
        _read_float(section, "gripper_open_deg", key, default=math.nan),
        _read_float(section, "gripper_closed_deg", key, default=math.nan),
    )


def _read_custom_gripper_ends(section: dict[str, Any], key: str) -> tuple[float, float]:
    """Reads a robot of your own's gripper ends: both or neither, apart.

    @param section The arm's mapping.
    @param key The section key, for messages.
    @return `(open_deg, closed_deg)`; both NaN when the section has neither.
    @throws ConfigError Only one is given, either is not a finite number, or they are equal.
    """
    given = [name for name in ("gripper_open_deg", "gripper_closed_deg") if name in section]
    if not given:
        return math.nan, math.nan
    if len(given) == 1:
        raise errors.ConfigError(gripper_ends_help(f"config section '{key}', which sets only '{given[0]}',"))
    open_deg, closed_deg = _read_gripper_ends(section, key)
    if open_deg == closed_deg:
        raise errors.ConfigError(
            f"config keys '{key}.gripper_open_deg' and '{key}.gripper_closed_deg' are both {open_deg:g}: the gripper "
            "would not move"
        )
    return open_deg, closed_deg


def _read_cameras(raw: dict[str, Any], base: pathlib.Path) -> list[CameraConfig]:
    """Parses the `cameras` list: non-empty, every entry a mapping, names unique.

    @param raw The top-level config mapping.
    @param base The config file's directory, which a relative camera class file is read from.
    @return One #CameraConfig per entry, in file order.
    @throws ConfigError The list is missing or empty, two entries share a name, or an entry is invalid.
    """
    entries = _require(raw, "cameras", "")
    if not isinstance(entries, list) or not entries:
        raise errors.ConfigError(f"config key 'cameras' must be a non-empty list, got {_describe(entries)}")
    cameras = [_read_camera(entry, f"cameras[{i}]", base) for i, entry in enumerate(entries)]
    names = [camera.name for camera in cameras]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise errors.ConfigError(f"config key 'cameras' has duplicate names {duplicates}; each becomes an image key")
    return cameras


def _read_live(raw: dict[str, Any], camera_names: list[str]) -> LiveConfig | None:
    """Parses the optional `live` mapping; None disables the agent.

    @param raw The top-level config mapping.
    @param camera_names The config's camera names, the set `live.cameras` may pick from and its default.
    @return The parsed #LiveConfig, or None when the section is absent.
    @throws ConfigError The section is not a mapping, has an unknown key or the moved `url`, has no `robot_id`, or
        names a camera the config does not have.
    """
    section = raw.get("live")
    if section is None:
        return None
    if not isinstance(section, dict):
        raise errors.ConfigError(f"config key 'live' must be a mapping, got {_describe(section)}")
    if "url" in section:
        raise errors.ConfigError(f"config key 'live.url' is gone: {_MOVED_KEYS['live.url']}; remove it from the file")
    _reject_unknown(section, LIVE_KEYS, "live")
    cameras = section.get("cameras", camera_names)
    if not isinstance(cameras, list) or not cameras or not all(isinstance(name, str) for name in cameras):
        raise errors.ConfigError(
            f"config key 'live.cameras' must be a non-empty list of camera names, got {_describe(cameras)}"
        )
    unknown = sorted(set(cameras) - set(camera_names))
    if unknown:
        raise errors.ConfigError(f"config key 'live.cameras' names cameras the config does not have: {unknown}")
    ignored = sorted(LIVE_IGNORED_KEYS & set(section))
    if ignored:
        _log.warning(
            "config: live.%s no longer set anything (Live View encodes at the platform's settings, and every run is "
            "recorded by its own dream client); remove them from the file",
            ", live.".join(ignored),
        )
    return LiveConfig(robot_id=_read_str(section, "robot_id", "live"), cameras=tuple(cameras))


def _read_camera(entry: Any, path: str, base: pathlib.Path) -> CameraConfig:
    """Parses one `cameras[i]` mapping.

    @param entry The raw list element; must be a mapping.
    @param path The dotted location of `entry` for messages, e.g. `"cameras[0]"`.
    @param base The config file's directory, which a relative camera class file is read from.
    @return The parsed #CameraConfig.
    @throws ConfigError The entry is not a mapping, has an unknown key, sets two or none of `device`, `serial`
        and `source`, has a `rotate` outside #_ROTATIONS, or a value of the wrong type or range.
    """
    if not isinstance(entry, dict):
        raise errors.ConfigError(f"config key '{path}' must be a mapping, got {_describe(entry)}")
    _reject_unknown(entry, _CAMERA_KEYS, path)
    device, serial, source = _read_capture_source(entry, path, base)
    rotate = _read_int(entry, "rotate", path, minimum=0, default=CameraConfig.rotate)
    if rotate not in _ROTATIONS:
        raise errors.ConfigError(f"config key '{path}.rotate' must be one of {_ROTATIONS}, got {rotate}")
    return CameraConfig(
        name=_read_str(entry, "name", path),
        device=device,
        serial=serial,
        source=source,
        width=_read_int(entry, "width", path, minimum=1),
        height=_read_int(entry, "height", path, minimum=1),
        fps=_read_int(entry, "fps", path, minimum=1, default=CameraConfig.fps),
        rotate=rotate,
    )


def _read_capture_source(
    entry: dict[str, Any], path: str, base: pathlib.Path
) -> tuple[int | str | None, str | None, str | None]:
    """Reads `device`, `serial` and `source` of one camera entry: exactly one is set, and each has its type.

    @param entry The camera mapping.
    @param path The dotted location of `entry` for messages, e.g. `"cameras[0]"`.
    @param base The config file's directory, which a relative camera class file is read from.
    @return `(device, serial, source)`: an int index or str path, a non-empty str, or a class name
        (#robots.user_class.path), the other two None.
    @throws ConfigError Two or none are set, `device` is not an int or str, `serial` is not a non-empty str, or
        `source` is not `module:Class` or `file.py:Class`.
    """
    device = entry.get("device")
    serial = entry.get("serial")
    source = entry.get("source")
    given = [key for key, value in (("device", device), ("serial", serial), ("source", source)) if value is not None]
    if not given:
        raise errors.ConfigError(
            f"config key '{path}' needs '{path}.device' (OpenCV index, path or address), '{path}.serial' (RealSense "
            f"serial) or '{path}.source' (a camera class of your own)"
        )
    if len(given) > 1:
        raise errors.ConfigError(
            f"config keys {' and '.join(f'{path}.{key}' for key in given)} are set; a camera names exactly one"
        )
    if source is not None:
        return None, None, user_class.path(_read_str(entry, "source", path), f"{path}.source", base)
    if device is not None and not (_is_int(device) or isinstance(device, str)):
        raise errors.ConfigError(
            f"config key '{path}.device' must be an int index or a str path, got {_describe(device)}"
        )
    if serial is not None and not (isinstance(serial, str) and serial):
        # A bare `serial: 260322275394` is an int to YAML; librealsense matches on the string.
        raise errors.ConfigError(
            f"config key '{path}.serial' must be a non-empty str (quote it), got {_describe(serial)}"
        )
    return device, serial, None


def _read_image_size(section: dict[str, Any], key: str, default: tuple[int, int]) -> tuple[int, int]:
    """Reads a top-level `WIDTHxHEIGHT` string as `(width, height)`; `default` is used only when the key is absent.

    @param section The mapping.
    @param key The key.
    @param default The size when the key is absent.
    @return `(width, height)`, both positive ints.
    @throws ConfigError The value is not a `WIDTHxHEIGHT` string of two positive ints.
    """
    if key not in section:
        return default
    value = section[key]
    width, _, height = str(value).partition("x")
    is_size = isinstance(value, str) and width.isdigit() and height.isdigit()
    if not is_size or int(width) < 1 or int(height) < 1:
        raise errors.ConfigError(
            f"config key '{key}' must be a WIDTHxHEIGHT string such as '640x360', got {_describe(value)}"
        )
    return int(width), int(height)


def _read_optional_str(section: dict[str, Any], key: str) -> str | None:
    """Reads a top-level string that may be absent or null: None then, a non-empty string otherwise.

    @param section The mapping.
    @param key The key.
    @return The string, or None when the key is absent or null.
    @throws ConfigError The value is present and not a non-empty string.
    """
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise errors.ConfigError(f"config key '{key}' must be a non-empty str or null, got {_describe(value)}")
    return value


def _read_optional_path(section: dict[str, Any], key: str) -> str | None:
    """Reads a top-level file path that may be absent or null, with a leading `~` expanded to the home directory.

    Nothing between the file and `open()` is a shell, so `~/robots-trace.csv` would otherwise name a directory
    called `~` in the working directory, which does not exist, and every Start would fail on `FileNotFoundError`.

    @param section The mapping.
    @param key The key.
    @return The expanded path, or None when the key is absent or null.
    @throws ConfigError The value is present and not a non-empty string.
    """
    value = _read_optional_str(section, key)
    return None if value is None else os.path.expanduser(value)


def _read_optional_task(raw: dict[str, Any]) -> str:
    """Reads the top-level `task`: absent is `""` (the prompt comes with the Start), present must be a
    non-empty string.

    @param raw The top-level config mapping.
    @return The task, or `""`.
    @throws ConfigError `task` is present but not a non-empty str.
    """
    if "task" not in raw:
        return ""
    return _read_str(raw, "task", "")


def _read_str(section: dict[str, Any], key: str, path: str, default: str | None = None) -> str:
    """Reads a non-empty string; `default` is used only when the key is absent.

    @param section The mapping to read from.
    @param key The key.
    @param path The dotted location of `section` for messages; `""` at the top level.
    @param default The value when the key is absent; None makes the key required.
    @return The string.
    @throws ConfigError The key is absent with no default, or the value is not a non-empty str.
    """
    if key not in section and default is not None:
        return default
    value = _require(section, key, path)
    if not isinstance(value, str) or not value:
        raise errors.ConfigError(f"config key '{_dotted(path, key)}' must be a non-empty str, got {_describe(value)}")
    return value


def _read_int(section: dict[str, Any], key: str, path: str, minimum: int, default: int | None = None) -> int:
    """Reads an int (never a bool) of at least `minimum`; `default` is used only when the key is absent.

    @param section The mapping to read from.
    @param key The key.
    @param path The dotted location of `section` for messages; `""` at the top level.
    @param minimum The smallest accepted value.
    @param default The value when the key is absent; None makes the key required.
    @return The int.
    @throws ConfigError The key is absent with no default, or the value is not an int of at least `minimum`.
    """
    if key not in section and default is not None:
        return default
    value = _require(section, key, path)
    if not _is_int(value) or value < minimum:
        raise errors.ConfigError(
            f"config key '{_dotted(path, key)}' must be an int >= {minimum}, got {_describe(value)}"
        )
    return int(value)


def _read_stall_ticks(raw: dict[str, Any], path: str, control_hz: float, action_timeout_s: float) -> int:
    ticks = _read_int(raw, "max_stall_ticks", "", minimum=0)
    if ticks / control_hz >= action_timeout_s:
        return ticks
    floor = max(MAX_STALL_TICKS, math.ceil(action_timeout_s * control_hz))
    _log.warning(
        "%s: max_stall_ticks %d holds the arms %.2f s at %g Hz, less than one late reply (action_timeout_s=%g), so "
        "one slow answer would end a run; an older dreamcli setup wrote it. Using %d; set max_stall_ticks to %d in "
        "the file to stop this message",
        path,
        ticks,
        ticks / control_hz,
        control_hz,
        action_timeout_s,
        floor,
        floor,
    )
    return floor


def _read_positive_float(section: dict[str, Any], key: str, path: str, default: float | None = None) -> float:
    """Reads a finite number > 0 (int or float, never a bool) as a float; `default` only when the key is absent.

    @param section The mapping to read from.
    @param key The key.
    @param path The dotted location of `section` for messages; `""` at the top level.
    @param default The value when the key is absent; None makes the key required.
    @return The number as a float.
    @throws ConfigError The key is absent with no default, or the value is not a positive finite number.
    """
    if key not in section and default is not None:
        return default
    value = _require(section, key, path)
    is_number = _is_int(value) or isinstance(value, float)
    if not is_number or not math.isfinite(value) or value <= 0:
        raise errors.ConfigError(
            f"config key '{_dotted(path, key)}' must be a positive finite number, got {_describe(value)}"
        )
    return float(value)


def _read_float(section: dict[str, Any], key: str, path: str, default: float) -> float:
    """Reads a finite number (int or float, never a bool) of either sign as a float; `default` only when the key is
    absent.

    @param section The mapping to read from.
    @param key The key.
    @param path The dotted location of `section` for messages; `""` at the top level.
    @param default The value when the key is absent.
    @return The number as a float.
    @throws ConfigError The value is not a finite number.
    """
    if key not in section:
        return default
    value = section[key]
    is_number = _is_int(value) or isinstance(value, float)
    if not is_number or not math.isfinite(value):
        raise errors.ConfigError(f"config key '{_dotted(path, key)}' must be a finite number, got {_describe(value)}")
    return float(value)


def _read_optional_positive_float(section: dict[str, Any], key: str, path: str, default: float | None) -> float | None:
    """Reads a finite number > 0 as a float, or null as None; `default` is used only when the key is absent.

    @param section The mapping to read from.
    @param key The key.
    @param path The dotted location of `section` for messages; `""` at the top level.
    @param default The value when the key is absent; may itself be None.
    @return The number as a float, or None when the value is null.
    @throws ConfigError The value is present and neither null nor a positive finite number.
    """
    if key not in section:
        return default
    value = section[key]
    if value is None:
        return None
    is_number = _is_int(value) or isinstance(value, float)
    if not is_number or not math.isfinite(value) or value <= 0:
        raise errors.ConfigError(
            f"config key '{_dotted(path, key)}' must be a positive finite number or null, got {_describe(value)}"
        )
    return float(value)


def _read_bool(section: dict[str, Any], key: str, path: str, default: bool) -> bool:
    """Reads a YAML bool (`1` and `"true"` are rejected); `default` is used only when the key is absent.

    @param section The mapping to read from.
    @param key The key.
    @param path The dotted location of `section` for messages; `""` at the top level.
    @param default The value when the key is absent.
    @return The bool.
    @throws ConfigError The value is not a bool.
    """
    if key not in section:
        return default
    value = section[key]
    if not isinstance(value, bool):
        raise errors.ConfigError(f"config key '{_dotted(path, key)}' must be a bool, got {_describe(value)}")
    return value


def _require(section: dict[str, Any], key: str, path: str) -> Any:
    """Returns `section[key]` or raises naming the key.

    @param section The mapping to read from.
    @param key The key.
    @param path The dotted location of `section` for messages; `""` at the top level.
    @return The raw value, unchecked.
    @throws ConfigError The key is absent.
    """
    if key not in section:
        raise errors.ConfigError(f"config key '{_dotted(path, key)}' is required")
    return section[key]


def _reject_unknown(section: dict[str, Any], allowed: frozenset[str], path: str) -> None:
    """Raises on keys outside `allowed`; a typo must not silently fall back to a default.

    @param section The mapping to check.
    @param allowed The keys the section may carry.
    @param path The dotted location of `section` for messages; `""` at the top level.
    @throws ConfigError The section has a key outside `allowed`; the message lists the offenders and `allowed`.
    """
    unknown = sorted(str(key) for key in section.keys() - allowed)
    if unknown:
        where = f"'{path}'" if path else "the top level"
        raise errors.ConfigError(f"config has unknown keys {unknown} under {where}; allowed: {sorted(allowed)}")


def _is_int(value: Any) -> bool:
    """True for an int that is not a bool (`True` is an int in Python, but never a count).

    @param value Any parsed YAML value.
    @return True when `value` is an int and not a bool.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _dotted(path: str, key: str) -> str:
    """Joins a section path and a key for messages: `"cameras[0].width"`, or `"control_hz"` at the top.

    @param path The dotted location of the section; `""` at the top level.
    @param key The key inside it.
    @return `"path.key"`, or just `key` when `path` is empty.
    """
    return f"{path}.{key}" if path else key


def _describe(value: Any) -> str:
    """Renders a value with its type for messages, so `True` and `1` are told apart.

    @param value The offending value.
    @return `repr(value)` followed by the type name in parentheses.
    """
    return f"{value!r} ({type(value).__name__})"


def _read_non_negative_float(section: dict[str, Any], key: str, path: str, default: float) -> float:
    """Reads a finite number >= 0 (int or float, never a bool) as a float; `default` only when the key is absent.

    @param section The mapping to read from.
    @param key The key.
    @param path The dotted location of `section` for messages; `""` at the top level.
    @param default The value when the key is absent.
    @return The number as a float.
    @throws ConfigError The value is not a non-negative finite number.
    """
    if key not in section:
        return default
    value = section[key]
    is_number = _is_int(value) or isinstance(value, float)
    if not is_number or not math.isfinite(value) or value < 0:
        raise errors.ConfigError(
            f"config key '{_dotted(path, key)}' must be a non-negative finite number, got {_describe(value)}"
        )
    return float(value)

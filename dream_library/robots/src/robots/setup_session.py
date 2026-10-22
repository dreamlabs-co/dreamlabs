"""The setup wizard's arms while the operator moves them by hand, and what the page has settled so far.

Every arm is opened the one way `dreamcli robot run` opens it, its body's `open_arm(cfg, side)` (#robots.bodies,
body_of()), and used only through the #robots.followers.FollowerArm that returns. While the operator moves arms by hand
(finding which arm is which, the home pose, the gripper's range) each arm is let go with `disengage()`, its driver's
passive mode, and read on and on: a YAM floats under gravity compensation, a bus arm (Makers, Metal, SO-101) goes limp
with its motors off (#robots.motor_arm.MotorArm.disengage), so the page tells the operator to support it first
(#LIMP_NOTE). Nothing is written to a disengaged arm. Before anything moves under power the arms are closed and
opened again, which holds each where it is (#robots.setup_motion).

- #ArmBank: the arms open now, read on a thread (#robots.setup_session.Session.tick) into the latest
  #robots.setup_checks.Reading per device.
- #Session: what the page has said so far (the robot, which device is which arm, the home pose), the arms above,
  and the powered checks (#robots.setup_motion.Motion) once started.

@see Session, Robot, robot_choices()
"""

import dataclasses
import datetime
import logging
import math
import os
import pathlib
import re
import tempfile
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from dream import client_ik

from robots import (
    beta_robots,
    bodies,
    config,
    control,
    errors,
    feetech,
    followers,
    metal_gripper,
    setup_checks,
    setup_devices,
    setup_model,
    setup_motion,
    so101,
    so101_calibration,
)
from robots.setup_checks import Reading

## How often the disengaged arms are read, per second.
READ_HZ: float = 20.0
SETTLE_S: float = 1.5

GRIPPER_READ_RANGE_DEG: dict[str, tuple[float, float]] = {
    "makers": (config.ArmConfig.gripper_closed_deg, config.ArmConfig.gripper_open_deg),
    "metal": metal_gripper.LIMITS_DEG,
}
GRIPPER_MIN_SPAN_DEG: float = 20.0
## The reach area's grid, metres: a painted hand point is rounded to it, so a hand held still adds one point.
REACH_GRID_M: float = 0.02
## The least area per arm, m^2, before the page's Continue (20 by 20 cm): the hand at home alone would be a 4 cm disk.
REACH_MIN_AREA_M2: float = 0.04
## Why a robot of your own skips the gripper step: its driver reports no gripper motor angle.
GRIPPER_SKIPPED = (
    "Your driver reports no gripper motor angle (its gripper_deg() returns None, as a robots.followers.Driver does "
    "unless you fill it in), so it keeps its gripper's range itself and this step is skipped. To measure the range "
    "here, have gripper_deg() return the angle read() saw, and map the gripper with the arm's gripper_closed_deg and "
    "gripper_open_deg (robots.driver_template)."
)

## The home pose the page draws for an SO-101 to put each arm in, degrees per joint slot: the arm seated as MakerMods'
## stations rest, folded down onto its base with the wrist camera facing up and the gripper closed, lifted just off
## the seat into the range dream's model was recorded on (the seat is past its elbow and shoulder ends,
## #robots.so101_calibration.past_reference()). The operator puts the arms there by hand; what they read is what Save
## keeps.
SO101_HOME_DEG: tuple[float, ...] = (0.0, -95.0, 80.0, 0.0, -90.0, 0.0)

## Where `lerobot-calibrate` writes an SO-101 follower's calibration files, one per robot id.
SO101_CALIBRATION_DIR = (
    pathlib.Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so101_follower"
)

_log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Robot:
    """The robot the operator picked.

    Attributes:
    - body: The config's `body` (#robots.config.BODY_NAMES, or #robots.config.CUSTOM_BODY).
    - label: What the page calls it.
    - ik_arm: dream's URDF of its arm (`dream_ik`), the model every check draws.
    - roles: Its camera roles (its wire's, #robots.adapter.Wire).
    - note: What the page says about it beside its tile and before it lets the arms go, or None.
    - unavailable: Why it cannot be picked on this install, or None.
    - embodiment: An SO-101's `embodiment`, the account's embodiment its fine-tune was trained on; else None.
    - custom: A robot of the customer's own: the config's `custom` section; else None.
    - calibration_files: An SO-101's LeRobot calibration files; each arm opens with the one its servos hold.
    - embodiment_id: The platform's id of its embodiment, what Save registers the robot as: #EMBODIMENTS for ours,
      the account embodiment's `emb_` id for an SO-101 or a robot of your own; None until that is picked.
    """

    body: str
    label: str
    ik_arm: str
    roles: tuple[str, ...]
    note: str | None = None
    unavailable: str | None = None
    embodiment: str | None = None
    custom: config.CustomConfig | None = None
    calibration_files: tuple[str, ...] = ()
    embodiment_id: str | None = None


## The platform's embodiment each built-in body registers as (`POST /v1/robots`) and its fine-tunes are trained on.
EMBODIMENTS: dict[str, str] = {"yam": "yam", "makers": "makermods_maker_arm", "metal": "makermods_metal"}

MAKERMODS_ROLES: tuple[str, ...] = ("scene", "wrist_l", "wrist_r")

ARM_KINDS: dict[str, frozenset[str]] = {
    "yam": frozenset({"can"}),
    "makers": frozenset({"can", "slcan"}),
    "metal": frozenset({"can", "slcan"}),
    "so101": frozenset({"serial"}),
}

BODY_LABELS: dict[str, str] = {"yam": "YAM", "makers": "Maker Arm", "metal": "Metal Arm", "so101": "SO-101"}

## The arms of dream's IK a robot of your own picks from, by the arm's own name (own_arms()): every arm but the ones
## the first step's tiles already are (robot_choices()) and #SAME_ARM, which are the beta robots
## (#robots.beta_robots) with no tile of their own, sorted by dream's name.
OWN_ARM_LABELS: dict[str, str] = {
    robot.ik: robot.label
    for robot in sorted(beta_robots.robots().values(), key=lambda r: r.ik)
    if robot.body == "custom"
}

## dream's arms the list leaves out as a second entry for an arm it has: `b601` is the reBot's URDF and tool point as
## `seeed_faas`, which the reBot's checkpoints and example were made on; `yamumi` is the YAM's arm with GenRobot's DAS
## gripper, the YAM tile's arm with another tool point.
SAME_ARM: dict[str, str] = {"b601": "seeed_faas", "yamumi": "yam"}

NOT_ANSWERING_HELP = (
    "it may still hold where it stopped, and the wizard cannot reach it to let it go. Support it, switch its power "
    "off, check its cable, then plug it back and switch it on."
)

_HOT = re.compile(
    r"(?P<motor>\w+) (?P<sensor>[\w ]+?) reached (?P<c>\d+) C \(the run stops at (?P<stop>\d+) C\); the arm parks\. "
    r"(?P<then>.+)"
)


def _what_happened(side: str, why: str) -> tuple[str, str]:
    reason = why.removeprefix(f"arm '{side}': ")
    hot = _HOT.fullmatch(reason)
    if hot is None:
        return f"The {side} arm: {reason}", ""
    motor = hot["motor"].replace("_", " ")
    said = f"The {side} arm's {motor} motor reached {hot['c']} C at its {hot['sensor']} (the limit is {hot['stop']} C)"
    return said, f"{hot['then']}."


def parked_words(side: str, why: str, limp: bool = False) -> str:
    """What the page says about an arm parked and switched off, once and plainly (browser B20).

    @param side The arm.
    @param why Its alarm.
    @param limp It was let go limp before it was off.
    @return The sentence.
    """
    what, then = _what_happened(side, why)
    if limp:
        return f"{what}; it went limp where it stood, not parked at rest, and is switched off now. {then}".strip()
    return f"{what}, so it was parked and switched off. {then}".strip()


def still_holding_words(side: str) -> str:
    """What the page and the exit say about an arm that drops when limp and stopped answering under power.

    @param side The arm.
    @return The sentence.
    """
    return (
        f"the {side} arm stopped answering; it still holds under power so it does not drop: support it, then switch "
        "its power off"
    )


def off_failed_words(side: str, why: str) -> str:
    """What the page says about an arm whose park did not switch it off, as parked_words() does.

    @param side The arm.
    @param why Why it was to be parked.
    @return The sentence.
    """
    what, _ = _what_happened(side, why)
    return (
        f"{what}. Switching it off failed, so it may still be powered: support it, and cut its power if it still "
        "holds."
    )


## What the page says on a body with no brakes, at its tile and before the arms go limp.
LIMP_NOTE = (
    "This arm has no brakes. When the wizard turns off the motors, the arm drops. Support the arm with your hand "
    "when finding the arms and moving to home position."
)


def body_of(robot: Robot) -> bodies.Body:
    """The robot's #robots.bodies.Body, as #robots.bodies.for_config picks it from a config.

    @param robot The robot.
    @return The body.
    """
    if robot.custom is not None:
        return bodies.custom_body(robot.custom)
    if robot.body == "so101":
        assert robot.embodiment is not None, "so101_robot() sets it"
        return bodies.so101_body(robot.embodiment)
    return bodies.BODIES[robot.body]


def reaches(robot: Robot) -> bool:
    """Whether the robot gets a reach area (step Reach area): dream's IK solves its hands (every body but the YAM).

    @param robot The robot.
    @return True when its body has an `ik`.
    """
    return body_of(robot).ik is not None


def robot_choices() -> list[Robot]:
    """The robots the first step offers: each body of #robots.bodies.BODIES, then the SO-101.

    @return One #Robot per body (a YAM's arm is dream's `yam`, the others' their `ik["arm"]`), then the SO-101, whose
        camera roles come with its embodiment (so101_robot()); unavailable when this install's dream IK lacks the arm.
    """
    try:
        specs, missing = setup_model.spec_names(), None
    except errors.ConfigError as exc:
        specs, missing = [], str(exc)
    choices = []
    for name, body in [*bodies.BODIES.items(), ("so101", None)]:
        ik_arm = so101.ARM.ik if body is None else (body.ik["arm"] if body.ik is not None else name)
        unavailable = missing or (None if ik_arm in specs else f"this install's dream IK has no {ik_arm}")
        if body is None or unavailable:
            roles: tuple[str, ...] = ()
        else:
            roles = MAKERMODS_ROLES if name in ("makers", "metal") else tuple(body.wire().camera_roles)
        choices.append(
            Robot(
                body=name,
                label=BODY_LABELS[name],
                ik_arm=ik_arm,
                roles=roles,
                note=None if name == "yam" else LIMP_NOTE,
                unavailable=unavailable,
                embodiment_id=EMBODIMENTS.get(name),
            )
        )
    return choices


def own_arms(existing: str | None) -> list[dict[str, str]]:
    """The arms of dream's IK a robot of your own can pick, each by its plain name (#OWN_ARM_LABELS).

    @param existing The station file's `custom.ik`, or None: an arm the list leaves out is listed too when the file
        names it, by dream's name.
    @return `{"name", "label"}` per arm, in #OWN_ARM_LABELS's order, the file's own last.
    @throws ConfigError dream's IK is not installed.
    """
    specs = setup_model.spec_names()
    arms = [{"name": name, "label": label} for name, label in OWN_ARM_LABELS.items() if name in specs]
    if existing in specs and existing not in OWN_ARM_LABELS:
        arms.append({"name": existing, "label": f"{existing} (your station file's)"})
    return arms


def fits(robot: Robot, device: dict[str, Any]) -> bool:
    """Whether a device can be one of the robot's arms, by its kind (#ARM_KINDS); one the operator typed always can.

    @param robot The robot.
    @param device A #robots.setup_devices.arm_adapters entry, or a typed one (#robots.setup_devices.typed_arm).
    @return True when the robot's driver opens that kind of device.
    """
    kinds = ARM_KINDS.get(robot.body)
    return kinds is None or device["kind"] in kinds or device["kind"] == setup_devices.TYPED


def so101_robot(files: list[str], embodiment: str) -> Robot:
    """The SO-101 with the operator's calibration files and the embodiment its fine-tune was trained on.

    @param files The files' paths (so101_calibration_files() offers the ones on this box); none is fine, since an arm
        that holds none of them is calibrated on the arms step (Session.start_calibration()).
    @param embodiment `<username>/<name>`, looked up on the account for the camera roles.
    @return The #Robot.
    @throws ValueError The SO-101 cannot be picked here.
    @throws ConfigError A file is not an SO-101's LeRobot calibration, or the account has no such embodiment.
    """
    found = next(robot for robot in robot_choices() if robot.body == "so101")
    if found.unavailable is not None:
        raise ValueError(f"the SO-101 cannot be picked here: {found.unavailable}")
    paths = tuple(str(pathlib.Path(path.strip()).expanduser().resolve()) for path in files if path.strip())
    for path in paths:
        feetech.read_calibration(path, so101.JOINTS)
    roles = tuple(bodies.so101_body(embodiment).wire().camera_roles)
    return dataclasses.replace(found, roles=roles, embodiment=embodiment, calibration_files=paths)


def so101_calibration_files() -> list[str]:
    """The calibration files `lerobot-calibrate` wrote for SO-101 followers on this box.

    @return Their paths, sorted; empty when there is none.
    """
    if not SO101_CALIBRATION_DIR.is_dir():
        return []
    return sorted(str(path) for path in SO101_CALIBRATION_DIR.glob("*.json"))


def so101_matches(files: list[str], adapters: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Which calibration files the SO-101s plugged in hold, each arm's servos read once from their EEPROM.

    Read-only: each serial adapter (#ARM_KINDS) is opened, its servos' calibration read and the port closed, with
    nothing written and no servo switched on (#robots.feetech.held_calibration). An adapter that does not answer as an
    SO-101 (LeRobot missing, another device, a servo silent) and a file that is not an SO-101's match nothing.

    @param files The calibration files' paths.
    @param adapters `robots.setup_devices.arm_adapters()` entries.
    @return Per file, the labels of the adapters whose servos hold it, in the adapters' order.
    """
    found: dict[str, list[str]] = {path: [] for path in files}
    parsed = {}
    for path in files:
        try:
            parsed[path] = feetech.read_calibration(path, so101.JOINTS)
        except errors.ConfigError:
            continue
    if not parsed:
        return found
    for device in adapters:
        if device["kind"] not in ARM_KINDS["so101"]:
            continue
        try:
            held = feetech.held_calibration(device["open"], so101.JOINTS)
        except Exception as exc:  # whatever is on the port, it only matches nothing
            _log.info("no SO-101 calibration read from %s: %s", device["open"], exc)
            continue
        for path, calibration in parsed.items():
            if not feetech.calibration_differences(calibration, held):
                found[path].append(device["label"])
    return found


def custom_robot(ik_arm: str, driver: str, embodiment: str) -> Robot:
    """A robot of the customer's own: their driver, dream's URDF of its arm, and their fine-tune's embodiment.

    @param ik_arm dream's name for the arm (`setup_model.spec_names()`), or an arm file's path (`dream.arm_file`).
    @param driver The driver class, `package.module:Class` or `/path/to/file.py:Class` (#robots.bodies.load_driver).
    @param embodiment `<username>/<name>`, looked up on the account for the camera roles.
    @return The #Robot.
    @throws ValueError The driver is not named `module:Class`.
    @throws ConfigError The account has no such embodiment, or it needs a `grip_scale`.
    """
    module_name, _, attribute = driver.strip().rpartition(":")
    if not module_name or not attribute.isidentifier():
        raise ValueError("name your robot's driver as package.module:Class or /path/to/file.py:Class")
    custom = config.CustomConfig(driver=driver.strip(), ik=ik_arm, embodiment=embodiment.strip())
    roles = tuple(bodies.custom_body(custom).wire().camera_roles)
    return Robot(body=config.CUSTOM_BODY, label="My own robot", ik_arm=ik_arm, roles=roles, custom=custom)


## How an arm is opened: `(robot, cfg, side) -> arm` (open_body_arm(); tests pass fakes).
Opener = Callable[[Robot, config.ArmConfig, str], followers.FollowerArm]


def open_body_arm(robot: Robot, cfg: config.ArmConfig, side: str) -> followers.FollowerArm:
    """Opens one arm with its body's driver, as `dreamcli robot run` does (body_of()).

    @param robot The robot.
    @param cfg The arm's config.
    @param side `"left"` or `"right"`, or the device's id while the side is not known yet.
    @return The arm, powered and holding.
    @throws Exception The driver's own error when the arm does not open.
    """
    return body_of(robot).open_arm(cfg, side)


def arm_config(
    robot: Robot, channel: str, settings: dict[str, Any], side: str, home: npt.NDArray[np.float64]
) -> config.ArmConfig:
    """One arm's config: the device it is on, the body's own arm keys and the home, read as the station reads them.

    @param robot The robot.
    @param channel The device to open, as its driver takes it (a CAN link, a by-id path).
    @param settings The body's other arm keys (a YAM's `gripper`, a Metal arm's gripper ends, an SO-101's
        `calibration`, ...).
    @param side `left` or `right`, for messages.
    @param home The arm's home joints, radians: the one the page set, or the URDF's rest pose until then. Only a
        move home reads it, and the wizard never makes one.
    @return The config.
    @throws ConfigError A key is unknown to the body or its value is wrong.
    """
    section = {**settings, "can_channel": channel, "home": [float(q) for q in home]}
    return config.read_arm(section, "follower_l" if side == "left" else "follower_r", robot.body)


class ArmBank:
    """The arms open while the operator moves them by hand: opened, disengaged, read on and on.

    Thread-safe: the reader thread polls while the page's requests open, close and read.

    Attributes:
    - errors: Per device, why its last read failed.
    - calibrations: Per SO-101 device, the calibration file its servos hold (the one it opened with).

    @see open_disengaged(), poll(), latest(), close()
    """

    def __init__(self, opener: Opener, clock: Callable[[], float]) -> None:
        """Binds how arms are opened; nothing opens until open_disengaged().

        @param opener How an arm is opened (#open_body_arm; tests pass fakes).
        @param clock Monotonic clock, seconds.
        """
        self._opener = opener
        self._clock = clock
        self._lock = threading.Lock()
        self._arms: dict[str, followers.FollowerArm] = {}
        self._latest: dict[str, Reading] = {}
        self.errors: dict[str, str] = {}
        self.calibrations: dict[str, str] = {}

    def open_disengaged(
        self, robot: Robot, configs: dict[str, config.ArmConfig], sides: dict[str, str] | None = None
    ) -> dict[str, str | None]:
        """Opens each device's arm, disengages it for the hand, and reads it once.

        @param robot The robot.
        @param configs Per device id, its config.
        @param sides Per device id, its side once the operator has named it; a device not in it opens under its id,
            the side a driver gets while the arms are not told apart yet.
        @return Per device id, None when it opened and answered.
        """
        found: dict[str, str | None] = {}
        for device, cfg in configs.items():
            self.close([device])
            try:
                arm = self._opener(robot, cfg, (sides or {}).get(device, device))
            except Exception as exc:
                found[device] = self.errors[device] = setup_devices.explain(exc)
                continue
            try:
                arm.disengage()
                reading = setup_checks.reading_of(arm.read(), self._clock())
            except Exception as exc:
                found[device] = self.errors[device] = setup_devices.explain(exc)
                _close_quietly(arm)
                continue
            with self._lock:
                self._arms[device] = arm
                self._latest[device] = reading
                self.errors.pop(device, None)
                if cfg.calibration is not None:
                    self.calibrations[device] = cfg.calibration
            found[device] = None
        return found

    def adopt(self, device: str, arm: followers.FollowerArm) -> None:
        """Takes an arm already let go (disengaged) to be read by hand under its device, and reads it once.

        @param device The device id.
        @param arm The arm.
        """
        reading = setup_checks.reading_of(arm.read(), self._clock())
        with self._lock:
            self._arms[device] = arm
            self._latest[device] = reading
            self.errors.pop(device, None)

    def poll(self) -> dict[str, Reading]:
        """Reads every open arm once.

        @return Per device, its new reading; a device whose read failed is left out and its error kept in #errors.
        """
        with self._lock:
            arms = dict(self._arms)
        fresh: dict[str, Reading] = {}
        for device, arm in arms.items():
            try:
                fresh[device] = setup_checks.reading_of(arm.read(), self._clock())
            except Exception as exc:
                self.errors[device] = setup_devices.explain(exc)
        with self._lock:
            self._latest.update({device: reading for device, reading in fresh.items() if device in self._arms})
        return fresh

    def latest(self) -> dict[str, Reading]:
        """The last reading of every open arm.

        @return Per device id.
        """
        with self._lock:
            return dict(self._latest)

    def devices(self) -> list[str]:
        """The devices open now.

        @return Their ids.
        """
        with self._lock:
            return list(self._arms)

    def alarms(self) -> dict[str, str]:
        """Each open arm's motor guard alarm (`FollowerArm.alarm()`: a motor too hot, a joint blocked too long).

        @return Per device with an alarm up, its words.
        """
        with self._lock:
            arms = dict(self._arms)
        found = {device: arm.alarm() for device, arm in arms.items()}
        return {device: alarm for device, alarm in found.items() if alarm is not None}

    def gripper_deg(self) -> dict[str, float | None]:
        """Each open arm's gripper motor angle as of its last read, degrees (`FollowerArm.gripper_deg()`).

        @return Per device id; None where its driver reports none.
        """
        with self._lock:
            arms = dict(self._arms)
        return {device: arm.gripper_deg() for device, arm in arms.items()}

    def close(self, devices: list[str] | None = None) -> None:
        """Closes some arms, or every one.

        @param devices The ids, or None for all.
        """
        with self._lock:
            names = list(self._arms) if devices is None else [d for d in devices if d in self._arms]
            arms = [self._arms.pop(name) for name in names]
            for name in names:
                self._latest.pop(name, None)
        for arm in arms:
            _close_quietly(arm)


def _cell(xy: npt.NDArray[np.float64]) -> tuple[int, int]:
    """The #REACH_GRID_M cell a point seen from above is in."""
    i, j = np.round(xy / REACH_GRID_M)
    return int(i), int(j)


def _metres(cells: set[tuple[int, int]]) -> list[list[float]]:
    """#REACH_GRID_M cells as `[[x, y], ...]` sorted, metres to the millimetre."""
    return [[round(i * REACH_GRID_M, 3), round(j * REACH_GRID_M, 3)] for i, j in sorted(cells)]


def _close_quietly(arm: Any) -> None:
    """Closes an arm; a failing close is logged, since the arm is being let go either way."""
    try:
        arm.close()
    except Exception:
        _log.exception("setup: closing an arm failed")


def _let_go_quietly(arm: Any) -> None:
    try:
        arm.disengage()
    except Exception:
        _log.exception("setup: letting an arm go failed; support it and switch its power off")
    _close_quietly(arm)


class Session:
    """What the page has settled so far, and the arms: one per wizard process.

    Every method is called from the HTTP handler's threads and the reader thread, under one lock.

    Attributes:
    - robot: The #Robot picked, or None.
    - model: Its arm's URDF once loaded (#robots.setup_model.ArmModel), or None.
    - model_state: `none`, `loading`, `ready` or `error` (#model_error says why).
    - devices: The arm devices the page found (`setup_devices.arm_adapters()`), by id.
    - assigned: Per side, the device id the operator said is that arm.
    - settings: Per side, the body's other arm keys.
    - home: Per side, the home joints, radians.
    - gripper: Per side, the gripper's motor angle at each end the operator moved it to by hand, `{"open", "closed"}`
      in degrees, on a body of #GRIPPER_READ_RANGE_DEG or a robot of your own (record_gripper()).
    - gripper_skipped: On a robot of your own whose driver reports no gripper motor angle, why its gripper's ends are
      not measured here (#GRIPPER_SKIPPED), from when both arms are found; else None.
    - motion: The powered checks (#robots.setup_motion.Motion), once started.
    - powered: Whether the arms are open under power (the powered checks run, or ended holding); release() lets them
      go, after the operator holds them up.
    - not_answering: Per side, why the arm stopped answering while it was powered (an unplugged cable, its power
      off); nothing powers the arms again until they are found again (open_arms()).
    - parked: Per side, what the page says (parked_words()) about an arm parked and switched off: its motor guard's
      alarm (a motor too hot), during the powered checks or while it held after them, or a joint blocked
      during the checks; nothing powers the arms again until they are found again (open_arms()).
    - off_failed: Per side, why an arm was to be parked whose switch-off failed: it may still be
      powered, so it stays among the powered arms, where Let go and the exit still reach it.
    - still_holding: Per side, what to do about an arm that drops when limp and stopped answering under power
      (still_holding_words()): it is never switched off, which would drop it, so it still holds with nothing driving
      it; the page and the exit say so.
    - reach: Per side, the reach area painted so far: the hand's x and y on each read while #painting, in cells of
      #REACH_GRID_M in that arm's base frame (reach_area()). Found arms again, or another robot, start it empty.
    - painting: Whether the page is on the Reach area step, so each read paints (paint_reach()).
    - reach_skipped: Whether the operator skipped the Reach area step: Save then writes no `reach`.
    - uncalibrated: SO-101 devices whose servos hold none of the operator's calibration files: opened with what
      they hold, to be found, and calibrated once both arms are (start_calibration()); nothing past the arms step
      runs while one is found among the arms.
    - calibration: The SO-101 arm being calibrated (#robots.so101_calibration.ArmCalibration), or None.
    - calibrated: Per side, the calibration finished this session: `{"file", "moved_deg", "notes"}`.
    """

    def __init__(
        self,
        *,
        opener: Opener,
        model_factory: Callable[[str], setup_model.ArmModel],
        old_channels: dict[str, str],
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = time.sleep,
        home_before_off: bool = False,
    ) -> None:
        """Starts with nothing picked.

        @param opener How an arm is opened (#open_body_arm; tests pass fakes).
        @param model_factory `ik arm -> ArmModel` (#robots.setup_model.ArmModel; tests may share one).
        @param old_channels Per side, the device the existing config drove that arm on, so a device found to be the
            other arm is reported as swapped.
        @param clock Monotonic clock, seconds.
        @param sleep Blocking sleep, seconds.
        @param home_before_off Switch off the motors drives the held arms home first (release()): the station
            file's `home_before_motors_off`. False lets them go where they are, as before.
        """
        self._opener = opener
        self._home_before_off = home_before_off
        self._model_factory = model_factory
        self._old_channels = old_channels
        self._clock = clock
        self._sleep = sleep
        self.lock = threading.RLock()
        self.bank = ArmBank(opener, clock)
        self.robot: Robot | None = None
        self.model: setup_model.ArmModel | None = None
        self.model_state = "none"
        self.model_error: str | None = None
        self.devices: dict[str, dict[str, Any]] = {}
        self.assigned: dict[str, str] = {}
        self.settings: dict[str, dict[str, Any]] = {"left": {}, "right": {}}
        self.home: dict[str, npt.NDArray[np.float64]] = {}
        self.gripper: dict[str, dict[str, float]] = {"left": {}, "right": {}}
        self.gripper_skipped: str | None = None
        self.motion: setup_motion.Motion | None = None
        self.crossed: list[str] = []
        self.not_answering: dict[str, str] = {}
        self.parked: dict[str, str] = {}
        self.off_failed: dict[str, str] = {}
        self.still_holding: dict[str, str] = {}
        self._baseline: dict[str, Reading] = {}
        self._peak: dict[str, float] = {}
        self._motion_arms: dict[str, Any] = {}
        self._motion_thread: threading.Thread | None = None
        self.uncalibrated: set[str] = set()
        self.calibration: so101_calibration.ArmCalibration | None = None
        self.calibrated: dict[str, dict[str, Any]] = {}
        self._calibration_error: str | None = None
        self._held_dir: str | None = None
        self.reach: dict[str, set[tuple[int, int]]] = {}
        self.painting = False
        self.reach_skipped = False
        self._hands: dict[str, list[float]] = {}
        self._bases: dict[str, list[float] | None] = {}
        self._reach_error: str | None = None
        self._forget_reach()

    def choose(self, robot: Robot) -> None:
        """Picks the robot and builds its URDF model; everything picked for another robot is dropped.

        @param robot The robot.
        @throws ConfigError Its IK arm cannot be built (the model's error, also kept in #model_error).
        """
        with self.lock:
            self._require_let_go()
            self._drop_calibration()
            self.uncalibrated, self.calibrated = set(), {}
            self.bank.close()
            self.robot = robot
            self.assigned, self.home, self.crossed, self.not_answering, self.parked = {}, {}, [], {}, {}
            self.off_failed, self.still_holding = {}, {}
            self.gripper, self.gripper_skipped = {"left": {}, "right": {}}, None
            self.motion = None
            self.model, self.model_state, self.model_error = None, "loading", None
            self._forget_reach()
        try:
            model = self._model_factory(robot.ik_arm)
        except Exception as exc:
            with self.lock:
                self.model_state, self.model_error = "error", f"{type(exc).__name__}: {exc}"
            raise
        with self.lock:
            if self.robot == robot:
                self.model, self.model_state = model, "ready"

    def open_arms(self, devices: list[dict[str, Any]], settings: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Opens every device that may be an arm, disengaged, so the operator can move one and the page see which.

        @param devices The devices to try (`setup_devices.arm_adapters()` entries).
        @param settings Per side, the body's other arm keys; every device opens with the left arm's until the sides
            are known (assign()).
        @return Per device id, None when it opened and answered.
        @throws ConfigError A setting is wrong for the body.
        """
        with self.lock:
            robot = self._require_robot()
            rest = self._require_model().rest
            self._require_let_go()
            if not devices:
                raise ValueError("no device to open")
            unfit = [device["label"] for device in devices if not fits(robot, device)]
            if unfit:
                raise ValueError(f"{', '.join(unfit)} cannot be a {robot.label}'s arm; untick it")
            self.settings = {side: dict(settings[side]) for side in ("left", "right")}
            limp = {side: {**self.settings[side], **self._gripper_read_range()} for side in ("left", "right")}
            if robot.body != "so101":
                for side in ("left", "right"):
                    arm_config(robot, devices[0]["open"], limp[side], side, rest)
            self._drop_calibration()
            self.uncalibrated, self.calibrated = set(), {}
            self.bank.close()
            # A driver file of one's own is read again, so an edit since the last open is the one that runs.
            bodies.load_driver.cache_clear()
            self.devices = {d["id"]: dict(d) for d in devices}
            self.assigned, self.home, self.crossed, self.not_answering, self.parked = {}, {}, [], {}, {}
            self.off_failed, self.still_holding = {}, {}
            self.gripper, self.gripper_skipped = {"left": {}, "right": {}}, None
            self._forget_reach()
            configs: dict[str, config.ArmConfig] = {}
            refused: dict[str, str | None] = {}
            for device in devices:
                settings_left = dict(limp["left"])
                if robot.body == "so101":
                    try:
                        settings_left["calibration"] = self._so101_file(device, robot.calibration_files)
                    except Exception as exc:
                        refused[device["id"]] = setup_devices.explain(exc)
                        continue
                configs[device["id"]] = arm_config(robot, device["open"], settings_left, "left", rest)
            opened = self.bank.open_disengaged(robot, configs)
            if robot.note == LIMP_NOTE and None in opened.values():
                self._sleep(SETTLE_S)
                self.bank.poll()
            self._rebase()
            return {**refused, **opened}

    def release_arms(self) -> None:
        """Lets go of every arm found so far and forgets which is which, to look for the adapters again."""
        with self.lock:
            self._require_still()
            self._drop_calibration()
            self.uncalibrated, self.calibrated = set(), {}
            self.bank.close()
            self.devices, self.assigned, self.home, self.crossed = {}, {}, {}, []
            self.not_answering, self.parked, self.off_failed, self.still_holding = {}, {}, {}, {}
            self.gripper, self.gripper_skipped = {"left": {}, "right": {}}, None
            self._forget_reach()
            self._rebase()

    def tick(self) -> None:
        """One read of every disengaged arm and the alarms of those that hold: the reader thread's work."""
        with self.lock:
            if self._motion_thread is not None and self._motion_thread.is_alive():
                return
            self._park_alarmed()
            if self.calibration is not None:
                self.calibration.poll()
            fresh = self.bank.poll()
            for device, score in setup_checks.wiggle_scores(self._baseline, fresh).items():
                self._peak[device] = max(self._peak.get(device, 0.0), score)
            if self.painting:
                self._paint()

    def wiggle(self) -> dict[str, Any]:
        """Which device moved since the baseline, and every device's numbers.

        A device's score is the most any of its joints moved from the baseline on any read since, so a quick wiggle
        that is back in place before the page asks still counts.

        @return `{"devices": {id: {"joints_deg", "gripper", "moved", "score_deg", "side", "error", "calibration",
            "uncalibrated"}}, "moved", "several"}`; `calibration` is the file name an SO-101's motors agree with, else
            None; `uncalibrated`: an SO-101 arm whose motors agree with none, still to calibrate.
        """
        with self.lock:
            latest = self.bank.latest()
            scores = dict(self._peak)
            moved, several = setup_checks.moved_device(
                {d: s for d, s in scores.items() if d not in self.assigned.values()}
            )
            sides = {device: side for side, device in self.assigned.items()}
            paths = self.bank.calibrations
            rows = {
                device: {
                    "joints_deg": np.degrees(reading.joints).round(1).tolist(),
                    # NaN from a driver that maps its gripper with ends not measured yet, which JSON cannot carry.
                    "gripper": round(reading.gripper, 3) if math.isfinite(reading.gripper) else None,
                    "score_deg": round(math.degrees(scores.get(device, 0.0)), 1),
                    "moved": device == moved,
                    "side": sides.get(device),
                    "error": self.bank.errors.get(device),
                    "calibration": pathlib.Path(paths[device]).name
                    if device in paths and device not in self.uncalibrated
                    else None,
                    "uncalibrated": device in self.uncalibrated,
                }
                for device, reading in latest.items()
            }
            return {"devices": rows, "moved": moved, "several": several}

    def rebase_wiggle(self) -> None:
        """Takes every device's reading now as the baseline the next move is measured from."""
        with self.lock:
            self._rebase()

    def _rebase(self) -> None:
        """Every open device's reading now is the wiggle's baseline, and no device has moved since."""
        self._baseline = self.bank.latest()
        self._peak = {}

    def assign(self, device: str, side: str) -> dict[str, Any]:
        """The operator says which arm the device that moved is: `left`, `right`, or `none` (not this robot).

        When the existing config drove the other arm on this device, the reply says so: its left and right were
        swapped. Once both sides are known every other device is closed, and the two arms are opened again with
        their own side's settings.

        @param device The device id.
        @param side `left`, `right` or `none`.
        @return `{"assigned", "crossed", "done"}`: the sides so far, a message when the old config had this device as
            the other arm (else None), and whether both arms are known.
        @throws ValueError The device is not open or the side is unknown.
        """
        with self.lock:
            robot = self._require_robot()
            if device not in self.bank.devices() and device not in self.assigned.values():
                raise ValueError(f"device {device!r} is not open")
            if side not in ("left", "right", "none"):
                raise ValueError(f"'side' must be left, right or none, not {side!r}")
            self.assigned = {s: d for s, d in self.assigned.items() if d != device}
            self.reach = {"left": set(), "right": set()}
            crossed = None
            if side == "none":
                self.bank.close([device])
            else:
                self.assigned[side] = device
                old = self._old_side(device)
                if old is not None and old != side:
                    crossed = (
                        f"Your old config drove the {old} arm on this adapter ({self._old_channels[old]}), but you "
                        f"moved the {side} arm: left and right were swapped. Saving fixes it."
                    )
                    self.crossed.append(crossed)
            self._rebase()
            done = set(self.assigned) == {"left", "right"}
            if done:
                self._reopen_both(robot)
            return {"assigned": dict(self.assigned), "crossed": crossed, "done": done}

    def swap_sides(self) -> dict[str, Any]:
        """Both arms named as the other side in one step, each opened again with its new side's settings.

        One call rather than two assign() calls, so a failure leaves no arm named twice.

        @return assign()'s reply: the sides now, the old config's swap note if one side now matches it, done.
        @throws ValueError Both arms are not found yet, or an arm did not open again.
        """
        with self.lock:
            robot = self._require_robot()
            if set(self.assigned) != {"left", "right"}:
                raise ValueError("find both arms before swapping them")
            self.assigned = {"left": self.assigned["right"], "right": self.assigned["left"]}
            self.reach = {"left": set(), "right": set()}
            self.crossed = []
            for side, device in self.assigned.items():
                old = self._old_side(device)
                if old is not None and old != side:
                    self.crossed.append(
                        f"Your old config drove the {old} arm on this adapter ({self._old_channels[old]}), but you "
                        f"moved the {side} arm: left and right were swapped. Saving fixes it."
                    )
            self._rebase()
            self._reopen_both(robot)
            return {
                "assigned": dict(self.assigned),
                "crossed": self.crossed[0] if self.crossed else None,
                "done": True,
            }

    def _reopen_both(self, robot: Robot) -> None:
        """Once both arms are named: every other device closed, and each arm opened again as its side, with its side's
        settings, so a driver that keys anything by side (a gripper's calibration) has it from here on.

        @throws ValueError An arm did not open again.
        """
        self.bank.close([d for d in self.bank.devices() if d not in self.assigned.values()])
        rest = self._require_model().rest
        configs = {
            self.assigned[s]: arm_config(
                robot, self.devices[self.assigned[s]]["open"], self.arm_settings(s, limp=True), s, rest
            )
            for s in ("left", "right")
        }
        sides = {device: s for s, device in self.assigned.items()}
        failed = [
            f"the {sides[device]} arm did not open again with its own settings: {why}"
            for device, why in self.bank.open_disengaged(robot, configs, sides).items()
            if why is not None
        ]
        if failed:
            raise ValueError(" ".join(failed))
        if robot.body == config.CUSTOM_BODY:
            reported = self.bank.gripper_deg()
            silent = any(reported[self.assigned[s]] is None for s in ("left", "right"))
            self.gripper_skipped = GRIPPER_SKIPPED if silent else None

    def set_home(self) -> dict[str, Any]:
        """Takes both arms' joints now as their home pose: where the operator put them, every run's start.

        @return `{"home": {side: degrees}, "out_of_range": [#_out_of_range() of each joint past the URDF's range]}`,
            left arm first, each arm's joints in order.
        """
        with self.lock:
            model = self._require_model()
            self._require_let_go()
            self._require_answering()
            self._require_calibrated()
            joints = self.joints_now()
            self.home = {side: q.copy() for side, q in joints.items()}
            self.motion = None
            outside = {
                side: sorted({i + 1 for i in model.out_of_range(q)} | set(self._past_reference(side, q)))
                for side, q in self.home.items()
            }
            return {
                "home": {side: np.degrees(q).round(1).tolist() for side, q in self.home.items()},
                "out_of_range": [
                    self._out_of_range(model, side, j, q) for side, q in self.home.items() for j in outside[side]
                ],
            }

    def _out_of_range(
        self, model: setup_model.ArmModel, side: str, joint: int, q: npt.NDArray[np.float64]
    ) -> dict[str, Any]:
        """One home joint past its range, as the page names it: the reading and the range, degrees.

        The range is the URDF's, narrowed on an SO-101 to the one dream's model was recorded on (#_past_reference()).

        @param model The arm's model.
        @param side `left` or `right`.
        @param joint The joint, 1-based.
        @param q The home, radians.
        @return `{"side", "joint" (1-based), "deg" (None for a reading that is not a number), "lower_deg",
            "upper_deg"}`.
        """
        i = joint - 1
        lower, upper = float(model.lower[i]), float(model.upper[i])
        if self.robot is not None and self.robot.body == "so101":
            low, high = so101_calibration.reference_ends(side).get(i, (lower, upper))
            lower, upper = max(lower, low), min(upper, high)
        reading = float(q[i])
        return {
            "side": side,
            "joint": joint,
            "deg": round(math.degrees(reading), 1) if math.isfinite(reading) else None,
            "lower_deg": round(math.degrees(lower), 1),
            "upper_deg": round(math.degrees(upper), 1),
        }

    def paint_reach(self, on: bool) -> None:
        """Starts or stops painting the reach area: the page is on the Reach area step, or left it.

        Starting un-skips the step. Nothing is required: the step never blocks the wizard.

        @param on True on the step.
        """
        with self.lock:
            self.painting = on
            if on:
                self.reach_skipped = False

    def clear_reach(self, side: str) -> None:
        """Forgets one arm's painted area, to paint it again.

        @param side `left` or `right`.
        @throws ValueError Another side.
        """
        if side not in ("left", "right"):
            raise ValueError(f"'side' must be left or right, not {side!r}")
        with self.lock:
            self.reach[side] = set()

    def skip_reach(self) -> None:
        """The operator skipped the Reach area step, from any state: Save writes no `reach`."""
        with self.lock:
            self.painting, self.reach_skipped = False, True

    def reach_area(self, side: str) -> list[list[float]] | None:
        """One arm's `reach` as Save writes it: its painted points and its home hand, in metres to the millimetre.

        Never raises, so Save never fails because of this step: a home whose hand cannot be computed is left out.

        @param side `left` or `right`.
        @return `[[x, y], ...]` sorted, or None when the step was skipped, the robot has none, or nothing was painted.
        """
        with self.lock:
            if self.robot is None or not reaches(self.robot) or self.reach_skipped or not self.reach[side]:
                return None
            cells = set(self.reach[side])
            hand = None if side not in self.home else self._hand(self.home[side], side)
            if hand is not None:
                cells.add(_cell(hand))
            return _metres(cells)

    def _forget_reach(self) -> None:
        self.reach = {"left": set(), "right": set()}
        self.painting = False
        self.reach_skipped = False
        self._hands = {}
        self._bases = {}
        self._reach_error = None

    def _paint(self) -> None:
        """Adds each found arm's hand now to its reach area, and keeps where the hands are for the page."""
        latest = self.bank.latest()
        hands: dict[str, list[float]] = {}
        for side, device in self.assigned.items():
            hand = None if device not in latest else self._hand(latest[device].joints, side)
            if hand is not None:
                self.reach[side].add(_cell(hand))
                hands[side] = [round(float(v), 4) for v in hand]
        self._hands = hands

    def _hand(self, joints: npt.ArrayLike, side: str) -> npt.NDArray[np.float64] | None:
        """The arm's tool point seen from above, `[2]` metres in its base frame, as dream's IK computes it with the
        held joints at zero; None for a reading that is not a number.

        A failure is kept for the page (#_reach_error) and gives None: the stage never stalls the wizard.
        """
        model = self.model
        if model is None:
            return None
        try:
            q = np.array(joints, dtype=np.float64)
            if not np.all(np.isfinite(q)):
                return None
            q[list(model.held)] = 0.0
            xy = model.tool(q, side)[:2, 3]
        except Exception as exc:
            if self._reach_error is None:
                _log.exception("setup: the %s hand's position failed; the reach area skips this reading", side)
            self._reach_error = f"the {side} hand's position failed: {type(exc).__name__}: {exc}"
            return None
        return xy

    def record_gripper(self, end: str) -> dict[str, Any]:
        """Takes both grippers' motor angle now as their `open` or `closed` end, set fully by hand.

        @param end `open` or `closed`.
        @return status()'s `gripper`.
        """
        with self.lock:
            self._require_measures_gripper()
            if end not in ("open", "closed"):
                raise ValueError(f"'end' must be open or closed, not {end!r}")
            self._require_let_go()
            self._require_answering()
            latest = self.bank.latest()
            self._require_open(latest)
            for side, deg in self._gripper_angles(latest).items():
                self.gripper[side][end] = round(deg, 1)
            self.motion = None
            status = self._gripper_status()
            assert status is not None, "_require_measures_gripper() refuses a body whose ends are not measured"
            return status

    def _gripper_angles(self, latest: dict[str, Reading]) -> dict[str, float]:
        """Both grippers' motor angle now: back from the read range's grip, or a robot of your own's driver's.

        @param latest Per device, its last reading; both arms' are in it.
        @return Per side, degrees.
        @throws ValueError A custom driver reports no angle now.
        """
        body = self._require_robot().body
        if body in GRIPPER_READ_RANGE_DEG:
            closed_deg, open_deg = GRIPPER_READ_RANGE_DEG[body]
            grips = {side: latest[self.assigned[side]].gripper for side in ("left", "right")}
            return {side: closed_deg + grip * (open_deg - closed_deg) for side, grip in grips.items()}
        reported = self.bank.gripper_deg()
        angles: dict[str, float] = {}
        for side in ("left", "right"):
            deg = reported[self.assigned[side]]
            if deg is None:
                raise ValueError(f"the {side} arm's driver reports no gripper motor angle now: gripper_deg() was None")
            angles[side] = deg
        return angles

    def gripper_ends(self, side: str) -> dict[str, float]:
        """One arm's gripper ends as the station keeps them.

        @param side `left` or `right`.
        @return `{"gripper_closed_deg", "gripper_open_deg"}`.
        """
        with self.lock:
            self._require_measures_gripper()
            recorded = self.gripper[side]
            if "open" not in recorded or "closed" not in recorded:
                raise ValueError(
                    f"press Record open and Record closed for the {side} gripper first: open it fully by hand and "
                    "press Record open, then close it fully and press Record closed"
                )
            span = recorded["open"] - recorded["closed"]
            body = self._require_robot().body
            if body in GRIPPER_READ_RANGE_DEG:
                closed_deg, open_deg = GRIPPER_READ_RANGE_DEG[body]
                span *= math.copysign(1.0, open_deg - closed_deg)
            else:
                span = abs(span)  # a robot of your own's gripper opens whichever way its motor turns
            if span < GRIPPER_MIN_SPAN_DEG:
                raise ValueError(
                    f"the {side} gripper turned only {max(0, round(span))} degrees from Record closed to Record open; "
                    "open it fully by hand and Record open, then close it fully and Record closed"
                )
            return {
                "gripper_closed_deg": recorded["closed"],
                "gripper_open_deg": recorded["open"],
            }

    def arm_settings(self, side: str, limp: bool = False) -> dict[str, Any]:
        """One arm's settings: the page's, an SO-101's calibration file, and the gripper's ends.

        @param side `left` or `right`.
        @param limp For the limp arms read by hand, before the ends are measured.
        @return The arm section's keys other than the device and the home.
        """
        with self.lock:
            settings = dict(self.settings[side])
            body = self._require_robot().body
            if body == "so101":
                device = self.assigned.get(side)
                if device is None or device not in self.bank.calibrations:
                    self._require_open(self.bank.latest())
                    raise ValueError(f"the {side} arm is not an open SO-101; find both arms first")
                if not limp:
                    self._require_calibrated()
                settings["calibration"] = self.bank.calibrations[device]
            if self._measures_gripper():
                settings.update(self._gripper_read_range() if limp else self.gripper_ends(side))
            return settings

    def _gripper_read_range(self) -> dict[str, float]:
        found = GRIPPER_READ_RANGE_DEG.get(self._require_robot().body)
        if found is None:
            return {}
        return {"gripper_closed_deg": found[0], "gripper_open_deg": found[1]}

    def _measures_gripper(self) -> bool:
        body = self._require_robot().body
        return body in GRIPPER_READ_RANGE_DEG or (body == config.CUSTOM_BODY and self.gripper_skipped is None)

    def _require_measures_gripper(self) -> None:
        if self.gripper_skipped is not None:
            raise ValueError(self.gripper_skipped)
        if not self._measures_gripper():
            raise ValueError(f"a {self._require_robot().body} gripper's ends are not measured here")

    def _gripper_status(self) -> dict[str, Any] | None:
        if self.robot is None:
            return None
        if self.gripper_skipped is not None:
            return {"skipped": self.gripper_skipped}
        if not self._measures_gripper():
            return None
        ends: dict[str, Any] = {}
        for side in ("left", "right"):
            try:
                ends[side] = self.gripper_ends(side)
            except ValueError as exc:
                ends[side] = {"error": str(exc)}
        return {"recorded": {side: dict(self.gripper[side]) for side in ("left", "right")}, "ends": ends}

    def _require_open(self, latest: dict[str, Reading]) -> None:
        """A ValueError while an arm is not open, with why it last failed to open or read (the bank's #errors), which
        otherwise only reaches the terminal.

        @param latest The bank's last readings.
        """
        for side in ("left", "right"):
            device = self.assigned.get(side)
            if device not in latest:
                why = self.bank.errors.get(device) if device is not None else None
                if why:
                    # explain()'s sentences end in a full stop of their own.
                    raise ValueError(
                        f"the {side} arm is not open: {why.rstrip('.')}. Fix that, then find both arms again on step 1"
                    )
                raise ValueError(f"the {side} arm is not open; find both arms first")

    def joints_now(self) -> dict[str, npt.NDArray[np.float64]]:
        """Both arms' joints as last read.

        @return Per side, `[width]` as read.
        @throws ValueError The arms are not both known, or one has no reading.
        """
        with self.lock:
            latest = self.bank.latest()
            self._require_open(latest)
            return {side: latest[self.assigned[side]].joints.copy() for side in ("left", "right")}

    @property
    def powered(self) -> bool:
        """Whether the arms are open under power: the powered checks run, or ended with the arms holding."""
        return bool(self._motion_arms)

    def start_motion(self) -> None:
        """After the safety popup, opens both arms again powered and runs the powered checks on a thread."""
        with self.lock:
            robot = self._require_robot()
            model = self._require_model()
            self._require_still()
            self._require_answering()
            if not self.home:
                raise ValueError("set the home pose before anything moves")
            alarms = [*self.bank.alarms().values(), *(arm.alarm() for arm in self._motion_arms.values())]
            up = [alarm for alarm in alarms if alarm is not None]
            if up:
                raise ValueError(f"{up[0]}; nothing moves under power until it is cool and free again")
            configs = {
                side: arm_config(
                    robot, self.devices[self.assigned[side]]["open"], self.arm_settings(side), side, self.home[side]
                )
                for side in ("left", "right")
            }
            arms: dict[str, Any] = dict(self._motion_arms)
            dead = sorted(side for side, arm in arms.items() if not arm.is_alive())
            if dead:
                raise ValueError(
                    f"the {' and '.join(dead)} arm stopped answering while it held: hold both arms up with your hands "
                    "and click Turn off motors first"
                )
            if not arms:
                self.bank.close()
                try:
                    for side, cfg in configs.items():
                        arms[side] = self._opener(robot, cfg, side)
                except Exception as exc:
                    # The one that opened goes limp again: closed, it would hold under power with nothing driving it.
                    for arm in arms.values():
                        _let_go_quietly(arm)
                    if isinstance(exc, errors.RobotsError):
                        raise ValueError(setup_devices.explain(exc)) from exc
                    raise
            self._motion_arms = arms
            self.motion = setup_motion.Motion(
                robot.body, model, arms, dict(self.home), clock=self._clock, sleep=self._sleep
            )
            self._motion_thread = threading.Thread(target=self._run_motion, name="setup-motion", daemon=True)
            self._motion_thread.start()

    def _run_motion(self) -> None:
        motion = self.motion
        assert motion is not None, "start_motion() sets it"
        motion.run()
        with self.lock:
            self._park_alarmed()
            if motion.let_go:
                self._adopt_motion_arms(disengage=False)

    def _park_alarmed(self) -> None:
        motion = self.motion
        for side, arm in list(self._motion_arms.items()):
            assert motion is not None, "start_motion() sets it with the arms"
            if side in self.off_failed:
                continue
            if side not in motion.parked and side not in motion.off_failed:
                alarm = arm.alarm()
                if alarm is None:
                    continue
                motion.park(side, alarm)
            if side in motion.parked:
                self.parked[side] = parked_words(side, motion.parked[side], side in motion.limp_off)
                del self._motion_arms[side]
            else:
                self.off_failed[side] = motion.off_failed[side]

    def release(self) -> dict[str, Any]:
        """Lets go of the arms after the powered checks, once the operator holds them up.

        With `home_before_off` (the station file's `home_before_motors_off`) the held arms first go home on a
        thread, as the checks' own move home does (`setup_motion.Motion.home_before_off()`), and are let go when it
        ends, home or not: a failed or cut-short move never leaves the motors on. That is skipped, and the arms let go
        where they are as before, when an arm was parked, its switch-off failed, or it stopped answering, since the
        move would drive what cannot be driven.

        @return The session's status(); while the arms go home, its `motion` is `running`.
        """
        with self.lock:
            self._require_still()
            if not self._motion_arms:
                raise ValueError("the arms are not powered")
            motion = self.motion
            drivable = (
                motion is not None
                and not self.parked
                and not self.off_failed
                and not motion.parked
                and not motion.off_failed
                and all(arm.is_alive() for arm in self._motion_arms.values())
            )
            if not (self._home_before_off and drivable):
                self._adopt_motion_arms(disengage=True)
                return self.status()
            assert motion is not None, "drivable says so"
            motion.begin_home_off()
            sides = sorted(self._motion_arms)
            self._motion_thread = threading.Thread(
                target=self._home_then_release, args=(motion, sides), name="setup-home-off", daemon=True
            )
            self._motion_thread.start()
            return self.status()

    def _home_then_release(self, motion: setup_motion.Motion, sides: list[str]) -> None:
        motion.home_before_off(sides)
        with self.lock:
            self._adopt_motion_arms(disengage=True)

    def _adopt_motion_arms(self, disengage: bool) -> None:
        for side, arm in list(self._motion_arms.items()):
            if side in self.off_failed:
                if control.run_teardown(f"switch the {side} arm off again", arm.torque_off):
                    self.parked[side] = parked_words(side, self.off_failed.pop(side))
                    del self._motion_arms[side]
                continue
            del self._motion_arms[side]
            try:
                if disengage:
                    arm.disengage()
                self.bank.adopt(self.assigned[side], arm)
            except Exception as exc:
                _log.exception("setup: the %s arm stopped answering", side)
                self._lost(side, arm, str(exc))

    def _lost(self, side: str, arm: Any, why: str) -> None:
        _close_quietly(arm)
        if arm.drops_when_limp:
            self.still_holding[side] = still_holding_words(side)
            _log.error("setup: %s", self.still_holding[side])
        else:
            self.not_answering[side] = why

    def confirm_eyes(self, same: bool) -> dict[str, Any]:
        """Takes the operator's answer, after the powered checks, to whether the real arms moved like the drawing.

        @param same True when they did; False fails the powered checks (#robots.setup_motion.Motion.eyes()).
        @return The powered checks' status (#robots.setup_motion.Motion.status()).
        """
        with self.lock:
            if self.motion is not None and any(row["group"] == "eyes" for row in self.motion.results):
                raise ValueError("this run already has your answer; Run again to answer again")
            if self.motion is None or self.motion.state != "done":
                raise ValueError("run the powered checks to their end first")
            self.motion.eyes(same)
            return self.motion.status()

    def stop_motion(self) -> None:
        """The powered checks end and both arms hold where they are; a park under way ends at rest (the exit)."""
        motion = self.motion
        if motion is not None:
            motion.stop()

    def press_stop(self) -> None:
        """The page's STOP: as stop_motion(), but during a park every arm goes limp at once (browser B29)."""
        motion = self.motion
        if motion is not None:
            motion.press_stop()

    def wait_motion(self, timeout_s: float | None) -> bool:
        """Waits for the powered checks to end.

        @param timeout_s Most seconds to wait.
        @return True when they ended.
        """
        thread = self._motion_thread
        if thread is not None:
            thread.join(timeout_s)
            return not thread.is_alive()
        return True

    def close(self) -> None:
        """Stops the powered checks and lets every arm go, so no arm is left holding once the wizard exits."""
        self.stop_motion()
        self.wait_motion(None)
        with self.lock:
            self._drop_calibration()
        self.bank.close()
        with self.lock:
            for side, arm in list(self._motion_arms.items()):
                if side not in self.off_failed and arm.drops_when_limp and not arm.is_alive():
                    self._lost(side, arm, "")
                    continue
                try:
                    arm.torque_off()
                except Exception:
                    _log.exception("setup: an arm did not switch off; support it and switch its power off")
            self._motion_arms = {}

    def status(self) -> dict[str, Any]:
        """Everything the page polls.

        @return `{"robot", "model", "model_error", "powered", "holding", "not_answering", "still_holding", "parked",
            "off_failed", "assigned", "crossed", "home", "gripper", "motion", "checks_on", "reach", "home_before_off",
            "calibration", "home_target"}` (`home_target`: the pose an SO-101's Home pose step draws, else None;
            `home_before_off`: Switch off the motors takes the arms home first; `calibration`: an SO-101's,
            _calibration_status(), else None); `reach` is None
            on a robot without the Reach area step, else `{"painting", "skipped", "radius_m", "grid_m", "min_area_m2",
            "points": {side: [[x, y], ...]}, "hands": {side: [x, y]} while painting, "bases": {side: [x, y] or None},
            "error"}`, metres.
        """
        with self.lock:
            robot = None if self.robot is None else dataclasses.asdict(self.robot)
            return {
                "robot": robot,
                "model": self.model_state,
                "model_error": self.model_error,
                "powered": self.powered or bool(self.still_holding),
                "not_answering": dict(self.not_answering),
                "still_holding": dict(self.still_holding),
                "holding": sorted(self._motion_arms),
                "parked": dict(self.parked),
                "off_failed": {side: off_failed_words(side, why) for side, why in self.off_failed.items()},
                "assigned": dict(self.assigned),
                "crossed": list(self.crossed),
                "home": {side: np.degrees(q).round(1).tolist() for side, q in self.home.items()},
                "gripper": self._gripper_status(),
                "motion": None if self.motion is None else self.motion.status(),
                "checks_on": self._motion_thread is not None and self._motion_thread.is_alive(),
                "reach": self._reach_status(),
                "home_before_off": self._home_before_off,
                "calibration": self._calibration_status(),
                "home_target": self._home_target(),
            }

    def _base(self, side: str) -> list[float] | None:
        """Where the arm's first joint turns, seen from above, in its base frame: the centre of the circle the hand
        draws when only that joint turns (an SO-101's base frame is not at its shoulder). None when it cannot be told,
        or when the hand does not turn about that centre by about as much as the joint turned: a first joint that
        lifts the arm (a humanoid's shoulder) puts the centre of three nearly collinear hands kilometres away.
        """
        model = self.model
        if model is None:
            return None
        rest = model.rests[side].copy()
        turned = []
        for turn in (-0.6, 0.0, 0.6):
            rest[0] = model.rests[side][0] + turn
            hand = self._hand(rest, side)
            if hand is None:
                return None
            turned.append(hand)
        (ax, ay), (bx, by), (cx, cy) = turned
        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-9:
            return None
        a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
        x = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
        y = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        (ux, uy), (vx, vy) = (ax - x, ay - y), (cx - x, cy - y)
        seen = math.atan2(abs(ux * vy - uy * vx), ux * vx + uy * vy)
        if not abs(seen - 1.2) < 0.6:  # the joint turned 1.2 rad
            return None
        return [round(float(x), 4), round(float(y), 4)]

    def _reach_status(self) -> dict[str, Any] | None:
        if self.robot is None or not reaches(self.robot):
            return None
        if self.painting and not self._bases:
            self._bases = {side: self._base(side) for side in ("left", "right")}
        return {
            "painting": self.painting,
            "skipped": self.reach_skipped,
            "radius_m": client_ik.REACH_RADIUS_M,
            "grid_m": REACH_GRID_M,
            "min_area_m2": REACH_MIN_AREA_M2,
            "points": {side: _metres(self.reach[side]) for side in ("left", "right")},
            "hands": dict(self._hands) if self.painting else {},
            "bases": dict(self._bases),
            "error": self._reach_error,
        }

    def _old_side(self, device: str) -> str | None:
        """The side the existing config drove on this device, or None."""
        opened = self.devices[device]["open"]
        for side, channel in self._old_channels.items():
            if channel == opened:
                return side
            if channel.startswith("/dev/") and os.path.realpath(channel) == os.path.realpath(opened):
                return side
        return None

    def _so101_file(self, device: dict[str, Any], files: tuple[str, ...]) -> str:
        """The calibration file an SO-101 arm opens with on the arms step: the operator's file its servos hold, else
        the calibration they hold written to a file of the session's own, for an arm still to calibrate.

        An arm that holds none of the operator's files is still opened, so it can be moved to say which arm it is; it
        is in #uncalibrated until it is calibrated (start_calibration()), and nothing past the arms step takes it.

        @param device The adapter.
        @param files The operator's calibration files.
        @return The file's path.
        @throws ArmError The servos do not answer.
        """
        held = feetech.held_calibration(device["open"], so101.JOINTS)
        for path in files:
            if not feetech.calibration_differences(feetech.read_calibration(path, so101.JOINTS), held):
                self.uncalibrated.discard(device["id"])
                return path
        if self._held_dir is None:
            self._held_dir = tempfile.mkdtemp(prefix="dream-setup-held-")
        name = "".join(c if c.isalnum() else "-" for c in device["id"])
        written, _ = so101_calibration.write_file(pathlib.Path(self._held_dir), name, held, datetime.datetime.now)
        self.uncalibrated.add(device["id"])
        return str(written)

    def start_calibration(self, side: str) -> dict[str, Any] | None:
        """Starts calibrating one SO-101 arm: the arms step lets go of it and its servos are switched off.

        @param side `left` or `right`, an arm found on the arms step.
        @return _calibration_status().
        @throws ValueError The robot is not an SO-101, the arm is not found, or another arm is being calibrated.
        @throws ArmError The arm does not answer.
        @throws ConfigError dream's IK has no reference for the arm.
        """
        with self.lock:
            robot = self._require_robot()
            if robot.body != "so101":
                raise ValueError("only an SO-101 is calibrated here")
            self._require_let_go()
            device = self.assigned.get(side)
            if device is None or device not in self.devices:
                raise ValueError(f"find the {side} arm first: move it and press {side.capitalize()} arm")
            if self.calibration is not None:
                raise ValueError(f"the {self.calibration.side} arm is being calibrated; finish or cancel it first")
            ranges = so101_calibration.reference(side)
            port = self.devices[device]["open"]
            self.bank.close([device])
            bus = feetech.open_bus(port, so101.JOINTS, None)
            try:
                self.calibration = so101_calibration.ArmCalibration(side, port, bus, ranges)
            except Exception:
                bus.disconnect(disable_torque=False)
                self._reopen(side)
                raise
            self.calibrated.pop(side, None)
            self._calibration_error = None
            return self._calibration_status()

    def set_middle(self) -> dict[str, Any] | None:
        """The arm being calibrated is in its middle pose: its zero is taken (so101_calibration.ArmCalibration).

        @return _calibration_status().
        """
        with self.lock:
            self._require_calibration().set_middle()
            return self._calibration_status()

    def finish_calibration(self) -> dict[str, Any] | None:
        """Writes the arm's calibration to its servos and its file, and opens the arm again with it, limp.

        @return _calibration_status(), with the arm's result under `done`.
        @throws ValueError A joint has not gone end to end, or the middle pose was too far off.
        """
        with self.lock:
            calibration = self._require_calibration()
            device = self.assigned[calibration.side]
            name = so101_calibration.file_name(calibration.side, self.devices[device])
            result = calibration.finish(SO101_CALIBRATION_DIR, name, datetime.datetime.now)
            calibration.close()
            self.calibration = None
            robot = self._require_robot()
            if result.path not in robot.calibration_files:
                self.robot = dataclasses.replace(robot, calibration_files=(*robot.calibration_files, result.path))
            self.bank.calibrations[device] = result.path
            self.uncalibrated.discard(device)
            self.calibrated[calibration.side] = {
                "file": result.path,
                "moved_deg": result.moved_deg,
                "notes": list(result.notes),
            }
            self._reopen(calibration.side)
            return self._calibration_status()

    def cancel_calibration(self) -> dict[str, Any] | None:
        """Stops calibrating: the arm is opened again, limp, with the calibration it had.

        @return _calibration_status().
        """
        with self.lock:
            calibration = self.calibration
            self._drop_calibration()
            if calibration is not None:
                self._reopen(calibration.side)
            return self._calibration_status()

    def _reopen(self, side: str) -> None:
        """Opens one found arm again, limp, with its side's settings; a failure is kept for the page."""
        robot = self._require_robot()
        device = self.assigned[side]
        cfg = arm_config(
            robot, self.devices[device]["open"], self.arm_settings(side, limp=True), side, self._require_model().rest
        )
        why = self.bank.open_disengaged(robot, {device: cfg})[device]
        self._calibration_error = None if why is None else f"the {side} arm did not open again: {why}"

    def _drop_calibration(self) -> None:
        """Closes the arm being calibrated, if any, its servos left off."""
        if self.calibration is not None:
            self.calibration.close()
            self.calibration = None

    def _require_calibration(self) -> so101_calibration.ArmCalibration:
        if self.calibration is None:
            raise ValueError("no arm is being calibrated: press Calibrate on the arm first")
        return self.calibration

    def _require_calibrated(self) -> None:
        """A ValueError while an SO-101 arm found on the arms step is not calibrated, or one is being calibrated."""
        if self.calibration is not None:
            raise ValueError(f"finish calibrating the {self.calibration.side} arm on step 1 first")
        sides = [side for side, device in self.assigned.items() if device in self.uncalibrated]
        if sides:
            raise ValueError(f"calibrate the {' and '.join(sorted(sides))} arm on step 1 first")

    def _past_reference(self, side: str, q: npt.NDArray[np.float64]) -> list[int]:
        """An SO-101's home joints, 1-based, past the range dream's model was recorded on; none on another body."""
        on_so101 = self.robot is not None and self.robot.body == "so101"
        return [i + 1 for i in so101_calibration.past_reference(side, q.tolist())] if on_so101 else []

    def _home_target(self) -> list[list[float]] | None:
        """The pose the page draws on the Home pose step for the operator to put the arms in (#SO101_HOME_DEG), on
        dream's model: an SO-101's; None on another robot or before the model is ready."""
        if self.robot is None or self.robot.body != "so101" or self.model is None:
            return None
        return self.model.frame(np.radians(SO101_HOME_DEG[: len(self.model.rest)]))

    def _calibration_status(self) -> dict[str, Any] | None:
        """An SO-101's calibration for the page: `{"needed": {side: bool}, "active": {"side", "stage", "joints",
        "middle_frame"} or None, "done": {side: {"file", "moved_deg", "notes"}}, "error"}`; None on another robot."""
        if self.robot is None or self.robot.body != "so101":
            return None
        calibration = self.calibration
        model = self.model
        return {
            "needed": {side: device in self.uncalibrated for side, device in self.assigned.items()},
            "active": None
            if calibration is None
            else {
                "side": calibration.side,
                "stage": calibration.stage,
                "joints": [dataclasses.asdict(p) for p in calibration.progress()],
                # The middle pose is the URDF's zero: the page draws the arm there to show the pose to put it in.
                "middle_frame": None if model is None else model.frame(np.zeros_like(model.rest)),
            },
            "done": dict(self.calibrated),
            "error": self._calibration_error,
        }

    def _require_robot(self) -> Robot:
        """The picked robot, or a ValueError."""
        if self.robot is None:
            raise ValueError("pick your robot first")
        return self.robot

    def _require_model(self) -> setup_model.ArmModel:
        """The robot's model, or a ValueError saying why it is not there."""
        if self.model is None:
            raise ValueError(f"the arm's model is not ready ({self.model_error or self.model_state})")
        return self.model

    def _require_still(self) -> None:
        """A ValueError until the powered checks' thread ends."""
        motion, thread = self.motion, self._motion_thread
        if motion is None or thread is None or not thread.is_alive():
            return
        if motion.state == "running":
            raise ValueError("the arms are moving; press STOP at the bottom of the page first")
        parking = motion.parking
        if not parking:
            raise ValueError("the powered checks are ending; try again in a moment")
        arms = (
            f"the {parking[0]} arm reaches rest and switches"
            if len(parking) == 1
            else "both arms reach rest and switch"
        )
        raise ValueError(f"wait until {arms} off; keep clear while it moves")

    def _require_answering(self) -> None:
        if self.still_holding:
            side = sorted(self.still_holding)[0]
            raise ValueError(f"{self.still_holding[side]}. Then click Start over in Connect arms")
        if self.not_answering:
            sides = " and ".join(sorted(self.not_answering))
            raise ValueError(
                f"the {sides} arm stopped answering under power: {NOT_ANSWERING_HELP} Then click Start over in "
                "Connect arms"
            )
        if self.off_failed:
            side = sorted(self.off_failed)[0]
            raise ValueError(
                f"{off_failed_words(side, self.off_failed[side])} Then turn off the motors and click Scan "
                "again in Connect arms"
            )
        if self.parked:
            side = sorted(self.parked)[0]
            raise ValueError(f"{self.parked[side]} Then click Start over in Connect arms")

    def _require_let_go(self) -> None:
        self._require_still()
        if self._motion_arms:
            raise ValueError(
                "both arms still hold under power from the powered checks: support them with your hands and click "
                "Turn off motors first"
            )

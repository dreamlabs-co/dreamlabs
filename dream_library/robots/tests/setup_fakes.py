"""Fakes for the setup wizard: a generic arm with mistakes planted in its driver, two arms on two adapters, cameras,
and an operator's hands.

#FakeArm is the physical arm: its true joints in URDF units, which only the operator's hands move while it is
disengaged and which follow every write while it is powered. #Driver is what a driver does on top of it: it reports
the true joints through its read path and turns a command into a target through its write path, each with the mistakes
a #Mistakes plants (degrees for radians, a flipped joint, a zero offset, two joints in the wrong order, a gripper in
another unit or the other way round, a write path that disagrees with the read path). A clean driver is the identity
both ways. The wizard opens it the way it opens every arm, under a `robots.followers.FollowerArm`.

#Rig is the station: one #FakeArm per adapter, and the opener the wizard calls with `(robot, cfg, side)`; the adapter
is `cfg.can_channel`. #Hands moves a disengaged arm the way an operator would, calling `Session.tick()` after every
step as the reader thread does.
"""

import dataclasses
import datetime
import math
import pathlib
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from dream.datasets import Embodiment
from dream.errors import DreamServerError
from dream.models import Model

from robots import adapter, bodies, config, followers, setup_cameras, setup_session, types


@dataclasses.dataclass(frozen=True)
class Mistakes:
    """What a driver gets wrong, each None or False when it gets it right.

    Attributes:
    - read_units: The read reports `degrees`, or LeRobot's `lerobot` -100 to 100 (here: q / pi x 100), not radians.
    - flip: A joint the driver turns the other way (read and write alike).
    - offset: `(joint, radians)`: a joint whose zero is off from the URDF's by that much (read and write alike).
    - swap: `(a, b)`: two joints in the other order (read and write alike).
    - gripper: The gripper reads `inverted` (1 closed), in `mm` (0 to 80), or `still` (the same number always).
    - write_flip: A joint the write turns the other way than the read does.
    - write_gripper_flip: The write turns the gripper the other way round than the read does.
    - write_units: The write takes `degrees` while the read reports radians.
    - write_swap: `(a, b)`: two joints the write puts in the other order.
    - write_scale: The write multiplies every joint by this (1000: it takes milliradians).
    """

    read_units: str | None = None
    flip: int | None = None
    offset: tuple[int, float] | None = None
    swap: tuple[int, int] | None = None
    gripper: str | None = None
    write_flip: int | None = None
    write_gripper_flip: bool = False
    write_units: str | None = None
    write_swap: tuple[int, int] | None = None
    write_scale: float | None = None


## A driver that gets everything right.
CLEAN = Mistakes()

## How far a LeRobot-style read spreads one radian.
LEROBOT_PER_RAD = 100.0 / math.pi

## A gripper read in mm: the stroke fully open.
GRIPPER_MM = 80.0

## What a gripper the driver does not read reports, always.
GRIPPER_STILL = 0.5

## An embodiment as the account answers it for an SO-101 or a robot of one's own: six-wide arms, grips, tool poses.
EMBODIMENT: dict[str, Any] = {
    "state": {
        "right_arm": [0, 6],
        "right_grip": [6, 7],
        "right_eef_xyz": [7, 10],
        "right_eef_rot6": [10, 16],
        "left_arm": [16, 22],
        "left_grip": [22, 23],
        "left_eef_xyz": [23, 26],
        "left_eef_rot6": [26, 32],
    },
    "action": {
        "right_eef_xyz": [0, 3],
        "right_eef_rot6": [3, 9],
        "right_grip": [9, 10],
        "left_eef_xyz": [10, 13],
        "left_eef_rot6": [13, 19],
        "left_grip": [19, 20],
    },
    "video": {"scene": "observation.images.scene", "wrist_l": "observation.images.wrist_l", "wrist_r": "x.wrist_r"},
}


CREATED = datetime.datetime(2026, 10, 13, tzinfo=datetime.UTC)

MAKERMODS_OWN: dict[str, Any] = {
    **adapter.MAKERMODS_EMBODIMENT,
    "video": {name: adapter.MAKERMODS_EMBODIMENT["video"][name] for name in ("overhead", "left_wrist", "right_wrist")},
}


def account_embodiment(name: str, embodiment_config: dict[str, Any] = EMBODIMENT) -> Embodiment:
    """One of the account's embodiments as the data service lists it; its id is made from its name."""
    return Embodiment(name, embodiment_config, CREATED, "emb_" + name.replace("/", "_").replace("-", "_"))


def model(
    checkpoint: str, owner: str, embodiment: str | None, cameras: list[str] | None, servable: bool = True
) -> Model:
    """A model the account may run, as the platform lists it."""
    return Model(checkpoint, owner, "dream-1-base", embodiment, servable, CREATED, cameras)


YAM_CAMERAS = ["overhead", "wrist_l", "wrist_r"]


## A customer's embodiment whose cameras keep their dataset's names (LeRobot's ALOHA layout), none a name the wizard
## has a kind for (`robots.setup_cameras.ROLE_KINDS`).
CAM_NAMES: dict[str, Any] = {
    **EMBODIMENT,
    "video": {name: f"observation.images.{name}" for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")},
}

EMBODIMENTS = [
    *(account_embodiment(name) for name in ("tester/so101-pick", "tester/so101", "tester/my-arm", "tester/mine")),
    account_embodiment("tester/my-metal", MAKERMODS_OWN),
    account_embodiment("tester/cams", CAM_NAMES),
]

MODELS = [
    model("dreamlabs/dream-1-beta", "public", "yam", YAM_CAMERAS),
    model("dream-1-ft:tester-yam", "mine", "yam", YAM_CAMERAS),
    model("dream-1-ft:tester-metal", "mine", "makermods_metal", None),
    model("dream-1-ft:shared-metal", "public", "makermods_metal", None),
    model("dream-1-ft:tester-makers", "mine", "makermods_maker_arm", None),
    model("dream-1-ft:tester-my-metal", "mine", "emb_tester_my_metal", list(MAKERMODS_OWN["video"])),
    model("dream-1-ft:tester-so101", "mine", "emb_tester_so101_pick", list(EMBODIMENT["video"])),
    model("dream-1-ft:tester-mine", "mine", "emb_tester_mine", list(EMBODIMENT["video"])),
    model("dream-1-ft:tester-warming", "mine", "makermods_metal", None, servable=False),
    model("dream-1-ft:shared-unread", "public", "emb_0000aaaa0000aaaa", ["head", "wrist_l", "wrist_r"]),
    model("dream-1-ft:shared-front", "public", "emb_0000bbbb0000bbbb", ["front", "left", "right"]),
]


def no_training_data(model: str) -> dict[str, Any]:
    """The data service's answer for a fine-tune whose dataset is gone; the camera step shows our YAM pictures."""
    raise DreamServerError(f"The dataset {model} was trained on was deleted.", code="not_found")


def stub_embodiments(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every account embodiment is #EMBODIMENT but tester/cams (#CAM_NAMES), with no network; the cached bodies are
    built again."""
    monkeypatch.setattr(
        bodies, "embodiment_config", lambda name, key: CAM_NAMES if name == "tester/cams" else EMBODIMENT
    )
    bodies.so101_body.cache_clear()
    bodies.custom_body.cache_clear()


class Account:
    """The account's embodiments, which `robots.bodies` reads with no network; those the wizard registers
    (SetupServer's `add_embodiment`) join it, newest first, as on the data service."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, embodiments: list[Embodiment]) -> None:
        self.embodiments = list(embodiments)
        self.added: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(
            bodies,
            "embodiment_config",
            lambda name, key: {e.name: e.embodiment_config for e in self.embodiments}[name],
        )
        bodies.so101_body.cache_clear()
        bodies.custom_body.cache_clear()

    def add(self, name: str, wire: dict[str, Any]) -> Embodiment:
        """Registers `tester/<name>` with that config."""
        self.added.append((name, wire))
        created = account_embodiment(f"tester/{name}", wire)
        self.embodiments.insert(0, created)
        return created


class FakeArm:
    """One physical arm: its true joints (URDF units) and gripper (0 closed to 1 open).

    Attributes:
    - q: True joints, `[joints]` radians.
    - grip: True gripper opening.
    - powered: Whether its motors hold and follow writes (False while disengaged or closed).
    - disengages: How many times it was disengaged.
    - writes: How many writes reached it.
    - sag_rad: How far below a target a powered joint holds under gravity, radians: one for every joint, or per joint.
    - band_rad: How far a powered joint trails a target it moves to (friction, a hobby servo's dead band), radians:
      one for every joint, or per joint.
    - unplugged: Its cable is out: nothing reaches it and it answers nothing, and a powered one keeps holding its last
      target, as a Damiao motor with its CAN timeout off does.
    - unplug_after_writes: The write count at which the cable comes out, or None.
    - alarm: What its motor guard says now (a motor too hot, a joint blocked too long), or None.
    - floor: `(joint, angle)`: something under that joint, which a powered move cannot take below the angle, or None.
    - gripper_motor_deg: The gripper motor's angle closed and open, degrees, which its driver reports
      (`Driver.gripper_deg()`), or None when it reports none.
    """

    def __init__(
        self, q: npt.ArrayLike, grip: float = 1.0, sag_rad: npt.ArrayLike = 0.0, band_rad: float = 0.0
    ) -> None:
        self.q = np.array(q, dtype=np.float64)
        self.grip = float(grip)
        self.powered = False
        self.disengages = 0
        self.writes = 0
        self.sag_rad = np.asarray(sag_rad, dtype=np.float64)
        self.band_rad = band_rad
        self.unplugged = False
        self.unplug_after_writes: int | None = None
        self.alarm: str | None = None
        self.floor: tuple[int, float] | None = None
        self.gripper_motor_deg: tuple[float, float] | None = None

    def follow(self, target: npt.NDArray[np.float64]) -> None:
        """A powered arm told to go to `target`."""
        gap = target - self.sag_rad - self.q
        self.q = self.q + np.where(np.abs(gap) > self.band_rad, gap - np.sign(gap) * self.band_rad, 0.0)
        if self.floor is not None:
            joint, angle = self.floor
            self.q[joint] = max(self.q[joint], angle)


class Driver(followers.Driver):
    """A #robots.followers.Driver on a #FakeArm, with the planted #Mistakes."""

    def __init__(self, arm: FakeArm, mistakes: Mistakes) -> None:
        self.arm = arm
        self.mistakes = mistakes
        self.closed = False
        self.read_gripper_deg: float | None = None
        arm.powered = True

    def read(self) -> followers.Reading:
        """The true state through the read path; the gripper motor's angle is kept for gripper_deg()."""
        assert not self.closed, "read after close"
        joints, grip = self.reported(self.arm.q, self.arm.grip)
        motor = self.arm.gripper_motor_deg
        self.read_gripper_deg = None if motor is None else motor[0] + self.arm.grip * (motor[1] - motor[0])
        return followers.Reading(positions_rad=joints, velocities_rad_s=np.zeros_like(joints), gripper=grip)

    def gripper_deg(self) -> float | None:
        """The gripper motor's angle the last read saw (#FakeArm.gripper_motor_deg)."""
        return self.read_gripper_deg

    def write(self, positions_rad: npt.NDArray[np.float64], gripper: float) -> None:
        """A command through the write path; a powered arm goes there at once."""
        assert not self.closed, "write after close"
        self.arm.writes += 1
        if self.arm.unplug_after_writes is not None and self.arm.writes >= self.arm.unplug_after_writes:
            self.arm.unplugged = True
        if self.arm.unplugged or not self.arm.powered:
            return
        q, self.arm.grip = self.target(np.asarray(positions_rad, dtype=np.float64), gripper)
        self.arm.follow(q)

    def is_alive(self) -> bool:
        """False once the cable is out."""
        return not self.arm.unplugged

    def why_stopped(self) -> str:
        """Why it stopped answering."""
        return "its cable is out"

    def alarm(self) -> str | None:
        """The arm's motor guard alarm (#FakeArm.alarm)."""
        return self.arm.alarm

    def disengage(self) -> bool:
        """Lets go: the arm floats where it is until a hand moves it; with the cable out nothing reaches it."""
        if self.arm.unplugged:
            raise OSError("no reply on the bus")
        self.arm.powered = False
        self.arm.disengages += 1
        return True

    def torque_off(self) -> None:
        """Switches off and closes."""
        self.arm.powered = False
        self.closed = True

    def close(self) -> None:
        """Closes the handle; the arm keeps whatever it was doing."""
        self.closed = True

    def reported(self, q: npt.NDArray[np.float64], grip: float) -> tuple[npt.NDArray[np.float64], float]:
        """True joints and gripper as the read path reports them."""
        m = self.mistakes
        out = q.copy()
        if m.offset is not None:
            out[m.offset[0]] += m.offset[1]
        if m.flip is not None:
            out[m.flip] = -out[m.flip]
        if m.swap is not None:
            a, b = m.swap
            out[[a, b]] = out[[b, a]]
        if m.read_units == "degrees":
            out = np.degrees(out)
        elif m.read_units == "lerobot":
            out = out * LEROBOT_PER_RAD
        if m.gripper == "inverted":
            grip = 1.0 - grip
        elif m.gripper == "mm":
            grip = grip * GRIPPER_MM
        elif m.gripper == "still":
            grip = GRIPPER_STILL
        return out, grip

    def target(self, joints: npt.NDArray[np.float64], grip: float) -> tuple[npt.NDArray[np.float64], float]:
        """Where the write path sends the arm for a command: its own bugs, then the read path undone."""
        m = self.mistakes
        out = joints.copy()
        if m.write_flip is not None:
            out[m.write_flip] = -out[m.write_flip]
        if m.write_units == "degrees":
            out = np.radians(out)
        if m.write_scale is not None:
            out = out * m.write_scale
        if m.write_swap is not None:
            a, b = m.write_swap
            out[[a, b]] = out[[b, a]]
        if m.read_units == "degrees":
            out = np.radians(out)
        elif m.read_units == "lerobot":
            out = out / LEROBOT_PER_RAD
        if m.swap is not None:
            a, b = m.swap
            out[[a, b]] = out[[b, a]]
        if m.flip is not None:
            out[m.flip] = -out[m.flip]
        if m.offset is not None:
            out[m.offset[0]] -= m.offset[1]
        if m.gripper == "inverted":
            grip = 1.0 - grip
        elif m.gripper == "mm":
            grip = grip / GRIPPER_MM
        if m.write_gripper_flip:
            grip = 1.0 - grip
        return out, float(grip)


class Rig:
    """A station: one #FakeArm per adapter, each driven with its own #Mistakes, and the wizard's opener.

    Attributes:
    - arms: Per adapter (`cfg.can_channel`), the arm.
    - mistakes: Per adapter, what its driver gets wrong.
    - opened: Every `(body, adapter, side)` the wizard opened, in order.
    - configs: The config of each of those opens.
    - handles: Every #Driver handed out.
    """

    def __init__(self, arms: dict[str, FakeArm], mistakes: dict[str, Mistakes] | None = None) -> None:
        self.arms = arms
        self.mistakes = {name: CLEAN for name in arms} | (mistakes or {})
        self.opened: list[tuple[str, str, str]] = []
        self.configs: list[config.ArmConfig] = []
        self.handles: list[Driver] = []

    def open(self, robot: setup_session.Robot, cfg: config.ArmConfig, side: str) -> followers.FollowerArm:
        """The wizard's opener: a FollowerArm on the arm on `cfg.can_channel`, powered and holding."""
        if cfg.can_channel not in self.arms:
            raise OSError(f"no adapter {cfg.can_channel}")
        self.opened.append((robot.body, cfg.can_channel, side))
        self.configs.append(cfg)
        handle = Driver(self.arms[cfg.can_channel], self.mistakes[cfg.can_channel])
        self.handles.append(handle)
        return followers.FollowerArm(side, handle, home_rad=cfg.home)


def device(link: str, serial: str) -> dict[str, Any]:
    """An arm adapter as `robots.setup_devices.arm_adapters` lists one: a CAN link with a serial."""
    return {
        "id": f"can-{serial}",
        "kind": "can",
        "bus": "can",
        "open": link,
        "serial": serial,
        "usb": "1-2:1.0",
        "label": f"CAN adapter {serial}",
        "detail": f"CAN link {link}, up",
        "state": "up",
        "via": None,
    }


def net_link(net: pathlib.Path, name: str, *, up: bool, operstate: str | None = None, kind: str = "280") -> None:
    """Plants a link in a fake `/sys/class/net`."""
    entry = net / name
    entry.mkdir(parents=True, exist_ok=True)
    (entry / "type").write_text(f"{kind}\n")
    (entry / "flags").write_text("0x81\n" if up else "0x80\n")
    (entry / "operstate").write_text(f"{operstate or ('up' if up else 'down')}\n")


def slcand(proc: pathlib.Path, pid: int, *argv: str) -> None:
    """Plants an `slcand` in a fake `/proc` (`robots.robstride.PROC_DIR`): its name and its command line."""
    entry = proc / str(pid)
    entry.mkdir(parents=True)
    (entry / "comm").write_text("slcand\n")
    (entry / "cmdline").write_bytes(b"\0".join(word.encode() for word in ("slcand", *argv)) + b"\0")


class Hands:
    """The operator's hands on a disengaged arm: each move in small steps, the session reading after every one.

    Attributes:
    - session: The wizard's session.
    - rig: The station.
    - clock: The session's clock; each step advances it by #STEP_S.
    """

    ## Seconds between two readings of a hand move: the reader thread's period.
    STEP_S = 1.0 / setup_session.READ_HZ

    def __init__(self, session: setup_session.Session, rig: Rig, clock: Any) -> None:
        self.session = session
        self.rig = rig
        self.clock = clock

    def _step(self) -> None:
        self.clock.sleep(self.STEP_S)
        self.session.tick()

    def move(self, adapter: str, joint: int, angle: float, steps: int = 10, back: bool = True) -> None:
        """Turns one joint by `angle` radians (true, URDF way), then back to where it was."""
        arm = self.rig.arms[adapter]
        assert not arm.powered, "a hand moved a powered arm"
        start = arm.q[joint]
        for i in range(1, steps + 1):
            arm.q[joint] = start + angle * i / steps
            self._step()
        if back:
            for i in range(steps - 1, -1, -1):
                arm.q[joint] = start + angle * i / steps
                self._step()

    def pose(self, adapter: str, target: npt.ArrayLike, steps: int = 10) -> None:
        """Moves the whole arm to a pose in a straight line, the session reading on the way."""
        arm = self.rig.arms[adapter]
        assert not arm.powered, "a hand moved a powered arm"
        start, end = arm.q.copy(), np.asarray(target, dtype=np.float64)
        for i in range(1, steps + 1):
            arm.q = start + (end - start) * i / steps
            self._step()

    def grip(self, adapter: str, opening: float, steps: int = 5) -> None:
        """Opens or closes the gripper by hand to `opening` (0 closed to 1 open), the session reading on the way."""
        arm = self.rig.arms[adapter]
        assert not arm.powered, "a hand moved a powered gripper"
        start = arm.grip
        for i in range(1, steps + 1):
            arm.grip = start + (opening - start) * i / steps
            self._step()


class FakeReader:
    """A camera reader with the `CameraReader` shape the wizard uses: one frame, alive until stopped."""

    def __init__(self, camera: config.CameraConfig, rgb: npt.NDArray[np.uint8] | None) -> None:
        self.camera = camera
        self.error: BaseException | None = None
        self.stopped = False
        self.frame = None if rgb is None else types.Frame(rgb, camera.name, 0.0, ("x",))

    def is_alive(self) -> bool:
        return not self.stopped

    def stop(self) -> None:
        self.stopped = True


def sample_rgb(kind: str) -> npt.NDArray[np.uint8]:
    """Our shipped reference frame of a kind of camera, as RGB: what a camera pointing the way ours do would see."""
    return setup_cameras.decode_jpeg(setup_cameras.sample_jpeg(kind, 0))


class FakeCameras:
    """Cameras by the value a row names them by, each showing a fixed picture; the wizard's `open_camera`.

    Attributes:
    - pictures: Per camera value, the RGB it shows.
    - opened: Every camera value opened.
    """

    def __init__(self, pictures: dict[str, npt.NDArray[np.uint8]]) -> None:
        self.pictures = pictures
        self.opened: list[str] = []

    def __call__(self, camera: config.CameraConfig) -> FakeReader:
        value = str(camera.serial if camera.serial is not None else camera.device)
        self.opened.append(value)
        return FakeReader(camera, self.pictures[value])

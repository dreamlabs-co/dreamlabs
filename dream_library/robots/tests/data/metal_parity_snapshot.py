"""Writes metal_parity.json: MakerMods' own Metal driver run through one scenario on the Metal fake motors.

The source of truth for tests/test_metal_parity.py, which runs the same scenario through `robots.metal` and requires
the same CAN frames, byte for byte, the same motor readings and the same goals sent. The source is MakerMods' Metal
control loop (their LeRobot fork at `5ce2fe4`, MakerMods Lab at `f3ee37c`, their console), the files
#SOURCE_FILES, copied out of it unmodified. That extract is for reading, not a
runnable package, so this script builds a package tree from the copies plus small stand-ins (#STUBS) for the LeRobot
framework pieces it leaves out (base classes, decorators, the camera factory); none of them is on the frame path.
The fake motors are `tests/metal_fakes.py`, the ones the test runs this package on.

What runs is the console's own wrapper (`makermods_console.hardware.MetalRobot`: MakerLab's holding bus around
LeRobot's `MetalFollower`, the 50 ms reply wait, the preloaded connect, the per-motor reads with one retry, the
status check), driven the way its servo loop drives it, one `read()` and one `command()` per tick at 30 Hz. Two
things do not run: MakerLab's holding thread (`MetalGripperBus._hold`, which re-runs the gripper's regulation every
20 ms between commands; its frames depend on wall-clock interleaving with the servo loop, and `robots.metal_gripper`
replaces it with the driver's write on every tick), and the console's release (`MetalRobot.release`, which calls a
MakerLab torque helper the extract leaves out): the scenario ends with MakerLab's own
`MetalHoldingBus.disconnect(disable_torque=True)`, which disables every joint and releases the gripper.

Per scenario step the snapshot holds the number of frames sent and the SHA-256 of their `"<id>:<payload hex>"`
lines, and per tick the seven motor readings and the seven goals sent, in degrees, and the torque the gripper
pushes with (the fake motor's: what the holding law produced). Run it with `python -I` and an
interpreter that has python-can 4.5, numpy and pydantic 2, on a directory holding copies of #SOURCE_FILES at their
paths in the extract:

```bash
python -I metal_parity_snapshot.py --source /tmp/metal_src --out metal_parity.json
```

Time is a fake clock that moves only when something sleeps or waits on it: `time.time`, `time.monotonic`,
`time.perf_counter` and `time.sleep` are replaced before anything else is imported.
"""

import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import tempfile
import time
import types

## The extract's files this run uses, at their paths in it.
SOURCE_FILES = (
    "lerobot/motors/damiao/damiao.py",
    "lerobot/motors/damiao/tables.py",
    "lerobot/robots/metal_follower/config_metal_follower.py",
    "lerobot/robots/metal_follower/metal_follower.py",
    "makermodslab/gripper_settings.py",
    "makermodslab/metal_gripper.py",
    "makermodslab/metal_gripper_hold.py",
    "makermods_console/config.py",
    "makermods_console/hardware.py",
)

## The versions the extract's README names for those files.
SOURCE_VERSIONS = {"lerobot": "5ce2fe4", "makermodslab": "f3ee37c"}

## Stand-ins for what the extract leaves out, by path in the package tree. None of it sends or reads a frame.
STUBS = {
    "lerobot/__init__.py": "",
    "lerobot/utils/__init__.py": "",
    "lerobot/utils/decorators.py": """
import functools


def check_if_not_connected(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.is_connected:
            raise ConnectionError(f"{self} is not connected")
        return func(self, *args, **kwargs)

    return wrapper


def check_if_already_connected(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.is_connected:
            raise ConnectionError(f"{self} is already connected")
        return func(self, *args, **kwargs)

    return wrapper
""",
    "lerobot/utils/import_utils.py": """
_can_available = True


def require_package(name, extra=None, import_name=None):
    __import__(import_name or name)
""",
    "lerobot/utils/robot_utils.py": """
import time


def precise_sleep(seconds):
    time.sleep(seconds)
""",
    "lerobot/utils/utils.py": """
def enter_pressed():
    return False


def move_cursor_up(lines):
    pass
""",
    "lerobot/motors/__init__.py": "from .motors_bus import Motor, MotorCalibration, MotorNormMode\n",
    "lerobot/motors/motors_bus.py": """
import dataclasses
import enum


class MotorNormMode(str, enum.Enum):
    DEGREES = "degrees"


@dataclasses.dataclass
class Motor:
    id: int
    model: str
    norm_mode: MotorNormMode
    motor_type_str: str | None = None
    recv_id: int | None = None


@dataclasses.dataclass
class MotorCalibration:
    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int


NameOrID = str | int
Value = int | float


class MotorsBusBase:
    def __init__(self, port, motors, calibration=None):
        self.port = port
        self.motors = motors
        self.calibration = calibration if calibration else {}
""",
    "lerobot/motors/damiao/__init__.py": "from .damiao import DamiaoMotorsBus\n",
    "lerobot/cameras/__init__.py": """
class CameraConfig:
    pass


def make_cameras_from_configs(configs):
    return {}
""",
    "lerobot/lerobot_types.py": "RobotAction = dict\nRobotObservation = dict\n",
    "lerobot/robots/__init__.py": "",
    "lerobot/robots/config.py": """
import dataclasses
import pathlib


@dataclasses.dataclass
class RobotConfig:
    id: str | None = None
    calibration_dir: pathlib.Path | None = None

    @classmethod
    def register_subclass(cls, name):
        return lambda subclass: subclass
""",
    "lerobot/robots/robot.py": """
import json

from lerobot.motors import MotorCalibration


class Robot:
    def __init__(self, config):
        self.id = config.id
        self.calibration_fpath = config.calibration_dir / f"{config.id}.json"
        self.calibration = {}
        if self.calibration_fpath.is_file():
            entries = json.loads(self.calibration_fpath.read_text())
            self.calibration = {name: MotorCalibration(**entry) for name, entry in entries.items()}

    def __str__(self):
        return f"{self.id} {type(self).__name__}"
""",
    "lerobot/robots/utils.py": """
def ensure_safe_goal_position(goal_present_pos, max_relative_target):
    raise NotImplementedError("max_relative_target is None on the console")
""",
    "lerobot/robots/metal_follower/__init__.py": "",
    "makermodslab/__init__.py": "",
    "makermodslab/maker_can.py": "def install():\n    pass\n",
    "makermods_console/__init__.py": "",
    "cv2.py": "CAP_ANY, CAP_V4L2, CAP_AVFOUNDATION = 0, 200, 1200\n",
}

## The arm's channel; it appears in no frame.
CHANNEL = "/dev/serial/by-id/metal-right"

## One servo tick, seconds: 30 Hz, the rate `robots.metal` ticks at.
PERIOD_S = 1.0 / 30.0


def ramp(start: list[float], end: list[float], ticks: int) -> list[list[float]]:
    """`ticks` goals from `start` to `end`, one per tick, the last one `end` (the test's copy must agree)."""
    return [[a + (b - a) * (i + 1) / ticks for a, b in zip(start, end, strict=True)] for i in range(ticks)]


## Goals, motor degrees base to gripper (the gripper 0 closed, 137.5 open).
_REACH = [25.0, -60.0, 80.0, -10.0, 15.0, 30.0, 100.0]
_AWAY = [-20.0, -75.0, 100.0, 20.0, -25.0, -40.0, 60.0]
## Past every soft limit: pan above 160, lift above 0, elbow below 0, wrist flex above 81, wrist yaw below -85, wrist
## roll above 145, the gripper above 137.5.
_OUT_OF_LIMITS = [170.0, 20.0, -10.0, 90.0, -100.0, 150.0, 150.0]

## The scenario, step by step (tests/test_metal_parity.py replays it):
## - pose: the motors' angles before the arm opens, degrees by CAN id;
## - register: a motor register's value before the arm opens (the gripper left in force-position mode, 4);
## - connect: the arm opened and energized (the console's `MetalFollower.connect`; here open() and enable());
## - track: one goal, sent on each of `ticks` ticks, each tick a read and a command;
## - ramp: a goal moving from `from` to `to` over `ticks` ticks (ramp());
## - hold: `ticks` ticks that send the goal the last tick sent;
## - obstacle: a rigid object in the closing direction of a motor (the gripper, id 7), or none;
## - drop_every: drop the reply to every n-th frame, from the motors `ids` only, 0 for none;
## - pause: the clock moves on with no tick;
## - fault: a motor starts reporting a fault status;
## - failing_tick: a tick whose read raises (the fault); its frames are the reads it made;
## - close: every joint disabled and the gripper released, then the adapter closed.
SCENARIO: list[dict] = [
    {"op": "pose", "deg": {"1": 10.0, "2": -40.0, "3": 60.0, "4": 5.0, "5": -5.0, "6": 20.0, "7": 30.0}},
    {"op": "register", "id": 7, "rid": 10, "u32": 4},
    {"op": "connect"},
    # The startup sync onto a goal that sits still (1 degree a tick until within 3), then tracking at full rate.
    {"op": "track", "q": _REACH, "ticks": 80},
    {"op": "ramp", "from": _REACH, "to": _AWAY, "ticks": 45},
    {"op": "track", "q": _AWAY, "ticks": 10},
    # Soft-limit clamp.
    {"op": "track", "q": _OUT_OF_LIMITS, "ticks": 20},
    {"op": "hold", "ticks": 5},
    # The gripper closes on an object: the hold takes over, then an open releases it.
    {"op": "obstacle", "id": 7, "deg": 40.0},
    {"op": "track", "q": _REACH[:6] + [0.0], "ticks": 45},
    {"op": "track", "q": _REACH[:6] + [120.0], "ticks": 20},
    {"op": "obstacle", "id": 7, "deg": None},
    # Missed joint replies: a missed read is asked once more, a missed MIT or batched reply costs nothing.
    {"op": "drop_every", "n": 7, "ids": [1, 2, 3, 4, 5, 6]},
    {"op": "ramp", "from": _REACH[:6] + [120.0], "to": _AWAY, "ticks": 30},
    {"op": "drop_every", "n": 0, "ids": None},
    # A gap: the feedforward's time step is clamped.
    {"op": "pause", "s": 0.2},
    {"op": "track", "q": _REACH, "ticks": 10},
    # A joint faults: the next read refuses the arm.
    {"op": "fault", "id": 5, "status": 9},
    {"op": "failing_tick", "q": _REACH},
    {"op": "close"},
]


class FakeClock:
    """A clock that moves only when something sleeps on it (metal_fakes.FakeClock, before any import)."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, seconds)


def build_tree(source: pathlib.Path, tree: pathlib.Path) -> dict[str, str]:
    """Copies #SOURCE_FILES out of `source` and writes #STUBS around them.

    @return The SHA-256 of each source file copied.
    """
    digests = {}
    for relative in SOURCE_FILES:
        target = tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
        digests[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    for relative, text in STUBS.items():
        target = tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text.lstrip("\n"), encoding="utf-8")
    return digests


def main() -> None:
    """Runs the scenario on MakerMods' driver and writes the snapshot."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--source", required=True, help="A directory with copies of SOURCE_FILES at their paths.")
    parser.add_argument("--out", required=True, help="Where to write the snapshot.")
    args = parser.parse_args()

    clock = FakeClock()
    time.time = clock
    time.monotonic = clock
    time.perf_counter = clock
    time.sleep = clock.sleep

    tree = pathlib.Path(tempfile.mkdtemp(prefix="metal_parity_"))
    digests = build_tree(pathlib.Path(args.source), tree)
    sys.path.insert(0, str(tree))
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    import can
    import metal_fakes
    import numpy as np

    factory = metal_fakes.FakeCanFactory(clock)
    can.interface.Bus = lambda channel, **kwargs: factory(channel)

    import makermodslab.metal_gripper as metal_gripper
    from makermods_console.config import ArmConfig
    from makermods_console.hardware import MetalRobot

    # MakerLab's holding thread is not started (see the module docstring); everything else of the module stays.
    class NoThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def join(self, timeout=None):
            pass

    real_threading = metal_gripper.threading
    metal_gripper.threading = types.SimpleNamespace(
        Thread=NoThread,
        RLock=real_threading.RLock,
        Event=real_threading.Event,
        current_thread=real_threading.current_thread,
    )

    calibration = tree / "calibration" / "metal_right.json"
    calibration.parent.mkdir()
    limits = {
        "shoulder_pan": (-160, 160),
        "shoulder_lift": (-180, 0),
        "elbow_flex": (0, 180),
        "wrist_flex": (-123, 81),
        "wrist_yaw": (-85, 85),
        "wrist_roll": (-145, 145),
        "gripper": (0, 137),
    }
    calibration.write_text(
        json.dumps(
            {
                name: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": low, "range_max": high}
                for i, (name, (low, high)) in enumerate(limits.items())
            }
        )
    )
    arm = ArmConfig(
        name="right",
        port=CHANNEL,
        robot_id="metal_right",
        calibration_file=str(calibration),
        can_interface="slcan",
        gripper_closed_deg=0.0,
        gripper_open_deg=137.5,
        gripper_hold_torque_nm=0.5,
    )
    robot = MetalRobot(types.SimpleNamespace(arms=[arm]))
    follower = robot.robots["right"]

    readings: list[list[float]] = []
    goals: list[list[float]] = []
    gripper_torques: list[float] = []
    raised: list[str] = []

    def tick(goal: list[float]) -> None:
        t0 = time.monotonic()
        q = robot.read()["right"]
        readings.append([float(x) for x in q])
        applied = robot.command({"right": np.asarray(goal, dtype=np.float64)})["right"]
        goals.append([float(x) for x in applied])
        gripper_torques.append(factory.buses[CHANNEL].motors[7].torque_nm)
        time.sleep(max(0.0, PERIOD_S - (time.monotonic() - t0)))

    def frames() -> list[tuple[int, bytes]]:
        return factory.buses[CHANNEL].frames if CHANNEL in factory.buses else []

    steps: list[dict] = []
    for step in SCENARIO:
        sent = len(frames())
        op = step["op"]
        if op == "pose":
            for can_id, deg in step["deg"].items():
                factory.bus(CHANNEL).motors[int(can_id)].position_deg = deg
        elif op == "register":
            factory.bus(CHANNEL).motors[step["id"]].registers[step["rid"]] = step["u32"].to_bytes(4, "little")
        elif op == "connect":
            follower.connect(calibrate=False)
        elif op == "track":
            for _ in range(step["ticks"]):
                tick(step["q"])
        elif op == "ramp":
            for goal in ramp(step["from"], step["to"], step["ticks"]):
                tick(goal)
        elif op == "hold":
            for _ in range(step["ticks"]):
                tick(goals[-1])
        elif op == "obstacle":
            factory.buses[CHANNEL].motors[step["id"]].obstacle_deg = step["deg"]
        elif op == "drop_every":
            factory.buses[CHANNEL].drop_every = step["n"]
            factory.buses[CHANNEL].drop_ids = None if step["ids"] is None else set(step["ids"])
        elif op == "pause":
            time.sleep(step["s"])
        elif op == "fault":
            factory.buses[CHANNEL].motors[step["id"]].fault = step["status"]
        elif op == "failing_tick":
            try:
                tick(step["q"])
            except ValueError as exc:
                raised.append(str(exc))
            else:
                raise SystemExit("the failing tick did not fail")
        elif op == "close":
            follower.bus.disconnect(disable_torque=True)
        else:
            raise SystemExit(f"unknown step {op!r}")
        lines = "\n".join(f"{can_id}:{data.hex()}" for can_id, data in frames()[sent:])
        steps.append({"frames": len(frames()) - sent, "sha256": hashlib.sha256(lines.encode()).hexdigest()})

    bus = factory.buses[CHANNEL]
    snapshot = {
        "source": "metal-control-loop received 2026-10-12",
        "source_versions": SOURCE_VERSIONS,
        "source_sha256": digests,
        "scenario": SCENARIO,
        "steps": steps,
        "readings": readings,
        "goals": goals,
        "gripper_torques": gripper_torques,
        "raised": raised,
        "enabled_after_close": [motor.can_id for motor in bus.motors.values() if motor.enabled],
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=None, separators=(",", ":"))
        handle.write("\n")
    shutil.rmtree(tree)
    print(
        f"{len(bus.frames)} frames, {len(readings)} ticks, raised {raised}, "
        f"gripper torque from {min(gripper_torques):.3f} to {max(gripper_torques):.3f} N.m"
    )


if __name__ == "__main__":
    main()

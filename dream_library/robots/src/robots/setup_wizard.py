"""The setup wizard: a local web page that sets up any robot for dream and writes the station config.

`dreamcli setup` serves one page (`setup_wizard.html`, with `setup_wizard.js` and the 3D arm view `setup_viewer.js`
next to this module, and the arm meshes under `setup_view/`) on the stdlib server, and opens it in a browser when the
box has a display; on a headless station the operator opens the printed address from a laptop (its network name or an
interface address; over ssh the `ssh -L` tunnel it prints is the fallback). Every request must carry the run's token,
which the printed address holds (#robots.setup_access); without it the reply is 403, a plain page in words when a
browser opens the address. The page loads three.js from a CDN, so setup needs internet. It walks the operator through
the steps in `docs/SETUP_WIZARD.md`: pick the robot (SetupServer.choose()), find each arm
by moving it (devices named by serial number, else by USB port), give each camera its role and compare it with a
reference picture, put the arms in their home pose, and then, after a safety popup, let the wizard nudge every joint
and sweep the hand a little under power while the page draws the arm from its readings, and say whether each joint and
the hand moved like the drawing. Each check is judged against dream's URDF (#robots.setup_model) and says what is
likely off when it fails (#robots.setup_checks).

The account is not asked: the key, the gateway and the platform are the credentials file (`dream.credentials`,
`~/.dreamlabs/credentials`), read once when the command starts and shown on the page; without a key the command stops
and says where to write one. Save registers the station on the platform as its embodiment's id (register_robot())
and writes the file once `robots.config.load` and `robots.run.policy_config` accept it, exactly as the agent will; a
file the agent would refuse is never written. The file names no model (`checkpoint`): a YAM runs its default, and every
start on another robot names one, as the live view's Start does.

`dreamcli tcp` (tcp_main(), TcpServer) serves the same page with only its URDF step, the tool point tool, for a station
file written by hand: it answers #TCP_ROUTES alone and needs no key.

| Route | Reply |
| --- | --- |
| `GET /`, `/setup_wizard.js`, `/setup_viewer.js` | the page and its scripts |
| `GET /driver_template.py` | `robots.driver_template`, the file a driver of your own starts from |
| `GET /view/<path>` | a file under `setup_view/`: an arm's meshes (#robots.setup_view.files()) |
| `GET /api/state` | the robots to pick, the file and what it holds, the account, defaults, the pick so far, version |
| `GET /api/status` | the session (#robots.setup_session.Session.status), the wiggle, cameras, the arms now; polled |
| `GET /api/scan` | arm adapters that fit the robot, other devices, cameras, typed ones (SetupServer.scan_devices) |
| `GET /api/udev_rule` | the rule naming the found arms' CAN links by serial, or 400 |
| `GET /api/template?body=` | `{"yaml"}`: the station file to fill in by hand, Set it up myself (template()) |
| `GET /api/preview?source=&value=&width=&height=&fps=` | one JPEG from that camera, or 400 |
| `GET /api/cameras/frame?role=&view=` | the latest JPEG of an open role: `live`, or `model` (as sent) |
| `GET /api/reference?role=&episode=` | the reference frame for a role: the operator's dataset, else our YAM sample |
| `GET /api/cameras/match` | camera roles that look swapped (`robots.setup_cameras.swapped_roles`) |
| `GET /api/view` | how the page draws the arm in 3D (#robots.setup_view.drawing()) |
| `GET /api/embodiments` | the account's embodiments: `name`, `id`, `cameras` (SetupServer.embodiments()) |
| `GET /api/reference_arm` | the URDF step's example arm, a YAM with its tool point (#robots.setup_view.reference()) |
| `GET /api/files?dir=` | a folder of this computer, for the driver's file picker (#robots.setup_files.folder()) |
| `GET /api/classes?file=` | the classes a Python file defines, read, not run (#robots.setup_files.classes()) |
| `POST /api/urdf` | `{"urdf"}`: its links at the arm's zero, a guess of base and flange (SetupServer.read_urdf()) |
| `POST /api/arm_file` | the URDF, its links, the tool point, `tcp_calibration`, `fixed` (SetupServer.add_arm()) |
| `POST /api/robot` | `{"body"}`, with `calibration` and `embodiment` (null for a new one) on an SO-101; `{"custom"}` |
| `POST /api/arms/typed` | `{"address"}`: an arm the scan misses, as its driver opens it (SetupServer.type_arm()) |
| `POST /api/cameras/typed` | `{"address"}`: a camera the scan misses, address or class (SetupServer.type_camera()) |
| `POST /api/arms/open` | `{"devices": [ids], "settings": {"left", "right"}}`: open every candidate, disengaged |
| `POST /api/arms/rebase` | measure the next wiggle from now |
| `POST /api/arms/close` | let go of every arm found so far and forget which is which, to look again |
| `POST /api/arms/assign` | `{"device", "side"}`: the device that moved is this arm, or `none` |
| `POST /api/calibrate/start` | `{"side"}`: calibrate that SO-101 arm, its servos off (Session.start_calibration()) |
| `POST /api/calibrate/middle` | `{}`: the arm is in its middle pose (Session.set_middle()) |
| `POST /api/calibrate/finish` | `{}`: write the servos and the file (Session.finish_calibration()) |
| `POST /api/calibrate/cancel` | `{}`: stop calibrating; the arm keeps what it had (Session.cancel_calibration()) |
| `POST /api/cameras/open` | `{"cameras": [rows]}`: hold each role's camera open |
| `POST /api/cameras/stop` | release every camera |
| `POST /api/dataset` | `{"path"}` (empty clears): the operator's LeRobot v2.1 dataset |
| `POST /api/dataset/keys` | `{"keys": {role: video key}}` |
| `POST /api/home` | `{}`: take the arms' joints now as home (Session.set_home()) |
| `POST /api/gripper` | `{"end"}`, `open` or `closed`: both grippers, put there by hand (Session.record_gripper()) |
| `POST /api/reach/paint` | `{"on": bool}`: the page is on the Reach area step, reads paint (Session.paint_reach()) |
| `POST /api/reach/clear` | `{"side"}`: forget that arm's painted area (Session.clear_reach()) |
| `POST /api/reach/skip` | `{}`: the Reach area step skipped; Save writes no `reach` (Session.skip_reach()) |
| `POST /api/references/retry` | `{}`: read the fine-tune's training data again (SetupServer.retry_references()) |
| `POST /api/arms/swap` | `{}`: both arms named as the other side in one step (Session.swap_sides()) |
| `POST /api/motion/start` | `{"confirm": true}`: the powered checks, after the safety popup |
| `POST /api/motion/stop` | Stop: both arms hold where they are; during a park both go limp at once |
| `POST /api/motion/release` | let go of the powered arms once the operator holds them up (Session.release()) |
| `POST /api/motion/eyes` | `{"same": bool}`: whether the real arms moved like the drawing (Session.confirm_eyes()) |
| `POST /api/save` | `{"cameras": [rows]}`: 200 the save reply (#Saved), or 400 `{"error"}` |
| `POST /api/exit` | 200; the wizard process ends |
| `GET /api/connection` | check_connection(): does the platform take the key; only with `DREAM_SETUP_WIZARD_V2` on |
| `POST /api/motors/gripper_mit` | `{"device"}`: gripper motor to MIT; only with `DREAM_SETUP_GRIPPER_MIT` on |

@see main(), tcp_main(), SetupServer, build_config(), save(), register_robot()
"""

import dataclasses
import datetime
import errno
import html
import http.server
import importlib.resources
import importlib.util
import json
import logging
import math
import os
import pathlib
import re
import signal
import socket
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
import zipfile
from collections.abc import Callable
from typing import IO, Any

import dream
import yaml
from dream import arm_file, credentials, fine_tuning
from dream.datasets import Embodiment
from dream.models import Model
from shared.contracts import station

import robots
from robots import (
    adapter,
    beta_robots,
    config,
    errors,
    feetech,
    followers,
    kinematics,
    motor_mode,
    run,
    setup_access,
    setup_cameras,
    setup_devices,
    setup_files,
    setup_model,
    setup_session,
    setup_urdf,
    setup_view,
    so101,
)

## Default bind address: the station is usually headless, and the operator opens the page from a laptop.
DEFAULT_HOST = "0.0.0.0"

## Default port of the wizard page; the agent's API is on 8091.
DEFAULT_PORT = 8092

## Default port of `dreamcli tcp`'s page (tcp_main()), so it runs beside `dreamcli setup` (8092) and `dreamcli robot
## run --view` (8093, `robots.run_view.DEFAULT_PORT`).
TCP_PORT = 8094

## The `/api/...` routes `dreamcli tcp` answers (TcpServer), beside the page, its scripts and the meshes (`/view/`).
TCP_ROUTES = ("/api/state", "/api/reference_arm", "/api/urdf", "/api/arm_file")

## What `dreamcli setup` prints on its own line: the page is served by the process in that terminal, so a command
## typed into it, or a Ctrl-C to get the prompt back, ends the page. A beta operator lost the page exactly this way.
KEEP_TERMINAL_OPEN = (
    "Leave this terminal open while you use the setup page. Open a second terminal for other commands. Connected "
    "over SSH? Keep that session open too: closing it stops setup."
)

## What to do once Save wrote the file, printed when the page is done, the command on a line of its own to copy;
## `{live}` is the platform's Live view.
NEXT_STEPS = """Next, start the station; copy and paste:

    dreamcli robot up

then press Start in Live view: {live}
Or one run from the terminal: dreamcli robot run --task "<what to do>" (--dry-run first to check without moving)"""

## Seconds one registration request may take; the platform answers in well under a second.
REGISTER_TIMEOUT_S: float = 15.0

## The modules of the `live` extra. Without both, a file with `live.url` stops the station at boot, so Save leaves the
## section out and says so instead.
LIVE_EXTRA_MODULES = ("av", "websockets")

## The loop keys the file carries, with their standard values. The keys `AppConfig` defaults are left to it, so a
## default
## that moves in code moves for every wizard-written file; the tick rate (per body) and the refill deadline are such
## defaults (the package owns the loop's timing).
LOOP_KEYS: dict[str, Any] = {
    "action_horizon": 48,
    "stale_after_s": 0.2,
    "max_stall_ticks": config.MAX_STALL_TICKS,
    "max_ee_speed_m_s": 0.6,
    "face_blur": True,
}

## Each body's arm settings the page starts from when the file has none: its arm section's keys other than the
## device, the home pose and the measured gripper ends.
DEFAULT_SETTINGS: dict[str, str] = {
    "yam": "gripper: linear_4310\n",
    "makers": "gripper_hold_torque_nm: 0.5\n",
    "metal": "gripper_hold_torque_nm: 0.5\n",
}

WIZARD_KEYS = ("can_channel", "home", "reach", "gripper_open_deg", "gripper_closed_deg", "calibration")

PAGE_KEYS = ("body", "embodiment", "custom", "checkpoint", "follower_l", "follower_r", "cameras", "live")

CAMERA_SIZES: dict[str, dict[str, tuple[int, int, int]]] = {
    "yam": {"head": (1280, 720, 30), "wrist_left": (1280, 720, 30), "wrist_right": (1280, 720, 30)},
    "makers": {"head": (1280, 720, 30), "wrist_left": (640, 480, 30), "wrist_right": (640, 480, 30)},
    "metal": {"head": (1280, 720, 30), "wrist_left": (1280, 720, 30), "wrist_right": (1280, 720, 30)},
}

## A camera's size and rate on a robot of the customer's own: 16:9, the shape the model takes (640x360), so nothing is
## stretched. 1280x720 is the largest 16:9 colour mode a RealSense D405 gives, at up to 15 fps (a webcam's too); the
## model samples 5 fps, so 15 is plenty.
CUSTOM_CAMERA_SIZE = (1280, 720, 15)

## Largest request body accepted: an arm's URDF (SetupServer.add_arm()), tens of kB without its meshes, is the largest;
## a camera list is a few hundred bytes.
MAX_BODY_BYTES: int = 2 * 1024 * 1024

## The folder beside the config file that the arms of your own the page takes are written to (SetupServer.add_arm()).
ARMS_DIR = "arms"

## The page's scripts, served from beside this module.
SCRIPTS = ("setup_wizard.js", "setup_wizard_v2.js", "setup_viewer.js", "setup_page.js")

## Both pages' stylesheet, served from beside this module.
STYLESHEET = "setup_wizard.css"

## The file a driver of your own starts from, served for the page's link to it.
DRIVER_TEMPLATE = "driver_template.py"

## The header every written file starts with.
FILE_HEADER = (
    "# Station config written by `dreamcli setup`. `dreamcli robot up` runs the station agent on it.\n"
    "# Every key is documented in the robots README, *Config file*; the loop keys not written here keep the\n"
    "# package's defaults. Re-run `dreamcli setup` to change it, or edit it by hand.\n"
)

_log = logging.getLogger(__name__)


def build_config(
    robot: setup_session.Robot,
    arms: dict[str, dict[str, Any]],
    cameras: list[dict[str, Any]],
) -> dict[str, Any]:
    """The YAML mapping: the robot, each arm's device, settings, home and reach area, the cameras, the loop values.

    @param robot The robot picked.
    @param arms Per side, `{"can_channel", "settings", "home", "reach"}` with `home` in radians and `reach` the arm's
        `[[x, y], ...]` in metres, or None for no `reach` key.
    @param cameras Checked camera rows (`name` is the role).
    @return The mapping `robots.config.load` reads, in the file's order.
    """
    mapping: dict[str, Any] = {"body": robot.body}
    if robot.embodiment is not None:
        mapping["embodiment"] = robot.embodiment
    if robot.custom is not None:
        mapping["custom"] = {
            "driver": robot.custom.driver,
            "ik": robot.custom.ik,
            "embodiment": robot.custom.embodiment,
        }
    for side, key in (("left", "follower_l"), ("right", "follower_r")):
        arm = arms[side]
        mapping[key] = {
            "can_channel": arm["can_channel"],
            **arm["settings"],
            "home": [round(float(q), 4) for q in arm["home"]],
        }
        if arm["reach"] is not None:
            mapping[key]["reach"] = arm["reach"]
    mapping["cameras"] = [_camera_entry(row) for row in cameras]
    mapping.update(LOOP_KEYS)
    return mapping


def _camera_entry(row: dict[str, Any]) -> dict[str, Any]:
    """One `cameras` entry of the file from a checked row."""
    entry: dict[str, Any] = {"name": row["name"]}
    value = str(row["value"])
    if row["source"] == "device":
        entry["device"] = int(value) if value.isdigit() else value
    else:
        entry[setup_devices.CAMERA_KEYS[row["source"]]] = value
    entry.update({"width": row["width"], "height": row["height"], "fps": row["fps"], "rotate": row["rotate"]})
    return entry


## What a file filled in by hand leaves to its owner (template()): each arm's adapter, an SO-101's calibration files,
## a camera's device, the account embodiment and, on a robot of your own, dream's model of the arm.
TEMPLATE_BLANK = "REPLACE"

## The header's line width in a file filled in by hand (template()): what the page's box shows without a scroll.
TEMPLATE_WIDTH = 96

## How each body's arm adapter is named, for the header of a file filled in by hand (template()).
ADAPTER_HINTS: dict[str, str] = {
    "yam": "its SocketCAN link (`ip link` lists them), e.g. can0",
    "makers": "its CANable, /dev/serial/by-id/... (`ls -l /dev/serial/by-id/`), or its SocketCAN link",
    "metal": "its CANable, /dev/serial/by-id/... (`ls -l /dev/serial/by-id/`), or its SocketCAN link",
    "so101": "its servo bus adapter, /dev/serial/by-id/... (`ls -l /dev/serial/by-id/`)",
    config.CUSTOM_BODY: "whatever your driver opens: a CAN link, a serial path, an IP address",
}


def template(body: str) -> str:
    """The station file of "Set it up myself", to fill in by hand.

    It is what Save writes for that body (build_config()), with every value the steps measure or pick left as
    `REPLACE_...` and a header saying where each comes from. No check runs on it: `dreamcli robot run` and
    `dreamcli robot up` read it as they read a file the wizard wrote.

    @param body A body the page offers: one of #robots.setup_session.robot_choices(), or `custom`.
    @return The file's text.
    @throws ValueError The page offers no such body.
    """
    blank = TEMPLATE_BLANK
    embodiment = f"{blank}_USERNAME/{blank}_EMBODIMENT"
    ours = {robot.body: robot for robot in setup_session.robot_choices()}
    if body == config.CUSTOM_BODY:
        custom = config.CustomConfig(driver="my_robot.py:Arm", ik=f"{blank}_ARM", embodiment=embodiment)
        robot = setup_session.Robot(body, "My own robot", custom.ik, adapter.ZERO_SHOT_CAMERAS, custom=custom)
    elif body == "so101":
        robot = dataclasses.replace(ours[body], roles=adapter.ZERO_SHOT_CAMERAS, embodiment=embodiment)
    elif body in ours:
        robot = ours[body]
    else:
        raise ValueError(f"no station file for {body!r}: pick one of {[*ours, config.CUSTOM_BODY]}")
    start = prefill({}, body, robot.roles)
    arms: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        settings = yaml.safe_load(start["settings"][side]) or {}
        if body == "metal":
            settings.update({"gripper_open_deg": f"{blank}_OPEN_DEG", "gripper_closed_deg": f"{blank}_CLOSED_DEG"})
        if body == "so101":
            settings["calibration"] = f"{blank}_{side.upper()}_CALIBRATION_JSON"
        channel = f"{blank}_{side.upper()}_ARM_ADAPTER"
        arms[side] = {"can_channel": channel, "settings": settings, "home": [], "reach": None}  # a blank below
    cameras = [{**row, "value": f"{blank}_{row['name'].upper()}_CAMERA"} for row in start["cameras"]]
    names = ", ".join(row["name"] for row in cameras)
    notes = [
        f"Station config ({robot.label}), filled in by hand (`dreamcli setup`, Set it up myself): the file the "
        f"wizard's Save writes, with every value its steps measure or pick left as a {blank}_ blank. Nothing "
        "checks it before a run. Every key is documented in the robots README, *Config file*.",
        f"can_channel: each arm's adapter, {ADAPTER_HINTS[body]}.",
        "home: each arm's home pose, a list of one angle per joint in radians (an SO-101's five, then 0); how many "
        "it lists is how many joints the arm has, and its driver refuses another count. A run starts by moving both "
        "arms there. To read it, put a 0 per joint here for `dreamcli robot run --dry-run`, which writes nothing to "
        "the arms, and copy each arm's reading at rest from its log. Replace the zeros before any run without "
        "--dry-run.",
    ]
    if body == "metal":
        notes.append(
            "gripper_open_deg, gripper_closed_deg: the gripper motor's angle, degrees, jaws fully open and closed."
        )
    if body == "so101":
        notes.append("calibration: the arm's LeRobot calibration file, what `lerobot-calibrate` wrote for it.")
    notes.append(
        f"cameras ({names}): each camera's device (`ls -l /dev/v4l/by-path/`, or an rtsp:// address), or `serial` in "
        "its place for a RealSense (`rs-enumerate-devices`), or `source` for a camera class of your own "
        "(module:Class)."
    )
    if robot.embodiment is not None or robot.custom is not None:
        notes.append(
            "embodiment: your account's embodiment, <username>/<name>; the camera names are its video roles. The "
            "wizard's New registers one for Dream Labs' base models, with the cameras scene, wrist_l and wrist_r."
        )
    if robot.custom is not None:
        notes.append(
            "custom.driver: your driver class, file.py:Class beside this file or package.module:Class. custom.ik: "
            "dream's model of your arm (the wizard's list), or for an arm from its URDF the arm file `dreamcli tcp` "
            "writes: it takes the URDF and the tool point on a 3D drawing and prints the path."
        )
    notes.append(
        "Live view needs `live: {robot_id: ...}`, the robot the wizard's Save registers; else run in a terminal."
    )
    wrapped = [
        textwrap.fill(note, TEMPLATE_WIDTH, initial_indent="# ", subsequent_indent="#   ", break_on_hyphens=False)
        for note in notes[1:]
    ]
    first = textwrap.fill(
        notes[0], TEMPLATE_WIDTH, initial_indent="# ", subsequent_indent="# ", break_on_hyphens=False
    )
    header = "\n".join([first, "#", *wrapped]) + "\n"
    mapping = build_config(robot, arms, cameras)
    for key in ("follower_l", "follower_r"):  # no zeros: a Metal arm at zero stands straight up
        mapping[key]["home"] = f"{blank}_HOME_RAD"
    return header + "\n" + yaml.safe_dump(mapping, sort_keys=False, default_flow_style=None, allow_unicode=True)


def saved_channel(device: dict[str, Any], side: str) -> str:
    """The `can_channel` the file names an arm's device by, one that stays the same.

    @param device The arm's `robots.setup_devices.arm_adapters` entry.
    @param side `left` or `right`.
    @return The channel.
    """
    if device["kind"] == "can" and device["usb"]:
        return setup_devices.CAN_LINKS[side]
    return str(device["open"])


def kept_robot_id(path: pathlib.Path) -> str | None:
    """The robot the target file already names, so a re-save keeps the station's identity.

    The page does not show it, so without this a re-save to change a camera would register the station as a second
    robot. A file that is missing, not YAML, or without a `live.robot_id` gives None.

    @param path The config file Save is about to replace.
    @return The `live.robot_id`, or None.
    """
    raw = _read_existing(path)
    live = _section(raw, "live")
    return str(live["robot_id"]) if live.get("robot_id") else None


def register_robot(
    api_key: str, platform_url: str, robot_id: str | None, name: str, embodiment: str
) -> dict[str, Any]:
    """The robot this station is on the platform: the one the file names, else one named after the box, else new.

    `GET /v1/robots` with the key lists the account's robots. The robot `robot_id` names is reused, so a re-save
    keeps the station's identity; else one with this `name` and `embodiment`, so a file written again from scratch
    does not add a second robot with the box's name (two robots with that name and embodiment is an error: the box
    cannot tell which is its own); else `POST /v1/robots` registers one. Every call carries the key
    the station itself will use, so a key the platform refuses is found here, on Save, and not at the first
    inference.

    @param api_key The DreamAPI key.
    @param platform_url The platform, `http(s)://<host>[:<port>]` (the credentials').
    @param robot_id The robot the existing file names, or None.
    @param name The label a new robot gets on the platform.
    @param embodiment The platform's id of the robot's embodiment.
    @return The robot as the platform answers it: `id`, `name`, `embodiment`, `status`, `live`.
    @throws ValueError The platform cannot be reached, refuses the key, or answers something else; the message says.
    """
    origin = platform_url.rstrip("/")
    answer = _platform_call(origin, api_key, "GET", "/v1/robots")
    if not isinstance(answer, list) or not all(isinstance(robot, dict) for robot in answer):
        raise ValueError(f"{origin}/v1/robots answered something other than a list of robots")
    robots: list[dict[str, Any]] = [dict(robot) for robot in answer]
    for robot in robots:
        if robot_id is not None and robot.get("id") == robot_id:
            return robot
    named = [robot for robot in robots if robot.get("name") == name and robot.get("embodiment") == embodiment]
    if len(named) > 1:
        raise ValueError(
            f"{len(named)} robots on {origin} are called {name!r} ({', '.join(str(r.get('id')) for r in named)}), so "
            "this box cannot tell which is its own; delete the extra ones on the platform, or put the right id in "
            "station.yaml under live: robot_id"
        )
    if named:
        return named[0]
    created = _platform_call(origin, api_key, "POST", "/v1/robots", {"name": name, "embodiment": embodiment})
    if not isinstance(created, dict) or not created.get("id"):
        raise ValueError(f"{origin}/v1/robots registered the robot but answered no id")
    return dict(created)


def build_warning(platform_url: str, ours: str | None) -> str | None:
    """Says when this computer's dream library is another build than the one the platform runs.

    It compares the commit its wheel was built from (#robots.COMMIT) against `commit` of the platform's
    `GET /v1/meta`, so a page from an older install is not taken for the platform's.

    @param platform_url The platform, `http(s)://<host>[:<port>]` (the credentials').
    @param ours This install's commit, its first 12 characters; None from a source checkout, which asks nothing.
    @return The warning; None when the two match or the platform does not say its commit (`PLATFORM_SETTINGS_COMMIT`
        off).
    @throws OSError, ValueError The platform could not be asked, or answered no JSON.
    """
    if ours is None:
        return None
    with urllib.request.urlopen(f"{platform_url.rstrip('/')}/v1/meta", timeout=REGISTER_TIMEOUT_S) as reply:
        theirs: str | None = json.load(reply)["commit"]
    # The wheel's version keeps the commit's letters and digits only, the first 12, as the platform's build does.
    if theirs is None or re.sub(r"[^A-Za-z0-9]", "", theirs)[:12] == ours:
        return None
    return (
        f"setup wizard: warning: this computer's dream library was built from {ours}, but {platform_url} runs "
        f"{theirs[:12]}: install it again (pip install --upgrade dreamlabs), then run dreamcli setup again"
    )


def _platform_call(origin: str, api_key: str, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    """One JSON request to the platform with the key as bearer; every failure is a `ValueError` naming the cause."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}
    request = urllib.request.Request(origin + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=REGISTER_TIMEOUT_S) as reply:
            return json.loads(reply.read() or b"null")
    except urllib.error.HTTPError as error:
        detail = _error_detail(error)
        if error.code == 401:
            raise ValueError(f"{origin} refused the api key: {detail}") from error
        if error.code >= 500:  # The platform is down, not the request wrong; say so and when to retry
            raise ValueError(f"{origin} is unavailable ({error.code}: {detail}); try again in a minute") from error
        raise ValueError(f"{origin}{path} answered {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise ValueError(f"could not reach {origin} to register the robot: {error.reason}") from error
    except (OSError, ValueError) as error:
        raise ValueError(f"could not reach {origin} to register the robot: {error}") from error


def _error_detail(error: urllib.error.HTTPError) -> str:
    """What the platform said in a refusal: its envelope's message, else the start of the body, else the reason."""
    raw = error.read()
    try:
        return str(json.loads(raw)["error"]["message"])
    except (ValueError, KeyError, TypeError):
        return raw.decode("utf-8", "replace")[:200] or str(error.reason)


def render_yaml(mapping: dict[str, Any]) -> str:
    """The file text: #FILE_HEADER, then the mapping in its own order.

    @param mapping The config mapping.
    @return The YAML text.
    """
    return FILE_HEADER + "\n" + yaml.safe_dump(mapping, sort_keys=False, default_flow_style=None, allow_unicode=True)


@dataclasses.dataclass(frozen=True)
class Saved:
    """What Save did: the file it wrote, the robot the station is, and whether the file links it to the live view.

    Attributes:
    - text: The YAML text written.
    - robot: The platform's robot (register_robot()): `id`, `name`, `embodiment`, `status`, `live`.
    - platform: The platform the robot is registered on.
    - live_missing: The `live` extra's modules that are not installed. Empty means the file carries a `live`
      section; otherwise it carries none and the station runs without the live view until the extra is installed
      and the file is saved again.
    """

    text: str
    robot: dict[str, Any]
    platform: str
    live_missing: tuple[str, ...]


def _camera_names(embodiment_config: Any) -> list[str]:
    """The camera roles of an account embodiment's config: its `video` keys.

    An embodiment registered with no settings (an empty or missing config, as a test from the platform website left on
    an account) has none, and is listed with no cameras rather than stopping the page.

    @param embodiment_config An account embodiment's `embodiment_config` (`dream.datasets.Embodiment`), as the data
        service answered it.
    @return The camera roles, in the config's order; empty when there are none.
    """
    video = embodiment_config.get("video") if isinstance(embodiment_config, dict) else None
    return list(video) if isinstance(video, (dict, list)) else []


def save(mapping: dict[str, Any], path: pathlib.Path, account: credentials.Credentials, embodiment: str) -> Saved:
    """Registers the station on the platform, then writes `mapping` to `path` once the agent accepts it.

    @param mapping build_config()'s mapping.
    @param path The target.
    @param account The credentials; `api_key` is set (main() refuses to serve without one).
    @param embodiment The platform's id of the robot's embodiment, what it registers as.
    @return The #Saved.
    @throws ValueError The platform refused the key or could not be reached.
    @throws ConfigError The agent would refuse this file; the message says.
    """
    robot = register_robot(
        account.api_key or "",
        account.platform_url,
        kept_robot_id(path),
        setup_access.robot_label(),
        embodiment,
    )
    old = _read_existing(path)
    if old.get("body", config.BODY_NAMES[0]) != mapping["body"]:
        old = {}
    kept = {key: value for key, value in old.items() if key in config.APP_KEYS and key not in PAGE_KEYS}
    mapping = {**mapping, **kept}
    live_missing = tuple(module for module in LIVE_EXTRA_MODULES if importlib.util.find_spec(module) is None)
    if not live_missing:
        live = {k: v for k, v in _section(old, "live").items() if k in config.LIVE_KEYS and k != "robot_id"}
        if isinstance(live.get("cameras"), list):
            names = [camera["name"] for camera in mapping["cameras"]]
            live["cameras"] = [name for name in live["cameras"] if name in names]
            if not live["cameras"]:
                del live["cameras"]
        mapping["live"] = {"robot_id": robot["id"], **live}
    text = render_yaml(mapping)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        loaded = config.load(tmp)
        if loaded.max_stall_ticks != mapping["max_stall_ticks"]:
            mapping["max_stall_ticks"] = loaded.max_stall_ticks
            text = render_yaml(mapping)
            tmp.write_text(text, encoding="utf-8")
        run.policy_config(loaded)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    _log.info("setup: wrote %s; robot %s (%s) on %s", path, robot["id"], robot.get("name"), account.platform_url)
    return Saved(text=text, robot=robot, platform=account.platform_url, live_missing=live_missing)


def prefill(raw: dict[str, Any], body: str, roles: tuple[str, ...]) -> dict[str, Any]:
    """What the page starts from for a body: the existing file's arm settings and cameras when it is that body's.

    @param raw The existing file's mapping (`{}` when there is none).
    @param body The body picked.
    @param roles Its camera roles.
    @return `{"settings": {side: YAML text}, "cameras": [rows by role]}`.
    """
    same = raw.get("body", config.BODY_NAMES[0]) == body
    settings = {}
    for side, key in (("left", "follower_l"), ("right", "follower_r")):
        kept = {k: v for k, v in _section(raw, key).items() if k not in WIZARD_KEYS}
        section = kept if same else {}
        settings[side] = render_settings(section) if section else DEFAULT_SETTINGS.get(body, "")
    rows = raw.get("cameras") if same else None
    by_name = {row.get("name"): row for row in rows if isinstance(row, dict)} if isinstance(rows, list) else {}
    by_kind = {
        setup_cameras.ROLE_KINDS[name]: row for name, row in by_name.items() if name in setup_cameras.ROLE_KINDS
    }
    cameras = []
    for role in roles:
        kind = setup_cameras.ROLE_KINDS.get(role, role)
        width, height, fps = CAMERA_SIZES.get(body, {}).get(kind, CUSTOM_CAMERA_SIZE)
        entry = by_name.get(role) or by_kind.get(kind, {})
        source = next((source for source, key in setup_devices.CAMERA_KEYS.items() if key in entry), "device")
        value = entry.get(setup_devices.CAMERA_KEYS[source])
        cameras.append(
            {
                "name": role,
                "source": source,
                "value": "" if value is None else str(value),
                "width": entry.get("width", width),
                "height": entry.get("height", height),
                "fps": entry.get("fps", fps),
                "rotate": entry.get("rotate", 0),
            }
        )
    return {"settings": settings, "cameras": cameras}


def render_settings(section: dict[str, Any]) -> str:
    """An arm's settings as the YAML the page edits, rows inline.

    @param section The arm's keys other than the device and the home pose.
    @return The YAML text.
    """
    return yaml.safe_dump(section, sort_keys=False, default_flow_style=None)


def parse_settings(text: Any, side: str) -> dict[str, Any]:
    """Reads one arm's settings YAML from the page.

    @param text The YAML text.
    @param side `left` or `right`, for messages.
    @return The mapping (empty for blank text).
    """
    if not isinstance(text, str):
        raise ValueError(f"the {side} arm's settings must be text")
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"the {side} arm's settings are not YAML: {exc}") from exc
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"the {side} arm's settings must be keys and values")
    for key in WIZARD_KEYS:
        if key in parsed:
            raise ValueError(f"the {side} arm's settings set '{key}'; the wizard sets that itself")
    return parsed


def _pick_reply(robot: setup_session.Robot, rows: dict[str, Any]) -> dict[str, Any]:
    """The pick's reply; `cameras` is each role with its kind and the labels the page shows (camera_roles())."""
    return {
        "robot": dataclasses.asdict(robot),
        "cameras": [dataclasses.asdict(cam) for cam in setup_cameras.camera_roles(robot.roles).values()],
        "prefill": rows,
        "joints": list(kinematics.arm_layout(robot.ik_arm).joints),
        "reach": setup_session.reaches(robot),
    }


def _embodiment_note(body: dict[str, Any], embodiment: Embodiment, reused: bool) -> str | None:
    """What the page says about the embodiment a pick asked to be new (`embodiment` null, _picked_embodiment()).

    @param body The pick's `custom` part, or the whole body on an SO-101.
    @param embodiment The embodiment the pick landed on.
    @param reused Whether it was the account's already, from an earlier setup.
    @return That setup registered it, or used the account's own rather than a second one; None for one picked from the
        account's list.
    """
    if body.get("embodiment") is not None:
        return None
    if reused:
        return (
            f"Your account already has {embodiment.name}, the embodiment for this arm with these cameras, "
            "from an earlier setup. Setup uses it rather than registering a second one."
        )
    return f"Registered a new embodiment, {embodiment.name}, on your account."


def account_client(account: credentials.Credentials) -> dream.DreamClient:
    """A dream client with the account's key: the platform lists its models, the data service its embodiments.

    @param account The credentials.
    @return The client.
    """
    return dream.DreamClient(account.api_key)


class SetupServer:
    """The wizard's HTTP server on its own thread, its session, its cameras, and the reader thread.

    Attributes:
    - host: Bind address.
    - port: Bound port; the actual one once start() has bound (0 asks the OS for a free port).
    - config_path: The file Save writes.
    - account: The credentials Save registers with and the page shows.
    - token: The secret a request must carry (#robots.setup_access.admitted), or None for no check.
    - done: Set when the page asks the wizard to exit.
    - saved: What the last Save this session wrote, or None while nothing was saved.
    - agent_running: Whether the station agent is running now: it owns the arms and the cameras.
    - session: The arms and what the page has settled (#robots.setup_session.Session).
    - cameras: The cameras held open by role (#robots.setup_cameras.CameraBank).
    - dataset: The operator's LeRobot dataset, or None for our samples.
    - dataset_keys: Per role, the dataset's video key.

    @see start(), stop()
    """

    def __init__(
        self,
        config_path: pathlib.Path,
        account: credentials.Credentials,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        agent_running: Callable[[], bool] | None = None,
        token: str | None = None,
        open_arm: setup_session.Opener | None = None,
        model_factory: Callable[[str], setup_model.ArmModel] | None = None,
        open_camera: Callable[[config.CameraConfig], Any] | None = None,
        scan: Callable[[], dict[str, Any]] | None = None,
        gripper_mode: Callable[[str], str | None] | None = None,
        gripper_to_mit: Callable[[str], str] | None = None,
        held_calibration: Callable[[str], dict[str, Any]] | None = None,
        preview: Callable[..., bytes] | None = None,
        ask_models: Callable[[], list[Model]] | None = None,
        ask_embodiments: Callable[[], list[Embodiment]] | None = None,
        add_embodiment: Callable[[str, dict[str, Any]], Embodiment] | None = None,
        ask_training_data: Callable[[str], dict[str, Any]] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = time.sleep,
    ) -> None:
        """Binds the target file and the account; nothing listens until start().

        @param config_path The file Save writes.
        @param account The credentials (`dream.credentials.load()`), with a key.
        @param host Bind address.
        @param port Port; 0 for an ephemeral one (tests).
        @param agent_running Asked when a robot is picked.
        @param token The secret every request must carry.
        @param open_arm How an arm is opened (`robots.setup_session.open_body_arm`); tests pass fakes.
        @param model_factory `ik arm -> ArmModel`; tests may hand back one model built once.
        @param open_camera `CameraConfig -> started reader` (`robots.setup_cameras.open_camera_reader`); tests pass
            stand-ins.
        @param scan What the box can see (`robots.setup_devices.scan`); tests pass a stand-in.
        @param gripper_mode A YAM arm's gripper motor mode by its CAN link (read_gripper_mode()); tests pass one.
        @param gripper_to_mit Switches a YAM arm's gripper motor to MIT on its CAN link (switch_gripper_to_mit());
            tests pass one.
        @param held_calibration The calibration an SO-101 arm's servos hold, by its serial port
            (`robots.feetech.held_calibration`); tests pass one.
        @param preview One picture from a camera (`robots.setup_devices.preview_jpeg`); tests pass a stand-in.
        @param ask_models What the account may run.
        @param ask_embodiments The account's own embodiments.
        @param add_embodiment Registers an embodiment on the account, `(name, config) -> Embodiment`.
        @param ask_training_data Where one of the account's fine-tunes' training data is.
        @param clock The session's monotonic clock; tests pass one that moves only when slept on.
        @param sleep The powered checks' sleep; tests pass one that does not wait.
        """
        self.host = host
        self.port = port
        self.config_path = config_path
        self.account = account
        self.token = token
        self.done = threading.Event()
        self.saved: Saved | None = None
        self.agent_running: Callable[[], bool] = agent_running or (lambda: False)
        self.scan = scan or setup_devices.scan
        self.gripper_mode = gripper_mode or read_gripper_mode
        self.gripper_to_mit = gripper_to_mit or switch_gripper_to_mit
        self.held_calibration = held_calibration or (lambda port: feetech.held_calibration(port, so101.JOINTS))
        self.preview = preview or setup_devices.preview_jpeg
        self._ask_models = ask_models or (lambda: account_client(account).get_models())
        self._ask_embodiments = ask_embodiments or (lambda: account_client(account).get_embodiments())
        self._add_embodiment = add_embodiment or (
            lambda name, wire: account_client(account).add_embodiment(name, wire)
        )
        self._ask_training_data = ask_training_data or (
            lambda model: fine_tuning.training_data(model, api_key=account.api_key)
        )
        self._models: list[Model] | None = None
        self._embodiments: list[Embodiment] | None = None
        self.existing = _read_existing(config_path)
        self.session = setup_session.Session(
            opener=open_arm or setup_session.open_body_arm,
            model_factory=model_factory or setup_model.ArmModel,
            old_channels=_old_channels(self.existing),
            clock=clock,
            sleep=sleep,
            # Off unless the station file already says so: a file written by Save keeps the key.
            home_before_off=self.existing.get("home_before_motors_off") is True,
        )
        self.cameras = setup_cameras.CameraBank(
            open_camera or setup_cameras.open_camera_reader, config.AppConfig.policy_image_size
        )
        self.dataset: setup_cameras.Dataset | None = None
        self.dataset_keys: dict[str, str | None] = {}
        self._fetched = False
        self.reference_note: str | None = None
        self.reference_failure: dict[str, str] | None = None
        self.devices: list[dict[str, Any]] = []
        self.typed: dict[str, list[dict[str, Any]]] = {"arms": [], "cameras": []}
        self.reader_error: str | None = None
        self._saving = threading.Lock()
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None
        self._reader: threading.Thread | None = None

    def start(self) -> None:
        """Binds the socket, serves on a daemon thread, and starts reading the arms the page opens.

        @throws OSError The port is taken.
        """
        self._server = _Server((self.host, self.port), _Handler, self)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, name="setup-wizard", daemon=True)
        self._thread.start()
        self._reader = threading.Thread(target=self._read_arms, name="setup-arms", daemon=True)
        self._reader.start()
        _log.info("setup wizard listening on %s:%d", self.host, self.port)

    def stop(self) -> None:
        """Stops serving, lets the arms go (#robots.setup_session.Session.close), releases the cameras."""
        self.done.set()
        self.session.close()
        self.cameras.release()
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        for thread in (self._thread, self._reader):
            if thread is not None:
                thread.join(5.0)
        self._server = None
        self._thread = None
        self._reader = None

    def _read_arms(self) -> None:
        """The reader thread: reads every disengaged arm at `robots.setup_session.READ_HZ` until the wizard ends."""
        while not self.done.wait(1.0 / setup_session.READ_HZ):
            try:
                self.session.tick()
                self.reader_error = None
            except Exception as exc:
                _log.exception("setup: reading the arms failed")
                self.reader_error = f"{type(exc).__name__}: {exc}"

    def state(self) -> dict[str, Any]:
        """What `/api/state` answers.

        The robots, the target file, the account, the arms a robot of your own picks from
        (#robots.setup_session.own_arms()), the folder its driver's file picker opens in
        (#robots.setup_files.start()), the robot picked so far, this install's version (#robots.__version__,
        shown in the page's footer), and `tool`: `wizard` here, `tcp` for `dreamcli tcp` (TcpServer.state()).

        `station_name` is the name Save registers the station under (the box's hostname, register_robot()).

        @return The dict; the key and the model are never sent to the page.
        """
        existing_ik = _section(self.existing, "custom").get("ik")
        try:
            ik_arms = setup_session.own_arms(existing_ik)
            ik_error = None
        except errors.ConfigError as exc:
            ik_arms, ik_error = [], str(exc)
        calibrations = sorted({*setup_session.so101_calibration_files(), *_old_calibrations(self.existing)})
        matches = self._so101_matches([str(self.config_path.parent / path) for path in calibrations])
        return {
            "config_path": str(self.config_path),
            "exists": self.config_path.is_file(),
            "existing_body": self.existing.get("body", config.BODY_NAMES[0]) if self.existing else None,
            "existing_embodiment": self.existing.get("embodiment")
            or _section(self.existing, "custom").get("embodiment"),
            "existing_driver": _section(self.existing, "custom").get("driver"),
            "existing_ik": existing_ik,
            "driver_dir": str(setup_files.start(self.config_path)),
            "robots": [dataclasses.asdict(robot) for robot in setup_session.robot_choices()],
            "so101_calibrations": [
                {
                    "path": path,
                    "modified": _modified(self.config_path.parent / path),
                    "matches": matches.get(str(self.config_path.parent / path), []),
                }
                for path in calibrations
            ],
            "so101_calibration_dir": _home_short(setup_session.SO101_CALIBRATION_DIR),
            "not_answering_help": setup_session.NOT_ANSWERING_HELP,
            "picked": self._picked(),
            "ik_arms": ik_arms,
            "ik_error": ik_error,
            "connection": {"gateway_url": self.account.gateway_url, "platform_url": self.account.platform_url},
            "credentials_path": str(self.account.path),
            "agent_running": self.agent_running(),
            "station_name": socket.gethostname(),
            "gripper_switch": setup_gripper_mit(),
            "so101_match": setup_so101_match(),
            # Saved this run: a page opened again goes back to Save station rather than to the steps before it.
            "saved": None
            if self.saved is None
            else {"robot": self.saved.robot.get("name"), "platform": self.saved.platform},
            "dataset": self.dataset_in_use(),
            "version": robots.__version__,
            "tool": "wizard",
        }

    def _so101_matches(self, files: list[str]) -> dict[str, list[str]]:
        """Per SO-101 calibration file, the serial adapters plugged in whose servos hold it
        (#robots.setup_session.so101_matches), so step 0 can say which file is which arm's. Nothing is read when there
        is no file, while the wizard holds arms open, or while the station agent runs (a second reader on its bus
        would take status packets its arms wait for); a scan that fails matches nothing."""
        if not files or self.session.bank.devices() or self.agent_running():
            return {}
        try:
            adapters = self.scan()["arms"]
        except Exception as exc:  # the page still lists the files, unticked
            _log.info("no arm adapters scanned for the SO-101 calibration files: %s", exc)
            return {}
        return setup_session.so101_matches(files, adapters)

    def _picked(self) -> dict[str, Any] | None:
        robot = self.session.robot
        if robot is None:
            return None
        rows = prefill(self.existing, robot.body, robot.roles)
        held = self.cameras.rows()
        rows["cameras"] = [held[row["name"]] if row["name"] in held else row for row in rows["cameras"]]
        return _pick_reply(robot, rows)

    def status(self) -> dict[str, Any]:
        """What `/api/status` answers: the session, the wiggle while arms are open, the cameras and the references.

        @return The dict.
        """
        robot = self.session.robot
        reply = self.session.status()
        reply["wiggle"] = self.session.wiggle() if self.session.bank.devices() else None
        reply["now"] = setup_view.now(self.session)
        reply["reader_error"] = self.reader_error
        reply["cameras"] = {
            "open": self.cameras.roles(),
            "states": self.cameras.states(),
            "references": self._references(robot),
            "reference_note": self.reference_note,
            "reference_failure": self.reference_failure,
        }
        return reply

    def choose(self, body: Any) -> dict[str, Any]:
        """Picks the robot; the file's settings and cameras for it come back to prefill the page.

        Its cameras take its embodiment's names: the body's own on a YAM, Maker or Metal Arm
        (#robots.setup_session.robot_choices()), the account embodiment's `video` roles on an SO-101 or a robot of
        your own.

        @param body `{"body"}` for one of ours, with `calibration` and `embodiment` on an SO-101; `{"custom"}`. An
            `embodiment` of null is a new one (_picked_embodiment()).
        @return `{"robot", "prefill", "joints", "reused", "embodiment_note"}`; `reused`: a new embodiment was the
            account's already; `embodiment_note`: what the page says about a new embodiment (_embodiment_note()).
        @throws ConfigError The arm's URDF cannot be built, or an SO-101 calibration file cannot be read.
        """
        self._require_agent_down()
        reused, note = False, None
        if isinstance(body.get("custom"), dict):
            custom = body["custom"]
            ik = _text(custom, "custom.ik_arm", "ik_arm")
            embodiment, reused = self._picked_embodiment(custom, ik)
            note = _embodiment_note(custom, embodiment, reused)
            robot = setup_session.custom_robot(ik, _text(custom, "custom.driver", "driver"), embodiment.name)
            robot = dataclasses.replace(robot, embodiment_id=embodiment.id)
        elif body.get("body") == "so101":
            files = body.get("calibration")
            if not isinstance(files, list) or not all(isinstance(path, str) for path in files):
                raise ValueError("'calibration' must be a list of file paths")
            embodiment, reused = self._picked_embodiment(body, so101.ARM.ik)
            note = _embodiment_note(body, embodiment, reused)
            robot = setup_session.so101_robot(files, embodiment.name)
            robot = dataclasses.replace(robot, embodiment_id=embodiment.id)
        else:
            robot = _our_robot(_text(body, "body"))
        self.cameras.release()
        self.session.choose(robot)
        if self.dataset is None or self._fetched:
            self._fetch_training_data(robot)
        else:
            self.dataset_keys = self.dataset.guess_keys(setup_cameras.camera_roles(robot.roles))
        reply = _pick_reply(robot, prefill(self.existing, robot.body, robot.roles))
        return {**reply, "reused": reused, "embodiment_note": note}

    def _fetch_training_data(self, robot: setup_session.Robot) -> None:
        """References from the data the account's newest fine-tune on the robot's embodiment was trained on.

        A YAM, and a robot whose embodiment the account has no fine-tune of, keep our YAM pictures; so does a robot
        whose models cannot be listed, and the page says why.
        """
        self.dataset, self.dataset_keys, self._fetched, self.reference_note = None, {}, False, None
        self.reference_failure = None
        if robot.body == "yam":
            return
        try:
            models = self.account_models()
        except ValueError as exc:
            self.reference_note = f"{exc}; showing our YAM pictures."
            return
        mine = [m for m in models if m.owner == "mine" and m.embodiment == robot.embodiment_id]
        # The platform dates every fine-tune of the account's own; a None would only sort it oldest.
        oldest = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
        model = max(mine, key=lambda m: m.created_at or oldest, default=None)
        if model is None:
            return
        try:
            found = self._ask_training_data(model.id)
            self.dataset = setup_cameras.training_dataset(found["url"])
        except (dream.DreamError, OSError, ValueError, zipfile.BadZipFile) as exc:
            self.reference_note = (
                f"Could not read the data {model.id} was trained on ({exc}); showing our YAM pictures."
            )
            self.reference_failure = {"model": model.id, "error": str(exc)}
            return
        self._fetched = True
        # A role the embodiment does not name (the station kept its own names) is guessed from the keys.
        guessed = self.dataset.guess_keys(setup_cameras.camera_roles(robot.roles))
        self.dataset_keys = {role: found["video"].get(role) or guessed[role] for role in robot.roles}

    def account_models(self) -> list[Model]:
        """What the account may run, asked from the platform the first time a step needs it and kept for the session.

        @return `dream.DreamClient.get_models()`: the public models and the account's fine-tunes.
        """
        if self._models is None:
            try:
                self._models = self._ask_models()
            except dream.DreamError as exc:
                raise ValueError(f"could not list the models your account can run: {exc}") from exc
        return self._models

    def account_embodiments(self) -> list[Embodiment]:
        """The account's own embodiments, asked from the data service the first time a step needs them and kept.

        @return `dream.DreamClient.get_embodiments()`: the account's own embodiments, with their ids.
        """
        if self._embodiments is None:
            try:
                self._embodiments = self._ask_embodiments()
            except dream.DreamError as exc:
                raise ValueError(f"could not list your account's embodiments: {exc}") from exc
        return self._embodiments

    def embodiments(self) -> dict[str, Any]:
        """What `/api/embodiments` answers: the account's embodiments an SO-101 or a robot of your own may be.

        @return `{"embodiments": [{"name", "id", "cameras"}]}`, newest first; `cameras` is empty for one with no
            settings (_camera_names()).
        """
        return {
            "embodiments": [
                {"name": e.name, "id": e.id, "cameras": _camera_names(e.embodiment_config)}
                for e in self.account_embodiments()
            ]
        }

    def read_urdf(self, body: dict[str, Any]) -> dict[str, Any]:
        """Reads the URDF of an arm of your own for the page's URDF step, before anything is written.

        @param body `{"urdf"}`: the URDF's text.
        @return #robots.setup_urdf.read() plus `most_joints`, the most moving joints the loop drives per arm
            (`robots.followers.MAX_JOINTS`), which the page holds the chosen chain to, and `joint_map` (joint_map()).
        @throws ValueError The text is not a URDF whose links hang from one root.
        """
        tree = setup_urdf.read(_text(body, "urdf"))
        return {**tree, "most_joints": followers.MAX_JOINTS, "joint_map": self.joint_map()}

    def joint_map(self) -> dict[str, Any] | None:
        """The URDF joints the robot drives, which the page holds the chain's moving joints to.

        None here: the arm the URDF step builds is its own embodiment's (_zero_shot_embodiment()), so it drives every
        joint of the chain. `dreamcli tcp` reads the station file's (TcpServer.joint_map()).

        @return None.
        """
        return None

    def add_arm(self, body: dict[str, Any]) -> dict[str, Any]:
        """Takes the URDF of an arm of your own, so dream's IK solves it and Dream Labs' base models run on it.

        Writes the URDF and an arm file of it (`dream.arm_file`) to #ARMS_DIR beside the config file, as
        `<name>.urdf` and `<name>.json`, `<name>` the URDF file's name in lowercase letters, digits and dashes, and
        builds dream's model of the arm from them. The arm file's `rest` is the URDF's zero, moved inside the limits of
        a joint whose range does not hold it. The file records what the operator did with the tool point, and when
        (`robots.kinematics.TCP_CALIBRATION`), which every launch checks (`robots.run.check_tool_point()`). A second
        upload of the same name writes the files again. The arm's embodiment is the pick's (_picked_embodiment()).

        @param body `{"filename", "urdf", "base", "flange", "tcp", "approach", "jaw", "tcp_calibration", "fixed"}`:
            the URDF's file name and text, the link the hand's pose is measured from, the link the gripper hangs off,
            the tool's point (metres) and its approach and jaw axes in the flange's frame, `calibrated` (Set tool
            center point, recorded with `dream.arm_file.tool_digest()`, which dream checks at `policy()`) or `skipped`
            (Skip calibration, confirmed), and the chain's joints the operator said the robot holds still (the arm
            file's `fixed`, `dream.arm_file.FIXED_KEY`; written only when there are any).
        @return `{"ik", "joints"}`: the arm file's path (the config's `custom.ik`) and the arm's joints from `base`.
        @throws ValueError A field is missing or malformed.
        @throws ConfigError dream cannot build the arm from the URDF, links and axes given; the message says why.
        """
        name = re.sub(r"[^a-z0-9]+", "-", pathlib.Path(_text(body, "filename")).stem.lower()).strip("-")[:64]
        if not name:
            raise ValueError("name the URDF file with at least one letter or digit")
        urdf, base, flange = _text(body, "urdf"), _text(body, "base"), _text(body, "flange")
        tool = {key: _vector(body, key) for key in ("tcp", "approach", "jaw")}
        status = _text(body, kinematics.TCP_CALIBRATION)
        if status not in kinematics.TCP_STATUSES:
            raise ValueError(
                f"'{kinematics.TCP_CALIBRATION}' must be one of {list(kinematics.TCP_STATUSES)}, got {status!r}"
            )
        fixed = body.get(arm_file.FIXED_KEY)
        if not isinstance(fixed, list) or not all(isinstance(joint, str) for joint in fixed):
            raise ValueError(f"'{arm_file.FIXED_KEY}' must be a list of the chain's joints your robot holds still")
        folder = self.config_path.parent / ARMS_DIR
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.json"
        try:
            _, joints, limits, _ = arm_file._chain(urdf.encode("utf-8"), base, flange, tuple(fixed), path)
        except dream.DreamConfigError as exc:
            raise errors.ConfigError(str(exc)) from exc
        rest = [min(max(0.0, lo), hi) for lo, hi in limits]
        (folder / f"{name}.urdf").write_text(urdf + "\n", encoding="utf-8")
        at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
        arm: dict[str, Any] = {"urdf": f"{name}.urdf", "base": base, "flange": flange, **tool, "rest": rest}
        record = {"status": status, "at": at}
        if status == "calibrated":
            record["digest"] = arm_file.tool_digest(arm, (folder / f"{name}.urdf").read_bytes())
        arm[kinematics.TCP_CALIBRATION] = record
        if fixed:
            arm[arm_file.FIXED_KEY] = fixed
        path.write_text(json.dumps(arm, indent=2) + "\n", encoding="utf-8")
        kinematics.arm_layout(str(path))
        return {"ik": str(path), "joints": list(joints)}

    def _picked_embodiment(self, body: dict[str, Any], ik: str) -> tuple[Embodiment, bool]:
        """The embodiment an SO-101 or a robot of your own is picked on: the account's one `embodiment` names, or with
        `embodiment` null a new one from the operator's own steps (_zero_shot_embodiment()), and if it was reused."""
        if body.get("embodiment") is None:
            return self._zero_shot_embodiment(ik)
        return self._embodiment(_text(body, "embodiment")), False

    def _zero_shot_embodiment(self, ik: str) -> tuple[Embodiment, bool]:
        """The embodiment Dream Labs' base models run on for this arm, on the account.

        Named after the arm (dream's name of it, or an arm file's name, in lowercase letters, digits and dashes), it is
        the wire the arm's joints go out on, with the cameras the camera step's roles name
        (`robots.adapter.zero_shot_embodiment`), or a beta robot's own layout and cameras
        (`robots.beta_robots.embodiment_config`); a fine-tune's data of the robot is uploaded under it too. An
        embodiment of that name the account has is reused when its config is this one, and refused when it is not.

        @param ik dream's name of the arm, or an arm file's path (the config's `custom.ik`).
        @return The account's embodiment, with its id, and whether it was the account's already.
        @throws ValueError The account has another embodiment of that name, or the data service could not register it.
        @throws ConfigError dream's IK has no such arm, or the arm has more joints than the loop drives.
        """
        arm = pathlib.Path(ik).stem if ik.endswith(".json") else ik  # dream registers an arm file under a hash
        name = re.sub(r"[^a-z0-9]+", "-", arm.lower()).strip("-")[:64]
        layout = kinematics.arm_layout(ik)
        beta = beta_robots.by_ik(ik)
        if beta is None:
            wire = adapter.zero_shot_embodiment(layout.width, layout.client_gripper)
        else:
            wire = beta_robots.embodiment_config(beta)
        found = next((e for e in self.account_embodiments() if e.name.partition("/")[2] == name), None)
        if (
            found is not None
            and layout.client_gripper
            and "right_gripper" not in (found.embodiment_config or {}).get("state", {})
        ):
            # One New registered before it named the grippers: the layout with them goes beside it.
            name = f"{name}-gripper"
            found = next((e for e in self.account_embodiments() if e.name.partition("/")[2] == name), None)
        if found is not None:
            if {key: (found.embodiment_config or {}).get(key) for key in wire} != wire:
                cameras = ", ".join(_camera_names(found.embodiment_config)) or "none"
                raise ValueError(
                    f"your account already has an embodiment {found.name} of another layout (cameras {cameras}): "
                    "pick it in the list to use it; a URDF of another arm needs a file name of its own"
                )
            return found, True
        try:
            created = self._add_embodiment(name, wire)
        except dream.DreamError as exc:
            raise ValueError(f"could not register the embodiment {name!r} on your account: {exc}") from exc
        self._embodiments = [created, *self.account_embodiments()]
        return created, False

    def _embodiment(self, name: str) -> Embodiment:
        embodiments = self.account_embodiments()
        found = next((e for e in embodiments if e.name == name.strip()), None)
        if found is None:
            names = ", ".join(e.name for e in embodiments) or "none yet"
            raise ValueError(f"your account has no embodiment {name!r}; it has: {names}")
        if found.id is None:
            raise ValueError(f"the data service did not give the id of {found.name}; update the dream client")
        return found

    def scan_devices(self) -> dict[str, Any]:
        """What `/api/scan` answers: the devices that can be the picked robot's arms, every other one, the cameras.

        @return `robots.setup_devices.scan()` with `arms` cut to the devices that fit the robot, and the arms and
            cameras the operator typed (type_arm(), type_camera()) after the ones found.
        """
        found = self.scan()
        robot = self.session.robot
        arms = [device for device in found["arms"] if robot is None or setup_session.fits(robot, device)]
        others = [device for device in found["arms"] if device not in arms]
        if robot is not None and robot.body == "yam":
            arms = [self._with_gripper_mode(device) for device in arms]
        self.devices = arms + self.typed["arms"]
        return {**found, "arms": self.devices, "cameras": found["cameras"] + self.typed["cameras"], "others": others}

    def _with_gripper_mode(self, device: dict[str, Any]) -> dict[str, Any]:
        """A YAM adapter with its gripper motor's mode, so the page can show the fix for one not in MIT mode.

        Read only while setup does not hold the adapter open: an open arm's link carries its readings.

        @param device A scanned arm device.
        @return The device with `gripper_mode` (a #robots.motor_mode.MODES name, or None when unread).
        """
        if device.get("kind") != "can" or device["id"] in self.session.bank.devices():
            return device
        return {**device, "gripper_mode": self.gripper_mode(device["open"])}

    def type_arm(self, body: Any) -> dict[str, Any]:
        """Adds an arm the scan does not list, by the address its driver opens; the page ticks it like a found one.

        @param body `{"address"}`.
        @return scan_devices() after.
        @throws ValueError The address is empty or not one line.
        """
        device = setup_devices.typed_arm(_text(body, "address"))
        if device not in self.typed["arms"]:
            self.typed["arms"].append(device)
        return self.scan_devices()

    def so101_match(self) -> dict[str, Any]:
        """Which SO-101 calibration file is which plugged-in arm's, so the page need not leave it to a guess.

        Each SO-101 adapter's servos are read once for the calibration they hold, writing no register, unless setup
        holds the adapter open already; each file offered is parsed as a LeRobot calibration and compared with each.

        @return `{"files": [{"path", "valid", "why", "matches"}], "adapters": [{"id", "label", "error"}]}`: `valid`
            false with `why` for a file that is no SO-101 calibration, `matches` the ids of the adapters whose servos
            hold it; `error` for an adapter that could not be read.
        """
        held: dict[str, dict[str, Any]] = {}
        adapters = []
        for device in self.scan()["arms"]:
            if device.get("kind") != "serial":
                continue
            error = None
            if device["id"] in self.session.bank.devices():
                error = "setup holds this arm open; click Start over in Connect arms to check it"
            else:
                try:
                    held[device["id"]] = self.held_calibration(device["open"])
                except Exception as exc:  # noqa: BLE001 - one arm that cannot be read leaves the others' answers
                    error = str(exc)
            adapters.append({"id": device["id"], "label": device["label"], "error": error})
        files = []
        for path in dict.fromkeys([*setup_session.so101_calibration_files(), *_old_calibrations(self.existing)]):
            try:
                calibration = feetech.read_calibration(path, so101.JOINTS)
            except (errors.RobotsError, OSError, ValueError) as exc:
                files.append({"path": path, "valid": False, "why": str(exc), "matches": []})
                continue
            matches = [id_ for id_, servos in held.items() if not feetech.calibration_differences(calibration, servos)]
            files.append({"path": path, "valid": True, "why": None, "matches": matches})
        return {"files": files, "adapters": adapters}

    def switch_gripper(self, body: Any) -> dict[str, Any]:
        """Switches one YAM arm's gripper motor to MIT mode, letting go of every arm first.

        Every arm is let go first so setup's own reads of the link cannot answer for the motor.

        @param body `{"device"}`: a scanned adapter's id.
        @return `{"mode"}`: what the motor's reply to the write reported; it runs that mode after a power cycle.
        @throws ValueError The device is not a scanned CAN adapter, or the motor did not answer.
        """
        device_id = _text(body, "device")
        device = next((d for d in self.devices if d["id"] == device_id and d.get("kind") == "can"), None)
        if device is None:
            raise ValueError(f"{device_id!r} is not a CAN adapter setup found; click Find adapters again")
        self.session.release_arms()
        return {"mode": self.gripper_to_mit(device["open"])}

    def type_camera(self, body: Any) -> dict[str, Any]:
        """Adds a camera the scan does not list: an address OpenCV opens, or a camera class of your own.

        @param body `{"address"}`.
        @return scan_devices() after.
        @throws ValueError The address is empty.
        """
        camera = setup_devices.typed_camera(_text(body, "address"))
        if camera not in self.typed["cameras"]:
            self.typed["cameras"].append(camera)
        return self.scan_devices()

    def open_arms(self, body: Any) -> dict[str, Any]:
        """Opens the devices the page ticked, disengaged, with the settings it gives.

        @param body `{"devices": [ids], "settings": {"left": text, "right": text}}`.
        @return Per device id, None when it answered, else why not.
        @throws ValueError A device is not one the last scan found, or a setting is not YAML.
        @throws ConfigError A setting is wrong for the body.
        """
        wanted = body.get("devices")
        if not isinstance(wanted, list) or not wanted:
            raise ValueError("'devices' must list at least one device")
        if not self.devices:
            self.scan_devices()
        known = {device["id"]: device for device in self.devices}
        missing = [ident for ident in wanted if ident not in known]
        if missing:
            raise ValueError(f"devices {missing} are not plugged in now; click Find adapters again")
        texts = _mapping(body, "settings")
        settings = {side: parse_settings(texts.get(side, ""), side) for side in ("left", "right")}
        return self.session.open_arms([known[ident] for ident in wanted], settings)

    def open_cameras(self, body: Any) -> dict[str, Any]:
        """Holds every role's camera open for the live feeds, releasing any other.

        @param body `{"cameras": [rows]}`, one per role of the robot.
        @return Per role, its frame size and shape warning (CameraBank.open()).
        @throws ValueError A row is malformed or a role is missing, or a camera delivers nothing.
        """
        rows = self._camera_rows(body)
        self.cameras.release()
        return {row["name"]: self.cameras.open(row) for row in rows}

    def retry_references(self) -> dict[str, Any]:
        """Reads the newest fine-tune's training data again after it failed (the camera step's Try again).

        @return `{"failure"}`: what failed this time, or None once the data is read.
        @throws ValueError No robot is picked yet.
        """
        robot = self.session.robot
        if robot is None:
            raise ValueError("pick a robot first")
        self._fetch_training_data(robot)
        return {"failure": self.reference_failure}

    def set_dataset(self, body: Any) -> dict[str, Any]:
        """Takes the operator's dataset for the references, or drops it for our samples.

        @param body `{"path"}`; an empty path drops it.
        @return `{"path", "video_keys", "episodes", "keys"}`, or `{"path": None}` once dropped.
        @throws ValueError The path is not a LeRobot v2.1 dataset.
        """
        raw = body.get("path")
        if not isinstance(raw, str):
            raise ValueError("'path' must be a string")
        self._fetched, self.reference_note, self.reference_failure = False, None, None
        if not raw.strip():
            self.dataset, self.dataset_keys = None, {}
            return {"path": None}
        dataset = setup_cameras.Dataset(pathlib.Path(raw.strip()).expanduser())
        self.dataset = dataset
        robot = self.session.robot
        self.dataset_keys = dataset.guess_keys(setup_cameras.camera_roles(robot.roles) if robot else {})
        return self.dataset_in_use()

    def dataset_in_use(self) -> dict[str, Any]:
        """The operator's dataset as set_dataset() answered it, for a page opened again after one was given.

        @return `{"path", "fetched", "video_keys", "episodes", "keys"}`, `fetched` true for a fine-tune's own training
            data rather than a folder the operator gave; or `{"path": None}` with none.
        """
        if self.dataset is None:
            return {"path": None}
        return {
            "path": str(self.dataset.path),
            "fetched": self._fetched,
            "video_keys": self.dataset.video_keys,
            "episodes": len(self.dataset.episodes),
            "keys": self.dataset_keys,
        }

    def set_dataset_keys(self, body: Any) -> dict[str, Any]:
        """Says which of the dataset's video keys is each role.

        @param body `{"keys": {role: key or null}}`.
        @return The keys now.
        @throws ValueError There is no dataset, or a key is not one of its cameras.
        """
        if self.dataset is None:
            raise ValueError("give your dataset's path first")
        keys = _mapping(body, "keys")
        for role, key in keys.items():
            if key is not None and key not in self.dataset.video_keys:
                raise ValueError(f"the dataset has no camera {key!r}; it has {self.dataset.video_keys}")
            self.dataset_keys[str(role)] = key
        return {"keys": dict(self.dataset_keys)}

    def reference_jpeg(self, role: str, episode: int) -> bytes:
        """The reference frame a role is compared with: the dataset's first frame of an episode, else our sample.

        @param role A camera role of the robot.
        @param episode Which episode (or which of our samples), wrapping round.
        @return JPEG bytes.
        @throws ValueError No robot is picked, the role is not one of its, or it has no reference (no dataset key and
            no kind: `/api/status` says `none`).
        """
        robot = self._robot()
        cams = setup_cameras.camera_roles(robot.roles)
        if role not in cams:
            raise ValueError(f"{robot.label} has no camera role {role!r}; it has {list(robot.roles)}")
        key = self.dataset_keys.get(role)
        if self.dataset is not None and key:
            return self.dataset.first_frame(key, episode)
        kind = cams[role].kind
        if kind is None:
            raise ValueError(f"the {role} camera has no reference picture: give your dataset and pick its key")
        return setup_cameras.sample_jpeg(kind, episode)

    def camera_jpeg(self, role: str, view: str) -> bytes:
        """The latest frame of an open camera role, for its live feed on the page.

        Single frames the page asks for again as each loads, not one held-open stream per feed: a browser keeps at most
        six connections to the page's server, and six feeds would hold every one of them.

        @param role The role.
        @param view `live` (the camera's frame) or `model` (resized as the client sends it).
        @return The JPEG.
        """
        if view not in ("live", "model"):
            raise ValueError(f"'view' must be live or model, not {view!r}")
        if not self.cameras.is_open(role):
            state = self.cameras.states().get(role)
            if state is not None:
                raise ValueError(f"the {role} camera stopped ({state['error']}); opening it again")
            raise ValueError(f"the {role} camera is not open; open the cameras first")
        jpeg = self.cameras.live_jpeg(role) if view == "live" else self.cameras.model_jpeg(role)
        if jpeg is None:
            raise ValueError(f"the {role} camera has no frame yet")
        return jpeg

    def camera_match(self) -> dict[str, Any]:
        """Which open roles look swapped against their references (the first episode's frame).

        @return `{"swapped": swapped_roles(), "checked": [roles]}`.
        """
        live = {}
        for role in self.cameras.roles():
            frame = self.cameras.frame(role)
            if frame is not None:
                live[role] = frame
        robot = self._robot()
        refs = self._references(robot)
        references = {
            role: setup_cameras.decode_jpeg(self.reference_jpeg(role, 0))
            for role in live
            if refs[role]["source"] != "none"
        }
        return {"swapped": setup_cameras.swapped_roles(live, references), "checked": sorted(live)}

    def view(self) -> dict[str, Any]:
        """How the page draws the picked robot's arm in 3D.

        @return #robots.setup_view.drawing().
        @throws ValueError No robot or model yet.
        @throws ConfigError The robot has no drawing, or its meshes are broken.
        """
        robot = self._robot()
        model = self.session.model
        if model is None:
            raise ValueError("dream's model of the arm is not ready yet")
        return setup_view.drawing(robot.body, model)

    def save(self, body: Any) -> Saved:
        """Writes the station file from what the page settled and the camera rows it posts.

        @param body `{"cameras": [rows]}`.
        @return The #Saved.
        @throws ValueError The arms or the home pose are not set, or a row is malformed.
        @throws ConfigError The agent would refuse the file.
        """
        robot = self._robot()
        session = self.session
        if set(session.assigned) != {"left", "right"}:
            raise ValueError("find both arms before saving")
        if not session.home:
            raise ValueError("set the home pose before saving")
        arms = {
            side: {
                "can_channel": saved_channel(session.devices[session.assigned[side]], side),
                "settings": session.arm_settings(side),
                "home": session.home[side],
                "reach": session.reach_area(side),
            }
            for side in ("left", "right")
        }
        assert robot.embodiment_id is not None, "robot_choices() and choose() set it"
        mapping = build_config(robot, arms, self._camera_rows(body))
        with self._saving:
            self.saved = save(mapping, self.config_path, self.account, robot.embodiment_id)
        return self.saved

    def udev_rule(self) -> dict[str, Any]:
        """The udev rule for the arms found so far on CAN links, with the names in effect now.

        @return `robots.setup_devices.can_udev_rule()` plus `in_effect`: whether each found arm's link already has its
            name.
        @throws ValueError No arm on a CAN link is found yet.
        """
        session = self.session
        found = {side: session.devices[device] for side, device in session.assigned.items()}
        can = {side: device for side, device in found.items() if device["kind"] == "can" and device["usb"]}
        rule = setup_devices.can_udev_rule(can)
        rule["in_effect"] = {side: device["open"] == setup_devices.CAN_LINKS[side] for side, device in can.items()}
        return rule

    def _camera_rows(self, body: Any) -> list[dict[str, Any]]:
        """The posted camera rows, checked, one per role of the robot."""
        robot = self._robot()
        rows = body.get("cameras")
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("'cameras' must be a list of objects")
        checked = [_camera_row(row, i) for i, row in enumerate(rows)]
        names = [row["name"] for row in checked]
        if sorted(names) != sorted(robot.roles):
            raise ValueError(f"give each camera role exactly once: {list(robot.roles)}, got {names}")
        return checked

    def _references(self, robot: setup_session.Robot | None) -> dict[str, Any]:
        """Where each role's reference comes from, for the page's labels: `dataset`, `sample` (ours, by its kind), or
        `none` (a role of no kind with no dataset key)."""
        if robot is None:
            return {}
        cams = setup_cameras.camera_roles(robot.roles)

        def source(role: str) -> str:
            if self.dataset is not None and self.dataset_keys.get(role):
                return "dataset"
            return "sample" if cams[role].kind is not None else "none"

        return {
            role: {
                "source": source(role),
                "key": self.dataset_keys.get(role),
                "episodes": len(self.dataset.episodes) if self.dataset is not None else 3,
            }
            for role in robot.roles
        }

    def _robot(self) -> setup_session.Robot:
        """The robot picked, or a ValueError."""
        robot = self.session.robot
        if robot is None:
            raise ValueError("pick your robot first")
        return robot

    def _require_agent_down(self) -> None:
        """A ValueError while the station agent runs: it holds the arms and the cameras."""
        if self.agent_running():
            raise ValueError("the station agent is running and holds the arms and cameras; run dreamcli robot down")


class TcpServer(SetupServer):
    """`dreamcli tcp`: the wizard's URDF step alone, the tool point tool, for a station file written by hand.

    The wizard's server answering #TCP_ROUTES only: the page opens on the URDF step alone (state()), and its "Set
    tool center point" writes the arm file SetupServer.add_arm() writes, beside the station file. It opens no arm or
    camera; of the account it asks only the station file's embodiment, whose joint map the page holds the chain to
    (joint_map()).
    """

    def state(self) -> dict[str, Any]:
        """SetupServer.state() with `tool` `tcp`: the page shows the URDF step and nothing else.

        @return The dict.
        """
        return {**super().state(), "tool": "tcp"}

    def joint_map(self) -> dict[str, Any] | None:
        """The URDF joints the station file's embodiment (`custom.embodiment`) drives, asked of the account.

        They are its config's `urdf.joints` (`shared.datasets.embodiments.Urdf.joints`). With none to read, the page
        asks about every moving joint of the chain; when the account could not be asked, it says why first, never
        falling back silently (2026-10-18).

        @return `{"embodiment", "joints", "error"}`: the embodiment's name, None when the file names none; its joints,
            sorted, None when its config has no URDF or the account could not be asked; and why it could not be
            asked, one line, else None.
        """
        name = _section(self.existing, "custom").get("embodiment")
        if name is None:
            return {"embodiment": None, "joints": None, "error": None}
        if self.account.api_key is None:
            error = f"could not read {name}'s joints: your credentials hold no API key"
            return {"embodiment": name, "joints": None, "error": error}
        try:
            config = self._embodiment(name).embodiment_config
        except ValueError as exc:
            return {"embodiment": name, "joints": None, "error": f"could not read {name}'s joints: {exc}"}
        if "urdf" not in config:
            return {"embodiment": name, "joints": None, "error": None}
        joints = sorted({joint for names in config["urdf"]["joints"].values() for joint in names})
        return {"embodiment": name, "joints": joints, "error": None}

    def add_arm(self, body: dict[str, Any]) -> dict[str, Any]:
        """SetupServer.add_arm(), with the arm file's path in the terminal too.

        @param body The page's upload: the URDF, its two links and the tool.
        @return SetupServer.add_arm()'s reply.
        """
        reply = super().add_arm(body)
        status = body[kinematics.TCP_CALIBRATION]
        _log.info(
            "dreamcli tcp: wrote %s (tool point %s); set custom.ik in %s to that path",
            reply["ik"],
            status,
            self.config_path,
        )
        return reply


def _tcp_serves(path: str) -> bool:
    """Whether `dreamcli tcp` (TcpServer) answers `path`: the page, its stylesheet and scripts, the meshes and
    #TCP_ROUTES."""
    return (
        path == "/"
        or path.lstrip("/") == STYLESHEET
        or path.lstrip("/") in SCRIPTS
        or path.startswith(setup_view.URL)
        or path in TCP_ROUTES
    )


class _Server(http.server.ThreadingHTTPServer):
    """The stdlib server plus a reference to its #SetupServer, which the handler reads through `self.server`."""

    ## Lets the wizard restart within TIME_WAIT of its last run.
    allow_reuse_address = True
    ## Request threads die with the process rather than block its exit.
    daemon_threads = True

    def __init__(
        self, address: tuple[str, int], handler: type[http.server.BaseHTTPRequestHandler], wizard: SetupServer
    ):
        """Binds the address and keeps the wizard for the handler.

        @param address `(host, port)`.
        @param handler The request handler class.
        @param wizard The wizard the routes act on.
        """
        super().__init__(address, handler)
        self.wizard = wizard


class _Handler(http.server.BaseHTTPRequestHandler):
    """Routes one request (the module's table)."""

    server: _Server

    def _admitted(self) -> bool:
        token = self.server.wizard.token
        if token is None or setup_access.admitted(token, self.headers):
            return True
        url = urllib.parse.urlsplit(self.path)
        if self.command == "GET" and setup_access.matches(token, setup_access.offered_in_query(url.query)):
            self.send_response(303)
            self.send_header("Set-Cookie", setup_access.token_cookie(token))
            self.send_header("Location", setup_access.without_token(url.path, url.query))
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self._refuse(403, "this setup page needs its private address: open the one `dreamcli setup` printed")
        return False

    def do_GET(self) -> None:
        """The page, its scripts and every `/api/...` GET; anything else 404."""
        if not self._admitted():
            return
        url = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(url.query)
        wizard = self.server.wizard
        if isinstance(wizard, TcpServer) and not _tcp_serves(url.path):
            self._refuse(404, f"dreamcli tcp serves the tool point page only, not GET {url.path}")
            return
        try:
            if url.path == "/":
                self._reply_bytes(200, "text/html; charset=utf-8", page_html(setup_wizard_v2()).encode("utf-8"))
            elif url.path == "/api/connection" and setup_wizard_v2():
                wizard = self.server.wizard
                self._reply_json(200, check_connection(wizard.account.api_key or "", wizard.account.platform_url))
            elif url.path.lstrip("/") == STYLESHEET:
                self._reply_bytes(200, "text/css; charset=utf-8", _package_text(STYLESHEET).encode("utf-8"))
            elif url.path.lstrip("/") in SCRIPTS:
                script = _package_text(url.path.lstrip("/")).encode("utf-8")
                self._reply_bytes(200, "text/javascript; charset=utf-8", script)
            elif url.path == "/" + DRIVER_TEMPLATE:
                self._reply_bytes(200, "text/plain; charset=utf-8", _package_text(DRIVER_TEMPLATE).encode("utf-8"))
            elif url.path.startswith(setup_view.URL) and url.path[len(setup_view.URL) :] in setup_view.files():
                path = url.path[len(setup_view.URL) :]
                kind = setup_view.CONTENT_TYPES[pathlib.PurePosixPath(path).suffix]
                self._reply_bytes(200, kind, setup_view.files()[path].read_bytes())
            elif url.path == "/api/state":
                self._reply_json(200, wizard.state())
            elif url.path == "/api/status":
                self._reply_json(200, wizard.status())
            elif url.path == "/api/scan":
                self._reply_json(200, wizard.scan_devices())
            elif url.path == "/api/so101/match" and setup_so101_match():
                self._reply_json(200, wizard.so101_match())
            elif url.path == "/api/udev_rule":
                self._reply_json(200, wizard.udev_rule())
            elif url.path == "/api/preview":
                # A picture opens its camera on its own; the live feeds may be holding it open.
                wizard.cameras.release()
                self._reply_bytes(200, "image/jpeg", wizard.preview(**_preview_args(query)))
            elif url.path == "/api/cameras/frame":
                jpeg = wizard.camera_jpeg(_query(query, "role"), _query(query, "view", "live"))
                self._reply_bytes(200, "image/jpeg", jpeg)
            elif url.path == "/api/reference":
                episode = int(_query(query, "episode", "0"))
                self._reply_bytes(200, "image/jpeg", wizard.reference_jpeg(_query(query, "role"), episode))
            elif url.path == "/api/cameras/match":
                self._reply_json(200, wizard.camera_match())
            elif url.path == "/api/view":
                self._reply_json(200, wizard.view())
            elif url.path == "/api/embodiments":
                self._reply_json(200, wizard.embodiments())
            elif url.path == "/api/reference_arm":
                self._reply_json(200, setup_view.reference())
            elif url.path == "/api/template":
                self._reply_json(200, {"yaml": template(_query(query, "body"))})
            elif url.path == "/api/files":
                self._reply_json(200, setup_files.folder(pathlib.Path(_query(query, "dir"))))
            elif url.path == "/api/classes":
                self._reply_json(200, setup_files.classes(pathlib.Path(_query(query, "file"))))
            else:
                self._refuse(404, f"no such route: GET {url.path}; open the address `dreamcli setup` printed")
        except (ValueError, errors.RobotsError) as exc:
            self._reply_json(400, {"error": str(exc)})
        except Exception as exc:
            _log.exception("setup wizard: GET %s failed", url.path)
            self._reply_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self) -> None:
        """Every `/api/...` POST; anything else 404."""
        if not self._admitted():
            return
        route = self.path.split("?", 1)[0]
        wizard = self.server.wizard
        if isinstance(wizard, TcpServer) and not _tcp_serves(route):
            self._refuse(404, f"dreamcli tcp serves the tool point page only, not POST {route}")
            return
        session = wizard.session
        try:
            body = self._read_json()
            if not isinstance(body, dict):
                raise ValueError("the request body must be a JSON object")
            if route == "/api/robot":
                reply: Any = wizard.choose(body)
            elif route == "/api/urdf":
                reply = wizard.read_urdf(body)
            elif route == "/api/arm_file":
                reply = wizard.add_arm(body)
            elif route == "/api/arms/typed":
                reply = wizard.type_arm(body)
            elif route == "/api/cameras/typed":
                reply = wizard.type_camera(body)
            elif route == "/api/arms/open":
                reply = {"opened": wizard.open_arms(body)}
            elif route == "/api/arms/rebase":
                session.rebase_wiggle()
                reply = {"ok": True}
            elif route == "/api/arms/close":
                session.release_arms()
                reply = {"ok": True}
            elif route == "/api/motors/gripper_mit" and setup_gripper_mit():
                reply = wizard.switch_gripper(body)
            elif route == "/api/arms/swap":
                reply = session.swap_sides()
            elif route == "/api/arms/assign":
                reply = session.assign(_text(body, "device"), _text(body, "side"))
            elif route == "/api/calibrate/start":
                reply = session.start_calibration(_text(body, "side"))
            elif route == "/api/calibrate/middle":
                reply = session.set_middle()
            elif route == "/api/calibrate/finish":
                reply = session.finish_calibration()
            elif route == "/api/calibrate/cancel":
                reply = session.cancel_calibration()
            elif route == "/api/cameras/open":
                reply = {"opened": wizard.open_cameras(body)}
            elif route == "/api/cameras/stop":
                wizard.cameras.release()
                reply = {"ok": True}
            elif route == "/api/references/retry":
                reply = wizard.retry_references()
            elif route == "/api/dataset":
                reply = wizard.set_dataset(body)
            elif route == "/api/dataset/keys":
                reply = wizard.set_dataset_keys(body)
            elif route == "/api/home":
                reply = session.set_home()
            elif route == "/api/gripper":
                reply = session.record_gripper(_text(body, "end"))
            elif route == "/api/reach/paint":
                if not isinstance(body.get("on"), bool):
                    raise ValueError("'on' must be true or false")
                session.paint_reach(body["on"])
                reply = {"ok": True}
            elif route == "/api/reach/clear":
                session.clear_reach(_text(body, "side"))
                reply = {"ok": True}
            elif route == "/api/reach/skip":
                session.skip_reach()
                reply = {"ok": True}
            elif route == "/api/motion/start":
                if body.get("confirm") is not True:
                    raise ValueError("confirm the safety checklist first")
                session.start_motion()
                reply = {"ok": True}
            elif route == "/api/motion/stop":
                session.press_stop()
                reply = {"ok": True}
            elif route == "/api/motion/release":
                reply = session.release()
            elif route == "/api/motion/eyes":
                if not isinstance(body.get("same"), bool):
                    raise ValueError("'same' must be true or false")
                reply = session.confirm_eyes(body["same"])
            elif route == "/api/save":
                saved = wizard.save(body)
                reply = {
                    "config_path": str(wizard.config_path),
                    "yaml": saved.text,
                    "robot": {key: saved.robot.get(key) for key in ("id", "name", "embodiment")},
                    "platform": saved.platform,
                    "live_missing": list(saved.live_missing),
                }
            elif route == "/api/exit":
                self._reply_json(200, {"ok": True})
                wizard.done.set()
                return
            else:
                self._reply_json(404, {"error": f"no such route: POST {route}"})
                return
            self._reply_json(200, reply)
        except (ValueError, errors.RobotsError) as exc:
            self._reply_json(400, {"error": str(exc)})
        except Exception as exc:
            _log.exception("setup wizard: POST %s failed", route)
            self._reply_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def _read_json(self) -> Any:
        """Reads the request body as JSON; an empty body is an empty object.

        @return The decoded value.
        @throws ValueError The body is not valid JSON or is larger than #MAX_BODY_BYTES.
        """
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise ValueError(f"request body of {length} bytes is larger than {MAX_BODY_BYTES}")
        raw = self.rfile.read(length) if length else b""
        if not raw.strip():
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"request body is not valid JSON: {exc.msg}") from exc

    def _refuse(self, status: int, error: str) -> None:
        opened = self.command == "GET" and not self.path.startswith("/api/")
        if opened and "text/html" in (self.headers["Accept"] or ""):
            page = ERROR_PAGE.format(said=html.escape(error[:1].upper() + error[1:]))
            self._reply_bytes(status, "text/html; charset=utf-8", page.encode("utf-8"))
        else:
            self._reply_json(status, {"error": error})

    def _reply_json(self, status: int, payload: dict[str, Any]) -> None:
        """Writes one JSON reply.

        @param status The HTTP status code.
        @param payload The body, JSON-encoded.
        """
        self._reply_bytes(status, "application/json", json.dumps(payload, default=str).encode("utf-8"))

    def _reply_bytes(self, status: int, content_type: str, data: bytes) -> None:
        """Writes one reply with a body.

        @param status The HTTP status code.
        @param content_type The `Content-Type` header.
        @param data The body.
        """
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (the stdlib's signature)
        """Routes the server's access lines to the module logger at DEBUG instead of stderr.

        @param format The stdlib's format string.
        @param args Its arguments.
        """
        _log.debug("setup wizard: " + format, *args)


ERROR_PAGE = (
    '<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Dream setup</title>'
    '<meta name="viewport" content="width=device-width, initial-scale=1"></head>'
    '<body style="font-family: system-ui, sans-serif; max-width: 40rem; margin: 3rem auto; padding: 0 1rem">'
    "<h1>Dream setup</h1><p>{said}.</p></body></html>"
)


def setup_so101_match() -> bool:
    """Whether the page checks SO-101 calibration files against the plugged-in arms, and its route answers.

    @return `station.SETUP_SO101_MATCH`, which is off when `DREAM_SETUP_SO101_MATCH` is unset.
    """
    return station.SETUP_SO101_MATCH


def setup_gripper_mit() -> bool:
    """Whether the page offers to switch a YAM gripper motor to MIT, and its route answers.

    @return `station.SETUP_GRIPPER_MIT`, which is off when `DREAM_SETUP_GRIPPER_MIT` is unset.
    """
    return station.SETUP_GRIPPER_MIT


def setup_wizard_v2() -> bool:
    """Whether the redesigned page and its connection route are served.

    @return `station.SETUP_WIZARD_V2`, which is on unless `DREAM_SETUP_WIZARD_V2` is set to `0`.
    """
    return station.SETUP_WIZARD_V2


def page_html(v2: bool = False) -> str:
    """The wizard page, read beside this module on every request (so an edit shows).

    @param v2 Serve `setup_wizard_v2.html`, the redesigned page, instead of `setup_wizard.html` (setup_wizard_v2()).
        Each page runs its own script (`setup_wizard_v2.js`, `setup_wizard.js`) against the same routes.
    @return The HTML.
    """
    return _package_text("setup_wizard_v2.html" if v2 else "setup_wizard.html")


def check_connection(api_key: str, platform_url: str) -> dict[str, Any]:
    """Whether the platform can be reached and takes the key, asked before the operator does any step.

    Read-only: `GET /v1/robots` with the key, the same first call Save's register_robot() makes, so the page finds a
    network or key problem up front instead of after every step is done.

    @param api_key The DreamAPI key.
    @param platform_url The platform, `http(s)://<host>[:<port>]` (the credentials').
    @return `{"state", "message", "platform"}`: `state` is `ok`, `bad_key` (401), `unreachable` (no answer at all)
        or `error` (any other answer); `message` is the cause as the request met it.
    """
    origin = platform_url.rstrip("/")
    request = urllib.request.Request(
        origin + "/v1/robots", headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=REGISTER_TIMEOUT_S) as reply:
            reply.read()
        return {"state": "ok", "message": f"{origin} took the key", "platform": origin}
    except urllib.error.HTTPError as error:
        detail = _error_detail(error)
        state = "bad_key" if error.code == 401 else "error"
        return {"state": state, "message": f"{error.code} {detail}", "platform": origin}
    except urllib.error.URLError as error:
        return {"state": "unreachable", "message": str(error.reason), "platform": origin}
    except OSError as error:
        return {"state": "unreachable", "message": str(error), "platform": origin}


def _package_text(name: str) -> str:
    """A text file shipped beside this module."""
    return importlib.resources.files("robots").joinpath(name).read_text(encoding="utf-8")


def main(
    config_path: pathlib.Path,
    *,
    host: str | None = None,
    port: int | None = None,
    browser: bool = True,
    agent_running: Callable[[], bool] | None = None,
    out: IO[str] = sys.stdout,
) -> int:
    """Serves the wizard until the page says it is done, or Ctrl-C, SIGTERM or SIGHUP; the `dreamcli setup` command.

    @param config_path The file Save writes (`robots.cli.Paths.config`).
    @param host Bind address; #DEFAULT_HOST when None.
    @param port Port; #DEFAULT_PORT when None.
    @param browser Open the page, and Live view once saved, in this box's browser when it has a display; the
        addresses are printed either way.
    @param agent_running Whether the station agent is running (`robots.cli` asks its pid file); None: never.
    @param out Where the address goes.
    @return The exit code: 0 when the wizard ended, 1 when the credentials are malformed or name no key,
        #robots.setup_access.TOKEN_ENV is not a usable token, or it could not bind.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Read before the page is served: a malformed file or a missing key stops the command rather than every Save.
    try:
        account = credentials.load()
    except dream.DreamConfigError as exc:
        print(f"setup wizard: {exc}", file=out)
        return 1
    if not account.api_key:
        print(
            f"setup wizard: no api key: neither the environment nor {account.path} sets DREAM_API_KEY; the Setup API"
            f" page on {account.platform_url} gives the lines that write that file",
            file=out,
        )
        return 1
    try:
        warning = build_warning(account.platform_url, robots.COMMIT)
    except (OSError, ValueError) as exc:
        warning = f"setup wizard: could not ask {account.platform_url} which build it runs, to compare: {exc}"
    if warning is not None:
        print(warning, file=out)
    try:
        token = setup_access.new_token()
    except errors.ConfigError as exc:
        print(f"setup wizard: {exc}", file=out)
        return 1
    server = SetupServer(
        config_path,
        account,
        host=host or DEFAULT_HOST,
        port=DEFAULT_PORT if port is None else port,
        agent_running=agent_running,
        token=token,
    )
    ending = (
        "press Ctrl-C to end it without saving; an arm still powered after the checks then parks and switches off, so "
        "keep clear while it moves and hold it up before it drops"
    )
    if not _serve(
        server,
        token,
        browser=browser,
        out=out,
        command="setup",
        label="setup wizard",
        intro=f"setup wizard: serving the setup page; writes {server.config_path}",
        ending=ending,
    ):
        return 1
    for words in server.session.still_holding.values():
        print(f"setup wizard: {words}", file=out)
    if server.saved is not None:
        live = f"{account.platform_url}/live"
        print(f"setup wizard: done, saved {server.config_path}", file=out)
        print(NEXT_STEPS.format(live=live), file=out)
        if browser and _has_display():
            webbrowser.open(live)
    else:
        print(f"setup wizard: nothing was saved; {server.config_path} is as it was", file=out)
    return 0


def tcp_main(config_path: pathlib.Path, *, host: str | None, port: int | None, browser: bool, out: IO[str]) -> int:
    """Serves the URDF step alone (TcpServer) until Ctrl-C, SIGTERM or SIGHUP; the `dreamcli tcp` command.

    For a station file written by hand (`dreamcli setup`, Set it up myself): the operator gives the arm's URDF, its end
    and its tool point on the page, and "Set tool center point" writes the arm file to #ARMS_DIR beside `config_path`,
    the path the file's `custom.ik` takes. The credentials give the page the platform's address; no key is needed,
    though with one the page holds the arm's joints to the station file's embodiment (TcpServer.joint_map()).

    @param config_path The station file (`robots.cli.Paths.config`); the arm files are written beside it.
    @param host Bind address; #DEFAULT_HOST when None.
    @param port Port; #TCP_PORT when None.
    @param browser Open the page in this box's browser when it has a display; the addresses are printed either way.
    @param out Where the address goes.
    @return The exit code: 0 when it ended, 1 when the credentials are malformed, #robots.setup_access.TOKEN_ENV is
        not a usable token, or it could not bind.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        account = credentials.load()
        token = setup_access.new_token()
    except (dream.DreamConfigError, errors.ConfigError) as exc:
        print(f"dreamcli tcp: {exc}", file=out)
        return 1
    server = TcpServer(
        config_path, account, host=host or DEFAULT_HOST, port=TCP_PORT if port is None else port, token=token
    )
    served = _serve(
        server,
        token,
        browser=browser,
        out=out,
        command="tcp",
        label="dreamcli tcp",
        intro=f"dreamcli tcp: serving the tool point page; writes arm files to {config_path.parent / ARMS_DIR}",
        ending="press Ctrl-C to end it; each Set tool center point writes the arm file again and prints its path",
    )
    return 0 if served else 1


def _serve(
    server: SetupServer,
    token: str,
    *,
    browser: bool,
    out: IO[str],
    command: str,
    label: str,
    intro: str,
    ending: str,
) -> bool:
    """Serves `server`'s page until it is done, or Ctrl-C, SIGTERM or SIGHUP, after printing where to open it.

    @param server The wizard (main()) or the tool point tool (tcp_main()), not started.
    @param token The secret its printed addresses carry.
    @param browser Open the page in this box's browser when it has a display.
    @param out Where the lines go.
    @param command The `dreamcli` command, for the hint when the port is taken: `setup` or `tcp`.
    @param label What its lines start with.
    @param intro The first line: what it serves and writes.
    @param ending The last line before it serves: how to end it.
    @return False when it could not listen (said on `out`), else True once it has ended.
    """
    try:
        server.start()
    except OSError as exc:
        hint = (
            f"another dreamcli {command} is probably still running in some terminal: end it with Ctrl-C there, or "
            f"pass another port, such as --port {server.port + 1}"
            if exc.errno == errno.EADDRINUSE
            else "pass --host or --port to listen elsewhere"
        )
        print(f"cannot listen on {server.host}:{server.port}: {exc.strerror}; {hint}", file=out)
        return False
    before = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    signal.signal(signal.SIGTERM, _interrupt)
    signal.signal(signal.SIGHUP, _interrupt)
    try:
        local = f"http://127.0.0.1:{server.port}/?{setup_access.TOKEN_QUERY}={token}"
        opening = browser and _has_display()
        print(intro, file=out)
        print(KEEP_TERMINAL_OPEN, file=out)
        everywhere = server.host in ("0.0.0.0", "::")
        if opening and not os.environ.get("SSH_CONNECTION"):
            # The page opens on this box's own screen: one address, for a browser that did not open, and no list of
            # network addresses, which a beginner at the robot's computer reads as a step to do.
            lines = setup_access.local_lines(local)
        else:
            if everywhere:
                addresses = setup_access.listen_addresses()
            else:  # bound to loopback, the page opens on this box or through the tunnel only
                addresses = [] if server.host == "127.0.0.1" else [server.host]
            lines = setup_access.reach_lines(
                server.port,
                token,
                ssh_connection=os.environ.get("SSH_CONNECTION"),
                addresses=addresses,
                tailnet=setup_access.tailnet_name() if everywhere else None,
            )
        for line in lines:
            print(line, file=out)
        print(ending, file=out)
        # Flushed: piped stdout is block-buffered, so these lines would sit unseen.
        out.flush()
        if opening:
            webbrowser.open(local)
        while not server.done.wait(0.5):
            pass
    except KeyboardInterrupt:
        print(f"\n{label}: ended", file=out)
    finally:
        # A second Ctrl-C while stop() parks the arms would leave one powered half way, so it waits.
        for signum in before:
            signal.signal(signum, signal.SIG_IGN)
        server.stop()
        for signum, handler in before.items():
            signal.signal(signum, handler)
    return True


def _interrupt(signum: int, _frame: object) -> None:
    raise KeyboardInterrupt(signal.Signals(signum).name)


def _has_display() -> bool:
    """True when a browser can open a window here; without one `webbrowser` would start a console browser.

    @return Whether a graphical display is available.
    """
    return sys.platform != "linux" or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _our_robot(body: str) -> setup_session.Robot:
    found = next((r for r in setup_session.robot_choices() if r.body == body), None)
    if found is None:
        raise ValueError(f"no robot {body!r} here")
    if found.unavailable is not None:
        raise ValueError(f"{found.label} cannot be picked here: {found.unavailable}")
    return found


def _read_existing(path: pathlib.Path) -> dict[str, Any]:
    """The existing file's mapping, read leniently for a prefill: `{}` when it is missing or not a YAML mapping."""
    try:
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, yaml.YAMLError) as exc:
        _log.warning("setup: %s is not readable as YAML (%s); starting from the defaults", path, exc)
        return {}
    return raw if isinstance(raw, dict) else {}


def _old_channels(raw: dict[str, Any]) -> dict[str, str]:
    """Per side, the device the existing file drives that arm on, so a swap of left and right is caught."""
    found = {}
    for side, key in (("left", "follower_l"), ("right", "follower_r")):
        channel = _section(raw, key).get("can_channel")
        if isinstance(channel, str) and channel:
            found[side] = channel
    return found


def _old_calibrations(raw: dict[str, Any]) -> list[str]:
    """The SO-101 calibration files the existing file names, so the page offers them again."""
    found = (_section(raw, key).get("calibration") for key in ("follower_l", "follower_r"))
    return [path for path in found if isinstance(path, str) and path]


def _modified(path: pathlib.Path) -> float | None:
    """When a calibration file last changed, in seconds since the epoch; None when it is not on this box (the existing
    file can name one since moved)."""
    return path.stat().st_mtime if path.is_file() else None


def read_gripper_mode(link: str) -> str | None:
    """A YAM arm's gripper motor mode, read on its CAN link (#robots.motor_mode.read_modes); changes nothing.

    @param link The arm's CAN link.
    @return The mode's name (`mit`, `pos_speed`, ...), or None when the motor did not answer or the link failed.
    """
    try:
        ((_, mode),) = motor_mode.read_modes(link, [motor_mode.ALL_MOTORS[-1]])
    except Exception:  # noqa: BLE001 - a scan shows the adapters even when this read fails
        return None
    return None if mode is None else motor_mode.mode_name(mode)


def switch_gripper_to_mit(link: str) -> str:
    """Switches a YAM arm's gripper motor to MIT and stores it (#robots.motor_mode.set_mit).

    @param link The arm's CAN link.
    @return The mode the write's reply reported (`mit`); the motor runs it after a power cycle.
    @throws ValueError The motor did not answer the read or the write.
    """
    ((_, _, after),) = motor_mode.set_mit(link, [motor_mode.ALL_MOTORS[-1]])
    if after is None:
        raise ValueError(f"the gripper motor on {link} did not answer; check the arm is on, then try again")
    return motor_mode.mode_name(after)


def _home_short(path: pathlib.Path) -> str:
    """A path as the page shows it, the home folder written `~`."""
    home = pathlib.Path.home()
    return f"~/{path.relative_to(home)}" if path.is_relative_to(home) else str(path)


def _preview_args(query: dict[str, list[str]]) -> dict[str, Any]:
    """The preview route's query as preview_jpeg()'s arguments.

    @param query The parsed query.
    @return `source`, `value`, `width`, `height`, `fps`.
    @throws ValueError A field is missing or malformed.
    """
    row = _camera_row({"name": "preview", "rotate": 0, **{k: v[0] for k, v in query.items() if v}}, 0)
    return {k: row[k] for k in ("source", "value", "width", "height", "fps")}


def _query(query: dict[str, list[str]], key: str, default: str | None = None) -> str:
    """One query parameter, or `default`; a ValueError when it is missing and there is no default."""
    values = query.get(key)
    if values and values[0]:
        return values[0]
    if default is None:
        raise ValueError(f"'{key}' is required")
    return default


def _camera_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    """Checks one camera row from the page.

    @param row The row.
    @param index Its position, for messages.
    @return The row with `value` stripped and the ints as ints.
    @throws ValueError A field is missing or of the wrong type.
    """
    where = f"cameras[{index}]"
    source = _text(row, f"{where}.source", "source")
    if source not in setup_devices.CAMERA_SOURCES:
        raise ValueError(f"'{where}.source' must be one of {setup_devices.CAMERA_SOURCES}, got {source!r}")
    return {
        "name": _text(row, f"{where}.name", "name"),
        "source": source,
        "value": _text(row, f"{where}.value", "value"),
        "width": _whole(row, "width", f"{where}.width"),
        "height": _whole(row, "height", f"{where}.height"),
        "fps": _whole(row, "fps", f"{where}.fps"),
        "rotate": _whole(row, "rotate", f"{where}.rotate", minimum=0),
    }


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """`raw[key]` when it is a mapping, else an empty one: how a prefill reads a file leniently.

    @param raw The file's mapping.
    @param key The key.
    @return The section, or `{}`.
    """
    value = raw.get(key)
    return value if isinstance(value, dict) else {}


def _mapping(body: dict[str, Any], key: str, path: str = "") -> dict[str, Any]:
    """`body[key]` as a mapping.

    @param body The object.
    @param key The key.
    @param path The dotted location of `body`, for messages.
    @return The mapping.
    @throws ValueError It is missing or not an object.
    """
    value = body.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"'{path + '.' if path else ''}{key}' must be an object")
    return value


def _text(body: dict[str, Any], label: str, key: str | None = None) -> str:
    """`body[key]` as a stripped non-empty string.

    @param body The object.
    @param label The dotted name for messages; also the key when `key` is None.
    @param key The key, when it differs from the label.
    @return The string.
    @throws ValueError It is missing, not a string, or blank.
    """
    value = body.get(key if key is not None else label)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{label}' is required")
    return value.strip()


def _vector(body: dict[str, Any], key: str) -> list[float]:
    """`body[key]` as three finite numbers `[x, y, z]`.

    @throws ValueError It is missing or is not three finite numbers.
    """
    value = body.get(key)
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x) for x in value)
    ):
        raise ValueError(f"'{key}' must be three numbers [x, y, z]")
    return [float(x) for x in value]


def _whole(body: dict[str, Any], key: str, label: str, minimum: int = 1) -> int:
    """`body[key]` as an int of at least `minimum`; a numeric string is accepted, a bool is not.

    @param body The object.
    @param key The key.
    @param label The dotted name for messages.
    @param minimum The smallest accepted value.
    @return The int.
    @throws ValueError It is missing, not a whole number, or below `minimum`.
    """
    value = body.get(key)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"'{label}' must be a whole number >= {minimum}")
    return value

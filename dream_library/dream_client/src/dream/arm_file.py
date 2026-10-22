"""An arm of your own for the `ik` config key: dream's IK model of it, from its URDF and a JSON arm file; internal.

The `ik` key's `arm` is one of the arms dream's IK ships (`dream.client_ik.ARMS`) or the path of an arm file, a JSON
object with every one of these keys:

- `urdf`: the arm's URDF, a path relative to the arm file. One arm; each side runs it in its own base frame.
- `base`: the link the hand's pose is measured from. A model trained on YAM arms reads it as YAM's base: x forward,
  y to the arm's left, z up.
- `flange`: the link the gripper hangs off.
- `tcp`: `[x, y, z]`, metres in the flange's frame: the point between the closed fingertips.
- `approach`: `[x, y, z]` in the flange's frame: the way the fingers point.
- `jaw`: `[x, y, z]` in the flange's frame: the way the fingers close (only its part across `approach` counts).
- `rest`: `[...]`, the arm's joints in a working pose, radians: the pose the solve leans toward, on both sides.

It may also carry `fixed`: `[...]`, names of joints on the chain the robot holds still though the URDF lets them move
(a passive mount), which the setup tools write once a person said so; and `tcp_calibration`, the setup tools' record
of whether the operator marked the tool point or skipped it, and when (`robots.kinematics.tcp_calibration`):
`{"status": "calibrated" | "skipped", "at": <time>}`, plus `digest`, tool_digest() of the file as the operator
confirmed it, once they pressed Set tool center point. register() never refuses a file over it, but warns once when
the tool point is not one a person confirmed (tool_point_problem()).

The URDF is cut to the chain from `base` to `flange`, so a gripper's own joints and every other link drop out; each
joint on the chain is fixed, revolute or prismatic with limits, or continuous, which the solve takes as revolute
within [-pi, pi]; a joint in `fixed` is cut as a fixed joint at its URDF zero. The arm's joints, in `state.<side>_arm`
and in the chunks, are the chain's other moving joints in order from `base`, in radians (metres for a prismatic one).
The solve keeps each joint 2 degrees inside its limits (dream's margin) and under its URDF velocity, at most YAM's
3 rad/s (a velocity of 0, as a URDF writes no limit, takes 3). The model is registered with dream's IK under a name
from the arm file's and the URDF's bytes, so an edited file is a new arm and the same file is built once per process.
"""

import hashlib
import inspect
import json
import logging
import math
import os
import pathlib
import tempfile
import warnings
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from dream.errors import DreamConfigError

## Every key of an arm file.
KEYS = ("urdf", "base", "flange", "tcp", "approach", "jaw", "rest")

## The key an arm file may add: the setup tools' record of the tool point.
TCP_CALIBRATION = "tcp_calibration"

## The key an arm file may add naming the chain's joints the robot holds still, each cut as a fixed joint.
FIXED_KEY = "fixed"

## The keys an arm file may add.
OPTIONAL_KEYS = (TCP_CALIBRATION, FIXED_KEY)

## The keys that place the tool, which tool_digest() hashes with the URDF.
TOOL_KEYS = ("base", "flange", "tcp", "approach", "jaw")

## The logger the tool point's warning goes to where warnings are errors; a child of `dream`, so the log file gets it.
_log = logging.getLogger("dream.arm_file")

## The `dream` package's directory: the warning names the first line outside it, the caller's own `policy()` call.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep

## The sides an arm file's arm runs on, as every arm dream's IK ships.
SIDES = ("right", "left")

## The joint speed cap, rad/s, where the URDF's is higher or unset: YAM's (`dream_ik.specs.YAM_ARM.max_speed`).
MAX_SPEED = 3.0

## The radius of the collision spheres along the arm, metres, as every arm dream's IK ships.
LINK_RADIUS = 0.04

## The pull toward `rest` per joint, the base joint's and the others', as every arm dream's IK ships.
REST_WEIGHTS = (0.15, 0.3)


def register(path: str) -> str:
    """Builds dream's IK model of the arm an arm file describes and adds it to dream's arms.

    @param path The arm file.
    @return The name dream's IK knows the arm by (`file:` and 12 hex digits of the arm file's and the URDF's bytes):
        the key of `dream_ik.specs.EMBODIMENTS` and the arm `dream_ik.solver.ChunkIK` takes.
    @throws DreamConfigError The arm file or its URDF is missing or malformed, or the chain cannot be solved for.
    """
    from dream_ik.specs import EMBODIMENTS, ArmSpec, Embodiment, ToolPose, _tool_axes

    file = pathlib.Path(path).expanduser()
    arm = _read(file)
    urdf_path = file.parent / arm["urdf"]
    if not urdf_path.is_file():
        raise DreamConfigError(f"arm file {file}: its urdf {str(urdf_path)!r} is not a file")
    urdf = urdf_path.read_bytes()
    name = "file:" + hashlib.sha256(file.read_bytes() + b"\0" + urdf).hexdigest()[:12]
    if name in EMBODIMENTS:
        return name
    chain, joints, limits, speeds = _chain(urdf, arm["base"], arm["flange"], tuple(arm.get(FIXED_KEY, ())), file)
    rest = arm["rest"]
    if len(rest) != len(joints):
        raise DreamConfigError(
            f"arm file {file}: rest has {len(rest)} joints, the chain from {arm['base']!r} to {arm['flange']!r} has "
            f"{len(joints)}: {list(joints)}"
        )
    for joint, value, (lo, hi) in zip(joints, rest, limits, strict=True):
        if not lo <= value <= hi:
            raise DreamConfigError(f"arm file {file}: rest puts {joint} at {value}, outside [{lo}, {hi}]")
    if not np.linalg.norm(np.cross(arm["approach"], arm["jaw"])) > 1e-6:
        raise DreamConfigError(
            f"arm file {file}: approach {arm['approach']} and jaw {arm['jaw']} must be nonzero and not parallel"
        )
    # Cut out of the URDF, so it is written where dream's IK reads it: one file per arm, written whole, then renamed.
    chain_path = pathlib.Path(tempfile.gettempdir()) / f"dream_arm_{name[5:]}.urdf"
    partial = chain_path.with_suffix(f".{os.getpid()}.part")
    partial.write_text(chain)
    partial.replace(chain_path)
    tool = ToolPose(offset=tuple(arm["tcp"]), rotation=_tool_axes(tuple(arm["approach"]), tuple(arm["jaw"])))
    spec = ArmSpec(
        urdf=chain_path,
        joints=joints,
        flange=arm["flange"],
        tools={"tcp": tool},
        rest_pose=tuple(rest),
        rest_weight=(REST_WEIGHTS[0],) + (REST_WEIGHTS[1],) * (len(joints) - 1),
        max_speed=speeds,
        joint_sign=None,
        link_radius=LINK_RADIUS,
        jaw=None,
        joint_units=None,
    )
    EMBODIMENTS[name] = Embodiment(arm=spec, sides=SIDES, obstacles={side: () for side in SIDES})
    problem = tool_point_problem(arm, urdf)
    if problem is not None:
        _warn(
            f"arm file {file}: {problem}, so the hand poses may land off target. Run `dreamcli tcp`, set the tool "
            "center point on the real gripper, and put the arm file it writes in config key 'ik'"
        )
    return name


def tool_digest(arm: dict[str, Any], urdf: bytes) -> str:
    """The hash of the tool point a person confirmed: the URDF's bytes and the arm file's #TOOL_KEYS.

    `dreamcli tcp` records it in `tcp_calibration` when the operator presses Set tool center point, and
    tool_point_problem() computes it again, so a tool point or URDF edited since no longer matches.

    @param arm The arm file's keys.
    @param urdf The URDF's bytes, as on disk.
    @return 16 hex digits.
    """
    tool = json.dumps({key: arm[key] for key in TOOL_KEYS}, sort_keys=True)
    return hashlib.sha256(urdf + b"\0" + tool.encode()).hexdigest()[:16]


def tool_point_problem(arm: dict[str, Any], urdf: bytes) -> str | None:
    """Why an arm file's tool point is not one a person confirmed on the real gripper, or None when it is.

    @param arm The arm file's keys.
    @param urdf The URDF's bytes, as on disk.
    @return The reason, in words: the tool point is the flange itself, no record says a person set it, the record
        says they skipped it, or the tool point or URDF changed since; a record from before tool_digest() is trusted.
    """
    if not any(arm["tcp"]):
        return "its tool center point is the flange itself (tcp [0, 0, 0]), not between the closed fingertips"
    if TCP_CALIBRATION not in arm:
        return f"no {TCP_CALIBRATION} record says a person set its tool center point (it was written by hand)"
    record = arm[TCP_CALIBRATION]
    if not isinstance(record, dict) or not {"status", "at"} <= set(record) <= {"status", "at", "digest"}:
        return f"its {TCP_CALIBRATION} record is not one the setup tools write: {record!r}"
    if record["status"] != "calibrated":
        return f"its tool center point was not set at setup ({TCP_CALIBRATION} status {record['status']!r})"
    if "digest" in record and record["digest"] != tool_digest(arm, urdf):
        return f"its tcp, approach, jaw, base, flange or URDF changed since its tool point was set ({record['at']})"
    return None


def _warn(text: str) -> None:
    """Issues `text` as a `UserWarning` at the caller's line: one line, never also a log line a console would echo.

    Never raises: the tool point's check warns and never refuses, so where a filter turns warnings into errors
    (`-W error`) the warning is dropped and `text` goes on the `dream` logger instead.

    @param text The warning's text.
    """
    # Level 1 is this frame; walking out of the package one frame per level ends on the caller's line and its level.
    level, frame = 1, inspect.currentframe()
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        level, frame = level + 1, frame.f_back
    del frame  # a frame held in a local keeps its whole stack alive
    try:
        warnings.warn(text, stacklevel=level)
    except Warning:
        _log.warning(text)


def _read(file: pathlib.Path) -> dict[str, Any]:
    """The arm file's keys, checked for shape.

    @param file The arm file.
    @return The parsed object.
    @throws DreamConfigError The file is missing, is not JSON, or a key is missing, extra (but #OPTIONAL_KEYS) or
        malformed.
    """
    if not file.is_file():
        raise DreamConfigError(f"config key 'ik' names the arm file {str(file)!r}, which is not a file")
    try:
        arm = json.loads(file.read_text())
    except json.JSONDecodeError as error:
        raise DreamConfigError(f"arm file {file} is not JSON: {error}") from None
    if not isinstance(arm, dict) or not set(KEYS) <= set(arm) <= {*KEYS, *OPTIONAL_KEYS}:
        got = sorted(arm) if isinstance(arm, dict) else type(arm).__name__
        raise DreamConfigError(
            f"arm file {file} must be an object with the keys {list(KEYS)}, and maybe {list(OPTIONAL_KEYS)}, got {got}"
        )
    for key in ("urdf", "base", "flange"):
        if not isinstance(arm[key], str) or not arm[key]:
            raise DreamConfigError(f"arm file {file}: {key} must be a name, got {arm[key]!r}")
    for key in ("tcp", "approach", "jaw"):
        if not _numbers(arm[key]) or len(arm[key]) != 3:
            raise DreamConfigError(f"arm file {file}: {key} must be 3 numbers [x, y, z], got {arm[key]!r}")
    if not _numbers(arm["rest"]) or not arm["rest"]:
        raise DreamConfigError(f"arm file {file}: rest must be the arm's joints in radians, got {arm['rest']!r}")
    fixed = arm.get(FIXED_KEY, [])
    if not isinstance(fixed, list) or not all(isinstance(name, str) for name in fixed):
        raise DreamConfigError(f"arm file {file}: fixed must be a list of joint names, got {fixed!r}")
    return arm


def _numbers(value: Any) -> bool:
    """Whether `value` is a list of finite numbers."""
    return isinstance(value, list) and all(
        isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x) for x in value
    )


def _chain(
    urdf: bytes, base: str, flange: str, fixed: tuple[str, ...], file: pathlib.Path
) -> tuple[str, tuple[str, ...], tuple[tuple[float, float], ...], tuple[float, ...]]:
    """The URDF cut to the chain from `base` to `flange`.

    A continuous joint is cut as a revolute one with limits [-pi, pi], which pyroki needs, and a joint `fixed` names as
    a fixed one at its URDF zero.

    @param urdf The arm's URDF.
    @param base The chain's first link.
    @param flange Its last.
    @param fixed The chain's joints the robot holds still (#FIXED_KEY).
    @param file The arm file, for messages.
    @return The chain's URDF (bare links: no meshes, which the solve does not read), its moving joints from `base`
        out, each one's `(lower, upper)` limits, and each one's speed cap.
    @throws DreamConfigError The URDF does not parse, lacks either link, has no chain between them, has a joint on it
        the solve cannot take, or `fixed` names a joint that is not a moving joint on it.
    """
    try:
        root = ET.fromstring(urdf)
    except ET.ParseError as error:
        raise DreamConfigError(f"arm file {file}: its URDF does not parse: {error}") from None
    links = {_attr(link, "name", file) for link in root.findall("link")}
    for key, link in (("base", base), ("flange", flange)):
        if link not in links:
            raise DreamConfigError(f"arm file {file}: {key} {link!r} is not a link of the URDF: {sorted(links)}")
    parent_joint = {_link(joint, "child", file): joint for joint in root.findall("joint")}
    chain: list[ET.Element] = []
    link = flange
    while link != base:
        if link not in parent_joint:
            raise DreamConfigError(f"arm file {file}: flange {flange!r} does not hang below base {base!r} in the URDF")
        chain.append(parent_joint[link])
        link = _link(parent_joint[link], "parent", file)
    chain.reverse()
    out = ET.Element("robot", name=root.get("name") or "arm")
    for link in [base] + [_link(joint, "child", file) for joint in chain]:
        ET.SubElement(out, "link", name=link)
    joints: list[str] = []
    limits: list[tuple[float, float]] = []
    speeds: list[float] = []
    held: list[str] = []
    for joint in chain:
        out.append(joint)
        kind, name = joint.get("type"), _attr(joint, "name", file)
        if kind == "fixed":
            continue
        if name in fixed:
            # Held still: only its origin is read, so the link sits where the URDF's zero puts it.
            joint.set("type", "fixed")
            for tag in ("axis", "limit", "mimic", "dynamics"):
                for child in joint.findall(tag):
                    joint.remove(child)
            held.append(name)
            continue
        limit = joint.find("limit")
        if kind == "continuous" and joint.find("mimic") is None:
            # A full turn each way at most; the limits go in the cut URDF too, which pyroki refuses without them.
            limit = limit if limit is not None else ET.SubElement(joint, "limit", effort="0", velocity="0")
            limit.set("lower", repr(-math.pi))
            limit.set("upper", repr(math.pi))
            joint.set("type", "revolute")
            kind = "revolute"
        if kind not in ("revolute", "prismatic") or joint.find("mimic") is not None:
            raise DreamConfigError(
                f"arm file {file}: joint {name!r} between {base!r} and {flange!r} is {kind}"
                f"{' with a mimic' if joint.find('mimic') is not None else ''}; the chain takes fixed joints, "
                "revolute and prismatic joints with limits, and continuous joints, none with a mimic"
            )
        lower, upper = (None, None) if limit is None else (limit.get("lower"), limit.get("upper"))
        if limit is None or lower is None or upper is None:
            raise DreamConfigError(f"arm file {file}: joint {name!r} has no lower and upper limit in the URDF")
        lo, hi = float(lower), float(upper)
        if not lo < hi:
            raise DreamConfigError(f"arm file {file}: joint {name!r} has limits [{lo}, {hi}], an empty range")
        velocity = float(limit.get("velocity") or 0.0)
        joints.append(name)
        limits.append((lo, hi))
        speeds.append(MAX_SPEED if velocity <= 0 else min(velocity, MAX_SPEED))
    unknown = [name for name in fixed if name not in held]
    if unknown:
        raise DreamConfigError(
            f"arm file {file}: fixed names {unknown[0]!r}, which is not a moving joint on the chain from {base!r} to "
            f"{flange!r}: {joints + held}"
        )
    if not joints:
        raise DreamConfigError(f"arm file {file}: the chain from {base!r} to {flange!r} has no moving joint")
    return ET.tostring(out, encoding="unicode"), tuple(joints), tuple(limits), tuple(speeds)


def _attr(element: ET.Element, key: str, file: pathlib.Path) -> str:
    """An attribute every URDF `<link>` and `<joint>` carries.

    @throws DreamConfigError The element lacks it: the URDF is malformed.
    """
    value = element.get(key)
    if value is None:
        raise DreamConfigError(f"arm file {file}: a <{element.tag}> of its URDF has no {key}")
    return value


def _link(joint: ET.Element, end: str, file: pathlib.Path) -> str:
    """The link a URDF joint's `<parent>` or `<child>` names.

    @throws DreamConfigError The joint lacks it: the URDF is malformed.
    """
    tag = joint.find(end)
    if tag is None:
        raise DreamConfigError(f"arm file {file}: joint {joint.get('name')!r} of its URDF has no <{end}> link")
    return _attr(tag, "link", file)

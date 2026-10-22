"""How the setup wizard draws an arm in 3D (`setup_viewer.js`): the maker's meshes riding dream's URDF of the arm, or
simple shapes from that URDF alone.

The page computes no kinematics: every pose it draws is #robots.setup_model.ArmModel.poses() of joints the server
chose (the home pose, and the arms' readings as they move), so the drawing moves exactly as the URDF the checks judge
against. This module says what hangs on each of its links, and the arms' poses now: now().

- #DRAWINGS: per robot, chosen on purpose, the folder of the maker's meshes under `setup_view/`, or #SHAPES.
- `setup_view/<folder>/view.json`: each part, the dream link it rides and its pose there; the gripper's jaws and how
  they move; where the meshes come from and their license. Written by `tests/data/setup_view_build.py`, and checked
  against poses of the maker's own URDF (`tests/data/setup_view_golden.json`) by `tests/test_setup_view.py`.

The page loads three.js itself, pinned, from a CDN (docs/SETUP_WIZARD.md, *The 3D arm*).

@see drawing(), files(), DRAWINGS
"""

import functools
import importlib.resources
import json
from typing import TYPE_CHECKING, Any

from robots import config, errors, setup_model, setup_session

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

## Drawn from dream's URDF alone: capsules for links, discs at joints, two small jaws at the tool.
SHAPES = "shapes"

## How each robot is drawn, by its body: the folder of the maker's meshes under `setup_view/`, or #SHAPES. Chosen per
## robot, never as a fallback: a body missing here cannot be drawn, and a folder with a mesh file missing is an error.
## A robot of one's own is drawn as shapes: its maker's meshes are not ours to ship, and it may look like no arm we
## know; the exception is an arm of dream's whose maker's meshes we do ship (#ARM_DRAWINGS). The YAM's are i2rt's,
## its linear_4310 gripper's included.
DRAWINGS: dict[str, str] = {
    "yam": "yam",
    "makers": "makers",
    "metal": "metal",
    "so101": "so101",
    config.CUSTOM_BODY: SHAPES,
}

## How a robot of one's own is drawn when its `ik` names one of dream's arms whose maker's meshes we ship, by that arm:
## the folder under `setup_view/`. A custom robot on any other arm, or on an arm file, stays #SHAPES. The reBot's are
## Seeed's (Seeed-Projects/reBot-DevArm), its two-finger gripper included.
ARM_DRAWINGS: dict[str, str] = {
    "seeed_faas": "rebot",
}

## The arm the URDF step draws beside the operator's, its tool marker at the fingertips: the example to copy.
REFERENCE_ARM = "yam"

## The page's path to a file under `setup_view/`.
URL = "/view/"

## What each servable file under `setup_view/` is sent as, by its extension.
CONTENT_TYPES: dict[str, str] = {
    ".stl": "model/stl",
    ".json": "application/json",
}


@functools.cache
def files() -> dict[str, "Traversable"]:
    """Every file the page may load from `setup_view/`, by its path there (`metal/link3.stl`).

    @return The files, by path; only the kinds in #CONTENT_TYPES.
    """
    found: dict[str, Traversable] = {}

    def walk(folder: "Traversable", prefix: str) -> None:
        for entry in folder.iterdir():
            if entry.is_dir():
                walk(entry, f"{prefix}{entry.name}/")
            elif any(entry.name.endswith(ext) for ext in CONTENT_TYPES):
                found[prefix + entry.name] = entry

    walk(importlib.resources.files("robots").joinpath("setup_view"), "")
    return found


def drawing(body: str, model: setup_model.ArmModel) -> dict[str, Any]:
    """What the page needs once to draw a robot's arm: dream's chain of it, and the meshes on each link.

    @param body The robot's body (#DRAWINGS); a robot of one's own on an arm in #ARM_DRAWINGS is drawn by that.
    @param model dream's URDF of the arm.
    @return #robots.setup_model.ArmModel.chain() plus `kind` (`meshes` or `shapes`), `parts` (each `{"link", "url",
        "pose"}`: the link index, the mesh's path, its pose on that link as `[qw, qx, qy, qz, x, y, z]` like
        #robots.setup_model.ArmModel.poses()) and `jaws`
        (each a part plus `move` (`slide` or `turn`), `axis` and `pivot` on that link, and `travel`, metres or
        radians from closed to open); both empty for shapes.
    @throws ConfigError The body has no drawing, or its meshes are for another URDF, name a link that URDF does not
        have, or a mesh file is missing.
    """
    if body not in DRAWINGS:
        raise errors.ConfigError(f"the setup page has no drawing of a {body!r} arm; add it to setup_view.DRAWINGS")
    chain = model.chain()
    folder = DRAWINGS[body]
    if body == config.CUSTOM_BODY:
        folder = ARM_DRAWINGS.get(model.spec, folder)
    if folder == SHAPES:
        return {"kind": SHAPES, **chain, "parts": [], "jaws": []}
    view = _view(folder)
    if view["arm"] != model.spec:
        raise errors.ConfigError(
            f"the {folder} meshes ride dream's {view['arm']!r} URDF, not {model.spec!r}; rebuild them "
            "(tests/data/setup_view_build.py)"
        )
    index = {name: i for i, name in enumerate(chain["links"])}

    def placed(part: dict[str, Any]) -> dict[str, Any]:
        if part["link"] not in index:
            raise errors.ConfigError(f"the {folder} meshes ride link {part['link']!r}, which {model.spec} has not")
        path = f"{folder}/{part['mesh']}"
        if path not in files():
            raise errors.ConfigError(f"the {folder} mesh {part['mesh']} is missing from setup_view/{folder}/")
        keep: dict[str, Any] = {key: part[key] for key in ("pose", "move", "axis", "pivot", "travel") if key in part}
        return {"link": index[part["link"]], "url": URL + path, **keep}

    return {
        "kind": "meshes",
        **chain,
        "parts": [placed(part) for part in view["parts"]],
        "jaws": [placed(jaw) for jaw in view["jaws"]],
    }


@functools.cache
def reference() -> dict[str, Any]:
    """dream's YAM at its rest pose, for the URDF step's example of where the tool marker goes.

    Its `tool_link` is the tool centre point between the closed fingertips, in dream's tool frame: z out of the
    gripper, x the way the fingers close (`dream_ik.specs._tool_axes`), the frame the step's marker is drawn in.

    @return drawing() of #REFERENCE_ARM plus `frame`, its link poses at rest (#robots.setup_model.ArmModel.frame()).
    @throws ConfigError dream's IK is not installed.
    """
    model = setup_model.ArmModel(REFERENCE_ARM)
    return {**drawing(REFERENCE_ARM, model), "frame": model.frame(model.rest)}


def _view(folder: str) -> dict[str, Any]:
    """A folder's view.json, or a ConfigError when it has none."""
    path = f"{folder}/view.json"
    if path not in files():
        raise errors.ConfigError(f"setup_view/{folder}/view.json is missing")
    view: dict[str, Any] = json.loads(files()[path].read_text(encoding="utf-8"))
    return view


def now(session: setup_session.Session) -> dict[str, dict[str, Any]]:
    """Each found arm as last read, on the URDF: what the page's 3D views follow live.

    Read from the arms the session holds by hand, or from the powered checks' arms while they are powered
    (#robots.setup_motion.Motion.latest).

    @param session The wizard's session.
    @return Per side with a reading, `{"frame", "grip"}`: the link poses (#robots.setup_model.ArmModel.frame()) and
        the gripper as read; empty before the model is ready.
    """
    with session.lock:
        model = session.model
        if model is None:
            return {}
        if session.motion is not None and session.powered:
            readings = dict(session.motion.latest)
        else:
            latest = session.bank.latest()
            readings = {side: latest[device] for side, device in session.assigned.items() if device in latest}
        return {
            side: {"frame": model.frame(reading.joints), "grip": round(float(reading.gripper), 3)}
            for side, reading in readings.items()
        }

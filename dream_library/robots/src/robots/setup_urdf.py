"""The URDF of an arm of your own, read for the setup wizard's URDF step: every link where the arm's zero puts it, what
its visuals draw, and the guess of the link the arm stands on and the one it ends at.

The page draws the links from these poses, with the meshes of a `.zip` the operator gave beside the URDF, and asks "Is
this the end of your arm?"; the operator clicks another link if not. It then turns the tool marker the operator moved
on the drawing into the arm file's `tcp`, `approach` and `jaw` in the end link's frame (`dream.arm_file`), from that
link's pose here. Nothing here builds dream's model of the arm: SetupServer.add_arm() does, from the links and the tool
the page sends.

@see read()
"""

import math
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import numpy.typing as npt

## The joints that move, as a URDF names them; every other joint (fixed, floating, planar) is drawn at its origin.
MOVING = ("revolute", "continuous", "prismatic")


def read(urdf: str) -> dict[str, Any]:
    """Every link of a URDF at the arm's zero, how they hang together, and the guess of its base and flange.

    @param urdf The URDF's text.
    @return `{"links", "parents", "joints", "moving", "poses", "visuals", "base", "flange"}`: the link names in the
        file's order; per link, the index of the link it hangs off (-1 for the root); per link, the name of the joint
        it hangs off (None for the root); per link, whether that joint moves; per link, its pose as
        `[qw, qx, qy, qz, x, y, z]` in the root's frame (the form of #robots.setup_model.ArmModel.poses()), rounded to
        10 micrometres, each moving joint at 0 or the limit nearest it (the arm file's `rest`,
        SetupServer.add_arm()); per link, what its `<visual>`s draw (_visuals()); the root's index; and the guess of
        the flange, the end of the longest chain (the link with no child after the most moving joints, then the most
        joints, then first in the file), less a gripper's fingers (_before_fingers()).
    @throws ValueError The text is not a URDF whose links hang from one root, or a visual's shape lacks its sizes.
    """
    try:
        robot = ET.fromstring(urdf)
    except ET.ParseError as exc:
        raise ValueError(f"this file is not a URDF: it does not parse as XML ({exc})") from exc
    elements = robot.findall("link")
    links = [link.get("name") or "" for link in elements]
    if robot.tag != "robot" or not links or "" in links:
        raise ValueError("this file is not a URDF: it has no <robot> with named links")
    index = {name: i for i, name in enumerate(links)}
    parents = [-1] * len(links)
    joints: list[ET.Element | None] = [None] * len(links)
    for joint in robot.findall("joint"):
        parent, child = (_link(joint, end) for end in ("parent", "child"))
        if parent not in index or child not in index:
            raise ValueError(f"joint {joint.get('name')!r} joins {parent!r} to {child!r}, not two links of the URDF")
        if joints[index[child]] is not None:
            raise ValueError(f"link {child!r} hangs off two joints")
        parents[index[child]], joints[index[child]] = index[parent], joint
    roots = [i for i, parent in enumerate(parents) if parent == -1]
    if len(roots) != 1:
        raise ValueError(f"the URDF's links must hang from one root, not from {[links[i] for i in roots]}")
    poses: dict[int, npt.NDArray[np.float64]] = {roots[0]: np.eye(4)}
    depth = [(0, 0)] * len(links)  # (moving joints, joints) from the root
    reached = [roots[0]]
    for link in reached:  # grows as it goes: each link's children after it
        for kid in (i for i, parent_index in enumerate(parents) if parent_index == link):
            kid_joint = joints[kid]
            assert kid_joint is not None
            poses[kid] = poses[link] @ _origin(kid_joint) @ _motion(kid_joint)
            moving = kid_joint.get("type") in MOVING
            depth[kid] = (depth[link][0] + moving, depth[link][1] + 1)
            reached.append(kid)
    if len(reached) != len(links):
        raise ValueError(f"links {sorted(set(links) - {links[i] for i in reached})} hang from no root: a loop")
    ends = [i for i in range(len(links)) if i not in parents]
    flange = _before_fingers(max(ends, key=lambda i: (*depth[i], -i)), parents, joints)
    return {
        "links": links,
        "parents": parents,
        "joints": [None if joint is None else joint.get("name") for joint in joints],
        "moving": [joint is not None and joint.get("type") in MOVING for joint in joints],
        "poses": [_pose(poses[i]) for i in range(len(links))],
        "visuals": [_visuals(link) for link in elements],
        "base": roots[0],
        "flange": flange,
    }


def _before_fingers(end: int, parents: list[int], joints: list[ET.Element | None]) -> int:
    """The link a gripper's fingers hang off when the chain to `end` slides after its last turning joint (a Metal's
    `finger_l` on `link6`): fingers are the gripper, not joints of the arm. Else `end`.

    @param end The end of the longest chain.
    @param parents Per link, the index of the link it hangs off (-1 for the root).
    @param joints Per link, the joint it hangs off (None for the root).
    @return The link before the first prismatic joint after the chain's last revolute or continuous one, or `end` when
        none slides after it or none turns.
    """
    path = [end]  # from the end to the root
    while parents[path[-1]] != -1:
        path.append(parents[path[-1]])
    kinds = [None if joint is None else joint.get("type") for joint in (joints[i] for i in path)]
    turning = [k for k, kind in enumerate(kinds) if kind in ("revolute", "continuous")]
    sliding = [k for k in range(turning[0] if turning else 0) if kinds[k] == "prismatic"]
    return parents[path[max(sliding)]] if sliding else end


def _link(joint: ET.Element, end: str) -> str:
    """The link a joint's `<parent>` or `<child>` names, or "" for none."""
    found = joint.find(end)
    return "" if found is None else found.get("link") or ""


def _numbers(element: ET.Element | None, key: str, default: str) -> npt.NDArray[np.float64]:
    """Three numbers of an attribute (`xyz`, `rpy`), the URDF's default when absent."""
    text = default if element is None else element.get(key) or default
    values = np.array([float(part) for part in text.split()], dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"a URDF {key} must be three numbers, got {text!r}")
    return values


def _visuals(link: ET.Element) -> list[dict[str, Any]]:
    """What a link's `<visual>`s draw, for the page to draw the arm as it looks rather than as plain links.

    @param link The `<link>`.
    @return Per visual, its `pose` on the link (`[qw, qx, qy, qz, x, y, z]`) and one of: `mesh`, the file name as the
        URDF writes it (`package://...`, a relative path), with `scale` (three numbers); `box`, its size `[x, y, z]`;
        `cylinder`, `[radius, length]` along the visual's z; `sphere`, its radius; all in metres. A visual with no
        shape, or a shape URDF has not, is left out: the page draws its link plain.
    @throws ValueError A shape lacks a size it needs.
    """
    out: list[dict[str, Any]] = []
    for visual in link.findall("visual"):
        geometry = visual.find("geometry")
        shape = None if geometry is None else next(iter(geometry), None)
        if shape is None:
            continue
        pose = {"pose": _pose(_origin(visual))}
        if shape.tag == "mesh":
            scale = [round(float(v), 6) for v in _numbers(shape, "scale", "1 1 1")]
            out.append({**pose, "mesh": shape.get("filename") or "", "scale": scale})
        elif shape.tag == "box":
            out.append({**pose, "box": [round(float(v), 6) for v in _numbers(shape, "size", "")]})
        elif shape.tag == "cylinder":
            out.append({**pose, "cylinder": [_number(shape, "radius"), _number(shape, "length")]})
        elif shape.tag == "sphere":
            out.append({**pose, "sphere": _number(shape, "radius")})
    return out


def _number(element: ET.Element, key: str) -> float:
    """One number of a shape's attribute (`radius`, `length`), metres."""
    text = element.get(key)
    if text is None:
        raise ValueError(f"a URDF {element.tag} needs a {key}")
    return round(float(text), 6)


def _origin(element: ET.Element) -> npt.NDArray[np.float64]:
    """An element's `<origin>`, rpy as fixed x, y, z turns: where a joint's child sits on its parent at the joint's
    zero, or a visual on its link."""
    origin = element.find("origin")
    roll, pitch, yaw = _numbers(origin, "rpy", "0 0 0")
    out = np.eye(4)
    out[:3, :3] = _turn([0, 0, 1], yaw) @ _turn([0, 1, 0], pitch) @ _turn([1, 0, 0], roll)
    out[:3, 3] = _numbers(origin, "xyz", "0 0 0")
    return out


def _motion(joint: ET.Element) -> npt.NDArray[np.float64]:
    """A moving joint's own move at 0, or the limit nearest 0 when its range leaves 0 out."""
    kind = joint.get("type")
    out = np.eye(4)
    if kind not in MOVING:
        return out
    limit = joint.find("limit")
    lower, upper = (None, None) if limit is None else (limit.get("lower"), limit.get("upper"))
    q = 0.0 if kind == "continuous" or lower is None or upper is None else min(max(0.0, float(lower)), float(upper))
    axis = _numbers(joint.find("axis"), "xyz", "1 0 0")
    if kind == "prismatic":
        out[:3, 3] = axis / np.linalg.norm(axis) * q
    else:
        out[:3, :3] = _turn(axis, q)
    return out


def _turn(axis: npt.ArrayLike, angle: float) -> npt.NDArray[np.float64]:
    """The (3, 3) turn by `angle` radians about `axis` (Rodrigues)."""
    k = np.asarray(axis, dtype=np.float64)
    k = k / np.linalg.norm(k)
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float64)
    turn: npt.NDArray[np.float64] = np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * cross @ cross
    return turn


def _pose(pose: npt.NDArray[np.float64]) -> list[float]:
    """A (4, 4) pose as `[qw, qx, qy, qz, x, y, z]` (qw >= 0), rounded to 10 micrometres."""
    m = pose[:3, :3]
    trace = float(np.trace(m))
    if trace > 0:
        s = 2 * math.sqrt(trace + 1)
        q = [s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2 * math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2 * math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4]
    if q[0] < 0:
        q = [-v for v in q]
    return [round(float(v), 5) for v in (*q, *pose[:3, 3])]

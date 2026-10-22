"""Minimal dual-YAM MuJoCo environment. SI units; actions are [6 joints, opening]."""

import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"
CALIBRATION = json.loads((ASSETS / "sim-eval-yam.json").read_text())
WORLD_ROTATION = np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]])
WORLD_OFFSET = np.array([0.0, 0.4925, 0.0])
## The interactive viewer's starting view (not a policy camera).
VIEWER_LOOKAT = (0.0, 0.0, 0.9)
VIEWER_DISTANCE, VIEWER_AZIMUTH, VIEWER_ELEVATION = 1.5, 90.0, -25.0
## The center (top) policy camera's offset from the station calibration: this much lower, and pitched this much up
## about its own horizontal axis (the calibration points it 60 degrees down from 0.94 m above the table). The scene's
## view, not the station's: images from it do not match the calibrated station exactly.
TOP_CAMERA_LOWER_M = 0.15
TOP_CAMERA_TILT_UP_DEG = 5.0


def aim_viewer(viewer):
    """Points a MuJoCo viewer at the scene from the starting view.
    @param viewer The passive viewer.
    """
    viewer.cam.lookat[:] = VIEWER_LOOKAT
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = VIEWER_DISTANCE, VIEWER_AZIMUTH, VIEWER_ELEVATION


def numbers(values):
    """Serialize numeric values for MJCF attributes.
    @param values Numeric values to serialize.
    @return The values as one space-separated string.
    """
    return " ".join(str(float(v)) for v in values)


def camera(parent, name, position, target, up=(0, 0, 1)):
    """Add a camera aimed at a world-space target.
    @param parent Parent MJCF element.
    @param name Element name.
    @param position Camera position.
    @param target Desired target position.
    @param up Camera up direction.
    """
    z = np.asarray(position) - np.asarray(target)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    ET.SubElement(parent, "camera", name=name, pos=numbers(position), xyaxes=numbers(np.r_[x, y]), fovy="58")


def calibrated_camera(parent, name, entry, world=False, lower=0.0, tilt_up_deg=0.0):
    """Add a camera using the saved station calibration.
    @param parent Parent MJCF element.
    @param name Element name.
    @param entry Calibration entry.
    @param world World body or world-frame conversion flag.
    @param lower Metres to move the camera down from its calibrated position.
    @param tilt_up_deg Degrees to pitch it up about its own horizontal axis.
    @return The camera element added.
    """
    pos = np.array(entry["position"])
    quat = np.array(entry["quaternion_opengl"])
    quat /= np.linalg.norm(quat)
    if world:
        pos = WORLD_ROTATION @ pos + WORLD_OFFSET
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat)
        mujoco.mju_mat2Quat(quat, (WORLD_ROTATION @ rotation.reshape(3, 3)).flatten())
    pos = pos - [0.0, 0.0, lower]
    if tilt_up_deg:
        # A camera looks along its -z; a positive turn about its own x (right) axis raises that view.
        pitch = np.empty(4)
        mujoco.mju_axisAngle2Quat(pitch, np.array([1.0, 0.0, 0.0]), np.radians(tilt_up_deg))
        tilted = np.empty(4)
        mujoco.mju_mulQuat(tilted, quat, pitch)
        quat = tilted
    k = entry["intrinsics"]
    return ET.SubElement(
        parent,
        "camera",
        name=name,
        pos=numbers(pos),
        quat=numbers(quat),
        resolution="640 480",
        sensorsize=".0064 .0048",
        focalpixel=numbers([k["fx"], k["fy"]]),
        principalpixel=numbers([320 - k["cx"], k["cy"] - 240]),
    )


def add_fruit_visual(asset, body, fruit):
    """Add a scanned fruit mesh and its texture.
    @param asset MJCF asset section.
    @param body Parent body element.
    @param fruit Scene fruit name.
    """
    ET.SubElement(
        asset, "texture", name=f"{fruit}_texture", type="2d", file=str(ASSETS / "fruits" / fruit / "texture_map.png")
    )
    ET.SubElement(
        asset, "material", name=f"{fruit}_material", texture=f"{fruit}_texture", specular=".15", shininess=".12"
    )
    ET.SubElement(asset, "mesh", name=f"{fruit}_scan", file=f"fruits/{fruit}/visual.obj")
    ET.SubElement(
        body,
        "geom",
        type="mesh",
        mesh=f"{fruit}_scan",
        material=f"{fruit}_material",
        contype="0",
        conaffinity="0",
        mass="0",
    )


def add_bowl(asset, world):
    # Lathed ceramic cross-section: underside, curved exterior, rolled rim, interior.
    """Build a hollow ceramic bowl with compound contact geometry.
    @param asset MJCF asset section.
    @param world World body or world-frame conversion flag.
    """
    profile = [
        (0, 0.003),
        (0.029, 0.003),
        (0.036, 0.006),
        (0.045, 0.011),
        (0.055, 0.018),
        (0.065, 0.027),
        (0.074, 0.038),
        (0.081, 0.050),
        (0.083, 0.054),
        (0.082, 0.057),
        (0.079, 0.058),
        (0.076, 0.055),
        (0.073, 0.047),
        (0.067, 0.037),
        (0.058, 0.027),
        (0.048, 0.019),
        (0.036, 0.012),
        (0.025, 0.010),
        (0, 0.010),
    ]
    count = 96
    vertices = [
        [r * np.cos(2 * np.pi * i / count), r * np.sin(2 * np.pi * i / count), z]
        for r, z in profile
        for i in range(count)
    ]
    faces = []
    for j in range(len(profile) - 1):
        for i in range(count):
            a, b = j * count + i, j * count + (i + 1) % count
            faces.extend([[a, b, b + count], [a, b + count, a + count]])
    ET.SubElement(
        asset,
        "mesh",
        name="ceramic_bowl",
        vertex=numbers(np.ravel(vertices)),
        face=" ".join(map(str, np.ravel(faces))),
        smoothnormal="true",
    )
    body = ET.SubElement(world, "body", name="bowl", pos="0 -.04 .76")
    ET.SubElement(body, "geom", type="mesh", mesh="ceramic_bowl", material="brown_glaze", contype="0", conaffinity="0")
    # Convex annular wedges follow both surfaces of the visible ceramic profile.
    # A single hull fills the cavity; thick capsules protrude into it.
    outer = np.array(profile[:9])
    inner = np.array(list(reversed(profile[10:])))
    levels = sorted(set(outer[:, 1]) | set(inner[:, 1]) | {0.054, 0.057, 0.058})
    for band, (z0, z1) in enumerate(zip(levels, levels[1:])):
        rings = []
        for z in (z0, z1):
            ro = np.interp(z, [*outer[:, 1], 0.057, 0.058], [*outer[:, 0], 0.082, 0.079])
            ri = 0.0 if z < 0.010 else np.interp(z, inner[:, 1], inner[:, 0])
            rings.append((ri, ro, z))
        for i in range(48):
            points = [
                [r * np.cos(a), r * np.sin(a), z]
                for ri, ro, z in rings
                for r in (ri, ro)
                for a in (2 * np.pi * i / 48, 2 * np.pi * (i + 1) / 48)
            ]
            name = f"bowl_contact_{band}_{i}"
            ET.SubElement(asset, "mesh", name=name, vertex=numbers(np.ravel(points)))
            ET.SubElement(body, "geom", type="mesh", mesh=name, rgba="0 0 0 0")


def build_xml():
    """Use the URDF's joint transforms, not I2RT's MJCF joint frames.
    @return The scene as an MJCF document string.
    """
    root = ET.Element("mujoco", model="Dream YAM Sim")
    ET.SubElement(root, "compiler", angle="radian", meshdir=str(ASSETS))
    ET.SubElement(
        root,
        "option",
        timestep="0.002",
        integrator="implicitfast",
        jacobian="sparse",
        cone="elliptic",
        iterations="100",
        noslip_iterations="3",
    )
    ET.SubElement(root, "size", memory="128M")
    contacts = ET.SubElement(root, "contact")
    for side in ("left", "right"):
        # Adjacent base/shoulder parts share the revolute bearing.
        ET.SubElement(contacts, "exclude", body1=f"{side}_world", body2=f"{side}_link1")
        # Closed CAD finger tips overlap slightly; the slide limits model the stop.
        ET.SubElement(contacts, "exclude", body1=f"{side}_tip_left", body2=f"{side}_tip_right")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="4")
    ET.SubElement(visual, "headlight", ambient=".25 .25 .25", diffuse=".35 .35 .35", specular=".1 .1 .1")
    defaults = ET.SubElement(root, "default")
    ET.SubElement(defaults, "geom", friction="1.6 0.01 0.001", solref="0.015 1", solimp="0.9 0.95 0.001")
    ET.SubElement(defaults, "joint", damping="1", armature="0.02")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1=".78 .85 .90",
        rgb2=".94 .95 .93",
        width="512",
        height="3072",
    )
    for name in ("oak", "brown_glaze", "blue_glaze"):
        ET.SubElement(asset, "texture", name=name, type="2d", file=str(ASSETS / f"{name}.png"))
        ET.SubElement(
            asset,
            "material",
            name=name,
            texture=name,
            texrepeat="2 2" if name == "oak" else "1 1",
            specular=".12" if name == "oak" else ".3",
            shininess=".25",
        )
    tree = json.loads((ASSETS / "model.json").read_text())["root"]
    meshes = set()
    urdf = ET.parse(ASSETS / "yam.urdf").getroot()
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="-.8 .2 2.7", dir=".3 -.1 -1", diffuse=".55 .53 .50", ambient=".25 .25 .25")
    ET.SubElement(world, "light", pos="1 -.6 2", dir="-.4 .2 -1", diffuse=".4 .44 .48", castshadow="false")
    ET.SubElement(world, "geom", type="plane", size="0 0 .05", rgba=".77 .80 .81 1")
    ET.SubElement(world, "geom", name="table", type="box", pos="0 -.12 .73", size=".65 .55 .03", material="oak")
    top_camera = calibrated_camera(
        world,
        "top",
        CALIBRATION["cameras"]["top"],
        world=True,
        lower=TOP_CAMERA_LOWER_M,
        tilt_up_deg=TOP_CAMERA_TILT_UP_DEG,
    )
    # Mock housing and support follow the calibrated optical pose.
    top_pos = np.fromstring(top_camera.get("pos"), sep=" ")
    top_quat = np.fromstring(top_camera.get("quat"), sep=" ")
    stand = ET.SubElement(world, "body", name="center_camera_stand")
    mast_y = top_pos[1] + 0.07
    ET.SubElement(
        stand,
        "geom",
        type="box",
        pos=numbers([top_pos[0], mast_y, 0.768]),
        size=".055 .055 .008",
        rgba=".07 .075 .085 1",
    )
    ET.SubElement(
        stand,
        "geom",
        type="capsule",
        fromto=numbers([top_pos[0], mast_y, 0.776, top_pos[0], mast_y, top_pos[2]]),
        size=".012",
        rgba=".09 .095 .105 1",
    )
    ET.SubElement(
        stand,
        "geom",
        type="capsule",
        fromto=numbers([top_pos[0], mast_y, top_pos[2], *top_pos + [0, 0.02, 0.02]]),
        size=".009",
        rgba=".09 .095 .105 1",
    )
    housing = ET.SubElement(stand, "body", name="center_camera_housing", pos=numbers(top_pos), quat=numbers(top_quat))
    ET.SubElement(housing, "geom", type="box", pos="0 0 .014", size=".025 .021 .011", rgba=".78 .8 .82 1")
    ET.SubElement(housing, "geom", type="box", pos="0 0 .0025", size=".024 .020 .001", rgba=".15 .17 .20 1")
    for lens_x in (-0.011, 0.011):
        ET.SubElement(
            housing, "geom", type="cylinder", pos=f"{lens_x} 0 .001", size=".005 .0005", rgba=".025 .09 .12 1"
        )
    actuators = ET.SubElement(root, "actuator")
    # Convex parts of each robot mesh, packed by tools/pack_assets.py: "<source mesh>/<index>" -> (N, 3) vertices.
    with np.load(ASSETS / "collisions.npz") as packed:
        collision_meshes = {}
        for key in packed.files:
            source, index = key.rsplit("/", 1)
            collision_meshes.setdefault(source, {})[int(index)] = packed[key]

    def add_node(parent, node, side):
        name = node["name"]
        body = ET.SubElement(
            parent,
            "body",
            name=f"{side}_{name}",
            pos=numbers(node["position"]),
            quat=numbers(node["quaternion"]),
            gravcomp="1",
        )
        joint = node.get("joint")
        if joint:
            jname = f"{side}_{joint['name']}"
            ET.SubElement(
                body,
                "joint",
                name=jname,
                type=joint["type"],
                axis=numbers(joint["axis"]),
                range=numbers(joint["limits"]),
            )
            if name.startswith("tip_"):
                # Keep stock finger transforms and slides; contacts use the decomposed STL.
                alias = f"{side}_finger_{-1 if name == 'tip_left' else 1}"
                body.find("joint").set("name", alias)
                ET.SubElement(body, "inertial", pos="0 0 0", mass=".071", diaginertia=".00005 .00005 .00005")
                ET.SubElement(
                    actuators,
                    "position",
                    name=alias,
                    joint=alias,
                    kp="1000",
                    kv="30",
                    ctrlrange="0 .0475",
                    forcerange="-22 22",
                )
            else:
                inertial = urdf.find(f"link[@name='link_{name[4:]}']/inertial")
                inertia = inertial.find("inertia")
                ET.SubElement(
                    body,
                    "inertial",
                    pos=inertial.find("origin").get("xyz"),
                    mass=inertial.find("mass").get("value"),
                    fullinertia=" ".join(inertia.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")),
                )
                ET.SubElement(
                    actuators,
                    "position",
                    name=jname,
                    joint=jname,
                    kp="120",
                    kv="12",
                    ctrlrange=numbers(joint["limits"]),
                    forcerange="-40 40",
                )
        geometries = node["geometries"]
        if name in ("link2", "link3"):
            geometries = [
                dict(file=f"{name}_{color}.stl", position=[0, 0, 0], quaternion=[1, 0, 0, 0], white=color == "white")
                for color in ("black", "white")
            ]
        for geometry in geometries:
            file = geometry["file"]
            if file not in meshes:
                ET.SubElement(asset, "mesh", name=file, file=file, smoothnormal="true")
                meshes.add(file)
            ET.SubElement(
                body,
                "geom",
                type="mesh",
                mesh=file,
                pos=numbers(geometry["position"]),
                quat=numbers(geometry["quaternion"]),
                rgba=".94 .945 .95 1" if geometry.get("white") else ".065 .07 .08 1",
                contype="0",
                conaffinity="0",
                mass="0",
            )
        # Use decomposed stock geometry, independent of the cosmetic color split.
        for geometry in node["geometries"]:
            parts = collision_meshes[geometry["file"]]
            for index in sorted(parts):
                file = f"{geometry['file']}_part{index}"
                if file not in meshes:
                    ET.SubElement(asset, "mesh", name=file, vertex=numbers(parts[index].ravel()))
                    meshes.add(file)
                contact = dict(
                    name=f"{side}_{name}_contact_{index}",
                    type="mesh",
                    mesh=file,
                    pos=numbers(geometry["position"]),
                    quat=numbers(geometry["quaternion"]),
                    rgba="0 0 0 0",
                    mass="0",
                )
                if name.startswith("tip_"):
                    contact.update(
                        friction="6 .08 .008", condim="6", priority="1", solref=".012 1", solimp=".95 .99 .001"
                    )
                ET.SubElement(body, "geom", **contact)
        if name == "link6":
            calibrated_camera(body, side, CALIBRATION["cameras"][f"{side}_wrist"])
        if name == "grasp_site":
            ET.SubElement(body, "site", name=f"{side}_tcp", size=".004", rgba="0 1 0 0")
            ET.SubElement(body, "inertial", pos="0 0 -.09", mass=".1", diaginertia=".00003 .00004 .00005")
        for child in node["children"]:
            add_node(body, child, side)

    # Browser (x,y,z) -> MuJoCo (x,-z,y); left/right are center-camera-relative.
    for side, x in [("left", 0.305), ("right", -0.305)]:
        base = ET.SubElement(
            world, "body", name=f"{side}_base", pos=f"{x} .24 .772", quat=".7071067811865476 0 0 -.7071067811865476"
        )
        add_node(base, tree, side)
    apple = ET.SubElement(world, "body", name="apple", pos=".15 .105 .792")
    ET.SubElement(apple, "freejoint", name="apple_free")
    add_fruit_visual(asset, apple, "apple")
    ET.SubElement(apple, "geom", type="mesh", mesh="apple_scan", mass=".12", rgba="0 0 0 0")
    banana = ET.SubElement(world, "body", name="banana", pos="-.115 .085 .79")
    ET.SubElement(banana, "freejoint", name="banana_free")
    add_fruit_visual(asset, banana, "banana")
    # Convex slices follow the curved scan instead of undersized centerline capsules.
    vertices = np.array(
        [
            [float(x) for x in line.split()[1:4]]
            for line in (ASSETS / "fruits/banana/visual.obj").read_text().splitlines()
            if line.startswith("v ")
        ]
    )
    edges = np.linspace(vertices[:, 1].min(), vertices[:, 1].max(), 17)
    for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
        points = vertices[(vertices[:, 1] >= lo - 0.00025) & (vertices[:, 1] <= hi + 0.00025)]
        ET.SubElement(asset, "mesh", name=f"banana_collision_{i}", vertex=numbers(points.ravel()))
        ET.SubElement(banana, "geom", type="mesh", mesh=f"banana_collision_{i}", mass=".0028125", rgba="0 0 0 0")
    orange = ET.SubElement(world, "body", name="orange", pos="0 .13 .794")
    ET.SubElement(orange, "freejoint", name="orange_free")
    ET.SubElement(orange, "geom", type="ellipsoid", size=".032 .032 .030", mass=".11", rgba=".92 .33 .025 1")
    ET.SubElement(
        orange,
        "geom",
        type="ellipsoid",
        pos="0 0 .029",
        size=".004 .004 .001",
        rgba=".27 .30 .055 1",
        contype="0",
        conaffinity="0",
        mass="0",
    )
    bottle = ET.SubElement(world, "body", name="water_bottle", pos=".22 -.11 .845")
    ET.SubElement(bottle, "freejoint", name="bottle_free")
    ET.SubElement(bottle, "geom", type="cylinder", size=".029 .075", mass=".25", rgba=".36 .58 .65 1")
    ET.SubElement(
        bottle, "geom", type="ellipsoid", pos="0 0 .074", size=".029 .029 .020", mass=".015", rgba=".36 .58 .65 1"
    )
    ET.SubElement(bottle, "geom", type="cylinder", pos="0 0 .097", size=".013 .012", mass=".01", rgba=".12 .24 .3 1")
    ET.SubElement(
        bottle,
        "geom",
        type="cylinder",
        pos="0 0 -.005",
        size=".0295 .025",
        mass=".001",
        rgba=".9 .95 .97 1",
        contype="0",
        conaffinity="0",
    )
    # Molded bottle ribs and label bands are visual-only details.
    for z in (-0.063, -0.053, 0.043, 0.053):
        ET.SubElement(
            bottle,
            "geom",
            type="cylinder",
            pos=f"0 0 {z}",
            size=".0298 .0012",
            rgba=".43 .64 .70 1",
            contype="0",
            conaffinity="0",
            mass="0",
        )
    for z in (-0.028, 0.018):
        ET.SubElement(
            bottle,
            "geom",
            type="cylinder",
            pos=f"0 0 {z}",
            size=".0298 .0012",
            rgba=".17 .35 .40 1",
            contype="0",
            conaffinity="0",
            mass="0",
        )
    add_bowl(asset, world)
    # Shallow ceramic plate beside the bowl, with an open usable center.
    plate = ET.SubElement(world, "body", name="plate", pos="-.22 -.045 .764")
    ET.SubElement(plate, "geom", type="cylinder", size=".069 .004", material="blue_glaze")
    for i in range(64):
        a, b = 2 * np.pi * i / 64, 2 * np.pi * (i + 1) / 64
        ET.SubElement(
            plate,
            "geom",
            type="capsule",
            fromto=numbers([0.068 * np.cos(a), 0.068 * np.sin(a), 0.004, 0.068 * np.cos(b), 0.068 * np.sin(b), 0.004]),
            size=".005",
            material="blue_glaze",
        )

    return ET.tostring(root, encoding="unicode")


## Each arm's home: the six joints at zero and the gripper open, the pose a reset starts from.
HOME = np.r_[np.zeros(6), 1.0]


def home_path(state, seconds=2.0, hz=20.0):
    """A smooth path of step() targets from the arms' current state to #HOME.

    The joints and grippers ease in and out (smoothstep), so the arms start and stop gently.
    @param state The arms' state, as YamEnv.state() reports it.
    @param seconds How long the move takes.
    @param hz Targets per second, the rate step() runs at.
    @return A list of (left, right) targets, the last one #HOME for both.
    """
    count = max(1, round(seconds * hz))
    path = []
    for i in range(1, count + 1):
        s = i / count
        blend = s * s * (3 - 2 * s)
        path.append(tuple(state[side] + (HOME - state[side]) * blend for side in ("left", "right")))
    return path


class YamEnv:
    """Position servo control at 20 Hz, physics at 500 Hz. No API calls required."""

    def __init__(self):
        """Initialize the object and its owned state."""
        self.model = mujoco.MjModel.from_xml_string(build_xml())
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.reset()

    def reset(self):
        """Reset the scene and open the grippers.
        @return The state after the reset, as state() reports it.
        """
        mujoco.mj_resetData(self.model, self.data)
        for side in ("left", "right"):
            for sign in (-1, 1):
                name = f"{side}_finger_{sign}"
                self.data.joint(name).qpos[0] = 0.0475
                self.data.actuator(name).ctrl[0] = 0.0475
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self):
        """Read measured joint positions and normalized gripper openings.
        @return By arm name, seven values: the six joint angles in radians and the gripper opening from 0 to 1.
        """
        result = {}
        for side in ("left", "right"):
            q = [self.data.joint(f"{side}_joint{i}").qpos[0] for i in range(1, 7)]
            opening = sum(self.data.joint(f"{side}_finger_{s}").qpos[0] for s in (-1, 1)) / 0.095
            result[side] = np.r_[q, np.clip(opening, 0, 1)]
        return result

    def step(self, left, right):
        """Execute one 50 ms target; clamp finite joint angles to URDF limits.
        @param left Left arm joint and gripper action.
        @param right Right arm joint and gripper action.
        @return The state after the step, as state() reports it.
        """
        commands = []
        for side, action in [("left", left), ("right", right)]:
            action = np.asarray(action, dtype=float)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError("Each action must contain seven finite values.")
            for i, value in enumerate(action[:6], 1):
                name = f"{side}_joint{i}"
                limits = self.model.joint(name).range
                # Measured MuJoCo joints may drift past a soft limit. Project commands,
                # including bootstrap/hold targets, just as the Dream adapter does.
                commands.append((name, float(np.clip(value, limits[0], limits[1]))))
            if not 0 <= action[6] <= 1:
                raise ValueError("Gripper opening must be between zero and one.")
            commands.extend((f"{side}_finger_{s}", action[6] * 0.0475) for s in (-1, 1))
        for name, value in commands:
            self.data.actuator(name).ctrl[0] = value
        mujoco.mj_step(self.model, self.data, nstep=25)
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def images(self):
        """RGB uint8 images. Call after step; retain nine calls for a temporal window.
        @return By camera name (`top`, `left`, `right`), a 360x640x3 uint8 RGB image.
        """
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        images = {}
        for name in ("top", "left", "right"):
            self.renderer.update_scene(self.data, camera=name)
            images[name] = self.renderer.render()[60:420].copy()  # the reference station recording crop
        return images

    def close(self):
        """Release the offscreen renderer."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def main():
    """Run this module entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-side-panel", action="store_true", help="Show native MuJoCo controls")
    parser.add_argument("--headless", action="store_true", help="Run a two-second smoke check")
    parser.add_argument("--frames", type=Path, help="Save the three camera images as PNG")
    args = parser.parse_args()
    env = YamEnv()
    try:
        home = HOME
        if args.headless:
            for _ in range(40):
                env.step(home, home)
            print("Simulation OK:", env.state())
        else:
            import mujoco.viewer
            from collision_view import CollisionView

            collisions = CollisionView(env)
            with mujoco.viewer.launch_passive(
                env.model,
                env.data,
                key_callback=collisions.toggle,
                show_left_ui=args.show_side_panel,
                show_right_ui=args.show_side_panel,
            ) as viewer:
                aim_viewer(viewer)
                from camera_panel import CameraPanel

                cameras = CameraPanel(env, viewer)
                while viewer.is_running():
                    start = time.monotonic()
                    mujoco.mj_step(env.model, env.data, nstep=10)
                    collisions.update(viewer)
                    viewer.sync()
                    cameras.update()
                    time.sleep(max(0, 0.02 - (time.monotonic() - start)))
        if args.frames:
            from PIL import Image

            args.frames.mkdir(parents=True, exist_ok=True)
            for name, rgb in env.images().items():
                Image.fromarray(rgb).save(args.frames / f"{name}.png")
            print(f"Camera images saved to {args.frames}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

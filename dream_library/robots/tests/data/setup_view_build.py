"""Builds the setup wizard's arm meshes (`src/robots/setup_view/<robot>/`) from each maker's own model of the arm.

Per robot (#ROBOTS) this fetches the maker's URDF, its meshes and its license at a pinned commit, checks that the
maker's chain rides dream's URDF of the arm (`dream_ik`) link for link, merges the meshes of each maker link into one,
simplifies it to the robot's triangle budget, and writes:

- `src/robots/setup_view/<robot>/<link>.stl`: binary STL, metres, in that maker link's frame;
- `src/robots/setup_view/<robot>/view.json`: the table the page draws from (#robots.setup_view): each part on the dream
  link it rides, its pose there, and the gripper's jaws with how they move;
- `src/robots/setup_view/<robot>/LICENSE`: the maker's license, as fetched;
- `tests/data/setup_view_golden.json`, the robot's entry: its maker URDF's part poses at a few fixed joint angles
  (golden()), which `tests/test_setup_view.py` checks the table against.

Run from `dream_library/robots` with pymeshlab beside the dev environment, for example
`uv run --with pymeshlab==2023.12.post3 python tests/data/setup_view_build.py metal so101 makers`. dream's SO-101
URDF is not in every dream_ik; `--dream-assets <dir>` points at one that has it.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import pathlib
import posixpath
import re
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pymeshlab
import pyroki as pk
import trimesh
import yourdfpy

HERE = pathlib.Path(__file__).resolve().parent
PACKAGE = HERE.parent.parent / "src" / "robots" / "setup_view"
GOLDEN = HERE / "setup_view_golden.json"


@dataclasses.dataclass(frozen=True)
class Gripper:
    """A gripper the maker models apart from the arm, as an MJCF, and its place on a link of the arm's URDF.

    Attributes:
    - mjcf: Its MJCF under the source's `root`. Its `gripper` body is the gripper; each child body with a slide or
      hinge joint is a jaw, and an `<equality><joint>` makes one jaw follow another.
    - link: The maker URDF link it mounts on, in place of that link's own meshes.
    - rpy: Its frame in that link's, URDF roll, pitch and yaw; its origin is the link's.
    """

    mjcf: str
    link: str
    rpy: tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class Source:
    """One robot's maker model and how it maps onto dream's URDF.

    Attributes:
    - arm: dream's IK name of the arm; `dream_urdf` is its URDF file in dream_ik's assets.
    - repository, commit, root: Where the maker's files are (`root` is the folder in the repository).
    - urdf, license: Paths under `root`.
    - license_name: What the license is, for view.json.
    - links: Per dream link, the maker links that ride it (the first entry of the first link is the anchor the two
      chains are lined up on: the base, unless the URDFs part ways before their first joint).
    - joints: Per dream joint, `(maker joint, sign)`.
    - jaw: `(maker joint, closed, open)`: the actuated jaw joint and its values with the gripper closed and open.
    - jaw_links: Per dream link, the maker jaw links hanging off it.
    - triangles: The robot's triangle budget, all links together.
    - gripper: A gripper the maker models apart from the arm, or None when the URDF has it (with_gripper()).
    """

    arm: str
    dream_urdf: str
    repository: str
    commit: str
    root: str
    urdf: str
    license: str
    license_name: str
    links: dict[str, list[str]]
    joints: dict[str, tuple[str, float]]
    jaw: tuple[str, float, float]
    jaw_links: dict[str, list[str]]
    triangles: int
    gripper: Gripper | None


ROBOTS: dict[str, Source] = {
    "metal": Source(
        arm="metal",
        dream_urdf="metal.urdf",
        repository="makermods-robotics/makermodslab",
        commit="64defd4e5e9641199b1af5b90587080f721374ba",
        root="frontend/dist/metal-urdf",
        urdf="metal_with_gripper.urdf",
        license="LICENSE",
        license_name="MIT",
        links={
            "base_link": ["base_link"],
            **{f"link{i}": [f"link{i}"] for i in range(1, 6)},
            "link6": ["link6", "gripper_base"],
        },
        joints={f"joint{i}": (f"joint{i}", 1.0) for i in range(1, 7)},
        jaw=("joint7", 0.0, -0.05),
        jaw_links={"link6": ["link7", "link8"]},
        triangles=34000,
        gripper=None,
    ),
    "so101": Source(
        arm="so101_faas",
        dream_urdf="so101_faas_left.urdf",
        repository="TheRobotStudio/SO-ARM100",
        commit="a758567c3978dfeefe282ede0500085a48fe8f78",
        root="Simulation/SO101",
        urdf="so101_new_calib.urdf",
        license="../../LICENSE",
        license_name="Apache-2.0",
        links={
            "arm_root": ["base_link"],
            "shoulder_link": ["shoulder_link"],
            "upper_arm_link": ["upper_arm_link"],
            "lower_arm_link": ["lower_arm_link"],
            "wrist_link": ["wrist_link"],
            "finger_lf": ["gripper_link"],
        },
        joints={
            "joint1": ("shoulder_pan", 1.0),
            "joint2": ("shoulder_lift", 1.0),
            "joint3": ("elbow_flex", 1.0),
            "joint4": ("wrist_flex", 1.0),
            "joint5": ("wrist_roll", 1.0),
        },
        jaw=("gripper", 0.0, 1.74533),
        jaw_links={"finger_lf": ["moving_jaw_so101_v1_link"]},
        triangles=36000,
        gripper=None,
    ),
    "makers": Source(
        arm="makers",
        dream_urdf="maker_arm_ik.urdf",
        repository="makermods-robotics/maker-arm-sdk",
        commit="b30d05a23d72e8c155a8e00f807aba8e8c705f68",
        root="urdf/maker_arm",
        urdf="robot.urdf",
        license="../../LICENSE",
        license_name="Apache-2.0",
        links={name: [name] for name in ("base_link", *(f"link_00{i}" for i in range(2, 8)))},
        joints={f"link_00{i}_joint": (f"link_00{i}_joint", -1.0 if i == 6 else 1.0) for i in range(2, 8)},
        jaw=("gripper_left_joint", 0.0, 0.0524125),
        jaw_links={"link_007": ["gripper_left", "gripper_right"]},
        triangles=36000,
        gripper=None,
    ),
    # Seeed's URDF of the reBot B601-DM is dream's b601_dm chain (the same links and joints, in the DM motor signs)
    # with the gripper: `end_link` is the gripper body, fixed to link6, and its two fingers slide on one rack (the
    # right mimics the left). Its mesh paths are relative to the URDF's folder, so that folder is the root. dream's
    # joint1 sits 45 mm lower on its base_link (the station's table drop, dream_ik's b601_dm.urdf header), so the two
    # bases are 45 mm apart along joint1's axis and only from link1 on do the chains agree: link1 comes first, the
    # anchor, and Seeed's base rides dream's base_link 45 mm down, where the real base is.
    "rebot": Source(
        arm="seeed_faas",
        dream_urdf="b601_dm.urdf",
        repository="Seeed-Projects/reBot-DevArm",
        commit="c16d03eaa31a2a1c0bb38b20423d60818f2a3665",
        root="Rebot_Arm_description/DM/urdf",
        urdf="ReBot_Arm_DM.urdf",
        license="../../../LICENSE",
        license_name="CERN-OHL-W-2.0",
        links={
            "link1": ["link1"],
            "base_link": ["base_link"],
            **{f"link{i}": [f"link{i}"] for i in range(2, 6)},
            "link6": ["link6", "end_link"],
        },
        joints={f"joint{i}": (f"joint{i}", 1.0) for i in range(1, 7)},
        jaw=("finger_left", 0.0, 0.05),
        jaw_links={"link6": ["finger_left_link", "finger_right_link"]},
        triangles=34000,
        gripper=None,
    ),
    # i2rt's URDF of the YAM is dream's (the same joints and links) without its gripper, whose model is an MJCF; its
    # link_6 mesh is not in the repository. The linear_4310 gripper sits on link_6's origin, on joint6's axis, as
    # dream's tool point has it (dream_ik.specs.TCP_OFFSET_M, 0.1347 m along link_6 z, is the MJCF's grasp_site). Its
    # turn is where i2rt's combined MJCF (robots/utils.py combine_arm_and_gripper_xml(), config/linear_4310.yml) puts
    # it on link_6, within 2e-5 rad at any joint angles; the MJCF's arm itself is up to 2 cm off the URDF's.
    "yam": Source(
        arm="yam",
        dream_urdf="yam.urdf",
        repository="i2rt-robotics/i2rt",
        commit="4e462ecc435491a46f9bceb465237afb41d7b66f",
        root="i2rt/robot_models/arm/yam",
        urdf="yam.urdf",
        license="../../../../LICENSE",
        license_name="MIT",
        links={name: [name] for name in ("base_link", *(f"link_{i}" for i in range(1, 7)))},
        joints={f"joint{i}": (f"joint{i}", 1.0) for i in range(1, 7)},
        jaw=("joint7", 0.0, 0.0475),
        jaw_links={"link_6": ["tip_left", "tip_right"]},
        triangles=34000,
        gripper=Gripper(
            mjcf="../../gripper/linear_4310/linear_4310.xml", link="link_6", rpy=(math.pi, 0.0, -math.pi / 2)
        ),
    ),
}


def fetch(source: Source, path: str, cache: pathlib.Path) -> pathlib.Path:
    """One file of the maker's folder at the pinned commit, downloaded once into `cache`."""
    relative = posixpath.normpath(f"{source.root}/{path}")
    local = cache / source.repository / source.commit / relative
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/{source.repository}/{source.commit}/{relative}"
        with urllib.request.urlopen(url, timeout=60) as reply:
            local.write_bytes(reply.read())
    return local


def mesh_path(filename: str) -> str:
    """A URDF mesh reference as a path under the maker's folder (`package://<pkg>/meshes/a.STL` -> `meshes/a.STL`)."""
    if filename.startswith("package://"):
        return filename.split("/", 3)[3]
    return filename


def with_gripper(source: Source, urdf_file: pathlib.Path, cache: pathlib.Path) -> pathlib.Path:
    """The maker's URDF with its gripper's MJCF (#Gripper) on it, written beside it in `cache`.

    The mount link's meshes give way to the gripper body's; each jaw is a link on a prismatic (slide) or revolute
    (hinge) joint of the MJCF's name, range and axis, and a jaw the MJCF's `<equality><joint>` ties to another
    mimics it. Every mesh path is made relative to the source's `root` (i2rt's `package://` is the URDF's folder).

    @return The written URDF.
    """
    grip = source.gripper
    assert grip is not None
    urdf = ET.parse(urdf_file).getroot()
    mjcf = ET.parse(fetch(source, grip.mjcf, cache)).getroot()
    meshdir = posixpath.join(posixpath.dirname(grip.mjcf), mjcf.find("compiler").get("meshdir", ""))
    files = {mesh.get("name"): posixpath.join(meshdir, mesh.get("file")) for mesh in mjcf.find("asset")}
    for mesh in urdf.iter("mesh"):
        mesh.set("filename", mesh.get("filename").removeprefix("package://"))
    mount = trimesh.transformations.euler_matrix(*grip.rpy, "sxyz")

    def placed(element: ET.Element) -> np.ndarray:
        out = trimesh.transformations.quaternion_matrix([float(v) for v in element.get("quat", "1 0 0 0").split()])
        out[:3, 3] = [float(v) for v in element.get("pos", "0 0 0").split()]
        return out

    def origin(parent: ET.Element, pose: np.ndarray) -> None:
        rpy = " ".join(repr(float(v)) for v in trimesh.transformations.euler_from_matrix(pose, "sxyz"))
        ET.SubElement(parent, "origin", xyz=" ".join(repr(float(v)) for v in pose[:3, 3]), rpy=rpy)

    def visuals(link: ET.Element, body: ET.Element, frame: np.ndarray) -> None:
        for geom in body.findall("geom"):
            if geom.get("type") == "mesh":
                visual = ET.SubElement(link, "visual")
                origin(visual, frame @ placed(geom))
                ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename=files[geom.get("mesh")])

    gripper = mjcf.find(".//body[@name='gripper']")
    link = next(link for link in urdf.findall("link") if link.get("name") == grip.link)
    for old in link.findall("visual") + link.findall("collision"):
        link.remove(old)
    visuals(link, gripper, mount)
    follows = {
        tie.get("joint2"): (tie.get("joint1"), *(float(c) for c in tie.get("polycoef").split()[:2]))
        for tie in mjcf.iter("joint")
        if tie.get("joint1") is not None
    }
    for body in gripper.findall("body"):
        joint = body.find("joint")
        assert not any(float(v) for v in joint.get("pos", "0 0 0").split()), f"{joint.get('name')} is off its body"
        jaw = ET.SubElement(urdf, "link", name=body.get("name"))
        visuals(jaw, body, np.eye(4))
        kind = {"slide": "prismatic", "hinge": "revolute"}[joint.get("type")]
        made = ET.SubElement(urdf, "joint", name=joint.get("name"), type=kind)
        origin(made, mount @ placed(body))
        ET.SubElement(made, "parent", link=grip.link)
        ET.SubElement(made, "child", link=body.get("name"))
        ET.SubElement(made, "axis", xyz=joint.get("axis"))
        lower, upper = joint.get("range").split()
        ET.SubElement(made, "limit", lower=lower, upper=upper, effort="1", velocity="1")
        if joint.get("name") in follows:
            # joint1 = c0 + c1 * joint2, so joint2 mimics joint1 times 1 / c1, offset by -c0 / c1.
            leader, c0, c1 = follows[joint.get("name")]
            ET.SubElement(made, "mimic", joint=leader, multiplier=repr(1 / c1), offset=repr(-c0 / c1))
    out = urdf_file.with_name(f"{urdf_file.stem}_{posixpath.basename(grip.mjcf).removesuffix('.xml')}.urdf")
    ET.ElementTree(urdf).write(out, encoding="unicode")
    return out


def matrix(wxyz_xyz: np.ndarray) -> np.ndarray:
    """A `[4, 4]` pose from pyroki's wxyz_xyz."""
    out = np.eye(4)
    out[:3, :3] = trimesh.transformations.quaternion_matrix(wxyz_xyz[:4])[:3, :3]
    out[:3, 3] = wxyz_xyz[4:7]
    return out


def pose7(pose: np.ndarray) -> list[float]:
    """A `[4, 4]` pose as view.json's `[qw, qx, qy, qz, x, y, z]`, the order of pyroki's (and the page's) poses."""
    quat = trimesh.transformations.quaternion_from_matrix(pose)
    quat = quat if quat[0] >= 0 else -quat
    return rounded([*quat, *pose[:3, 3]])


def rounded(values: object) -> list[float]:
    """Numbers for view.json: 7 decimals (a tenth of a micrometre), no negative zeros."""
    return [round(float(v), 7) + 0.0 for v in values]  # type: ignore[attr-defined]


def dumps(value: object) -> str:
    """JSON as view.json and the golden file are written: one line per list of numbers."""
    text = json.dumps(value, indent=1)
    return re.sub(r"\[\s+([^\[\]{}]*?)\s+\]", lambda m: "[" + " ".join(m.group(1).split()) + "]", text) + "\n"


def golden(source: Source, maker: yourdfpy.URDF, dream: pk.Robot) -> list[dict[str, object]]:
    """The maker's URDF's part poses at four fixed joint angles: zeros, then three seeded at random in dream's limits.

    @return Per angle `{"q": {dream joint: radians}, "parts": {maker link: pose7 in the maker's base}}`, jaws closed;
        dream joints the maker does not move are left out (zero).
    """
    rng = np.random.default_rng(0)
    lower, upper = np.asarray(dream.joints.lower_limits), np.asarray(dream.joints.upper_limits)
    links = [m for ms in source.links.values() for m in ms] + [j for js in source.jaw_links.values() for j in js]
    out = []
    for q in [np.zeros(len(lower)), *(rng.uniform(lower, upper) for _ in range(3))]:
        named = zip(dream.joints.actuated_names, q, strict=True)
        joints = {j: round(float(v), 7) for j, v in named if j in source.joints}
        cfg = dict.fromkeys(maker.actuated_joint_names, 0.0) | {source.jaw[0]: source.jaw[1]}
        maker.update_cfg(cfg | {source.joints[j][0]: source.joints[j][1] * v for j, v in joints.items()})
        out.append({"q": joints, "parts": {link: pose7(maker.get_transform(link, maker.base_link)) for link in links}})
    return out


def build(name: str, source: Source, dream_assets: pathlib.Path, cache: pathlib.Path) -> None:
    """Fetches, checks, merges, simplifies and writes one robot."""
    urdf_file = fetch(source, source.urdf, cache)
    maker_file = urdf_file if source.gripper is None else with_gripper(source, urdf_file, cache)
    maker = yourdfpy.URDF.load(str(maker_file), load_meshes=False, build_collision_scene_graph=False)
    dream = pk.Robot.from_urdf(
        yourdfpy.URDF.load(str(dream_assets / source.dream_urdf), load_meshes=False, build_collision_scene_graph=False)
    )
    names = list(dream.links.names)
    jaw_joint, closed, opened = source.jaw

    def maker_poses(q: np.ndarray, jaw: float) -> dict[str, np.ndarray]:
        cfg = dict.fromkeys(maker.actuated_joint_names, 0.0)
        for i, joint in enumerate(dream.joints.actuated_names):
            if joint in source.joints:
                cfg[source.joints[joint][0]] = source.joints[joint][1] * float(q[i])
        cfg[jaw_joint] = jaw
        maker.update_cfg(cfg)
        return {link: maker.get_transform(link, maker.base_link) for link in maker.link_map}

    def dream_poses(q: np.ndarray) -> dict[str, np.ndarray]:
        poses = np.asarray(dream.forward_kinematics(q))
        return {link: matrix(poses[i]) for i, link in enumerate(names)}

    # Each maker link's pose in its dream link, which must not change with the joints: else the meshes would come
    # apart from dream's chain.
    base_dream, base_maker = next(iter(source.links)), source.links[next(iter(source.links))][0]
    rng = np.random.default_rng(0)
    samples = [np.zeros(len(dream.joints.actuated_names))] + [
        rng.uniform(np.asarray(dream.joints.lower_limits), np.asarray(dream.joints.upper_limits)) for _ in range(8)
    ]
    rides: dict[tuple[str, str], np.ndarray] = {}
    for q in samples:
        d, m = dream_poses(q), maker_poses(q, closed)
        root = d[base_dream] @ np.linalg.inv(m[base_maker])
        for link, makers in source.links.items():
            for maker_link in makers:
                pose = np.linalg.inv(d[link]) @ root @ m[maker_link]
                first = rides.setdefault((link, maker_link), pose)
                gap = float(np.abs(pose - first).max())
                assert gap < 1e-6, f"{name}: maker link {maker_link} drifts {gap:.2e} on dream link {link}"

    out = PACKAGE / name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.stl"):
        old.unlink()
    meshes = {}
    for maker_link in [m for ms in source.links.values() for m in ms] + [
        j for js in source.jaw_links.values() for j in js
    ]:
        parts = []
        for visual in maker.link_map[maker_link].visuals:
            mesh = trimesh.load(fetch(source, mesh_path(visual.geometry.mesh.filename), cache), force="mesh")
            if visual.geometry.mesh.scale is not None:
                mesh.apply_scale(visual.geometry.mesh.scale)
            mesh.apply_transform(visual.origin if visual.origin is not None else np.eye(4))
            parts.append(mesh)
        merged = trimesh.util.concatenate(parts)
        merged.merge_vertices()
        merged.update_faces(merged.nondegenerate_faces())
        merged.remove_unreferenced_vertices()
        meshes[maker_link] = merged
    total = sum(len(mesh.faces) for mesh in meshes.values())
    for maker_link, mesh in meshes.items():
        target = max(500, round(source.triangles * len(mesh.faces) / total))
        if len(mesh.faces) > target:
            meshes_set = pymeshlab.MeshSet()
            vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
            meshes_set.add_mesh(pymeshlab.Mesh(vertices, np.ascontiguousarray(mesh.faces, dtype=np.int32)))
            meshes_set.meshing_decimation_quadric_edge_collapse(
                targetfacenum=target, preservenormal=True, planarquadric=True, qualitythr=0.3
            )
            simple = meshes_set.current_mesh()
            mesh = trimesh.Trimesh(simple.vertex_matrix(), simple.face_matrix(), process=False)
        mesh.export(out / f"{maker_link}.stl", file_type="stl")

    # The jaws: each jaw link's pose on its dream link with the gripper closed, and how its own joint moves it.
    q0 = samples[0]
    d0 = dream_poses(q0)
    jaws = []
    for link, jaw_links in source.jaw_links.items():
        for jaw_link in jaw_links:
            joint = next(j for j in maker.robot.joints if j.child == jaw_link)
            m = maker_poses(q0, closed)
            root = d0[base_dream] @ np.linalg.inv(m[base_maker])
            frame = np.linalg.inv(d0[link]) @ root @ m[joint.parent] @ joint.origin
            mimic = (joint.mimic.multiplier, joint.mimic.offset) if joint.mimic is not None else (1.0, 0.0)
            travel = mimic[0] * (opened - closed)
            jaws.append(
                {
                    "link": link,
                    "mesh": f"{jaw_link}.stl",
                    "from": jaw_link,
                    "pose": pose7(np.linalg.inv(d0[link]) @ root @ m[jaw_link]),
                    "move": {"prismatic": "slide", "revolute": "turn"}[joint.type],
                    "axis": rounded(frame[:3, :3] @ np.asarray(joint.axis, dtype=float)),
                    "pivot": rounded(frame[:3, 3]),
                    "travel": round(float(travel), 7),
                }
            )
    view = {
        "arm": source.arm,
        "source": {
            "repository": f"https://github.com/{source.repository}",
            "commit": source.commit,
            "urdf": f"{source.root}/{source.urdf}",
            "urdf_sha256": hashlib.sha256(urdf_file.read_bytes()).hexdigest(),
            "license": source.license_name,
            **(
                {}
                if source.gripper is None
                else {
                    "gripper": posixpath.normpath(f"{source.root}/{source.gripper.mjcf}"),
                    "gripper_sha256": hashlib.sha256(
                        fetch(source, source.gripper.mjcf, cache).read_bytes()
                    ).hexdigest(),
                }
            ),
        },
        "joints": {joint: list(pair) for joint, pair in source.joints.items()},
        "jaw": list(source.jaw),
        "parts": [
            {"link": link, "mesh": f"{maker_link}.stl", "from": maker_link, "pose": pose7(rides[(link, maker_link)])}
            for link, makers in source.links.items()
            for maker_link in makers
        ],
        "jaws": jaws,
    }
    (out / "view.json").write_text(dumps(view), encoding="utf-8")
    (out / "LICENSE").write_bytes(fetch(source, source.license, cache).read_bytes())
    goldens = json.loads(GOLDEN.read_text(encoding="utf-8")) if GOLDEN.exists() else {}
    goldens[name] = golden(source, maker, dream)
    GOLDEN.write_text(dumps(goldens), encoding="utf-8")
    size = sum(path.stat().st_size for path in out.glob("*.stl"))
    triangles = sum(len(trimesh.load(path, force="mesh").faces) for path in out.glob("*.stl"))
    print(f"{name}: {triangles} triangles in {len(list(out.glob('*.stl')))} meshes, {size / 1e6:.2f} MB")


def main() -> None:
    """Builds the robots named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("robots", nargs="+", choices=sorted(ROBOTS))
    parser.add_argument("--dream-assets", type=pathlib.Path, default=None, help="dream_ik's assets folder")
    parser.add_argument("--cache", type=pathlib.Path, default=pathlib.Path.home() / ".cache" / "setup_view_build")
    args = parser.parse_args()
    if args.dream_assets is None:
        import dream_ik.specs

        args.dream_assets = dream_ik.specs.ASSETS
    for name in args.robots:
        build(name, ROBOTS[name], args.dream_assets, args.cache)


if __name__ == "__main__":
    main()

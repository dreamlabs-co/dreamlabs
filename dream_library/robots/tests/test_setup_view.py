"""The setup wizard's 3D view: every shipped mesh sits where its maker's URDF puts it on dream's URDF, the drawing's
tool point is dream_ik's, a robot without meshes is drawn as shapes on purpose, never as a fallback, and the page loads
three.js pinned by version and hash.

The maker's URDF is checked through `tests/data/setup_view_golden.json`: its part poses at four fixed joint angles,
written by `tests/data/setup_view_build.py` (golden()) when it builds the meshes.
"""

import base64
import hashlib
import json
import pathlib
import re
import struct
import subprocess
import urllib.request

import numpy as np
import pytest

from robots import config, errors, setup_model, setup_view, setup_wizard

GOLDEN = json.loads((pathlib.Path(__file__).parent / "data" / "setup_view_golden.json").read_text(encoding="utf-8"))
MESHES = sorted(
    {folder for folder in setup_view.DRAWINGS.values() if folder != setup_view.SHAPES}
    | set(setup_view.ARM_DRAWINGS.values())
)

## The 0.1 mm the drawing may be off dream's kinematics, and the matching turn, radians.
POSITION_M = 1e-4
TURN_RAD = 1e-3


def view_json(folder: str) -> dict:
    return json.loads(setup_view.files()[f"{folder}/view.json"].read_text(encoding="utf-8"))


def arm_model(arm: str) -> setup_model.ArmModel:
    pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")
    if arm not in setup_model.spec_names():
        pytest.skip(f"this install's dream IK has no {arm}")
    return setup_model.ArmModel(arm)


def matrix(pose: list[float] | np.ndarray) -> np.ndarray:
    """`[qw, qx, qy, qz, x, y, z]` as `[4, 4]`."""
    w, x, y, z = np.asarray(pose[:4]) / np.linalg.norm(pose[:4])
    out = np.eye(4)
    out[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    out[:3, 3] = pose[4:7]
    return out


@pytest.mark.parametrize("folder", MESHES)
def test_every_mesh_rides_dreams_urdf_where_the_makers_urdf_puts_it(folder: str) -> None:
    """At the golden joint angles, each part on dream's link is within 0.1 mm of the maker's URDF's link, and the
    drawing's tool point is within 0.1 mm of dream_ik's tool."""
    view = view_json(folder)
    model = arm_model(view["arm"])
    index = {name: i for i, name in enumerate(model.link_names)}
    parts = view["parts"] + view["jaws"]
    assert len(GOLDEN[folder]) == 4 and all(
        set(sample["parts"]) == {p["from"] for p in parts} for sample in GOLDEN[folder]
    )
    anchor = None
    for sample in GOLDEN[folder]:
        q = np.array([sample["q"].get(name, 0.0) for name in model.joint_names])
        poses = model.poses(q)
        for part in parts:
            drawn = matrix(poses[index[part["link"]]]) @ matrix(part["pose"])
            theirs = matrix(sample["parts"][part["from"]])
            # The two URDFs' bases may differ by a fixed transform: the first part (the base) sets it.
            anchor = drawn @ np.linalg.inv(theirs) if anchor is None else anchor
            gap = np.linalg.inv(drawn) @ anchor @ theirs
            assert np.linalg.norm(gap[:3, 3]) < POSITION_M, (part["from"], gap[:3, 3])
            assert setup_model.rotation_angle(gap[:3, :3], np.eye(3)) < TURN_RAD, part["from"]
        np.testing.assert_allclose(poses[model.tool_link, 4:7], model.tool(q)[:3, 3], atol=POSITION_M)


@pytest.mark.parametrize("folder", MESHES)
def test_each_robots_meshes_are_small_licensed_and_all_there(folder: str) -> None:
    view = view_json(folder)
    names = {part["mesh"] for part in view["parts"] + view["jaws"]}
    assert all(f"{folder}/{name}" in setup_view.files() for name in names)
    folder_path = setup_view.files()[f"{folder}/view.json"].parent
    # CERN-OHL-W-2.0: Seeed's reBot, an open hardware licence; its meshes ship merged and simplified, with the licence
    # beside them and the source's repository and commit in view.json.
    assert (folder_path / "LICENSE").is_file() and view["source"]["license"] in ("MIT", "Apache-2.0", "CERN-OHL-W-2.0")
    size = sum(entry.stat().st_size for entry in pathlib.Path(str(folder_path)).iterdir())
    assert size < 2_000_000, f"{folder} is {size} bytes"


def test_the_page_loads_three_js_pinned_by_version_and_hash_and_says_when_it_cannot() -> None:
    """The import map names three.js 0.170.0 on the CDN, every file the viewer imports has its hash (and TGALoader,
    which the COLLADA loader imports, and BufferGeometryUtils, which the glTF loader imports), nothing is vendored, and
    a failed load shows the page's "needs internet" banner."""
    html = setup_wizard.page_html()
    start = html.index('<script type="importmap">') + len('<script type="importmap">')
    imports = json.loads(html[start : html.index("</script>", start)])
    cdn = "https://cdn.jsdelivr.net/npm/three@0.170.0/"
    assert imports["imports"] == {"three": f"{cdn}build/three.module.min.js", "three/addons/": f"{cdn}examples/jsm/"}
    viewer = (pathlib.Path(setup_wizard.__file__).parent / "setup_viewer.js").read_text(encoding="utf-8")
    addons = re.findall(r'^import .* from "three/addons/(.+)";$', viewer, re.M)
    assert addons == [
        "controls/OrbitControls.js",
        "loaders/ColladaLoader.js",
        "loaders/GLTFLoader.js",
        "loaders/OBJLoader.js",
        "loaders/STLLoader.js",
    ]
    used = [
        "build/three.module.min.js",
        *(f"examples/jsm/{path}" for path in addons),
        "examples/jsm/loaders/TGALoader.js",
        "examples/jsm/utils/BufferGeometryUtils.js",
    ]
    assert set(imports["integrity"]) == {cdn + path for path in used}
    assert all(value.startswith("sha384-") for value in imports["integrity"].values())
    assert not any(path.endswith(".js") for path in setup_view.files())
    assert 'id="offline"' in html and "onerror=\"document.getElementById('offline').hidden = false\"" in html


def glb(data: dict[str, object], binary: bytes) -> bytes:
    """A glTF binary: `data`, the file's JSON, then `binary`, its one buffer, each padded to 4 bytes."""
    text = json.dumps(data).encode()
    text += b" " * (-len(text) % 4)
    binary += bytes(-len(binary) % 4)
    chunks = struct.pack("<I4s", len(text), b"JSON") + text + struct.pack("<I4s", len(binary), b"BIN\0") + binary
    return struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks


def test_a_glb_or_gltf_mesh_is_drawn_in_its_links_frame_with_its_own_node_turns(tmp_path: pathlib.Path) -> None:
    """A robot whose URDF names .glb meshes is drawn with them, not as sticks. A made-up triangle in a glTF whose
    node turns it a quarter about x and lifts it: the view's own
    loader, under node with three.js as the page's import map pins it (each file checked against its hash), draws it
    where the node puts it in the link's frame, with no turn for glTF's y up, as ROS does; as a .glb, an upper-case
    .GLB, or a .gltf whose buffer is a file beside it. A .gltf whose buffer the zip lacks is not drawn, so the page
    names it. The same files given as a folder, chosen or dropped, are drawn the same as from the .zip. A
    compressed .glb (Draco or
    meshopt, whose decoders the page does not load) stopped the page with no drawing: it is not drawn, so the page
    draws its link plain and names it, and the rest of the arm still draws."""
    html = setup_wizard.page_html()
    start = html.index('<script type="importmap">') + len('<script type="importmap">')
    pinned = json.loads(html[start : html.index("</script>", start)])["integrity"]
    three = tmp_path / "node_modules" / "three"
    for address, integrity in pinned.items():
        body = urllib.request.urlopen(address, timeout=30).read()
        assert "sha384-" + base64.b64encode(hashlib.sha384(body).digest()).decode() == integrity, address
        target = three / address.split("/three@0.170.0/", 1)[1]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    exports = {".": "./build/three.module.min.js", "./addons/*": "./examples/jsm/*"}
    (three / "package.json").write_text(json.dumps({"name": "three", "type": "module", "exports": exports}))
    corners = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=np.float32)
    data = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        # A quarter turn about x, as a y-up export of a z-up part carries it, then 5 cm up.
        "nodes": [{"mesh": 0, "rotation": [np.sqrt(0.5), 0, 0, np.sqrt(0.5)], "translation": [0, 0, 0.05]}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [0, 0, 0],
                "max": [0.1, 0.2, 0],
            }
        ],
        "bufferViews": [{"buffer": 0, "byteLength": corners.nbytes}],
        "buffers": [{"byteLength": corners.nbytes}],
    }
    zipped = tmp_path / "zip" / "arm" / "meshes"
    zipped.mkdir(parents=True)
    (zipped / "part.glb").write_bytes(glb(data, corners.tobytes()))
    (zipped / "PART2.GLB").write_bytes(glb(data, corners.tobytes()))
    beside = {**data, "buffers": [{"byteLength": corners.nbytes, "uri": "part%20data.bin"}]}
    (zipped / "part.gltf").write_text(json.dumps(beside))
    (zipped / "part data.bin").write_bytes(corners.tobytes())
    (zipped / "lost.gltf").write_text(json.dumps({**beside, "buffers": [{"byteLength": 36, "uri": "lost.bin"}]}))
    draco = ["KHR_draco_mesh_compression"]
    squashed = {**data, "extensionsUsed": draco, "extensionsRequired": draco}
    (zipped / "draco.glb").write_bytes(glb(squashed, corners.tobytes()))
    meshopt = ["EXT_meshopt_compression"]
    packed = {"buffer": 0, "byteLength": corners.nbytes, "byteStride": 12, "count": 3, "mode": "ATTRIBUTES"}
    views = [{"buffer": 0, "byteLength": corners.nbytes, "extensions": {meshopt[0]: packed}}]
    squeezed = {**data, "extensionsUsed": meshopt, "extensionsRequired": meshopt, "bufferViews": views}
    (zipped / "meshopt.glb").write_bytes(glb(squeezed, corners.tobytes()))
    names = ("part.glb", "PART2.GLB", "part.gltf", "lost.gltf", "draco.glb", "meshopt.glb")
    paths = [f"arm/meshes/{name}" for name in names]
    page = pathlib.Path(__file__).parent / "setup_meshes.mjs"
    drawn_by = {}
    for given in ("zip", "list", "drop"):
        done = subprocess.run(
            ["node", str(page), str(tmp_path), given, *paths], capture_output=True, text=True, timeout=60, check=False
        )
        assert done.returncode == 0, done.stderr
        drawn_by[given] = json.loads(done.stdout)
    assert drawn_by["list"] == drawn_by["zip"] and drawn_by["drop"] == drawn_by["zip"]
    shown = drawn_by["zip"]
    for name in ("lost.gltf", "draco.glb", "meshopt.glb"):
        assert shown.pop(f"arm/meshes/{name}") is None, name
    for path, drawn in shown.items():
        assert drawn["meshes"] == 1 and drawn["painted"], path
        np.testing.assert_allclose(drawn["box"], [[0, 0, 0.05], [0.1, 0, 0.25]], atol=1e-6, err_msg=path)


def test_a_robot_is_drawn_as_its_table_says_and_a_broken_one_refuses() -> None:
    yam = arm_model("yam")
    assert setup_view.DRAWINGS[config.CUSTOM_BODY] == setup_view.SHAPES
    shapes = setup_view.drawing(config.CUSTOM_BODY, yam)
    assert shapes["kind"] == "shapes" and shapes["parts"] == [] and len(shapes["joints"]) == 6
    with pytest.raises(errors.ConfigError, match="no drawing"):
        setup_view.drawing("robot_from_mars", yam)
    with pytest.raises(errors.ConfigError, match="not 'yam'"):
        setup_view.drawing("metal", yam)


def test_a_robot_of_ones_own_on_the_rebot_arm_is_drawn_with_seeeds_meshes_and_its_gripper() -> None:
    """A custom robot whose `ik` is dream's `seeed_faas` arm gets the reBot's meshes and two sliding fingers; on any
    other arm it stays shapes."""
    rebot = arm_model("seeed_faas")
    drawn = setup_view.drawing(config.CUSTOM_BODY, rebot)
    assert drawn["kind"] == "meshes"
    assert all(part["url"].startswith(setup_view.URL + "rebot/") for part in drawn["parts"])
    assert len(drawn["jaws"]) == 2 and {jaw["move"] for jaw in drawn["jaws"]} == {"slide"}
    assert setup_view.drawing(config.CUSTOM_BODY, arm_model("yam"))["kind"] == setup_view.SHAPES


def test_a_missing_mesh_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    metal = arm_model("metal")
    drawn = setup_view.drawing("metal", metal)
    assert drawn["kind"] == "meshes" and all(part["url"].startswith(setup_view.URL) for part in drawn["parts"])
    files = dict(setup_view.files())
    del files["metal/link3.stl"]
    monkeypatch.setattr(setup_view, "files", lambda: files)
    with pytest.raises(errors.ConfigError, match="link3.stl is missing"):
        setup_view.drawing("metal", metal)

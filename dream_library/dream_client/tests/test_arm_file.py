"""Tests for `dream.arm_file`: an arm file of the Metal arm's URDF, with a gripper and a camera mount added, is dream's
Metal arm (the same chain, limits and tool poses); a file is built once and an edited one is a new arm; and every
malformed file is a `DreamConfigError` naming what to fix."""

from __future__ import annotations

import json
import math
import pathlib
import re
import warnings
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import pytest

from dream import arm_file
from dream.errors import DreamConfigError

dream_ik_specs = pytest.importorskip("dream_ik.specs")
from dream_ik.solver import ChunkIK  # noqa: E402

## dream's Metal arm, as an arm file names it: its URDF's links, its TCP and the approach and jaw of its tool axes.
METAL = {
    "urdf": "metal.urdf",
    "base": "base_link",
    "flange": "link6",
    "tcp": [0.135, 0.0, 0.0],
    "approach": [1.0, 0.0, 0.0],
    "jaw": [0.0, 1.0, 0.0],
    "rest": [0.0, -1.2, 1.5, 0.0, 0.0, 0.0],
}

## What a vendor URDF hangs off the Metal chain beside the arm: a gripper with a finger joint below the flange and a
## camera mount on the base, which the chain from base to flange leaves out.
EXTRAS = """
  <link name="gripper"/><link name="finger"/><link name="camera_mount"/>
  <joint name="gripper_mount" type="fixed"><parent link="link6"/><child link="gripper"/></joint>
  <joint name="finger_joint" type="prismatic"><parent link="gripper"/><child link="finger"/>
    <axis xyz="0 1 0"/><limit lower="0" upper="0.04" effort="10" velocity="0.1"/></joint>
  <joint name="camera_mount_joint" type="fixed"><parent link="base_link"/><child link="camera_mount"/>
    <origin xyz="0 0.3 0.5"/></joint>
</robot>"""


def write(tmp_path: pathlib.Path, urdf_text: str | None = None, **changes: Any) -> pathlib.Path:
    """An arm file of the Metal arm with a gripper and a camera mount, with `changes` to its keys, in `tmp_path`."""
    if urdf_text is None:
        urdf_text = (dream_ik_specs.ASSETS / "metal.urdf").read_text().replace("</robot>", EXTRAS)
    (tmp_path / "metal.urdf").write_text(urdf_text)
    path = tmp_path / "arm.json"
    path.write_text(json.dumps({**METAL, **changes}))
    return path


def test_an_arm_file_of_the_metal_urdf_is_dreams_metal_arm(tmp_path: pathlib.Path) -> None:
    name = arm_file.register(str(write(tmp_path)))
    ours, metal = ChunkIK(name, "tcp", 30.0), ChunkIK("metal", "tcp", 30.0)
    spec = dream_ik_specs.EMBODIMENTS[name].arm
    assert spec.joints == dream_ik_specs.EMBODIMENTS["metal"].arm.joints
    assert np.array_equal(ours.lo, metal.lo) and np.array_equal(ours.hi, metal.hi)
    assert np.array_equal(ours.rest, metal.rest)
    q = np.random.default_rng(0).uniform(metal.lo, metal.hi, (20, 6))  # (poses, joints)
    assert np.allclose(ours.fk(q), metal.fk(q), atol=1e-6)


def test_a_file_is_built_once_and_an_edited_one_is_a_new_arm(tmp_path: pathlib.Path) -> None:
    path = write(tmp_path)
    first = arm_file.register(str(path))
    assert arm_file.register(str(path)) == first
    edited = arm_file.register(str(write(tmp_path, tcp=[0.12, 0.0, 0.0])))
    assert edited != first
    assert dream_ik_specs.EMBODIMENTS[edited].arm.tools["tcp"].offset == (0.12, 0.0, 0.0)


def test_the_setup_tools_record_of_the_tool_point_is_allowed_and_keeps_the_arm(tmp_path: pathlib.Path) -> None:
    """`tcp_calibration`, robots' record of whether the operator marked the tool point, is the one key a file may add:
    the arm is the same, its chain written to a file of its own (named by the file's bytes) with the same text."""
    plain = dream_ik_specs.EMBODIMENTS[arm_file.register(str(write(tmp_path)))].arm
    marked = dream_ik_specs.EMBODIMENTS[
        arm_file.register(str(write(tmp_path, tcp_calibration={"status": "skipped", "at": "2026-10-15T08:00:00Z"})))
    ].arm
    assert marked.urdf != plain.urdf
    assert marked.urdf.read_text() == plain.urdf.read_text()
    assert marked.model_copy(update={"urdf": plain.urdf}) == plain


@pytest.fixture
def fresh_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    """dream's arms without the arm files earlier tests built, so a file is built, and checked, again."""
    shipped = {name: arm for name, arm in dream_ik_specs.EMBODIMENTS.items() if not name.startswith("file:")}
    monkeypatch.setattr(dream_ik_specs, "EMBODIMENTS", shipped)


def confirmed(**changes: Any) -> dict[str, Any]:
    """The record `dreamcli tcp` writes when the operator presses Set tool center point on write()'s file."""
    urdf = (dream_ik_specs.ASSETS / "metal.urdf").read_text().replace("</robot>", EXTRAS).encode()
    digest = arm_file.tool_digest({**METAL, **changes}, urdf)
    return {"status": "calibrated", "at": "2026-10-18T08:00:00+00:00", "digest": digest}


@pytest.mark.usefixtures("fresh_arms")
@pytest.mark.parametrize(
    "record",
    [confirmed, lambda: {"status": "calibrated", "at": "2026-10-15T08:00:00+00:00"}],
    ids=["digest", "before_the_digest"],
)
def test_a_tool_point_a_person_set_warns_nothing(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture, record: Any
) -> None:
    """Set with `dreamcli tcp`, now or before the record carried a digest: no warning and no log line."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        arm_file.register(str(write(tmp_path, tcp_calibration=record())))
    assert "arm file" not in caplog.text


@pytest.mark.usefixtures("fresh_arms")
@pytest.mark.parametrize(
    ("changes", "says"),
    [
        ({}, "no tcp_calibration record says a person set its tool center point"),
        ({"tcp_calibration": {"status": "skipped", "at": "2026-10-18T08:00:00+00:00"}}, "status 'skipped'"),
        ({"tcp_calibration": {"status": "set"}}, "record is not one the setup tools write"),
    ],
)
def test_a_tool_point_no_person_confirmed_warns_once_and_runs(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture, changes: dict[str, Any], says: str
) -> None:
    """No record (written by hand or by an agent), a skipped one, or a malformed one: the arm is built, and register()
    warns at the caller's line, naming the fix, once a file, and logs nothing an app's console would print again."""
    path = write(tmp_path, **changes)
    with pytest.warns(UserWarning, match=says) as caught:
        name = arm_file.register(str(path))
    assert name in dream_ik_specs.EMBODIMENTS
    ours = [w for w in caught if "arm file" in str(w.message)]
    assert [w.filename for w in ours] == [__file__]
    assert "Run `dreamcli tcp`, set the tool center point on the real gripper" in str(ours[0].message)
    assert says not in caplog.text
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert arm_file.register(str(path)) == name


@pytest.mark.usefixtures("fresh_arms")
def test_a_tool_point_or_urdf_edited_since_it_was_set_warns(tmp_path: pathlib.Path) -> None:
    """The record's digest is of the file the operator confirmed: a tcp an agent moved, or another URDF, warns."""
    with pytest.warns(UserWarning, match="changed since its tool point was set"):
        arm_file.register(str(write(tmp_path, tcp=[0.12, 0.0, 0.0], tcp_calibration=confirmed())))
    urdf = (dream_ik_specs.ASSETS / "metal.urdf").read_text()
    with pytest.warns(UserWarning, match="changed since its tool point was set"):
        arm_file.register(str(write(tmp_path, urdf_text=urdf, tcp_calibration=confirmed())))


@pytest.mark.usefixtures("fresh_arms")
def test_a_tool_point_at_the_flange_warns_even_when_confirmed(tmp_path: pathlib.Path) -> None:
    flange = {"tcp": [0.0, 0.0, 0.0]}
    with pytest.warns(UserWarning, match=r"the flange itself \(tcp \[0, 0, 0\]\)"):
        arm_file.register(str(write(tmp_path, **flange, tcp_calibration=confirmed(**flange))))


@pytest.mark.usefixtures("fresh_arms")
def test_the_warning_never_refuses_even_when_warnings_are_errors(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert arm_file.register(str(write(tmp_path))) in dream_ik_specs.EMBODIMENTS
    assert "no tcp_calibration record" in caplog.text


def test_the_speed_caps_are_the_urdfs_at_most_yams(tmp_path: pathlib.Path) -> None:
    urdf = (dream_ik_specs.ASSETS / "metal.urdf").read_text()
    speeds = dream_ik_specs.EMBODIMENTS[arm_file.register(str(write(tmp_path, urdf)))].arm.max_speed
    moving = [j for j in ET.fromstring(urdf).findall("joint") if j.get("type") != "fixed"]
    velocities = [float(j.find("limit").get("velocity") or 0.0) for j in moving]
    assert speeds == tuple(arm_file.MAX_SPEED if v <= 0 else min(v, arm_file.MAX_SPEED) for v in velocities)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"flange": "wrist"}, "flange 'wrist' is not a link of the URDF"),
        ({"base": "link3", "flange": "link1"}, "flange 'link1' does not hang below base 'link3'"),
        ({"flange": "finger"}, "rest has 6 joints, the chain from 'base_link' to 'finger' has 7"),
        ({"rest": [0.0, -1.2, 1.5, 0.0, 0.0]}, "rest has 5 joints"),
        ({"rest": [0.0, -1.2, 1.5, 0.0, 0.0, 99.0]}, "rest puts joint6 at 99.0, outside"),
        ({"jaw": [2.0, 0.0, 0.0]}, "must be nonzero and not parallel"),
        ({"tcp": [0.1, 0.0]}, "tcp must be 3 numbers"),
        ({"rest": {"right": [0.0] * 6}}, "rest must be the arm's joints in radians"),
        ({"urdf": "missing.urdf"}, "is not a file"),
        ({"hz": 30}, "must be an object with the keys"),
        ({"fixed": "joint6"}, "fixed must be a list of joint names"),
        ({"fixed": ["finger_joint"]}, "fixed names 'finger_joint', which is not a moving joint on the chain"),
        ({"fixed": ["joint6"]}, "rest has 6 joints, the chain from 'base_link' to 'link6' has 5"),
    ],
    ids=[
        "no_such_flange",
        "flange_above_base",
        "chain_takes_the_finger",
        "rest_too_short",
        "rest_outside_a_limit",
        "parallel_axes",
        "short_tcp",
        "rest_per_side",
        "no_urdf",
        "extra_key",
        "fixed_not_a_list",
        "fixed_off_the_chain",
        "fixed_left_in_rest",
    ],
)
def test_a_bad_arm_file_is_a_config_error(tmp_path: pathlib.Path, changes: dict[str, Any], message: str) -> None:
    with pytest.raises(DreamConfigError, match=message):
        arm_file.register(str(write(tmp_path, **changes)))


@pytest.mark.parametrize(
    ("limit", "speed"),
    [("", arm_file.MAX_SPEED), ('<limit effort="30" velocity="1.5"/>', 1.5)],
    ids=["no_limit", "velocity_only"],
)
def test_a_continuous_joint_on_the_chain_turns_a_full_turn_each_way(
    tmp_path: pathlib.Path, limit: str, speed: float
) -> None:
    """A continuous joint, as a URDF writes a hand mount that turns without end (a Vega-1's L_hand_mount_roll,
    2026-10-18: `dreamcli tcp` refused it), is solved as a revolute one within [-pi, pi] at its velocity, at most
    YAM's: pyroki builds the arm from the cut URDF, which it refuses with no limits on a continuous joint, and the
    arm's poses are Metal's."""
    urdf = (dream_ik_specs.ASSETS / "metal.urdf").read_text()
    continuous = re.sub(
        r'(<joint name="joint1") type="revolute">(.*?)<limit [^>]*/>',
        rf'\1 type="continuous">\2{limit}',
        urdf,
        count=1,
        flags=re.DOTALL,
    )
    assert 'name="joint1" type="continuous"' in continuous
    name = arm_file.register(str(write(tmp_path, continuous)))
    spec = dream_ik_specs.EMBODIMENTS[name].arm
    assert spec.max_speed[0] == speed
    cut = ET.parse(spec.urdf).getroot().find("joint[@name='joint1']")
    assert cut is not None and cut.get("type") == "revolute"
    ours, metal = ChunkIK(name, "tcp", 30.0), ChunkIK("metal", "tcp", 30.0)
    assert (ours.lo[0], ours.hi[0]) == pytest.approx((-math.pi + spec.limit_margin, math.pi - spec.limit_margin))
    q = np.random.default_rng(1).uniform(metal.lo, metal.hi, (20, 6))  # (poses, joints)
    assert np.allclose(ours.fk(q), metal.fk(q), atol=1e-6)


def test_a_joint_the_file_holds_still_is_cut_at_its_zero(tmp_path: pathlib.Path) -> None:
    """`fixed` names a chain joint the robot holds still though the URDF lets it move (a passive mount, set by a
    person on the setup page): the arm is the chain's other joints, and its poses are Metal's with that joint at 0."""
    name = arm_file.register(str(write(tmp_path, fixed=["joint6"], rest=METAL["rest"][:5])))
    spec = dream_ik_specs.EMBODIMENTS[name].arm
    assert spec.joints == dream_ik_specs.EMBODIMENTS["metal"].arm.joints[:5]
    ours, metal = ChunkIK(name, "tcp", 30.0), ChunkIK("metal", "tcp", 30.0)
    q = np.random.default_rng(2).uniform(metal.lo, metal.hi, (20, 6))  # (poses, joints)
    q[:, 5] = 0.0
    assert np.allclose(ours.fk(q[:, :5]), metal.fk(q), atol=1e-6)


## Dexmate's Vega-1 with dexterous hands on a continuous mount roll, as the platform draws it: in this repository, not
## the library's copy.
VEGA = [
    path
    for path in sorted((pathlib.Path(__file__).resolve().parents[3] / "platform" / "web" / "urdf").glob("*/vega_1*"))
    if b'<joint name="L_hand_mount_roll" type="continuous"' in path.read_bytes()
]


@pytest.mark.skipif(not VEGA, reason="the Vega-1 URDF is beside the platform's web app only")
def test_the_vega_1_left_arm_builds_through_its_continuous_hand_mount(tmp_path: pathlib.Path) -> None:
    """The chain `dreamcli tcp` refused on a Vega-1 (2026-10-18), base to left_hand_C_MC: a three-joint torso, the
    arm's seven and the hand mount's continuous roll. It builds either way a person answers the page: every joint
    moving (eleven, the mount within a full turn), or the torso and the mount held still (the arm's own seven), whose
    poses are the eleven's with those at 0."""
    (vega_urdf,) = VEGA
    (tmp_path / "vega.urdf").write_bytes(vega_urdf.read_bytes())
    vega = {**METAL, "urdf": "vega.urdf", "base": "base", "flange": "left_hand_C_MC"}
    arm = [f"L_arm_j{i}" for i in range(1, 8)]
    held = ["torso_j1", "torso_j2", "torso_j3", "L_hand_mount_roll"]
    rest = [0.0] * 11  # every joint's zero is inside its limits
    (tmp_path / "all.json").write_text(json.dumps({**vega, "rest": rest}))
    (tmp_path / "arm.json").write_text(json.dumps({**vega, "rest": rest[:7], "fixed": held}))
    every = arm_file.register(str(tmp_path / "all.json"))
    own = arm_file.register(str(tmp_path / "arm.json"))
    assert dream_ik_specs.EMBODIMENTS[every].arm.joints == ("torso_j1", "torso_j2", "torso_j3", *arm, held[3])
    assert dream_ik_specs.EMBODIMENTS[own].arm.joints == tuple(arm)
    eleven, seven = ChunkIK(every, "tcp", 30.0), ChunkIK(own, "tcp", 30.0)
    q = np.random.default_rng(3).uniform(seven.lo, seven.hi, (20, 7))  # (poses, arm joints)
    q11 = np.concatenate([np.zeros((20, 3)), q, np.zeros((20, 1))], axis=1)  # (poses, torso + arm + mount)
    assert np.allclose(seven.fk(q), eleven.fk(q11), atol=1e-6)


def test_a_file_that_is_not_json_is_a_config_error(tmp_path: pathlib.Path) -> None:
    path = write(tmp_path)
    path.write_text("urdf: metal.urdf")
    with pytest.raises(DreamConfigError, match="is not JSON"):
        arm_file.register(str(path))

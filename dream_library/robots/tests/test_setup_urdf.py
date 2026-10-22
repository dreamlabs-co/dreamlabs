"""The URDF step's reading of an uploaded URDF: every link where dream's own model of the arm puts it, the guess of the
link the arm stands on and the one it ends at, and a file that is no arm's URDF refused by what is wrong with it."""

import json
import math
import pathlib
import re

import numpy as np
import pytest
from dream import arm_file

from robots import setup_model, setup_session, setup_urdf

## An arm whose zero leaves its elbow's range, with a gripper of two fingers and a camera on its hand.
GRIPPER = """<robot name="two_finger">
  <link name="base"/><link name="upper"/><link name="hand"/><link name="camera"/>
  <link name="finger_l"/><link name="finger_r"/>
  <joint name="shoulder" type="revolute"><parent link="base"/><child link="upper"/>
    <origin xyz="0 0 0.1" rpy="0 0 1.5708"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="1"/></joint>
  <joint name="elbow" type="revolute"><parent link="upper"/><child link="hand"/>
    <origin xyz="0.2 0 0"/><axis xyz="0 1 0"/><limit lower="0.5" upper="1.5" velocity="1"/></joint>
  <joint name="mount" type="fixed"><parent link="hand"/><child link="camera"/><origin xyz="0 0 0.05"/></joint>
  <joint name="left" type="prismatic"><parent link="hand"/><child link="finger_l"/>
    <origin xyz="0.1 0.02 0"/><axis xyz="0 1 0"/><limit lower="0" upper="0.03" velocity="1"/></joint>
  <joint name="right" type="prismatic"><parent link="hand"/><child link="finger_r"/>
    <origin xyz="0.1 -0.02 0"/><axis xyz="0 -1 0"/><limit lower="0" upper="0.03" velocity="1"/></joint>
</robot>"""


def test_every_link_sits_where_dreams_model_of_the_arm_puts_it_at_the_arm_files_rest(tmp_path: pathlib.Path) -> None:
    """The Metal's URDF, read for the page, puts every link where dream's model of the arm (an arm file of it) does
    at the rest the wizard writes (each joint at 0 or its nearest limit), to 10 micrometres; the guess is its root and
    the end of its chain, the links the arm file names."""
    specs = pytest.importorskip("dream_ik.specs", reason="dream's IK needs Python 3.11 or newer")
    urdf = (specs.ASSETS / "metal.urdf").read_text()
    tree = setup_urdf.read(urdf)
    assert (tree["links"][tree["base"]], tree["links"][tree["flange"]]) == ("base_link", "link6")
    _, _, limits, _ = arm_file._chain(urdf.encode(), "base_link", "link6", (), tmp_path / "arm.json")
    rest = [min(max(0.0, lo), hi) for lo, hi in limits]
    (tmp_path / "arm.urdf").write_text(urdf)
    arm = {"urdf": "arm.urdf", "base": "base_link", "flange": "link6", "tcp": [0.1, 0, 0], "approach": [1, 0, 0]}
    (tmp_path / "arm.json").write_text(json.dumps({**arm, "jaw": [0, 1, 0], "rest": rest}))
    model = setup_model.ArmModel(str(tmp_path / "arm.json"))
    dream_poses = dict(zip(model.link_names, model.poses(np.asarray(rest)), strict=True))
    for name, pose in zip(tree["links"], tree["poses"], strict=True):
        assert np.allclose(pose[4:], dream_poses[name][4:], atol=2e-5), name
        turn, dream_turn = (np.asarray(q[:4]) / np.linalg.norm(q[:4]) for q in (pose, dream_poses[name]))
        assert abs(np.dot(turn, dream_turn)) == pytest.approx(1.0, abs=1e-8), name  # under 0.02 degrees apart


def test_a_gripper_of_its_own_is_drawn_at_the_zero_and_the_guess_ends_before_its_fingers() -> None:
    """Each joint's origin and move at the arm's zero, the elbow at its nearest limit, place each link; the guess of
    the end is the end of the longest chain less the fingers that slide after the elbow: the hand they hang off."""
    tree = setup_urdf.read(GRIPPER)
    assert tree["links"] == ["base", "upper", "hand", "camera", "finger_l", "finger_r"]
    assert tree["parents"] == [-1, 0, 1, 2, 2, 2]
    assert (tree["base"], tree["flange"]) == (0, 2)  # hand: finger_l is the longest chain, and it slides
    poses = {name: pose for name, pose in zip(tree["links"], tree["poses"], strict=True)}
    turn = math.sqrt(0.5)
    assert poses["upper"] == pytest.approx([turn, 0, 0, turn, 0, 0, 0.1], abs=1e-5)
    # The elbow at 0.5 rad about y, the hand 0.2 m along the upper arm's x, which the shoulder turned to the base's y.
    assert poses["hand"][4:] == pytest.approx([0, 0.2, 0.1], abs=1e-5)
    elbow = np.array([math.cos(0.25), 0, math.sin(0.25), 0])
    expected = [turn * elbow[0], -turn * elbow[2], turn * elbow[2], turn * elbow[0]]  # shoulder, then elbow
    assert poses["hand"][:4] == pytest.approx(expected, abs=1e-5)
    assert poses["camera"][4:] == pytest.approx([0, 0.2 + 0.05 * math.sin(0.5), 0.1 + 0.05 * math.cos(0.5)], abs=1e-5)


def test_a_joint_that_slides_before_the_last_turning_one_is_the_arms_own() -> None:
    """A lift that slides before the wrist turns is a joint of the arm, not a finger, so the guess keeps the wrist."""
    lift = """<robot name="lift"><link name="base"/><link name="carriage"/><link name="wrist"/>
  <joint name="lift" type="prismatic"><parent link="base"/><child link="carriage"/>
    <axis xyz="0 0 1"/><limit lower="0" upper="0.5" velocity="1"/></joint>
  <joint name="turn" type="revolute"><parent link="carriage"/><child link="wrist"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" velocity="1"/></joint>
</robot>"""
    tree = setup_urdf.read(lift)
    assert tree["links"][tree["flange"]] == "wrist"


def _onshape_urdf() -> str:
    """A straight six-joint arm as Onshape exports one: every joint hangs off a servo link fixed to the body before
    it, whose origin sits 0.3 m behind the joint in every pose."""
    links, joints, parent = ['<link name="base_link"/>'], [], "base_link"
    for i in range(1, 7):
        links += [f'<link name="servo{i}"/>', f'<link name="link{i}"/>']
        joints.append(
            f'<joint name="fix{i}" type="fixed"><parent link="{parent}"/><child link="servo{i}"/>'
            '<origin xyz="-0.3 0 0"/></joint>'
        )
        axis = "0 0 1" if i in (1, 4, 6) else "0 1 0"
        joints.append(
            f'<joint name="joint{i}" type="revolute"><parent link="servo{i}"/><child link="link{i}"/>'
            f'<origin xyz="0.3 0 0.1"/><axis xyz="{axis}"/><limit lower="-2" upper="2" velocity="1"/></joint>'
        )
        parent = f"link{i}"
    return '<robot name="onshape">' + "".join(links) + "".join(joints) + "</robot>"


def test_a_cad_exported_arm_is_drawn_through_its_joints_not_its_fixed_links(tmp_path: pathlib.Path) -> None:
    """An Onshape export of a five-joint arm: bones through the fixed servo links' origins drew a straight arm
    zig-zagged though its kinematics were right. The bones run from joint to joint and on to the tool, passing
    the fixed links by, so a straight arm is drawn straight."""
    pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")
    urdf = _onshape_urdf()
    arm_file._chain(urdf.encode(), "base_link", "link6", (), tmp_path / "arm.json")
    (tmp_path / "arm.urdf").write_text(urdf)
    arm = {"urdf": "arm.urdf", "base": "base_link", "flange": "link6", "tcp": [0, 0, 0.05], "approach": [0, 0, 1]}
    (tmp_path / "arm.json").write_text(json.dumps({**arm, "jaw": [0, 1, 0], "rest": [0.0] * 6}))
    model = setup_model.ArmModel(str(tmp_path / "arm.json"))
    names = [(model.link_names[p], model.link_names[c]) for p, c in model.chain()["segments"]]
    assert names == [
        ("base_link", "link1"),
        *[(f"link{i}", f"link{i + 1}") for i in range(1, 6)],
        ("link6", model.link_names[model.tool_link]),
    ]
    poses = model.poses(model.rest)
    ends = {end for pair in model.segments for end in pair}
    assert np.allclose(poses[sorted(ends), 4:6], 0.0, atol=1e-6), "every bone stands on the arm's straight line"
    assert np.allclose(poses[model.link_names.index("servo3"), 4], -0.3, atol=1e-6), "the servo links lie off it"


def test_a_five_joint_arm_file_is_five_wide_from_the_model_to_the_home(tmp_path: pathlib.Path) -> None:
    """An arm file of a five-joint URDF. An arm has as many joints as its home lists, so every joint vector of the
    model is five long, with no padded sixth slot, the home the wizard saves lists five, and dream's IK solves the
    five."""
    pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")
    links = '<link name="base_link"/>' + "".join(f'<link name="l{i}"/>' for i in range(1, 6))
    joints = "".join(
        f'<joint name="j{i}" type="revolute"><parent link="{"base_link" if i == 1 else f"l{i - 1}"}"/>'
        f'<child link="l{i}"/><origin xyz="0 0 0.1"/><axis xyz="{"0 0 1" if i in (1, 5) else "0 1 0"}"/>'
        '<limit lower="-2" upper="2" velocity="1"/></joint>'
        for i in range(1, 6)
    )
    (tmp_path / "arm.urdf").write_text(f'<robot name="five">{links}{joints}</robot>')
    arm = {"urdf": "arm.urdf", "base": "base_link", "flange": "l5", "tcp": [0, 0, 0.05], "approach": [0, 0, 1]}
    (tmp_path / "arm.json").write_text(json.dumps({**arm, "jaw": [0, 1, 0], "rest": [0.0, 0.3, 0.3, 0.3, 0.0]}))
    model = setup_model.ArmModel(str(tmp_path / "arm.json"))
    assert model.width == 5 and model.held == ()
    assert [len(v) for v in (model.rest, model.lower, model.upper)] == [5, 5, 5]
    assert model.rest.tolist() == [0.0, 0.3, 0.3, 0.3, 0.0]
    assert model.out_of_range(np.zeros(5)) == [] and model.out_of_range([0, 0, 0, 0, 3.0]) == [4]
    assert len(model.rise(model.rest, "left")) == 5 and len(model.chain()["joints"]) == 5
    robot = setup_session.Robot("custom", "My own robot", str(tmp_path / "arm.json"), roles=())
    assert len(setup_session.arm_config(robot, "/dev/ttyACM0", {}, "left", model.rest).home) == 5
    targets = {side: np.stack([model.tool(model.rest, side)] * 2) for side in model.sides}
    solved = model.solve(targets, {side: model.rests[side] for side in model.sides})
    assert all(solved[side]["q"].shape == (2, 5) for side in model.sides)


## A link of each visual kind URDF has, and one with no visual: what the page draws from, whatever the zip holds.
LOOKS = """<robot name="looks">
  <link name="base"><visual><origin xyz="0 0 0.05" rpy="0 0 1.5708"/>
    <geometry><mesh filename="package://looks/meshes/base.STL" scale="0.001 0.001 0.001"/></geometry></visual>
    <collision><geometry><mesh filename="package://looks/meshes/base_collision.stl"/></geometry></collision></link>
  <link name="arm"><visual><geometry><box size="0.04 0.04 0.3"/></geometry></visual>
    <visual><origin xyz="0 0 0.15"/><geometry><cylinder radius="0.02" length="0.1"/></geometry></visual></link>
  <link name="hand"><visual><geometry><sphere radius="0.03"/></geometry></visual>
    <visual><geometry><mesh filename="../meshes/hand.dae"/></geometry></visual></link>
  <link name="tcp"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="arm"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" velocity="1"/></joint>
  <joint name="j2" type="fixed"><parent link="arm"/><child link="hand"/><origin xyz="0 0 0.3"/></joint>
  <joint name="j3" type="fixed"><parent link="hand"/><child link="tcp"/><origin xyz="0 0 0.05"/></joint>
</robot>"""


def test_each_links_visuals_are_read_for_the_page_to_draw_as_the_arm_looks() -> None:
    """Per link, in the file's order: each visual's pose on its link and its mesh as the URDF names it (the page finds
    it in the .zip) with its scale, or its box, cylinder or sphere; collisions are not drawn, and a link with no visual
    has none, so the page draws it plain."""
    tree = setup_urdf.read(LOOKS)
    turn = math.sqrt(0.5)
    base, arm, hand, tcp = tree["visuals"]
    assert base == [
        {
            "pose": pytest.approx([turn, 0, 0, turn, 0, 0, 0.05], abs=1e-5),
            "mesh": "package://looks/meshes/base.STL",
            "scale": [0.001, 0.001, 0.001],
        }
    ]
    assert arm == [
        {"pose": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "box": [0.04, 0.04, 0.3]},
        {"pose": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15], "cylinder": [0.02, 0.1]},
    ]
    assert hand == [
        {"pose": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "sphere": 0.03},
        {"pose": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "mesh": "../meshes/hand.dae", "scale": [1.0, 1.0, 1.0]},
    ]
    assert tcp == []


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("<robot", "this file is not a URDF: it does not parse as XML"),
        (
            '<robot><link name="a"><visual><geometry><cylinder radius="0.1"/></geometry></visual></link></robot>',
            "a URDF cylinder needs a length",
        ),
        (
            '<robot><link name="a"><visual><geometry><box/></geometry></visual></link></robot>',
            "a URDF size must be three numbers, got ''",
        ),
        ('<sdf><link name="a"/></sdf>', "this file is not a URDF: it has no <robot> with named links"),
        ('<robot><link name="a"/><link name="b"/></robot>', "must hang from one root, not from ['a', 'b']"),
        (
            '<robot><link name="a"/><joint name="j" type="fixed"><parent link="a"/><child link="z"/></joint></robot>',
            "joint 'j' joins 'a' to 'z', not two links of the URDF",
        ),
        (
            '<robot><link name="a"/><link name="b"/><link name="c"/>'
            '<joint name="j" type="fixed"><parent link="a"/><child link="c"/></joint>'
            '<joint name="k" type="fixed"><parent link="b"/><child link="c"/></joint></robot>',
            "link 'c' hangs off two joints",
        ),
    ],
)
def test_a_file_that_is_no_arms_urdf_is_refused_by_what_is_wrong(text: str, message: str) -> None:
    """The page shows the reason under the file box."""
    with pytest.raises(ValueError, match=re.escape(message)):
        setup_urdf.read(text)

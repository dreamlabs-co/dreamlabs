"""The setup wizard's judgments one at a time (`robots.setup_checks`), and the URDF model they use."""

import math

import numpy as np
import numpy.typing as npt
import pytest

from robots import setup_checks, setup_model
from robots.setup_checks import Reading

YAM_TOLERANCE = setup_checks.TOLERANCES["yam"]


@pytest.fixture(scope="module")
def yam() -> setup_model.ArmModel:
    pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")
    return setup_model.ArmModel("yam")


@pytest.fixture(scope="module")
def metal() -> setup_model.ArmModel:
    pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")
    return setup_model.ArmModel("metal")


def nudge(
    joint: int,
    sent_deg: float,
    after_deg: list[float],
    *,
    during: tuple[tuple[float, list[float]], ...] = (),
    lifts: bool = False,
    followed: bool = True,
) -> setup_checks.Finding:
    """judge_nudge() on readings from 0 seconds (all zero) to 1.6 (`after_deg`), and any `(t_s, joints)` between; the
    move takes 1 second, the rest is its settle."""
    path = [Reading(0.0, np.zeros(6), 1.0)]
    path += [Reading(t_s, np.radians(joints), 1.0) for t_s, joints in during]
    path.append(Reading(1.6, np.radians(after_deg), 1.0))
    return setup_checks.judge_nudge(joint, math.radians(sent_deg), path, 1.0, lifts=lifts, followed=followed)


@pytest.mark.parametrize(
    ("joint", "after", "cause"),
    [
        (2, [0, 0, 5.0, 0, 0, 0], None),
        (2, [0, 0, 4.0, 0, 0, 0], None),
        (2, [0, 0, 2.5, 0.4, 0, 0], None),
        (2, [0, 0, 7.5, 0, 0, 0], None),
        (2, [0, 0, -5.0, 0, 0, 0], "write_flipped"),
        (2, [0, 0, 0, 5.0, 0, 0], "write_order"),
        (2, [0, 0, 1.5, 0, 0, 0], None),
        (2, [0, 0, 1.0, 0, 0, 0], None),
        (2, [0, 0, 0.9, 0, 0, 0], "joint_stuck"),
        (2, [0, 0, 0.3, 0, 0, 0], "joint_stuck"),
        (2, [0, 0, 9.0, 0, 0, 0], None),
        (2, [0, 0, 9.5, 0, 0, 0], None),
        (2, [0, 0, 10.5, 0, 0, 0], "write_size"),
        (2, [0, 0, 286.5, 0, 0, 0], "write_size"),
    ],
)
def test_a_nudge_names_a_write_that_went_wrong(joint: int, after: list[float], cause: str | None) -> None:
    assert nudge(joint, 5.0, after).cause == cause


def test_a_joint_that_moves_the_way_sent_5x_under_to_2x_over_passes() -> None:
    """On an SO-101: sent -5.0 degrees, the joint moved -1.8 and the check said nothing moved.
    The way sent by a fifth to twice the move passes ("5x under - 2x over"); under a fifth fails, and only about no
    move at all says nothing moved; over twice fails as a write in other units."""

    def judged(moved_deg: float) -> setup_checks.Finding:
        return nudge(1, -5.0, [0.0, moved_deg, 0.0, 0.0, 0.0, 0.0])

    assert judged(-1.8) == setup_checks.Finding("pass", "sent joint 2 -5.0 degrees; it moved -1.8", joint=1)
    assert judged(-0.8).message == "sent joint 2 -5.0 degrees; it moved under a fifth of that (-0.8)"
    assert judged(-0.2).message == "sent joint 2 -5.0 degrees; nothing moved (-0.2)"
    assert judged(-0.8).cause == judged(-0.2).cause == "joint_stuck"
    assert judged(1.8).cause == "write_flipped"
    assert judged(-9.5).state == "pass"
    assert judged(-10.5).cause == "write_size"


def test_a_joint_that_falls_short_names_one_cause_from_what_else_the_readings_say(
    metal: setup_model.ArmModel,
) -> None:
    """A Metal arm whose driver has no gravity torque: joints 2 and 3 move under a fifth of their nudge, joint 1
    having followed, and the check says why, not only that they barely moved. A
    joint that lifts the arm (by dream's URDF) and falls short once the arm has shown it takes commands is its weight;
    nothing moving on the arm's first joint is its power; one joint that does not lift, short, is stuck."""
    lifting = metal.lifting(metal.rest, "left")
    assert {1, 2} <= lifting and not {0, 5} & lifting, lifting
    for joint in (1, 2):
        after = [0.0] * 6
        after[joint] = 0.6
        finding = nudge(joint, 5.0, after, lifts=joint in lifting)
        assert (finding.cause, finding.joint) == ("weight", joint)
        assert finding.as_json()["title"] == "This arm cannot hold its own weight."
    assert nudge(0, 5.0, [0.0] * 6, followed=False).cause == "no_power"
    assert nudge(1, 5.0, [0.0, 0.6, 0.0, 0.0, 0.0, 0.0], lifts=True, followed=False).cause == "weight"
    assert nudge(5, 5.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.3], lifts=5 in lifting).cause == "joint_stuck"


def test_a_neighbour_drifting_while_the_joint_sent_falls_short_is_not_swapped_wiring(
    metal: setup_model.ArmModel,
) -> None:
    """A Metal arm and a Maker Arm: sent joint 3 +5.0 degrees, it fell short and joint 4 drifted
    -1.1, and the check said the driver writes joints in another order, on every run. Swapped wiring moves the other
    joint by about the whole nudge; a drift under half of it is the joint sent not following, named as such."""
    lifting = metal.lifting(metal.rest, "left")
    drifted = nudge(2, 5.0, [0.0, 0.0, 0.6, -1.1, 0.0, 0.0], lifts=2 in lifting)
    assert (drifted.cause, drifted.joint) == ("weight", 2)
    assert drifted.message == (
        "sent joint 3 +5.0 degrees; it moved under a fifth of that (+0.6), while joint 4 drifted -1.1"
    )
    still = nudge(2, 5.0, [0.0, 0.0, 0.2, -1.1, 0.0, 0.0])
    assert still.cause == "joint_stuck" and "it did not move (+0.2)" in still.message
    assert nudge(2, 5.0, [0.0, 0.0, 2.0, -1.1, 0.0, 0.0]).state == "pass"
    assert nudge(2, 5.0, [0.0, 0.0, 0.3, 2.6, 0.0, 0.0]).cause == "write_order"
    assert nudge(2, 5.0, [0.0, 0.0, 3.0, 4.0, 0.0, 0.0]).cause == "write_order"
    assert nudge(2, 5.0, [0.0, 0.0, 4.0, 3.0, 0.0, 0.0]).state == "pass"


def test_each_way_the_driver_writes_joints_wrong_has_its_own_headline_and_line_to_change() -> None:
    """The headline "Your driver writes joints differently from how it reads them" is the title of five causes, so
    alone it could not say which check failed. Each says what was seen; the impact stays the same."""
    order = nudge(2, 5.0, [0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    flipped = nudge(2, 5.0, [0.0, 0.0, -5.0, 0.0, 0.0, 0.0])
    size = nudge(2, 5.0, [0.0, 0.0, 286.5, 0.0, 0.0, 0.0])
    assert [finding.cause for finding in (order, flipped, size)] == ["write_order", "write_flipped", "write_size"]
    codes = ["write_order", "write_flipped", "write_size", "not_followed", "too_far"]
    words = [setup_checks.CAUSES[code] for code in codes]
    assert len({w.title for w in words}) == 5 and len({w.fix for w in words}) == 5
    assert len({w.impact for w in words}) == 1
    assert "order" in words[0].fix and "sign" in words[1].fix and "radians" in words[2].fix
    assert [w.title for w in words] == [
        "Another joint moved instead of the one we sent.",
        "This joint moved the wrong way.",
        "This joint moved over twice as far as we sent.",
        "The arm did not follow what we sent.",
        "This joint went farther from home than any test move.",
    ]


def test_readings_during_the_move_name_a_swing_past_and_a_slow_follow() -> None:
    swung = nudge(
        2, 5.0, [0.0, 0.0, 5.0, 0.0, 0.0, 0.0], during=((0.5, [0, 0, 4.0, 0, 0, 0]), (1.0, [0, 0, 11.0, 0, 0, 0]))
    )
    assert swung.cause == "overshoot"
    assert swung.message == "sent joint 3 +5.0 degrees; it swung to +11.0, then settled at +5.0"
    assert nudge(2, 5.0, [0.0, 0.0, 5.0, 0.0, 0.0, 0.0], during=((1.0, [0, 0, 9.0, 0, 0, 0]),)).state == "pass"
    creeping = nudge(2, 5.0, [0.0, 0.0, 0.9, 0.0, 0.0, 0.0], during=((1.0, [0, 0, 0.3, 0, 0, 0]),), lifts=True)
    assert creeping.cause == "slow" and creeping.message.endswith("(+0.9), still creeping on as it settled")
    settled_short = nudge(2, 5.0, [0.0, 0.0, 0.9, 0.0, 0.0, 0.0], during=((1.0, [0, 0, 0.8, 0, 0, 0]),), lifts=True)
    assert settled_short.cause == "weight"


def test_a_gripper_nudge_and_a_reach() -> None:
    assert setup_checks.judge_gripper_nudge(-0.2, 1.0, 0.81).state == "pass"
    assert setup_checks.judge_gripper_nudge(-0.2, 1.0, 1.0).cause == "gripper_still"
    assert setup_checks.judge_gripper_nudge(-0.2, 0.2, 0.4).cause == "gripper_write"
    assert setup_checks.judge_gripper_nudge(-0.2, 1.0, 0.4).cause == "gripper_write_size"
    pose = np.eye(4)
    sent = (pose, setup_model.pose_offset(pose, (0.03, 0.0, 0.0), (0.0, 0.0, 0.0)))
    near = setup_model.pose_offset(pose, (0.035, 0.0, 0.0), (0.0, 0.0, math.radians(2.0)))
    short = np.radians([0.0, 0.0, 0.0, 0.0, 6.0, 0.0])
    lifting = frozenset({1, 2, 3})

    def reach(reached: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]) -> setup_checks.Finding:
        return setup_checks.judge_reach("+x", False, sent, reached, YAM_TOLERANCE, short, lifting)

    assert reach((pose, near)).state == "pass"
    far = setup_model.pose_offset(pose, (0.07, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert reach((pose, far)).cause == "missed" and reach((pose, far)).joint == 4
    turned = setup_model.pose_offset(pose, (0.03, 0.0, 0.0), (0.0, math.radians(6.0), 0.0))
    assert reach((pose, turned)).cause == "missed"
    sag = setup_model.pose_offset(pose, (0.0, 0.0, -0.02), (math.radians(5.0), 0.0, 0.0))
    sagged = setup_model.pose_offset(sag, (0.03, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert reach((sag, sagged)).state == "pass"
    elbow = np.radians([0.0, 0.0, -6.0, 0.0, 1.0, 0.0])
    missed_by_the_elbow = setup_checks.judge_reach("+x", False, sent, (pose, far), YAM_TOLERANCE, elbow, lifting)
    assert (missed_by_the_elbow.cause, missed_by_the_elbow.joint) == ("weight", 2)
    assert missed_by_the_elbow.message.endswith("; joint 3 was 6.0 degrees off")


def test_a_sweep_move_shrunk_below_the_tolerance_is_judged_by_where_the_hand_ended() -> None:
    """An SO-101's left arm: from its seated home the sweep's -z is halved twice to 0.4 cm; the
    hand goes 0.07 cm (18%) with the shoulder 0.7 degrees off, inside the servos' dead band, yet ends 0.33 cm from
    where it was sent against a 1.5 cm tolerance, so it is not "missed". A move smaller than the tolerance is
    judged by where the hand ended; a large one still by its share."""
    pose = np.eye(4)
    so101 = setup_checks.tolerance_for("so101")
    lifting = frozenset({1, 2, 3})
    shoulder = np.radians([0.0, 0.7, 0.0, 0.0, 0.0, 0.0])
    sent = (pose, setup_model.pose_offset(pose, (0.0, 0.0, -0.004), (0.0, 0.0, 0.0)))
    ended = setup_model.pose_offset(pose, (0.0, 0.0, -0.0007), (0.0, math.radians(1.0), 0.0))
    small = setup_checks.judge_reach("-z", False, sent, (pose, ended), so101, shoulder, lifting)
    assert small.state == "pass", small.message
    # The same share of a 3 cm move is a miss: 2.4 cm short is past the tolerance.
    whole = (pose, setup_model.pose_offset(pose, (0.0, 0.0, -0.03), (0.0, 0.0, 0.0)))
    short = setup_model.pose_offset(pose, (0.0, 0.0, -0.0054), (0.0, math.radians(1.0), 0.0))
    assert setup_checks.judge_reach("-z", False, whole, (pose, short), so101, shoulder, lifting).state == "fail"
    # A small move whose hand ended past the tolerance still fails.
    away = setup_model.pose_offset(pose, (0.0, 0.0, 0.016), (0.0, 0.0, 0.0))
    assert setup_checks.judge_reach("-z", False, sent, (pose, away), so101, shoulder, lifting).state == "fail"


def test_a_turn_allows_twice_the_shift_and_only_a_lifting_joint_behind_its_move_is_weight() -> None:
    # A YAM: "turn about y" turned 114% with the tool 1.5 cm off (1.0 cm limit), joint 3 2.1 degrees off.
    pose = np.eye(4)
    sent = (pose, setup_model.pose_offset(pose, (0.0, 0.0, 0.0), (0.0, math.radians(10.0), 0.0)))
    swung = setup_model.pose_offset(pose, (0.015, 0.0, 0.0), (0.0, math.radians(11.4), 0.0))
    joint3 = np.radians([0.0, 0.0, 2.1, 0.0, 0.0, 0.0])
    lifting = frozenset({1, 2, 3})
    assert (
        setup_checks.judge_reach("turn about y", True, sent, (pose, swung), YAM_TOLERANCE, joint3, lifting).state
        == "pass"
    )
    far = setup_model.pose_offset(pose, (0.03, 0.0, 0.0), (0.0, math.radians(11.4), 0.0))
    went_past = np.radians([0.0, 0.0, 5.0, 0.0, 0.0, 0.0])
    behind = np.radians([0.0, 0.0, -5.0, 0.0, 0.0, 0.0])

    def cause(moved: npt.NDArray[np.float64]) -> str | None:
        return setup_checks.judge_reach(
            "turn about y", True, sent, (pose, far), YAM_TOLERANCE, joint3, lifting, moved
        ).cause

    assert (cause(went_past), cause(behind)) == ("missed", "weight")


def test_a_lifting_joint_behind_no_more_than_the_base_yaw_followed_its_nudge_is_not_weight() -> None:
    """An arm with DM4340P motors on joints 1 to 3 in POS_VEL, as on the Seeed reBot: every 4340P joint,
    the base yaw that lifts nothing included, moved 3.6 to 3.8 of its 5 degree nudge, joint 4 5.5, joints 5 and 6
    exactly 5; then "Sweep left +x: 1.3 of 1.5 cm, 90%; joint 2 3.9 degrees off" was called the arm's weight. A
    lifting joint that fell no farther short of its nudge than an unloaded joint is how the arm's joints follow; one
    short where the base yaw was exact is still its weight."""
    pose = np.eye(4)
    sent = (pose, setup_model.pose_offset(pose, (0.015, 0.0, 0.0), (0.0, 0.0, 0.0)))
    reached = (pose, setup_model.pose_offset(pose, (0.01349, 0.0, 0.0), (0.0, math.radians(6.0), 0.0)))
    short = np.radians([0.0, -3.9, 1.0, 0.5, 0.0, 0.0])
    moved = np.radians([0.0, 15.0, -10.0, -5.0, 0.0, 0.0])
    lifting = frozenset({1, 2, 3})

    def cause(nudged_deg: list[float] | None) -> str | None:
        nudged = None if nudged_deg is None else np.asarray(nudged_deg) / 5.0
        finding = setup_checks.judge_reach(
            "+x", False, sent, reached, setup_checks.CUSTOM_TOLERANCE, short, lifting, moved, nudged
        )
        assert finding.state == "fail" and finding.joint == 1
        assert finding.message.startswith("+x: moved 1.3 of 1.5 cm (90%)")
        return finding.cause

    assert cause([3.8, 3.6, 3.7, 5.5, 5.0, 5.0]) == "missed"
    assert cause([3.6, 3.8, 3.7, 4.7, 5.0, 5.0]) == "missed"
    assert cause([5.0, 3.6, 3.7, 5.0, 5.0, 5.0]) == "weight"
    assert cause([5.0, 2.5, 5.0, 5.0, 5.0, 5.0]) == "weight"
    assert cause([5.0, math.nan, 5.0, 5.0, 5.0, 5.0]) == "weight"
    assert cause(None) == "weight"


def test_the_wiggle_names_one_device_or_says_several() -> None:
    base = {d: Reading(0.0, np.zeros(6), 1.0) for d in ("a", "b")}
    moved = {"a": Reading(1.0, np.radians([0, 12, 0, 0, 0, 0]), 1.0), "b": Reading(1.0, np.radians([1] * 6), 1.0)}
    scores = setup_checks.wiggle_scores(base, moved)
    assert setup_checks.moved_device(scores) == ("a", False)
    both = dict(moved, b=Reading(1.0, np.radians([20, 0, 0, 0, 0, 0]), 1.0))
    assert setup_checks.moved_device(setup_checks.wiggle_scores(base, both)) == (None, True)
    assert setup_checks.moved_device({"a": 0.01}) == (None, False)


def test_every_cause_has_words_and_each_body_a_tolerance() -> None:
    for cause, words in setup_checks.CAUSES.items():
        shown = setup_checks.Finding("fail", "x", cause).as_json()
        assert (shown["title"], shown["impact"], shown["fix"]) == (words.title, words.impact, words.fix)
        assert all(text.endswith(".") for text in (words.title, words.impact, words.fix)), cause
    assert setup_checks.Finding("pass", "x").as_json()["title"] == ""
    assert setup_checks.Finding("pass", "x").as_json()["go_on"] == ""


def test_a_failed_check_says_whether_continuing_past_it_is_safe() -> None:
    """A failed Move under power says whether continuing is safe. A robot that runs, less precisely, may go on;
    one that would move unlike the model, or not at all, is fixed first."""
    safe = {cause for cause, words in setup_checks.CAUSES.items() if words.safe}
    assert safe == {"weight", "slow", "missed"}
    for cause in setup_checks.CAUSES:
        shown = setup_checks.Finding("fail", "x", cause).as_json()
        assert shown["safe"] is (cause in safe)
        assert shown["go_on"] == setup_checks.GO_ON[cause in safe]
    assert setup_checks.GO_ON[True].startswith("You can continue")
    assert setup_checks.GO_ON[False].startswith("Fix this before you continue")
    assert setup_checks.tolerance_for("custom") == setup_checks.CUSTOM_TOLERANCE
    assert setup_checks.tolerance_for("makers").pos_m > setup_checks.tolerance_for("yam").pos_m
    assert setup_checks.tolerance_for("so101") == setup_checks.TOLERANCES["so101"]


def test_the_models_chain_turns_the_way_the_urdf_does(yam: setup_model.ArmModel) -> None:
    """A positive turn of the base joint about its axis (+z) swings the tool to +y."""
    chain = yam.chain()
    base = chain["joints"][0]
    np.testing.assert_allclose(base["axis"], [0, 0, 1], atol=1e-6)
    turned = yam.rest.copy()
    turned[0] += 0.3
    assert yam.poses(turned)[yam.tool_link, 5] > yam.poses(yam.rest)[yam.tool_link, 5]
    assert len(chain["links"]) == len(chain["segments"]) + 1
    np.testing.assert_allclose(yam.poses(yam.rest)[yam.tool_link, 4:7], yam.tool(yam.rest)[:3, 3], atol=1e-6)
    assert yam.frame([np.nan] * 6) is None


def test_a_sweep_move_smaller_than_the_tolerance_is_judged_by_where_it_ended() -> None:
    """an SO-101 sweep shrank -z to 0.3 cm and the arm read 0 of it; that is inside its 1.5 cm tolerance."""
    so101 = setup_checks.TOLERANCES["so101"]
    pose, lifting, short = np.eye(4), frozenset({1, 2}), np.radians([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    sent = (pose, setup_model.pose_offset(pose, (0.0, 0.0, -0.003), (0.0, 0.0, 0.0)))
    assert setup_checks.judge_reach("-z 0.3 cm", False, sent, (pose, pose), so101, short, lifting).state == "pass"
    big = (pose, setup_model.pose_offset(pose, (0.0, 0.0, -0.03), (0.0, 0.0, 0.0)))
    assert setup_checks.judge_reach("-z 3 cm", False, big, (pose, pose), so101, short, lifting).state == "fail"

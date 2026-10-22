"""The setup wizard's checks against planted mistakes: where each driver mistake an operator makes is caught.

A two-arm station of fake arms (tests/setup_fakes.py) runs the session the way the page does: open every adapter
disengaged, find each arm by moving it, put the arms in their home pose and press OK, then the powered checks. A
clean driver passes everything on every arm dream has a URDF of. Each mistake below is planted in the left arm's driver
only, on a YAM homed at the start of an episode of our data, and is caught where the table says. "Your eyes" means
every check passes: only the operator sees it, at home where the drawing of the arm does not look like the real arm, or
during the powered checks, where they answer No to "Did every joint and hand move like the drawing?"
(test_a_no_from_your_eyes_fails_the_powered_step). Nothing compares home with a pose it should be (an expected pose
drawn for every arm is wrong on some, the Makers Arm and the SO-101 among them), so a driver that reads and writes a
joint the same wrong way is the eyes' to catch.
While a joint is nudged the page highlights it in the drawing, one joint at a time
(test_the_nudge_moves_one_joint_at_a_time_and_names_it_while_it_moves).

| Mistake | Caught at | Cause |
| --- | --- | --- |
| degrees instead of radians | home, joints outside the URDF's range | outside |
| LeRobot -100 to 100 | home, joints outside the URDF's range | outside |
| a joint's zero shifted 14.4 degrees (SO-101 horn re-seated) | your eyes, at home | eyes |
| zero offset (shoulder, 90 degrees) | your eyes, at home | eyes |
| a joint flipped where home is far from 0 (joint 3, home 7 degrees) | your eyes, at home or during the sweep | eyes |
| two joints swapped whose home values differ (joints 4 and 5) | your eyes, at home or during the sweep | eyes |
| a joint flipped where home is near 0 (joint 6, home 1 degree) | your eyes, during the sweep | eyes |
| two joints swapped whose home values are alike (joints 1 and 6) | your eyes, during the sweep | eyes |
| gripper inverted | your eyes, during the gripper nudge | eyes |
| gripper not read | powered, the gripper nudge | gripper_still |
| gripper in mm | powered, the hold check | gripper_write |
| left and right arms swapped | finding the arms | crossed |
| reads right, writes a joint flipped | powered, the nudge | write_flipped |
| reads right, writes degrees | powered, the hold check | not_followed |
| reads right, writes two joints swapped | powered, the nudge | write_order |
| reads right, writes the gripper the other way | powered, the hold check | gripper_write |
| reads right, writes 1000 times too big | powered, the first tick (the limit) | too_far |
"""

import dataclasses
import functools
import math
import pathlib
import threading
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import setup_fakes
import so101_fakes
import test_metal
import test_so101
import test_so101_calibration
from makers_fakes import FakeClock
from setup_fakes import CLEAN, FakeArm, Hands, Mistakes, Rig, device

from robots import (
    bodies,
    config,
    errors,
    feetech,
    followers,
    setup_checks,
    setup_model,
    setup_motion,
    setup_session,
    so101,
    yam,
)

pytest.importorskip("dream_ik", reason="dream's IK needs Python 3.11 or newer")


@pytest.fixture(autouse=True)
def _embodiments(monkeypatch: pytest.MonkeyPatch) -> None:
    """The account's embodiments without the network (setup_fakes.stub_embodiments())."""
    setup_fakes.stub_embodiments(monkeypatch)


## The station's two adapters: the left arm's and the right arm's.
LEFT, RIGHT = "can0", "can1"

## The arms the wizard can be set up for, by the dream URDF it checks against, with the body that opens them.
ROBOTS = {"yam": "yam", "makers": "makers", "metal": "metal"}

GRIP_OPEN_STOP = 114.0 / 137.5

## One spline tooth of a re-seated SO-101 servo horn, radians.
SPLINE_TOOTH_RAD = math.radians(14.4)

YAM_HOME_RAD = {
    "left": (-0.0374, 0.0101, 0.12, -0.2241, 0.0612, 0.0219),
    "right": (0.0507, 0.0082, 0.123, -0.2081, -0.0616, -0.0536),
}


@functools.cache
def model(spec: str) -> setup_model.ArmModel:
    """One URDF model per arm for the whole run: building one compiles its kinematics."""
    return setup_model.ArmModel(spec)


def robot_for(spec: str, custom: bool = False) -> setup_session.Robot:
    """The robot the page picks for an arm: ours, or one of the customer's own on that URDF."""
    if custom:
        return setup_session.custom_robot(spec, "my_robot.driver:MyArm", "tester/my-arm")
    return next(r for r in setup_session.robot_choices() if r.body == ROBOTS[spec])


def settings_for(robot: setup_session.Robot) -> dict[str, dict[str, Any]]:
    """Arm settings the body accepts: every body's defaults (a Makers or Metal gripper's ends are measured by hand on
    the home step, Station.set_home())."""
    return {"left": {}, "right": {}}


class Station:
    """A session on a fake two-arm station, driven as the page drives it."""

    def __init__(
        self,
        spec: str = "yam",
        mistakes: dict[str, Mistakes] | None = None,
        custom: bool = False,
        old_channels: dict[str, str] | None = None,
        home_before_off: bool = False,
    ) -> None:
        self.model = model(spec)
        self.rig = Rig({LEFT: FakeArm(self.model.rest), RIGHT: FakeArm(self.model.rest)}, mistakes)
        self.clock = FakeClock()
        self.session = setup_session.Session(
            opener=self.rig.open,
            model_factory=lambda _arm: self.model,
            old_channels=old_channels or {},
            clock=self.clock,
            sleep=self.clock.sleep,
            home_before_off=home_before_off,
        )
        self.hands = Hands(self.session, self.rig, self.clock)
        self.robot = robot_for(spec, custom)
        self.session.choose(self.robot)
        self.devices = [device(LEFT, "AAA111"), device(RIGHT, "BBB222")]

    def find_arms(self) -> list[dict[str, Any]]:
        """Opens both adapters disengaged, then moves the left arm and says so, then the right."""
        opened = self.session.open_arms(self.devices, settings_for(self.robot))
        assert opened == {"can-AAA111": None, "can-BBB222": None}
        replies = []
        for adapter, side in ((LEFT, "left"), (RIGHT, "right")):
            self.hands.move(adapter, 0, 0.3)
            wiggle = self.session.wiggle()
            assert wiggle["moved"] == f"can-{'AAA111' if adapter == LEFT else 'BBB222'}"
            replies.append(self.session.assign(wiggle["moved"], side))
        return replies

    def measure_grippers(self, opening: float = GRIP_OPEN_STOP, closing: float = 0.0) -> dict[str, Any]:
        """The operator opens both grippers by hand to their open stop and records it."""
        for end, grip in (("open", opening), ("closed", closing)):
            for adapter in (LEFT, RIGHT):
                self.hands.grip(adapter, grip)
            found = self.session.record_gripper(end)
        return found

    def set_home(self) -> dict[str, Any]:
        """The operator measures a Makers or Metal arm's grippers (measure_grippers()), puts both arms in their home
        pose and presses OK: a YAM at the start of an episode of our data (#YAM_HOME_RAD), another arm at the URDF's
        rest pose."""
        if self.robot.body in setup_session.GRIPPER_READ_RANGE_DEG:
            self.measure_grippers()
        for side, adapter in (("left", LEFT), ("right", RIGHT)):
            home = YAM_HOME_RAD[side] if self.robot.body == "yam" else self.model.rest
            self.hands.pose(adapter, np.array(home))
        return self.session.set_home()

    def motion(self) -> setup_motion.Motion:
        """The powered checks, run to their end."""
        self.session.start_motion()
        assert self.session.wait_motion(120.0)
        motion = self.session.motion
        assert motion is not None
        return motion


@pytest.mark.parametrize(("spec", "custom"), [("yam", False), ("makers", False), ("metal", False), ("makers", True)])
def test_a_clean_driver_passes_every_check_and_nothing_moves_before_the_popup(spec: str, custom: bool) -> None:
    """Both arms found and home set, then the hold check, home, every nudge and every sweep move pass, and a Yes from
    the operator's eyes keeps it so; no write reaches an arm before the powered checks, the powered phase opens each
    arm with the home the page set, and both arms end holding at home."""
    station = Station(spec, custom=custom)
    replies = station.find_arms()
    assert replies[-1]["done"] and replies[-1]["assigned"] == {"left": "can-AAA111", "right": "can-BBB222"}
    assert all(reply["crossed"] is None for reply in replies)
    assert station.set_home()["out_of_range"] == []
    assert all(arm.writes == 0 and not arm.powered for arm in station.rig.arms.values())
    assert all(arm.disengages >= 1 for arm in station.rig.arms.values())

    motion = station.motion()
    assert motion.state == "done", (motion.error, motion.results[-3:])
    failed = [row for row in motion.results if row["finding"].state != "pass"]
    assert failed == []
    groups = [row["group"] for row in motion.results]
    assert groups.count("hold") == 2 and groups.count("nudge") == 2 * 7
    skipped = len({note["move"] for note in motion.notes if note.get("skipped")})
    assert groups.count("sweep") == 2 * (len(setup_motion.SWEEP_MOVES) - skipped)
    assert all("out of the arm's reach" in note["text"] or "would turn" in note["text"] for note in motion.notes), (
        motion.notes
    )
    assert spec != "makers" or motion.notes == [], motion.notes
    for side, adapter in (("left", LEFT), ("right", RIGHT)):
        np.testing.assert_allclose(station.rig.arms[adapter].q, station.session.home[side], atol=1e-6)
        assert station.rig.arms[adapter].powered
    for cfg, (_, _, side) in zip(station.rig.configs[-2:], station.rig.opened[-2:], strict=True):
        np.testing.assert_allclose(cfg.home, station.session.home[side])
    np.testing.assert_allclose(station.rig.configs[0].home, station.model.rest)
    assert station.session.confirm_eyes(True)["state"] == "done"


@pytest.mark.parametrize(
    ("mistakes", "where", "cause"),
    [
        pytest.param(Mistakes(read_units="degrees"), "home", "outside", id="degrees"),
        pytest.param(Mistakes(read_units="lerobot"), "home", "outside", id="lerobot"),
        pytest.param(Mistakes(offset=(1, SPLINE_TOOTH_RAD)), "eyes", None, id="spline-tooth"),
        pytest.param(Mistakes(offset=(1, math.pi / 2)), "eyes", None, id="zero-offset"),
        pytest.param(Mistakes(flip=2), "eyes", None, id="flipped-far-from-zero"),
        pytest.param(Mistakes(swap=(3, 4)), "eyes", None, id="swapped-unlike"),
        pytest.param(Mistakes(flip=5), "eyes", None, id="flipped-near-zero"),
        pytest.param(Mistakes(swap=(0, 5)), "eyes", None, id="swapped-alike"),
        pytest.param(Mistakes(gripper="inverted"), "eyes", None, id="gripper-inverted"),
        pytest.param(Mistakes(gripper="still"), "nudge", "gripper_still", id="gripper-still"),
        pytest.param(Mistakes(gripper="mm"), "hold", "gripper_write", id="gripper-mm"),
        pytest.param(Mistakes(write_flip=1), "nudge", "write_flipped", id="write-flipped"),
        pytest.param(Mistakes(write_flip=0), "hold", "not_followed", id="write-flipped-off-zero"),
        pytest.param(Mistakes(write_units="degrees"), "hold", "not_followed", id="write-degrees"),
        pytest.param(Mistakes(write_swap=(1, 5)), "nudge", "write_order", id="write-swapped"),
        pytest.param(Mistakes(write_gripper_flip=True), "hold", "gripper_write", id="write-gripper-flipped"),
        pytest.param(Mistakes(write_scale=1000.0), "limit", "too_far", id="write-1000x"),
    ],
)
def test_each_planted_mistake_is_caught_where_the_table_says(
    mistakes: Mistakes, where: str, cause: str | None
) -> None:
    """`home`: set home names left arm joints outside the URDF's range. A powered group (`hold`, `nudge`, `limit`):
    the powered checks stop at that group on the left arm with the cause. `eyes`: everything passes. A
    joint written flipped is caught by the nudge where home is near zero (joint 2, 0.6 degrees), and already by the
    hold check where it is not (joint 1, 2 degrees, so the flipped write moves it 4)."""
    station = Station("yam", {LEFT: mistakes})
    station.find_arms()
    found = station.set_home()["out_of_range"]
    outside = {side: [o["joint"] for o in found if o["side"] == side] for side in ("left", "right")}
    assert outside["right"] == [], outside
    if where == "home":
        assert outside["left"], outside
        return
    assert outside["left"] == [], outside
    motion = station.motion()
    if where == "eyes":
        assert motion.state == "done" and all(row["finding"].state == "pass" for row in motion.results), motion.error
        return
    last = motion.results[-1]
    assert motion.state == "failed" and all(row["finding"].state == "pass" for row in motion.results[:-1])
    assert (last["group"], last["side"], last["finding"].cause) == (where, "left", cause), last
    let_go = cause in setup_checks.LET_GO_CAUSES
    assert motion.let_go is let_go and station.session.powered is not let_go
    assert all(arm.powered is not let_go for arm in station.rig.arms.values())


def test_the_nudge_moves_one_joint_at_a_time_and_names_it_while_it_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    """While a joint is nudged the status names that arm and joint, which the page highlights in the drawing: each
    joint of the left arm in order, then of the right, one at a time, every other joint still; none once done."""
    station = Station("yam")
    station.find_arms()
    station.set_home()
    seen: list[tuple[str, int, int]] = []
    write = setup_fakes.Driver.write

    def watched(driver: setup_fakes.Driver, positions_rad: Any, gripper: float) -> None:
        motion = station.session.motion
        nudging = None if motion is None else motion.status()["nudging"]
        if nudging is not None and driver.arm is station.rig.arms[LEFT if nudging["side"] == "left" else RIGHT]:
            home = station.session.home[nudging["side"]]
            moved = np.flatnonzero(np.abs(np.asarray(positions_rad) - home) > 1e-9)
            assert set(moved) <= {nudging["joint"]}, (nudging, moved)
            seen.append((nudging["side"], nudging["joint"], len(moved)))
        write(driver, positions_rad, gripper)

    monkeypatch.setattr(setup_fakes.Driver, "write", watched)
    motion = station.motion()
    order = list(dict.fromkeys((side, joint) for side, joint, _ in seen))
    assert order == [(side, joint) for side in ("left", "right") for joint in range(6)]
    assert all(any(n == 1 for s, j, n in seen if (s, j) == pair) for pair in order)
    assert motion.state == "done" and motion.status()["nudging"] is None


def test_a_no_from_your_eyes_fails_the_powered_step() -> None:
    """After the powered checks pass, a No to "did each joint and the hand move like the drawing?" fails the step with
    the cause that names joint directions and order. No answer is taken before the checks end."""
    station = Station("yam", {LEFT: Mistakes(flip=5)})
    station.find_arms()
    station.set_home()
    with pytest.raises(ValueError, match="powered checks to their end"):
        station.session.confirm_eyes(True)
    station.motion()
    status = station.session.confirm_eyes(False)
    last = status["results"][-1]
    assert status["state"] == "failed" and (last["group"], last["finding"]["cause"]) == ("eyes", "eyes")
    assert "directions" in last["finding"]["fix"] and "order" in last["finding"]["fix"]


def test_a_new_home_or_gripper_range_drops_the_run_from_before_and_one_answer_is_taken() -> None:
    station = Station("metal")
    station.find_arms()
    station.set_home()
    station.motion()
    station.session.confirm_eyes(True)
    with pytest.raises(ValueError, match="already has your answer"):
        station.session.confirm_eyes(True)
    assert [row["group"] for row in station.session.status()["motion"]["results"]].count("eyes") == 1
    station.session.release()
    station.set_home()
    assert station.session.status()["motion"] is None
    station.motion()
    station.session.release()
    station.measure_grippers()
    assert station.session.status()["motion"] is None


@pytest.mark.parametrize("custom", [False, True])
def test_once_both_arms_are_named_each_opens_again_as_its_side_not_its_adapter_id(custom: bool) -> None:
    """While the arms are found a driver gets the adapter's id; once both are named, `left` and `right`, so a driver
    that keys its gripper's calibration by side reads its gripper on the home step (on a reBot the drawn fingers
    never moved, the driver having reported 0 for want of a side)."""
    station = Station(custom=custom)
    station.find_arms()
    body = station.robot.body
    assert station.rig.opened[:2] == [(body, LEFT, "can-AAA111"), (body, RIGHT, "can-BBB222")]
    assert station.rig.opened[-2:] == [(body, LEFT, "left"), (body, RIGHT, "right")]


def test_left_and_right_swapped_against_the_old_config_are_caught_when_the_arms_are_found() -> None:
    """The old file drove the left arm on the adapter that is really the right arm's: moving the left arm names the
    other adapter, and the page says the two were swapped."""
    station = Station("yam", old_channels={"left": RIGHT, "right": LEFT})
    replies = station.find_arms()
    assert "left and right were swapped" in replies[0]["crossed"]
    assert "drove the right arm" in replies[0]["crossed"]
    assert len(station.session.crossed) == 2


def test_two_arms_moved_at_once_name_neither_and_a_device_of_another_robot_is_let_go() -> None:
    """Both adapters moving is `several` with no device named; after a fresh baseline one moves; an adapter said to be
    another robot's is closed and never asked about again."""
    station = Station("yam")
    session, hands = station.session, station.hands
    session.open_arms(station.devices, settings_for(station.robot))
    hands.move(LEFT, 0, 0.3)
    hands.move(RIGHT, 0, 0.3)
    assert session.wiggle()["several"] and session.wiggle()["moved"] is None
    session.rebase_wiggle()
    hands.move(RIGHT, 1, 0.3)
    assert session.wiggle()["moved"] == "can-BBB222"
    session.assign("can-BBB222", "none")
    assert "can-BBB222" not in session.bank.devices()
    assert station.rig.handles[1].closed


def test_a_device_that_does_not_open_is_named_and_the_rest_still_open() -> None:
    """An adapter whose driver raises is reported with the error; the other opens."""
    station = Station("yam")
    devices = [*station.devices, device("can9", "ZZZ999")]
    opened = station.session.open_arms(devices, settings_for(station.robot))
    assert opened["can-AAA111"] is None and "no adapter can9" in opened["can-ZZZ999"]


def test_limp_arms_settle_before_the_move_that_finds_them_is_measured() -> None:
    station = Station("metal")
    waited: list[float] = []

    def sag(seconds: float) -> None:
        waited.append(seconds)
        for arm in station.rig.arms.values():
            arm.q[1] += 0.2
        station.clock.sleep(seconds)

    station.session._sleep = sag
    opened = station.session.open_arms(station.devices, settings_for(station.robot))
    station.session._sleep = station.clock.sleep
    assert opened == {"can-AAA111": None, "can-BBB222": None} and waited == [setup_session.SETTLE_S]
    station.hands.move(LEFT, 0, 0.3)
    wiggle = station.session.wiggle()
    assert wiggle["moved"] == "can-AAA111" and not wiggle["several"]


def test_a_device_another_robot_drives_is_refused_before_anything_opens() -> None:
    station = Station("yam")
    servo_bus = dict(device(LEFT, "AAA111"), kind="serial", label="USB serial adapter CH340 5A7A")
    with pytest.raises(ValueError, match="USB serial adapter CH340 5A7A cannot be a YAM's arm; untick it"):
        station.session.open_arms([servo_bus, station.devices[1]], settings_for(station.robot))
    assert station.rig.opened == []


def test_after_a_failure_the_arms_are_let_go_and_found_again_without_restarting() -> None:
    station = Station("yam")
    session, rig = station.session, station.rig
    gone = rig.arms.pop(RIGHT)
    opened = session.open_arms(station.devices, settings_for(station.robot))
    assert opened["can-AAA111"] is None and "no adapter can1" in str(opened["can-BBB222"])
    rig.arms[RIGHT] = gone
    session.release_arms()
    assert rig.handles[0].closed and session.bank.devices() == [] and session.devices == {}
    assert station.find_arms()[1]["done"]


def test_a_driver_file_edited_while_the_page_is_open_is_read_again_when_the_arms_are_opened(
    tmp_path: pathlib.Path,
) -> None:
    """After an edit to the driver file the page kept running the driver it first read until dreamcli setup was
    restarted. A station reads its driver once; the page reads it again each time
    it opens the arms."""
    path = tmp_path / "edited_arm.py"
    driver = f"{path}:MyArm"
    path.write_text("class MyArm:\n    EDIT = 'first'\n", encoding="utf-8")
    assert bodies.load_driver(driver).EDIT == "first"
    path.write_text("class MyArm:\n    EDIT = 'edited'\n", encoding="utf-8")
    assert bodies.load_driver(driver).EDIT == "first"

    Station("makers", custom=True).find_arms()

    assert bodies.load_driver(driver).EDIT == "edited"


def test_each_so101_arm_opens_with_the_calibration_file_its_servos_hold(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if so101.ARM.ik not in setup_model.spec_names():
        pytest.skip(f"this dream IK has no {so101.ARM.ik}")
    monkeypatch.setattr(bodies, "embodiment_config", lambda name, key: test_so101.SO101_EMBODIMENT)
    bodies.so101_body.cache_clear()
    elbow = feetech.Calibration(3, 0, 40, 900, 3000)
    left_file = so101_fakes.write_calibration(tmp_path / "left.json")
    right_file = so101_fakes.write_calibration(tmp_path / "right.json", elbow_flex=(3, 40, 900, 3000))
    left, right = "/dev/serial/by-id/usb-1a86_5B8E1-if00", "/dev/serial/by-id/usb-1a86_5B8E2-if00"
    servos = so101_fakes.FakeFeetechFactory()
    servos.eeprom[right] = {"elbow_flex": elbow}
    monkeypatch.setattr(feetech, "open_bus", servos)
    arm_model = model(so101.ARM.ik)
    rig = Rig({left: FakeArm(arm_model.rest), right: FakeArm(arm_model.rest)})
    clock = FakeClock()
    session = setup_session.Session(
        opener=rig.open, model_factory=lambda _arm: arm_model, old_channels={}, clock=clock, sleep=clock.sleep
    )
    session.choose(setup_session.so101_robot([str(right_file), str(left_file)], "tester/so101-pick"))
    devices = [dict(device(left, "AAA111"), kind="serial"), dict(device(right, "BBB222"), kind="serial")]
    assert session.open_arms(devices, {"left": {}, "right": {}}) == {"can-AAA111": None, "can-BBB222": None}
    assert [cfg.calibration for cfg in rig.configs] == [str(left_file), str(right_file)]
    assert all(not bus.connected for buses in servos.buses.values() for bus in buses), "the EEPROM read closes"
    servos.eeprom[right] = {"elbow_flex": dataclasses.replace(elbow, homing_offset=77)}
    opened = session.open_arms(devices, {"left": {}, "right": {}})
    assert opened == {"can-AAA111": None, "can-BBB222": None}, "an arm that holds no file opens, to be calibrated"
    assert session.uncalibrated == {"can-BBB222"}
    rows = session.wiggle()["devices"]
    assert rows["can-AAA111"]["calibration"] == "left.json" and not rows["can-AAA111"]["uncalibrated"]
    assert rows["can-BBB222"]["calibration"] is None and rows["can-BBB222"]["uncalibrated"]
    held = feetech.read_calibration(rig.configs[-1].calibration or "", so101.JOINTS)
    assert held["elbow_flex"].homing_offset == 77, "it opens with what its servos hold, in a file of the wizard's own"


def test_an_so101_arm_is_calibrated_on_the_arms_step_and_nothing_past_it_runs_until_both_are(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No calibration file at all: both arms open and are found by moving them; the home pose waits; each arm is
    calibrated (servos off, middle pose, every joint end to end) into LeRobot's folder and opened again with its new
    file; Cancel leaves an arm as it was."""
    if so101.ARM.ik not in setup_model.spec_names():
        pytest.skip(f"this dream IK has no {so101.ARM.ik}")
    monkeypatch.setattr(bodies, "embodiment_config", lambda name, key: test_so101.SO101_EMBODIMENT)
    monkeypatch.setattr(setup_session, "SO101_CALIBRATION_DIR", tmp_path / "lerobot")
    bodies.so101_body.cache_clear()
    left, right = "/dev/serial/by-id/usb-1a86_5B8E1-if00", "/dev/serial/by-id/usb-1a86_5B8E2-if00"
    hands = {left: test_so101_calibration.Hand("left"), right: test_so101_calibration.Hand("right")}

    def open_bus(port: str, joints: Any, calibration: Any) -> so101_fakes.CalibratingBus:
        bus = hands[port].bus
        bus.connected, bus.torque_off = True, False
        return bus

    monkeypatch.setattr(feetech, "open_bus", open_bus)
    arm_model = model(so101.ARM.ik)
    rig = Rig({left: FakeArm(arm_model.rest), right: FakeArm(arm_model.rest)})
    clock = FakeClock()
    session = setup_session.Session(
        opener=rig.open, model_factory=lambda _arm: arm_model, old_channels={}, clock=clock, sleep=clock.sleep
    )
    session.choose(setup_session.so101_robot([], "tester/so101-pick"))
    devices = [dict(device(left, "AAA111"), kind="serial"), dict(device(right, "BBB222"), kind="serial")]
    assert session.open_arms(devices, {"left": {}, "right": {}}) == {"can-AAA111": None, "can-BBB222": None}
    session.assign("can-AAA111", "left")
    session.assign("can-BBB222", "right")
    assert session.status()["calibration"]["needed"] == {"left": True, "right": True}
    with pytest.raises(ValueError, match="calibrate the left and right arm on step 1 first"):
        session.set_home()

    status = session.start_calibration("left")["active"]
    assert status["stage"] == "middle" and status["middle_frame"], status
    assert "can-AAA111" not in session.bank.devices(), "the arms step lets go of the arm being calibrated"
    with pytest.raises(ValueError, match="the left arm is being calibrated"):
        session.start_calibration("right")
    session.set_middle()
    hands[left].sweep(session.calibration)  # type: ignore[arg-type]
    session.tick()
    done = session.finish_calibration()
    written = tmp_path / "lerobot" / "left_AAA111.json"
    assert done["done"]["left"]["file"] == str(written) and written.is_file()
    assert done["done"]["left"]["moved_deg"]["elbow_flex"] == pytest.approx(9.0, abs=0.2)
    assert done["needed"] == {"left": False, "right": True}
    assert str(written) in (session.robot.calibration_files if session.robot else ())
    assert "can-AAA111" in session.bank.devices() and rig.configs[-1].calibration == str(written)

    session.start_calibration("right")
    session.cancel_calibration()
    assert session.status()["calibration"]["needed"]["right"] and "can-BBB222" in session.bank.devices()
    with pytest.raises(ValueError, match="calibrate the right arm on step 1 first"):
        session.arm_settings("right")

    session.start_calibration("right")
    session.set_middle()
    hands[right].sweep(session.calibration)  # type: ignore[arg-type]
    session.finish_calibration()
    assert session.uncalibrated == set()
    assert session.arm_settings("right")["calibration"] == str(tmp_path / "lerobot" / "right_BBB222.json")
    session.set_home()


def test_an_arm_that_does_not_move_under_power_fails_the_nudge() -> None:
    """Writes that never reach the motors: the hold check passes (nothing moved), the first nudge does not, and with
    no joint of the arm moved the cause is its power."""
    station = Station("yam")
    station.find_arms()
    station.set_home()
    original = Rig.open

    def open_dead(rig: Rig, robot: setup_session.Robot, cfg: config.ArmConfig, side: str) -> Any:
        arm = original(rig, robot, cfg, side)
        handle = rig.handles[-1]
        handle.arm.powered = False
        handle.write = lambda positions_rad, gripper: None  # type: ignore[method-assign]
        return arm

    station.session._opener = lambda robot, cfg, side: open_dead(station.rig, robot, cfg, side)
    motion = station.motion()
    assert motion.state == "failed" and motion.results[-1]["finding"].cause == "no_power"


def test_a_metal_arm_that_cannot_hold_its_weight_fails_the_shoulder_nudge_naming_its_weight() -> None:
    """A Metal arm whose driver has no gravity torque: joint 1 follows its nudge, joint 2 moves under a fifth of
    it. Here the shoulder and elbow trail what is sent by 4.4 degrees."""
    station = Station("metal")
    for arm in station.rig.arms.values():
        arm.band_rad = np.radians([0.0, 4.4, 4.4, 0.0, 0.0, 0.0])
    station.find_arms()
    station.set_home()
    motion = station.motion()
    first, last = (row for row in motion.results if row["group"] == "nudge")
    assert (first["side"], first["finding"].joint, first["finding"].state) == ("left", 0, "pass")
    assert (last["side"], last["finding"].joint, last["finding"].cause) == ("left", 1, "weight"), last
    assert last["finding"].message == "sent joint 2 +5.0 degrees; it moved under a fifth of that (+0.6)"
    assert motion.state == "failed" and not motion.let_go


@pytest.mark.parametrize(
    ("band_deg", "cause"), [([1.3, 1.3, 1.3, 0, 0, 0], "missed"), ([0, 1.3, 1.3, 0, 0, 0], "weight")]
)
def test_a_sweep_miss_is_weight_only_when_the_lifting_joints_trail_more_than_the_base_yaw(
    band_deg: list[float], cause: str
) -> None:
    """An arm with weak lifting joints: its DM4340P joints 1 to 3, the base yaw that lifts nothing included, moved
    3.7 of their 5 degree nudge, and the sweep's +x miss was called the arm's weight. Here those joints trail what is
    sent by 1.3 degrees; with the base yaw exact, the same trail on the lifting joints is still their weight."""
    station = Station("yam")
    for arm in station.rig.arms.values():
        arm.band_rad = np.radians(band_deg)
    station.find_arms()
    station.set_home()
    motion = station.motion()
    nudges = [row["finding"].message for row in motion.results if row["group"] == "nudge"][:3]
    assert nudges[1:] == ["sent joint 2 +5.0 degrees; it moved +3.7", "sent joint 3 +5.0 degrees; it moved +3.7"]
    last = motion.results[-1]
    assert (last["group"], last["label"], last["finding"].cause) == ("sweep", "+x", cause), last


def test_stop_ends_the_powered_checks_with_both_arms_holding() -> None:
    """Stop before the run gets far: the run ends `stopped`, every arm told to hold, and Stop is idempotent."""
    station = Station("yam")
    station.find_arms()
    station.set_home()
    station.session.stop_motion()
    station.session.start_motion()
    motion = station.session.motion
    assert motion is not None
    motion.stop()
    assert station.session.wait_motion(30.0)
    assert motion.state == "stopped" and "operator" in (motion.error or "")
    station.session.stop_motion()


@pytest.mark.parametrize("spec", ["makers", "metal"])
def test_an_arm_that_sags_and_lags_under_power_passes_every_check(spec: str) -> None:
    station = Station(spec)
    for arm in station.rig.arms.values():
        arm.sag_rad = np.radians([0.0, 3.0, 2.0, 0.0, 0.0, 0.0])
        arm.band_rad = math.radians(0.3)
    station.find_arms()
    station.set_home()
    home = station.session.home["left"]
    low = station.model.tool(home - station.rig.arms[LEFT].sag_rad, "left")[:3, 3]
    sag_m = float(np.linalg.norm(low - station.model.tool(home, "left")[:3, 3]))
    assert sag_m > setup_checks.tolerance_for(spec).pos_m, sag_m
    motion = station.motion()
    assert motion.state == "done", (motion.error, motion.results[-3:])
    assert all(row["finding"].state == "pass" for row in motion.results)


def test_a_maker_arm_shoulder_that_sinks_in_the_hold_check_is_its_weight_not_its_driver() -> None:
    """A Maker Arm's shoulder_lift, with no gravity torque, sinks 3 degrees when sent where it is; the hold check
    does not call that a driver mismatch. A lifting joint that lowers the tool a few degrees is `weight`."""
    station = Station("makers")
    for arm in station.rig.arms.values():
        arm.sag_rad = np.radians([0.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    station.find_arms()
    station.set_home()
    last = station.motion().results[-1]
    assert (last["group"], last["finding"].cause, last["finding"].joint) == ("hold", "weight", 1), last
    assert setup_checks.CAUSES["weight"].safe


def test_a_driver_writing_1000x_stops_on_the_first_tick_and_the_arms_are_let_go() -> None:
    station = Station("yam", {LEFT: Mistakes(write_scale=1000.0)})
    station.find_arms()
    station.set_home()
    writes = station.rig.arms[LEFT].writes
    motion = station.motion()
    assert motion.state == "failed" and motion.results[-1]["finding"].cause == "too_far", motion.results[-1:]
    assert station.rig.arms[LEFT].writes == writes + 1
    assert motion.let_go and not station.session.powered
    assert not any(arm.powered for arm in station.rig.arms.values())
    assert set(station.session.bank.devices()) == {"can-AAA111", "can-BBB222"}


def test_after_a_stop_or_a_failed_check_the_arms_hold_until_let_go_and_the_steps_before_work_again() -> None:
    station = Station("yam", {LEFT: Mistakes(write_flip=1)})
    station.find_arms()
    station.set_home()
    motion = station.motion()
    assert motion.state == "failed" and not motion.let_go and station.session.powered
    assert all(arm.powered for arm in station.rig.arms.values())
    with pytest.raises(ValueError, match="Turn off motors"):
        station.session.set_home()
    with pytest.raises(ValueError, match="Turn off motors"):
        station.session.open_arms(station.devices, settings_for(station.robot))
    status = station.session.release()
    assert status["powered"] is False and not any(arm.powered for arm in station.rig.arms.values())
    with pytest.raises(ValueError, match="not powered"):
        station.session.release()
    station.set_home()
    handles = len(station.rig.handles)
    station.motion()
    assert len(station.rig.handles) == handles + 2 and all(h.closed for h in station.rig.handles[:handles])


def _held_then_pushed_off_home(station: Station) -> dict[str, npt.NDArray[np.float64]]:
    """Runs the checks, stops them with both arms holding, then pushes each arm's base joint 8 degrees off home, as an
    arm that sagged or was held off its pose would be; returns the home pose."""
    station.find_arms()
    station.set_home()
    station.session.start_motion()
    motion = station.session.motion
    assert motion is not None
    motion.stop()
    assert station.session.wait_motion(30.0) and station.session.powered
    for arm in station.rig.arms.values():
        arm.q = arm.q + np.radians([8.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    return dict(station.session.home)


def test_switch_off_the_motors_leaves_the_arms_where_they_are_unless_the_station_file_asks_for_home_first() -> None:
    """Today's behaviour, kept while `home_before_motors_off` is unset: the arms go limp where they hold."""
    station = Station("yam")
    _held_then_pushed_off_home(station)
    held = {adapter: arm.q.copy() for adapter, arm in station.rig.arms.items()}

    station.session.release()

    assert not station.session.powered
    for adapter, arm in station.rig.arms.items():
        assert np.array_equal(arm.q, held[adapter]), "nothing was written: they went limp where they held"


def test_switch_off_the_motors_drives_the_arms_home_first_then_switches_them_off() -> None:
    """With `home_before_motors_off`, Switch off the motors moves both held arms home at the checks' own
    pace (the run shows as moving, so STOP shows), then lets them go and says so; the run's results are kept."""
    station = Station("yam", home_before_off=True)
    home = _held_then_pushed_off_home(station)
    motion = station.session.motion
    assert motion is not None
    results, state = list(motion.results), motion.state

    station.session.release()

    assert station.session.wait_motion(30.0)
    assert not station.session.powered and not any(arm.powered for arm in station.rig.arms.values())
    for side, adapter in (("left", LEFT), ("right", RIGHT)):
        assert np.allclose(station.rig.arms[adapter].q, home[side], atol=math.radians(1.0)), side
    assert motion.results == results and motion.state == state
    assert motion.notes[-1] == {"text": setup_motion.HOMED_NOTE}


def test_while_the_arms_go_home_before_switch_off_the_run_shows_as_moving_so_the_page_shows_stop() -> None:
    station = Station("yam", home_before_off=True)
    _held_then_pushed_off_home(station)
    motion = station.session.motion
    assert motion is not None

    motion.begin_home_off()

    assert motion.status()["state"] == "running" and "home" in motion.status()["phase"]


def test_a_home_move_that_fails_before_switch_off_still_switches_the_motors_off_and_says_where() -> None:
    """A joint that runs the wrong way on the way home trips the checks' own watchdog: the move ends, the motors go
    off anyway (never left on because homing failed), and the note says the arms were not home."""
    station = Station("yam", {LEFT: Mistakes(write_flip=1)}, home_before_off=True)
    station.find_arms()
    station.set_home()
    motion = station.motion()
    assert motion.state == "failed" and station.session.powered, motion.error
    station.rig.arms[LEFT].q = station.rig.arms[LEFT].q + np.radians([0.0, 8.0, 0.0, 0.0, 0.0, 0.0])

    station.session.release()

    assert station.session.wait_motion(30.0)
    assert not station.session.powered and not any(arm.powered for arm in station.rig.arms.values())
    assert motion.notes[-1]["text"].startswith("The arms did not reach home before the motors went off"), motion.notes


def test_an_arm_unplugged_under_power_is_named_and_the_other_is_still_let_go() -> None:
    station = Station("yam")
    station.find_arms()
    station.set_home()
    left, right = station.rig.arms[LEFT], station.rig.arms[RIGHT]
    left.unplug_after_writes = left.writes + 5
    motion = station.motion()
    assert motion.state == "failed" and motion.let_go and "ArmError" not in (motion.error or ""), motion.error
    status = station.session.status()
    assert status["powered"] is False and list(status["not_answering"]) == ["left"]
    assert left.powered and not right.powered
    for step in (station.session.start_motion, station.session.set_home):
        with pytest.raises(ValueError, match="the left arm stopped answering under power: it may still hold"):
            step()
    left.unplugged, left.unplug_after_writes = False, None
    station.session.release_arms()
    station.find_arms()
    station.set_home()
    assert station.motion().state == "done" and station.session.status()["not_answering"] == {}


def test_let_go_reaches_the_other_arm_when_one_stopped_answering_while_holding() -> None:
    station = Station("yam", {LEFT: Mistakes(write_flip=1)})
    station.find_arms()
    station.set_home()
    motion = station.motion()
    assert motion.state == "failed" and not motion.let_go and station.session.powered
    station.rig.arms[LEFT].unplugged = True
    status = station.session.release()
    assert status["powered"] is False and list(status["not_answering"]) == ["left"]
    assert not station.rig.arms[RIGHT].powered
    with pytest.raises(ValueError, match="not powered"):
        station.session.release()


HOT = "j2 mos reached 66 C (the run stops at 65 C); the arm parks. Let it cool below 55 C before the next run"


def test_a_motor_too_hot_while_the_arms_hold_after_the_checks_parks_that_arm() -> None:
    station = Station("yam")
    station.find_arms()
    station.set_home()
    assert station.motion().state == "done" and station.session.powered
    left, right = station.rig.arms[LEFT], station.rig.arms[RIGHT]
    left.alarm = HOT
    station.session.tick()
    status = station.session.status()
    assert list(status["parked"]) == ["left"] and "66 C" in status["parked"]["left"]
    assert not left.powered and right.powered and status["powered"] is True and status["holding"] == ["right"]
    with pytest.raises(ValueError, match=r"The left arm's j2 motor reached 66 C at its mos \(the limit is 65 C\), so"):
        station.session.start_motion()
    assert station.session.release()["powered"] is False and not right.powered
    station.session.release_arms()
    station.find_arms()
    station.set_home()
    with pytest.raises(ValueError, match="66 C.*nothing moves under power until it is cool"):
        station.session.start_motion()
    left.alarm = None
    assert station.motion().state == "done" and station.session.status()["parked"] == {}


def test_a_parked_arm_is_said_once_and_in_plain_words() -> None:
    rig = test_metal.fake_rig()
    rig.motor("elbow_flex").temp_rotor_c = 80
    rig.tick(70, target=test_metal.pose_target(test_metal.POSE_DEG, 25.0))
    alarm = followers.FollowerArm("left", rig.arm, home_rad=np.zeros(6)).alarm()
    assert alarm is not None and setup_session.parked_words("left", alarm) == (
        "The left arm's elbow flex motor reached 80 C at its coil (the limit is 65 C), so it was parked and switched "
        "off. Let it cool below 55 C before the next run."
    )
    assert setup_session.off_failed_words("right", "arm 'right': motor 3 stopped answering").startswith(
        "The right arm: motor 3 stopped answering. Switching it off failed, so it may still be powered"
    )


def test_stop_during_a_park_says_both_arms_went_limp_and_never_calls_the_cut_arm_parked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    station = Station("yam")
    station.find_arms()
    station.set_home()
    left, right = station.rig.arms[LEFT], station.rig.arms[RIGHT]
    parking, stopped = threading.Event(), threading.Event()
    torque_off = setup_fakes.Driver.torque_off

    def travelling_off(driver: setup_fakes.Driver) -> None:
        if driver.arm is left:
            parking.set()
            assert stopped.wait(5.0)
        torque_off(driver)

    monkeypatch.setattr(setup_fakes.Driver, "torque_off", travelling_off)
    monkeypatch.setattr(
        setup_fakes.Driver, "alarm", lambda driver: HOT if driver.arm is left and left.writes > 3 else None
    )
    station.session.start_motion()
    assert parking.wait(5.0)
    assert station.session.status()["motion"]["state"] == "parking"
    station.session.press_stop()
    stopped.set()
    assert station.session.wait_motion(5.0)
    status = station.session.status()
    assert status["motion"]["stop_cut"] and status["motion"]["let_go"] and not status["checks_on"], status["motion"]
    assert status["parked"] == {
        "left": "The left arm's j2 motor reached 66 C at its mos (the limit is 65 C); it went limp where it stood, "
        "not parked at rest, and is switched off now. Let it cool below 55 C before the next run."
    }
    assert status["powered"] is False and status["holding"] == [] and not left.powered and not right.powered
    for row in status["motion"]["results"]:
        assert "We parked" not in row["finding"]["fix"], row


def test_a_motor_that_stops_answering_reads_plainly_and_nothing_of_the_end_shows_until_the_arms_are_let_go(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    station = Station("yam")
    station.find_arms()
    station.set_home()
    left, right = station.rig.arms[LEFT], station.rig.arms[RIGHT]
    letting_go, go_on = threading.Event(), threading.Event()
    disengage = setup_fakes.Driver.disengage

    def slow_let_go(driver: setup_fakes.Driver) -> bool:
        if driver.arm is right:
            letting_go.set()
            assert go_on.wait(5.0)
        return disengage(driver)

    silent = (
        "MotorSilent: can-AAA111: elbow_flex (id 3, reply id 0x13) did not answer; check the arm's 24 V power and its "
        "CAN wiring"
    )
    monkeypatch.setattr(setup_fakes.Driver, "why_stopped", lambda _driver: silent)
    monkeypatch.setattr(setup_fakes.Driver, "disengage", slow_let_go)
    left.unplug_after_writes = left.writes + 5
    station.session.start_motion()
    try:
        assert letting_go.wait(5.0)
        during = station.session.status()
    finally:
        go_on.set()
    assert station.session.wait_motion(5.0)
    after = station.session.status()
    plain = "The left arm's elbow flex motor stopped answering: check the arm's power and the cable to its adapter"
    assert during["motion"]["error"] == plain and during["checks_on"], during
    assert after["motion"]["error"] == plain and not after["checks_on"] and after["holding"] == [], after
    assert list(after["not_answering"]) == ["left"] and not right.powered


class YamRobot:
    """An i2rt robot over a FakeArm, as i2rt behaves: close() sends nothing, so the motors keep their last target."""

    def __init__(self, arm: FakeArm, channel: str) -> None:
        self.arm = arm
        self.events: list[str] = []
        self.motor_chain = type("Chain", (), {"running": True, "channel": channel, "motor_list": [[1], [2]]})()

    def get_observations(self) -> dict[str, Any]:
        return {"joint_pos": self.arm.q.copy(), "joint_vel": np.zeros(6), "gripper_pos": np.array([self.arm.grip])}

    def command_joint_pos(self, positions: Any) -> None:
        if self.arm.powered:
            self.arm.follow(np.asarray(positions[:6], dtype=np.float64))
            self.arm.grip = float(positions[6])

    def enter_gravity_comp_idle(self) -> None:
        self.events.append("gravity_comp")

    def close(self) -> None:
        self.events.append("close")


def test_a_yam_joint_blocked_on_the_way_home_lets_both_arms_float(monkeypatch: pytest.MonkeyPatch) -> None:
    station = Station("yam")
    station.find_arms()
    station.set_home()
    left, right = station.rig.arms[LEFT], station.rig.arms[RIGHT]
    robots = {"left": YamRobot(left, "can_left"), "right": YamRobot(right, "can_right")}
    monkeypatch.setattr(yam, "_switch_off_motors", lambda name, *_: robots[name].events.append("disable sweep"))

    def opener(_robot: Any, cfg: Any, side: str) -> followers.FollowerArm:
        robots[side].arm.powered = True
        return followers.FollowerArm(side, yam.I2rtDriver(side, robots[side]), home_rad=cfg.home)

    station.session._opener = opener
    lifted = right.q[2] + 0.5
    right.q[2] = lifted
    right.floor = (2, lifted - math.radians(2.0))
    motion = station.motion()
    last = motion.results[-1]
    assert motion.state == "failed" and motion.let_go, motion.error
    assert (last["group"], last["side"], last["finding"].cause) == ("watchdog", "both", "not_followed")
    assert robots["left"].events == robots["right"].events == ["gravity_comp"]
    assert station.session.status()["parked"] == {}


def test_a_joint_that_runs_away_from_what_was_sent_still_lets_both_arms_go(monkeypatch: pytest.MonkeyPatch) -> None:
    station = Station("yam")
    station.find_arms()
    station.set_home()
    left, right = station.rig.arms[LEFT], station.rig.arms[RIGHT]
    write = setup_fakes.Driver.write
    drift = [0.0]

    def running_away(driver: setup_fakes.Driver, positions_rad: Any, gripper: float) -> None:
        write(driver, positions_rad, gripper)
        motion = station.session.motion
        if driver.arm is right and motion is not None and motion.phase == "moving to home":
            drift[0] += 0.02
            right.q[2] += drift[0]

    monkeypatch.setattr(setup_fakes.Driver, "write", running_away)
    motion = station.motion()
    last = motion.results[-1]
    assert motion.state == "failed" and motion.let_go, motion.error
    assert (last["group"], last["side"], last["finding"].cause) == ("watchdog", "both", "not_followed")
    assert not left.powered and not right.powered and station.session.status()["parked"] == {}


def test_a_brakeless_joint_pushed_past_its_reach_while_it_strays_parks_that_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    station = Station("yam")
    station.find_arms()
    station.set_home()
    left, right = station.rig.arms[LEFT], station.rig.arms[RIGHT]
    write = setup_fakes.Driver.write
    pushed = [0.0]

    def pushed_on(driver: setup_fakes.Driver, positions_rad: Any, gripper: float) -> None:
        write(driver, positions_rad, gripper)
        motion = station.session.motion
        if driver.arm is right and motion is not None and motion.phase == "moving to home":
            pushed[0] += 0.06
            right.q[2] += pushed[0]

    monkeypatch.setattr(setup_fakes.Driver, "drops_when_limp", True)
    monkeypatch.setattr(setup_fakes.Driver, "write", pushed_on)
    motion = station.motion()
    last = motion.results[-1]
    assert motion.state == "failed" and not motion.let_go, motion.error
    assert (last["group"], last["side"], last["finding"].cause) == ("watchdog", "right", "blocked"), last
    assert last["finding"].message.startswith("joint 3 stayed 28 degrees from where we sent it while it moved")
    assert list(station.session.status()["parked"]) == ["right"] and not right.powered and left.powered


def test_finishing_while_an_alarmed_arm_parks_waits_for_the_park_then_switches_the_other_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    station = Station("yam")
    station.find_arms()
    station.set_home()
    assert station.motion().state == "done" and station.session.powered
    left, right = station.rig.arms[LEFT], station.rig.arms[RIGHT]
    parking, arrived = threading.Event(), threading.Event()
    calls: list[tuple[str, str]] = []
    torque_off = setup_fakes.Driver.torque_off

    def travelling_off(driver: setup_fakes.Driver) -> None:
        side = "left" if driver.arm is left else "right"
        calls.append((side, threading.current_thread().name))
        if side == "left":
            parking.set()
            assert arrived.wait(5.0)
        torque_off(driver)

    monkeypatch.setattr(setup_fakes.Driver, "torque_off", travelling_off)
    left.alarm = HOT
    reader = threading.Thread(target=station.session.tick, name="setup-arms")
    reader.start()
    assert parking.wait(5.0)
    closer = threading.Thread(target=station.session.close, name="wizard-exit")
    closer.start()
    closer.join(0.2)
    assert closer.is_alive(), "the exit did not wait for the park"
    arrived.set()
    reader.join(5.0)
    closer.join(5.0)
    assert not closer.is_alive() and calls == [("left", "setup-arms"), ("right", "wizard-exit")]
    assert not left.powered and not right.powered


@pytest.mark.parametrize("then", ["let go", "let go, fails again", "exit"])
def test_a_park_whose_switch_off_fails_stays_powered_for_let_go_and_the_exit(
    then: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    station = Station("yam")
    station.find_arms()
    station.set_home()
    assert station.motion().state == "done"
    left, right = station.rig.arms[LEFT], station.rig.arms[RIGHT]
    torque_off = setup_fakes.Driver.torque_off
    tries: list[str] = []

    def left_fails(driver: setup_fakes.Driver) -> None:
        if driver.arm is left:
            tries.append("left")
            if len(tries) == 1 or then == "let go, fails again":
                raise errors.ArmError("arm 'left': motors [2] on can0 may still be energized; cut the arm's power")
        torque_off(driver)

    monkeypatch.setattr(setup_fakes.Driver, "torque_off", left_fails)
    left.alarm = HOT
    station.session.tick()
    station.session.tick()
    status = station.session.status()
    assert status["parked"] == {} and list(status["off_failed"]) == ["left"] and "66 C" in status["off_failed"]["left"]
    assert status["holding"] == ["left", "right"] and left.powered and right.powered and tries == ["left"]
    with pytest.raises(ValueError, match=r"The left arm's j2 motor .*\. Switching it off failed, so it may still be"):
        station.session.start_motion()
    if then == "let go":
        status = station.session.release()
        assert list(status["parked"]) == ["left"] and status["off_failed"] == {} and status["powered"] is False
        assert tries == ["left", "left"] and not left.powered and not right.powered
    elif then == "let go, fails again":
        status = station.session.release()
        assert status["parked"] == {} and list(status["off_failed"]) == ["left"] and status["holding"] == ["left"]
        assert tries == ["left", "left"] and left.powered and not right.powered
        station.session.close()
        assert tries == ["left", "left", "left"]
    else:
        station.session.close()
        assert tries == ["left", "left"] and not left.powered and not right.powered


def test_run_again_drives_the_arms_that_hold_and_exit_switches_them_off() -> None:
    station = Station("yam")
    station.find_arms()
    station.set_home()
    station.motion()
    handles = list(station.rig.handles)
    assert station.motion().state == "done"
    assert station.rig.handles == handles and not any(handle.closed for handle in handles[-2:])
    assert station.session.powered
    station.session.close()
    assert all(handle.closed for handle in station.rig.handles) and not station.session.powered
    assert not any(arm.powered for arm in station.rig.arms.values())


def test_run_again_on_a_held_arm_that_stopped_answering_asks_to_let_go_first() -> None:
    station = Station("yam")
    station.find_arms()
    station.set_home()
    station.motion()
    handles = list(station.rig.handles)
    station.rig.arms[LEFT].unplugged = True
    with pytest.raises(ValueError, match="the left arm stopped answering while it held: .* Turn off motors first"):
        station.session.start_motion()
    assert station.rig.handles == handles and station.session.powered


def test_a_powered_open_whose_other_arm_fails_lets_the_opened_one_go_again() -> None:
    station = Station("yam")
    station.find_arms()
    station.set_home()
    opener = station.session._opener

    def right_fails(robot: Any, cfg: Any, side: str) -> Any:
        if side == "right":
            raise OSError("no reply on the bus")
        return opener(robot, cfg, side)

    station.session._opener = right_fails
    with pytest.raises(OSError, match="no reply"):
        station.session.start_motion()
    left = station.rig.handles[-1]
    assert left.arm is station.rig.arms[LEFT] and left.closed and not left.arm.powered
    assert not station.session.powered


def test_a_sweep_move_out_of_reach_is_shrunk_and_said_or_skipped() -> None:
    """A move the IK reaches only at half size runs at half size; one it never reaches is skipped; both are noted."""
    station = Station("yam")
    station.find_arms()
    station.set_home()
    real = station.model.solve
    calls = {"n": 0}

    def solve(targets: dict[str, Any], start: dict[str, Any]) -> dict[str, Any]:
        out = real(targets, start)
        calls["n"] += 1
        travel = max(float(np.linalg.norm(t[-31, :3, 3] - t[0, :3, 3])) for t in targets.values())
        if travel > 0.02:
            for side in out:
                out[side] = dict(out[side], pos_miss=out[side]["pos_miss"] + 0.02)
        return out

    station.model.solve = solve  # type: ignore[method-assign]
    try:
        motion = station.motion()
    finally:
        del station.model.solve
    assert motion.state == "done", motion.error
    assert any("ran at half size" in note["text"] for note in motion.notes)
    assert calls["n"] > len(setup_motion.SWEEP_MOVES)


def test_home_outside_the_urdfs_range_is_named_and_the_custom_robot_needs_no_settings() -> None:
    """set_home() names the joints a degrees driver puts outside the URDF; a robot of one's own refuses settings."""
    station = Station("yam", {LEFT: Mistakes(read_units="degrees"), RIGHT: CLEAN})
    station.find_arms()
    found = station.set_home()
    assert found["out_of_range"] and {o["side"] for o in found["out_of_range"]} == {"left"}
    assert set(found) == {"home", "out_of_range"}
    custom = setup_session.custom_robot("yam", "x:Y", "tester/my-arm")
    with pytest.raises(errors.ConfigError, match="gripper"):
        setup_session.arm_config(custom, "can0", {"gripper": "linear_4310"}, "left", np.zeros(6))


@pytest.mark.parametrize(("spec", "ends"), [("metal", (0.0, 114.0)), ("makers", (-2.5, -80.0))])
def test_the_grippers_range_is_measured_by_hand_and_kept_inside_the_open_stop(
    spec: str, ends: tuple[float, float]
) -> None:
    station = Station(spec)
    station.find_arms()
    for adapter in (LEFT, RIGHT):
        station.hands.pose(adapter, station.model.rest)
    station.session.set_home()
    with pytest.raises(ValueError, match="press Record open and Record closed for the left gripper first"):
        station.session.start_motion()
    found = station.measure_grippers()
    closed, opened = ends
    kept = {"gripper_closed_deg": closed, "gripper_open_deg": opened}
    assert found["ends"] == {"left": kept, "right": kept}, found
    assert station.session.arm_settings("left")["gripper_open_deg"] == opened
    motion = station.motion()
    assert motion.state == "done", motion.error
    cfg = station.rig.configs[-1]
    assert (cfg.gripper_closed_deg, cfg.gripper_open_deg) == ends


def test_a_gripper_not_opened_fully_or_on_a_body_that_needs_no_range_is_refused() -> None:
    station = Station("metal")
    station.find_arms()
    found = station.measure_grippers(opening=0.1)
    assert "turned only 14 degrees" in found["ends"]["left"]["error"], found
    with pytest.raises(ValueError, match="turned only 14 degrees"):
        station.session.gripper_ends("left")
    makers = Station("makers")
    makers.find_arms()
    unmoved = makers.measure_grippers(opening=0.5, closing=0.5)
    assert "turned only 0 degrees" in unmoved["ends"]["left"]["error"], unmoved
    yam = Station("yam")
    yam.find_arms()
    with pytest.raises(ValueError, match="not measured here"):
        yam.session.record_gripper("open")
    with pytest.raises(ValueError, match="'end' must be open or closed"):
        station.session.record_gripper("half")


## A Seeed reBot jaw motor's angle closed and open, degrees: it closes toward increasing angles.
REBOT_JAW_DEG = (15.0, -326.7)


def test_a_robot_of_ones_own_measures_its_grippers_range_from_the_motor_angle_its_driver_reports() -> None:
    """Onboarding a robot of one's own needed the gripper moved closed then open by hand outside the page. A driver
    that reports its gripper motor's angle (`Driver.gripper_deg()`) gets the gripper step: the
    limp arms open without ends (NaN, the grip read unknown on the page), the ends are the angles it reported, either
    way round, and the powered checks and Save get them."""
    station = Station("makers", custom=True)
    for arm in station.rig.arms.values():
        arm.gripper_motor_deg = REBOT_JAW_DEG
    station.find_arms()
    assert all(math.isnan(cfg.gripper_open_deg) for cfg in station.rig.configs)
    station.rig.arms[LEFT].grip = math.nan
    station.session.tick()
    assert station.session.wiggle()["devices"]["can-AAA111"]["gripper"] is None
    station.rig.arms[LEFT].grip = 1.0
    status = station.session.status()["gripper"]
    assert "press Record open and Record closed" in status["ends"]["left"]["error"], status
    with pytest.raises(ValueError, match="press Record open and Record closed for the left gripper first"):
        station.session.arm_settings("left")
    short = station.measure_grippers(opening=0.03)
    assert "turned only 10 degrees" in short["ends"]["left"]["error"], short
    found = station.measure_grippers(opening=1.0)
    closed, opened = REBOT_JAW_DEG
    kept = {"gripper_closed_deg": closed, "gripper_open_deg": opened}
    assert found["ends"] == {"left": kept, "right": kept}, found
    assert station.session.arm_settings("right") == kept
    station.set_home()
    motion = station.motion()
    assert motion.state == "done", motion.error
    cfg = station.rig.configs[-1]
    assert (cfg.gripper_closed_deg, cfg.gripper_open_deg) == REBOT_JAW_DEG


def test_a_robot_of_ones_own_whose_driver_reports_no_gripper_angle_skips_the_step_and_says_why() -> None:
    station = Station("makers", custom=True)
    station.find_arms()
    assert station.session.status()["gripper"] == {"skipped": setup_session.GRIPPER_SKIPPED}
    with pytest.raises(ValueError, match="reports no gripper motor angle"):
        station.session.record_gripper("open")
    assert station.session.arm_settings("left") == {}
    station.set_home()
    assert station.motion().state == "done"


def test_a_robot_of_ones_own_opens_with_the_driver_it_names(tmp_path: pathlib.Path) -> None:
    """open_body_arm() opens a custom robot through `robots.bodies.custom_body`, the driver class its file names, under
    a FollowerArm; a name that is not `module:Class` is refused at the pick."""
    (tmp_path / "my_arm.py").write_text(
        "from robots import driver_template\n\nclass MyArm(driver_template.TemplateArm):\n    pass\n",
        encoding="utf-8",
    )
    robot = setup_session.custom_robot("yam", f"{tmp_path / 'my_arm.py'}:MyArm", "tester/my-arm")
    cfg = config.ArmConfig(can_channel="can0", home=(0.1,) * 6)
    arm = setup_session.open_body_arm(robot, cfg, "left")
    np.testing.assert_allclose(arm.home_rad, (0.1,) * 6)
    assert arm.read().positions_rad.shape == (6,)
    arm.close()
    with pytest.raises(ValueError, match="module:Class"):
        setup_session.custom_robot("yam", "my_arm", "tester/my-arm")


def test_the_reach_area_paints_on_a_grid_keeps_a_new_home_and_is_cleared_by_clear_or_finding_the_arms_again() -> None:
    """Each read while painting adds the hand's cell, 2 cm apart, each a point in the base frame to the millimetre;
    the hand is dream's IK's tool point (tool(), the held joints at zero). Save adds the home hand. Clear empties one
    arm; a new home keeps the painting; finding the arms again starts it empty. A YAM has no reach area."""
    station = Station("metal")
    station.find_arms()
    station.set_home()
    session = station.session
    assert session.reach_area("left") is None, "nothing painted: no reach"
    session.paint_reach(True)
    station.hands.move(LEFT, 0, 0.6)
    status = session.status()["reach"]
    left = status["points"]["left"]
    hand = station.model.tool(station.rig.arms[LEFT].q, "left")[:2, 3]
    grid = setup_session.REACH_GRID_M
    assert len(left) > 4 and status["points"]["right"], status
    assert all(round(v / grid) * grid == pytest.approx(v) for point in left for v in point), left
    assert [round(float(v) / grid) * grid for v in hand] in [pytest.approx(p) for p in left], "the hand now is painted"
    assert status["hands"]["left"] == pytest.approx(hand, abs=1e-4) and status["radius_m"] == 0.04
    home = station.model.tool(station.model.rest, "left")[:2, 3]
    assert [round(float(v) / grid) * grid for v in home] in [pytest.approx(p) for p in session.reach_area("left")]
    session.clear_reach("right")
    assert session.status()["reach"]["points"]["right"] == [] and session.reach_area("right") is None
    station.set_home()
    assert session.status()["reach"]["points"]["left"] == left, "a new home keeps the painting"
    session.release_arms()
    assert session.status()["reach"]["points"] == {"left": [], "right": []} and not session.painting
    assert Station("yam").session.status()["reach"] is None


def test_each_home_joint_out_of_range_comes_with_its_reading_and_the_models_range() -> None:
    """The line "Outside Dream's model of the arms: left arm joints 2 ..." alone does not say which way to pose the
    arm.
    set_home() gives each such joint's reading and the range, so the page says which way."""
    station = Station("yam", {LEFT: Mistakes(read_units="degrees"), RIGHT: CLEAN})
    station.find_arms()
    found = station.set_home()
    assert found["out_of_range"] and {o["side"] for o in found["out_of_range"]} == {"left"}
    for o in found["out_of_range"]:
        i = o["joint"] - 1
        assert o["deg"] == found["home"]["left"][i] and not o["lower_deg"] <= o["deg"] <= o["upper_deg"], o
        ends = (station.model.lower[i], station.model.upper[i])
        assert (o["lower_deg"], o["upper_deg"]) == tuple(round(math.degrees(e), 1) for e in ends), o


def test_an_arm_that_did_not_open_again_says_why_on_every_later_step() -> None:
    """An arm that fails to open again once both are found: the home step and every step after it say why, not
    only "the left arm is not open; find both arms first" with the reason in the terminal."""
    station = Station("yam")
    session, rig = station.session, station.rig
    assert session.open_arms(station.devices, settings_for(station.robot)) == {"can-AAA111": None, "can-BBB222": None}
    station.hands.move(LEFT, 0, 0.3)
    session.assign(session.wiggle()["moved"], "left")
    station.hands.move(RIGHT, 0, 0.3)
    moved = session.wiggle()["moved"]
    gone = rig.arms.pop(RIGHT)
    with pytest.raises(ValueError, match="the right arm did not open again"):
        session.assign(moved, "right")
    for step in (session.set_home, session.joints_now):
        with pytest.raises(ValueError, match=r"the right arm is not open: .*no adapter can1.*find both arms again"):
            step()
    # setup_devices.explain()'s sentences end in a full stop: the line keeps one, not "again.. Fix that".
    session.bank.errors[moved] = "The adapter is gone. Plug it back in, then click Scan again."
    with pytest.raises(ValueError, match=r"then click Scan again\. Fix that, then find both arms again on step 1$"):
        session.joints_now()
    rig.arms[RIGHT] = gone
    session.release_arms()
    station.find_arms()
    assert station.set_home()["out_of_range"] == []


@pytest.mark.parametrize(
    ("spec", "centred"),
    [
        ("metal", True),
        ("so101_faas", True),
        ("unitree_r1_faas", True),
        ("ai_worker_faas", False),
        ("openarm2_faas", False),
        ("robo_t_faas", False),
        ("mantis_faas", False),
        ("bracketbot_faas", False),
    ],
)
def test_the_reach_area_is_drawn_about_the_first_joint_only_where_it_turns_the_hand_about_an_upright_axis(
    spec: str, centred: bool
) -> None:
    """The reach preview: the first joint of four beta robots' arms lifts the arm, so the circle through
    the hands it turned centred kilometres away or nowhere (NaN, which no JSON carries), and the page drew millions of
    rings and froze. Those arms are drawn about their base frame; one whose first joint turns the hand about an upright
    axis keeps its centre, within a metre."""
    if spec not in setup_model.spec_names():
        pytest.skip(f"this dream IK has no {spec}")
    station = Station("metal")
    station.session.model = model(spec)  # the arm's URDF, on a body whose fakes are as wide as the station's
    station.session.paint_reach(True)
    bases = station.session.status()["reach"]["bases"]
    for side in ("left", "right"):
        if centred:
            assert all(math.isfinite(v) and abs(v) < 1.0 for v in bases[side]), bases
        else:
            assert bases[side] is None, bases

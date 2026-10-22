"""Tests for `robots.gripper_force`: the effort filter and the grip force cap's control law.

The law is exercised against a pure-python stand-in for the grasp it regulates: the arm reaches the commanded
gripper position unless an object blocks it, and the motor PD turns the leftover squeeze into effort
(`effort = stiffness * (object - command)`), with noise and the occasional spike on top, as a DM motor under
impedance control reports. What the tests pin is the behaviour the cap exists for: the effort settles *at* the
cap rather than above it, the position freezes there instead of hunting, a softening grasp is re-gripped, and
free motion releases the back-off.

The grip force cap is tested against this package's tick rate and its `cap`-on-the-instance
signature.
"""

import numpy as np
import numpy.typing as npt
import pytest

from robots import gripper_force

## The station's tick period, seconds: the cap runs once per tick at `control_hz` (20 Hz), not on the 250 Hz
## command path.
TICK_S = 1.0 / 20.0

## Where the simulated object stops the fingers, in gripper units.
OBJECT_POS = 0.45

## Effort the simulated motor PD reports per unit of squeeze past the object.
STIFFNESS = 40.0


def _simulate(
    cap: gripper_force.GripperForceCap,
    *,
    ticks: int = 400,
    target: float = 0.2,
    object_pos: float = OBJECT_POS,
    stiffness: float = STIFFNESS,
    t0_s: float = 0.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Closes the simulated gripper onto an object and runs the cap over it, one call per tick.

    @param cap The cap under test; its state carries across calls, as it does across ticks.
    @param ticks How many ticks to run.
    @param target The commanded gripper opening, held for the whole run, as a policy holding a close would.
    @param object_pos Where the object stops the fingers, in gripper units.
    @param stiffness Effort the motor PD reports per unit of squeeze past the object.
    @param t0_s Time of the first tick, seconds.
    @return The commanded and measured-effort series, one entry per tick.
    """
    measured = 0.9
    efforts: list[float] = []
    commands: list[float] = []
    noise = np.random.default_rng(0)
    for i in range(ticks):
        command = cap.update(target, measured, efforts[-1] if efforts else 0.0, t0_s + i * TICK_S)
        # The arm reaches the command unless the object is in the way.
        measured = max(command, object_pos)
        squeeze = max(0.0, object_pos - command)
        spike = 8.0 if i % 97 == 0 else 0.0
        efforts.append(stiffness * squeeze + noise.normal(0, 0.4) + spike)
        commands.append(command)
    return np.array(commands), np.array(efforts)


# --- the effort filter ---------------------------------------------------------------------------


def test_the_filter_drops_a_single_sample_spike_and_settles_on_the_signal() -> None:
    """The median-of-three is what kills a spike: a low-pass alone would smear it over its whole time constant."""
    effort_filter = gripper_force.EffortFilter()
    outputs = [effort_filter.update(2.0 + (50.0 if i == 50 else 0.0), i * TICK_S) for i in range(100)]

    assert max(outputs) < 10.0
    assert abs(outputs[-1] - 2.0) < 0.2


def test_the_filter_passes_its_first_sample_through_and_forgets_it_on_reset() -> None:
    """There is nothing to blend the first sample with, so it is its own output; reset() makes the next one first."""
    effort_filter = gripper_force.EffortFilter()

    assert effort_filter.update(3.0, 0.0) == 3.0
    assert effort_filter.update(3.0, 0.0) == 3.0  # A clock that did not advance blends nothing either.
    effort_filter.reset()

    assert effort_filter.update(7.0, 1.0) == 7.0


# --- the control law -----------------------------------------------------------------------------


def test_the_effort_settles_at_the_cap_and_the_position_freezes_there() -> None:
    """The equilibrium the law is built for: a constant effort *at* the cap, the command parked, no hunting around
    it and no creep back toward the raw target, which is what would over-squeeze. The cap is told the simulated
    gripper's stiffness, so its hard floor sits at that same equilibrium."""
    cap = gripper_force.GripperForceCap(5.0, stiffness=STIFFNESS)

    commands, efforts = _simulate(cap, ticks=800)

    settled_effort, settled_command = efforts[500:], commands[500:]
    assert abs(settled_effort.mean() - 5.0) < 0.15 * 5.0
    assert settled_effort.mean() < 5.0 * 1.05
    assert settled_command.std() < 0.005
    # The analytic equilibrium: command = object - cap / stiffness = 0.45 - 5 / 40.
    assert abs(settled_command.mean() - 0.325) < 0.02


def test_a_softer_object_is_re_gripped_and_a_removed_one_releases_the_back_off() -> None:
    """Under the cap the offset shrinks until the effort is back at it (an object compressing, or slipping), and
    with no contact left it bleeds away, so the next grasp starts from zero. A compressing object lets the jaw
    close further; the motor PD's stiffness is the gripper's, whatever it holds."""
    cap = gripper_force.GripperForceCap(5.0, stiffness=STIFFNESS)
    _simulate(cap, ticks=800)
    gripped = cap.offset
    assert gripped > 0.05

    _simulate(cap, ticks=800, object_pos=OBJECT_POS - 0.05, t0_s=100.0)
    assert cap.offset < gripped - 0.02

    for i in range(300):
        cap.update(0.2, 0.4, 0.0, 200.0 + i * TICK_S)
    assert cap.offset < 0.02


def test_a_deliberate_open_bleeds_the_back_off_away_even_in_contact() -> None:
    """A target above the measured position by more than the deadband is the policy opening the gripper, not a
    squeeze to regulate: the cap gets out of the way instead of adding to the opening."""
    cap = gripper_force.GripperForceCap(5.0, stiffness=STIFFNESS)
    _simulate(cap, ticks=800)
    assert cap.offset > 0.05

    for i in range(300):
        # Wide open asked for, still reading the effort of the grasp being let go of.
        cap.update(0.9, 0.45, 5.0, 100.0 + i * TICK_S)

    assert cap.offset < 0.02


def test_the_back_off_never_closes_the_gripper_and_stops_at_max_offset() -> None:
    """The cap may only open the gripper, and a cap no grasp can meet stops at `max_offset` rather than running
    the command to fully open."""
    cap = gripper_force.GripperForceCap(0.1, max_offset=0.2)

    commands, _ = _simulate(cap, ticks=800)

    assert commands.min() >= 0.2
    assert cap.offset == pytest.approx(0.2)
    assert commands[-1] == pytest.approx(0.4)


def test_a_long_gap_between_ticks_integrates_at_most_one_step() -> None:
    """A tick that arrives late (a hold, a stall, a long refill wait) must not move the offset by the whole gap.

    The first tick puts the command on the floor, `cap / stiffness` closed of the jaw; the late one integrates a
    single bounded step on top of it, still short of the jaw (the ceiling)."""
    cap = gripper_force.GripperForceCap(5.0, stiffness=STIFFNESS)
    cap.update(0.2, OBJECT_POS, 10.0, 0.0)
    on_the_floor = cap.offset

    cap.update(0.2, OBJECT_POS, 10.0, 30.0)

    assert on_the_floor == pytest.approx(OBJECT_POS - 5.0 / STIFFNESS - 0.2)
    assert cap.offset == pytest.approx(on_the_floor + 0.4 * (10.0 / 5.0 - 1.0) * 0.1)


# --- the pass-through paths ----------------------------------------------------------------------


def test_a_driver_that_reports_no_effort_passes_the_command_through() -> None:
    """There is nothing to regulate on, so the target reaches the arms as the policy sent it."""
    cap = gripper_force.GripperForceCap(5.0)

    assert cap.update(0.3, 0.5, None, 0.0) == 0.3


def test_a_cap_of_zero_passes_the_command_through() -> None:
    """`grip_force_cap: 0` is how a station hands grip force back to the motor PD."""
    cap = gripper_force.GripperForceCap(0.0)

    assert cap.update(0.3, 0.5, 3.0, 0.0) == 0.3


def test_the_command_is_clipped_into_range() -> None:
    """The arms refuse a gripper outside `[0, 1]`, so the cap never hands one down."""
    cap = gripper_force.GripperForceCap(5.0)

    assert cap.update(1.2, 0.5, 1.0, 0.0) == 1.0
    assert cap.update(-0.2, 0.5, 1.0, TICK_S) == 0.0


def test_reset_drops_the_back_off_and_the_filter_history() -> None:
    """A ramp is a fresh run: an offset carried across it would open the gripper for a squeeze that is over."""
    cap = gripper_force.GripperForceCap(5.0, stiffness=STIFFNESS)
    _simulate(cap, ticks=800)
    assert cap.offset > 0.05

    cap.reset()

    assert cap.offset == 0.0
    assert cap.effort == 0.0
    assert cap.speed == 0.0
    # With no history the next tick integrates nothing; the jaw at the target leaves the floor out of it.
    assert cap.update(0.2, 0.2, 10.0, 100.0) == 0.2


def test_in_contact_the_command_never_sits_further_closed_of_the_jaw_than_the_cap_needs() -> None:
    """The hard floor: a full close on a rigid object is never more than `cap / stiffness` closed of the
    measured jaw, so the motor PD is never asked for more than the cap, from the first tick in contact."""
    cap = gripper_force.GripperForceCap(2.0)

    command = cap.update(0.0, 0.5, 10.0, 0.0)

    assert command == pytest.approx(0.5 - 2.0 / gripper_force.DEFAULT_STIFFNESS)


def test_a_fast_free_close_is_not_contact_however_hard_the_motor_works() -> None:
    """A jaw still closing fast draws effort just accelerating; read as contact, the floor would throttle the close."""
    cap = gripper_force.GripperForceCap(2.0)
    command = 0.0
    for tick in range(10):
        # A close starts from rest at low effort, and the motor loads up as the jaw accelerates.
        command = cap.update(0.0, 0.9 - 0.08 * tick, 0.1 if tick == 0 else 10.0, tick * TICK_S)

    assert cap.speed > cap.contact_speed
    assert command == 0.0


def test_a_tick_without_an_effort_reading_keeps_the_back_off() -> None:
    """A feedback gap holds the grasp where it is instead of snapping back to the raw close and re-squeezing."""
    cap = gripper_force.GripperForceCap(2.0)
    held = cap.update(0.0, 0.5, 10.0, 0.0)

    assert cap.update(0.0, 0.5, None, TICK_S) == pytest.approx(held)


def test_in_contact_the_back_off_never_opens_past_the_jaw() -> None:
    """The ceiling: an effort reading still high from a moment ago (the filter lags) must not open the command past
    the measured jaw, which would let go of the object and start a squeeze-and-release cycle."""
    cap = gripper_force.GripperForceCap(2.0)
    for tick in range(20):
        command = cap.update(0.0, 0.5, 30.0, tick * TICK_S)
        assert command <= 0.5

    assert command == pytest.approx(0.5)

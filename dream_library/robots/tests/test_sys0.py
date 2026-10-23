"""System 0: the bundled network, the tracker's shaping and supervisor, the motor path's hand-offs, the arm's
process, the switch."""

import asyncio
import multiprocessing
import threading
import time
import types as pytypes
from collections.abc import Callable
from typing import Any

import conftest
import numpy as np
import pytest
import test_control

from robots import config, control, errors, types
from robots.sys0 import arm_process, fast_loop, motors, policy, tracker

## Joint limits wide enough that nothing in these tests reaches them.
WIDE_RANGE = np.tile([-3.0, 3.0], (6, 1))


class LinearKinematics:
    """A grasp point that moves one metre per radian of the first joint and nothing else."""

    def ee_position(self, joints_rad: np.ndarray) -> np.ndarray:
        return np.array([float(joints_rad[0]), 0.0, 0.0])


class JointCommands:
    """i2rt's command block, with the named constructor write_motor_command() builds from."""

    def __init__(self, n: int) -> None:
        self.pos, self.vel, self.kp, self.kd, self.torques = (np.zeros(n) for _ in range(5))

    @classmethod
    def init_all_zero(cls, n: int) -> "JointCommands":
        return cls(n)


class FakeMotorRobot(conftest.FakeRobot):
    """A FakeRobot with i2rt's command block and motor loop: what System 0 writes into."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.motor_chain = [object()] * 7
        self.remapper = pytypes.SimpleNamespace(to_robot_joint_pos_space=lambda pos: np.array(pos))
        self._commands = JointCommands(7)
        self._command_lock = threading.Lock()
        self._kp = np.array([80.0, 80.0, 80.0, 40.0, 10.0, 10.0, 20.0])
        self._kd = np.array([5.0, 5.0, 5.0, 1.5, 1.5, 1.5, 0.5])
        self.updates = 0

    def _clip_robot_joint_pos_command(self, pos: np.ndarray) -> np.ndarray:
        return pos

    def command_joint_pos(self, positions: np.ndarray) -> None:
        super().command_joint_pos(positions)
        block = JointCommands(7)
        block.pos, block.kp, block.kd = np.array(positions), self._kp, self._kd
        self._commands = block

    def update(self) -> None:
        self.updates += 1


def _models() -> motors.Models:
    return motors.Models(
        net=policy.load(),
        kinematics={arm: LinearKinematics() for arm in motors.ARMS},
        joint_range_rad={arm: WIDE_RANGE for arm in motors.ARMS},
    )


def _action(value: float, gripper: float = 0.5) -> types.Action:
    command = types.ArmCommand(positions_rad=np.full(6, value), gripper=gripper)
    return types.Action(left=command, right=command)


def _plans(value: float, t_s: float = 0.0) -> dict[str, tracker.Plan]:
    rows = np.append(np.full(6, value), 0.5)[None, :]
    return {arm: tracker.Plan(t_s=np.array([t_s]), rows=rows) for arm in motors.ARMS}


class _Writer:
    """The command writer's surface, recording what it is asked."""

    def __init__(self) -> None:
        self.period_s, self.written = 0.004, 0
        self.calls: list[str] = []
        self.restarts = 0

    def submit(self, action: types.Action, now_s: float) -> None:
        self.calls.append("submit")

    def hold(self) -> None:
        self.calls.append("hold")

    def once(self, now_s: float) -> None:
        self.calls.append("once")
        self.written += 1

    def restart(self) -> None:
        self.restarts += 1


def _path(clock: conftest.ManualClock | None = None) -> tuple[motors.MotorPath, _Writer, dict[str, FakeMotorRobot]]:
    robots = {arm: FakeMotorRobot() for arm in motors.ARMS}
    arms = {arm: conftest.yam_arm(arm, robot) for arm, robot in robots.items()}
    writer = _Writer()
    path = motors.MotorPath(writer, arms, _models(), max_ee_speed_m_s=10.0, clock=clock or conftest.ManualClock())
    return path, writer, robots


# --- the network -----------------------------------------------------------------------------------------


def test_bundled_network_answers_what_the_station_checkpoint_answers() -> None:
    """Golden actions from the tracker's PyTorch checkpoint, for three fixed observations."""
    net = policy.load()
    golden = {
        0.0: [-0.023347, -0.030628, 0.042198, -0.025848, 0.019952, -0.010469, -0.002534],
        0.5: [-0.250205, -0.147674, -0.323899, -0.077817, 0.34451, 0.218255, 0.010584],
        -1.5: [0.496135, 0.281623, 0.645868, 0.046949, -0.772805, -0.34266, -0.151382],
    }
    for scale, expected in golden.items():
        obs = net.obs_mean + scale * net.obs_std * np.sin(np.arange(policy.OBS_DIM))
        np.testing.assert_allclose(policy.act(net, obs), expected, atol=2e-6)
    assert net.meta["actuator"] == "hybrid_torque_full"


def test_observation_is_state_then_errors_from_the_measured_pose() -> None:
    q = np.arange(7.0)
    rows = np.tile(q + 1.0, (policy.FUTURE_FRAMES + 1, 1))
    obs = policy.observation(q, np.full(7, 2.0), np.full(7, 3.0), rows)
    assert obs.shape == (policy.OBS_DIM,)
    np.testing.assert_array_equal(obs[:7], q)
    np.testing.assert_allclose(obs[7:14], 0.2)
    np.testing.assert_array_equal(obs[14:21], 3.0)
    np.testing.assert_array_equal(obs[21:], 1.0)
    with pytest.raises(ValueError, match="reference rows"):
        policy.observation(q, q, q, rows[:3])


# --- the command and the supervisor ----------------------------------------------------------------------


def test_decode_sets_torque_with_stock_damping_and_no_stiffness_and_leaves_the_gripper_to_the_cap() -> None:
    action = np.array([1.0, -1.0, 0.5, 0.0, 2.0, -0.25, 1.0])
    command = tracker.decode(action, np.append(np.full(6, 0.1), 0.95), WIDE_RANGE)
    np.testing.assert_allclose(command.torque, [11.2, -11.2, 5.6, 0.0, 4.0, -1.0])  # 0.4 of 28/10 N*m, clipped
    np.testing.assert_array_equal(command.kp, 0.0)
    np.testing.assert_array_equal(command.kd, tracker.NOMINAL_KD)
    np.testing.assert_allclose(command.target[:6], 0.1)
    assert command.target[6] == 0.95  # the loop's capped gripper, whatever the network's seventh output says


def _supervised(
    sup: tracker.Supervisor, now_s: float, qd: float = 0.0, q: float = 0.0, ref: float = 0.0, tau: float = 0.0
) -> tracker.MotorCommand:
    command = tracker.MotorCommand(np.zeros(7), np.zeros(6), tracker.NOMINAL_KD.copy(), np.full(6, tau))
    return sup.filter(now_s, np.full(7, q), np.full(7, qd), np.full(7, ref), command)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"qd": 5.0}, "overspeed on joint 1"),
        ({"q": 3.1}, "joint 1 past its limit"),
    ],
)
def test_supervisor_holds_on_overspeed_and_a_joint_past_its_limit(kwargs: dict[str, float], reason: str) -> None:
    sup = tracker.Supervisor(WIDE_RANGE, tracker.SupervisorConfig())
    out = _supervised(sup, 0.0, **kwargs)
    assert sup.latched_reason == reason
    np.testing.assert_array_equal(out.kp, tracker.NOMINAL_KP)
    np.testing.assert_array_equal(out.torque, 0.0)


def test_supervisor_tolerates_a_joint_at_its_limit_as_the_yam_rests() -> None:
    sup = tracker.Supervisor(WIDE_RANGE, tracker.SupervisorConfig())
    _supervised(sup, 0.0, q=3.04)
    assert sup.latched_reason == ""


def test_supervisor_leash_trips_only_when_the_error_is_held() -> None:
    sup = tracker.Supervisor(WIDE_RANGE, tracker.SupervisorConfig())
    for k in range(50):  # 0.5 rad off for 0.196 s: not yet
        _supervised(sup, k * 0.004, q=0.5)
    assert sup.latched_reason == ""
    _supervised(sup, 0.21, q=0.5)
    assert sup.latched_reason.startswith("tracking error")


def test_supervisor_watchdog_trips_on_a_late_tick_and_resume_restarts_it() -> None:
    sup = tracker.Supervisor(WIDE_RANGE, tracker.SupervisorConfig())
    _supervised(sup, 0.0)
    sup.resume()
    _supervised(sup, 1.0)  # a new stretch: the gap before it is no late tick
    assert sup.latched_reason == ""
    _supervised(sup, 1.08)
    assert sup.latched_reason.startswith("watchdog")
    sup.resume()
    assert sup.latched_reason.startswith("watchdog")  # a hold outlives a new stretch


def test_supervisor_clamps_and_rate_limits_the_torque_then_holds_when_it_stays_clamped() -> None:
    sup = tracker.Supervisor(WIDE_RANGE, tracker.SupervisorConfig())
    out = _supervised(sup, 0.0, tau=100.0)
    np.testing.assert_allclose(out.torque, 0.05 * tracker.TORQUE_LIMIT)  # one tick's step from zero
    for k in range(1, 30):
        out = _supervised(sup, k * 0.004, tau=100.0)
    assert sup.latched_reason == "torque clamped too long"
    assert sup.clamped_ticks >= 26


# --- the reference ----------------------------------------------------------------------------------------


def test_plan_sample_interpolates_between_rows_and_holds_past_either_end() -> None:
    plan = tracker.Plan(t_s=np.array([1.0, 2.0]), rows=np.array([[0.0] * 7, [1.0] * 7]))
    np.testing.assert_allclose(tracker.sample(plan, np.array([0.0, 1.5, 3.0]))[:, 0], [0.0, 0.5, 1.0])


def test_capped_reference_stays_inside_the_limits_and_the_speed_caps() -> None:
    limits = np.tile([-1.0, 1.0], (6, 1))
    arm = tracker.ArmTracker(policy.load(), limits, LinearKinematics(), max_ee_speed_m_s=0.5)
    arm.restart()
    rows = np.zeros((policy.FUTURE_FRAMES + 1, 7))
    rows[:, 1] = 5.0  # far past joint 2's limit
    rows[:, 0] = np.linspace(0.0, 2.0, policy.FUTURE_FRAMES + 1)  # 2 m/s at the grasp point
    out = arm.capped(rows, np.zeros(7), 0.0)
    assert out[:, 1].max() <= 1.0 - tracker.REF_LIMIT_MARGIN_RAD + 1e-9
    assert abs(out[0, 1]) <= 0.8 * 4.5 * 0.004 + 1e-9  # the first row moves one tick from the measured pose
    assert np.all(np.abs(np.diff(out[:, 0])) <= 0.5 * policy.FUTURE_DT_S * 1.01 + 1e-9)


def test_tracker_at_rest_on_a_still_plan_asks_for_little_torque() -> None:
    arm = tracker.ArmTracker(policy.load(), WIDE_RANGE, LinearKinematics(), max_ee_speed_m_s=0.6)
    arm.restart()
    pose = np.append([0.0, 0.6, 0.6, 0.0, 0.0, 0.0], 0.5)
    plan = tracker.Plan(t_s=np.array([0.0]), rows=pose[None, :])
    for k in range(50):
        command = arm.tick(k * 0.004, pose, np.zeros(7), plan)
    assert arm.supervisor.latched_reason == ""
    assert np.all(np.abs(command.torque) < 0.25 * tracker.TORQUE_FRACTION * tracker.TORQUE_LIMIT)


# --- the motor path ---------------------------------------------------------------------------------------


def test_motor_path_leaves_the_ramp_to_the_writer_and_drives_the_motors_from_a_plan() -> None:
    path, writer, robots = _path()
    path.submit(_action(0.0), 0.0, None)
    path.once(0.0)
    assert writer.calls == ["submit", "once"] and not path.is_driving and path.written == 1
    path.submit(_action(0.0), 0.05, _plans(0.0))
    path.once(0.05)
    assert path.is_driving and path.written == 2
    for robot in robots.values():
        np.testing.assert_array_equal(robot._commands.kp[:6], 0.0)
        np.testing.assert_array_equal(robot._commands.kd[:6], tracker.NOMINAL_KD)
        assert robot._commands.kp[6] == 20.0  # the gripper keeps the driver's own gains


def test_motor_path_hands_the_arms_back_at_a_new_ramp_from_the_submitted_command() -> None:
    path, writer, _ = _path()
    path.submit(_action(0.0), 0.0, _plans(0.0))
    path.once(0.0)
    ramp = _action(0.1)
    path.submit(ramp, 0.05, None)
    path.once(0.05)
    assert not path.is_driving and writer.restarts == 1 and writer.calls[-1] == "once"


def test_motor_path_hold_freezes_the_plan_while_driving_and_holds_the_writer_otherwise() -> None:
    path, writer, _ = _path()
    path.hold()
    assert writer.calls == ["hold"]
    path.submit(_action(0.2), 0.0, {arm: tracker.Plan(np.array([0.0, 1.0]), np.zeros((2, 7))) for arm in motors.ARMS})
    path.once(0.0)
    path.hold()
    assert writer.calls == ["hold", "submit"]
    assert path._submission is not None and path._submission.plans is not None
    still = path._submission.plans["left"]
    np.testing.assert_allclose(tracker.sample(still, np.array([5.0]))[0, :6], 0.2)


def test_a_safety_hold_stops_the_run_and_stop_leaves_the_arms_on_their_pd() -> None:
    path, _, robots = _path()
    robots["right"].joint_vel = np.full(6, 9.0)  # far past the trip speed
    path.submit(_action(0.0), 0.0, _plans(0.0))
    path.once(0.0)
    assert isinstance(path.error, errors.ControlLoopError) and "right arm: overspeed" in str(path.error)
    np.testing.assert_array_equal(robots["right"]._commands.kp[:6], tracker.NOMINAL_KP)
    path.stop()
    assert not path.is_driving
    np.testing.assert_array_equal(robots["left"]._commands.kp[:6], [80.0, 80.0, 80.0, 40.0, 10.0, 10.0])


def test_the_motor_loop_check_replaces_a_stale_block_with_a_pd_hold() -> None:
    clock = conftest.ManualClock()
    path, _, robots = _path(clock)
    path.submit(_action(0.0), 0.0, _plans(0.0))
    path.once(0.0)
    left = path._motors["left"]
    assert isinstance(left, motors.ArmMotors)
    left._guard()
    assert left.supervisor.latched_reason == ""
    clock.now_s += motors.BLOCK_STALE_S + 0.01
    left._guard()
    assert left.supervisor.latched_reason.startswith("watchdog")
    np.testing.assert_array_equal(robots["left"]._commands.kp[:6], tracker.NOMINAL_KP)
    assert isinstance(path.error, errors.ControlLoopError) and "left arm: watchdog" in str(path.error)


def test_start_hooks_each_arms_motor_loop_and_stop_unhooks_it() -> None:
    path, _, robots = _path()
    path.start()
    try:
        assert "update" in vars(robots["left"])
        robots["left"].update()
        assert robots["left"].updates == 1
    finally:
        path.stop()
    assert "update" not in vars(robots["left"])


# --- the arm's process (served from a thread here: the same pipes, no i2rt) ---------------------------------


class ArmKinematics(LinearKinematics):
    """LinearKinematics with the joint limits `ArmKinematics` carries, as `motors.build_arm()` returns it."""

    joint_range_rad = WIDE_RANGE


def _served(robot: FakeMotorRobot | None = None) -> tuple[arm_process.RemoteArm, FakeMotorRobot, threading.Thread]:
    """A RemoteArm whose arm's process is a thread running the real server over real pipes."""
    robot = robot or FakeMotorRobot()
    arm = conftest.yam_arm("left", robot)
    ours, theirs = multiprocessing.Pipe(duplex=True)
    events_in, events_out = multiprocessing.Pipe(duplex=False)
    server = arm_process._Server("left", arm, (policy.load(), ArmKinematics()), events_out)

    def serve() -> None:
        try:
            server.serve(theirs)
        finally:
            server.shut_down()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    process = pytypes.SimpleNamespace(
        is_alive=thread.is_alive, exitcode=None, join=thread.join, terminate=lambda: None
    )
    return arm_process.RemoteArm("left", process, ours, events_in, home_rad=np.zeros(6)), robot, thread


def _wait_for(condition: Callable[[], bool], timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def test_an_arm_process_serves_the_arm_and_drops_pd_writes_while_its_tracker_drives() -> None:
    remote, robot, thread = _served()
    remote.write(types.ArmCommand(positions_rad=np.full(6, 0.1), gripper=0.4))
    np.testing.assert_allclose(remote.read().commanded_rad, 0.1)
    assert remote.is_alive()
    sys0 = remote.sys0_motors(10.0)
    sys0.start()
    sys0.follow(_plans(0.0)["left"])
    _wait_for(lambda: robot._commands.kp[0] == 0.0)  # the tracker's block: torque, no stiffness
    written = len(robot.commands)
    remote.write(types.ArmCommand(positions_rad=np.full(6, 0.3), gripper=0.4))
    remote.hold()
    remote.read()  # answered after the write and the hold, so both were handled
    assert len(robot.commands) == written
    sys0.stop()
    assert robot._commands.kp[0] == 80.0 and sys0.error is None  # left on its PD
    remote.close()
    thread.join(2.0)
    assert robot.closed and not thread.is_alive() and not remote.is_alive()


def test_a_safety_hold_in_an_arm_process_stops_the_motor_path() -> None:
    (left, left_robot, _), (right, _, _) = _served(), _served()
    left_robot.joint_vel = np.full(6, 9.0)  # far past the trip speed
    path = motors.MotorPath(
        _Writer(), {"left": left, "right": right}, motors.IN_ARM_PROCESSES, max_ee_speed_m_s=10.0, clock=time.monotonic
    )
    path.start()
    try:
        path.submit(_action(0.0), time.monotonic(), _plans(0.0))
        _wait_for(lambda: path.error is not None)
        assert isinstance(path.error, errors.ControlLoopError) and "left arm: overspeed" in str(path.error)
    finally:
        path.stop()
        left.close()
        right.close()
    np.testing.assert_array_equal(left_robot._commands.kp[:6], [80.0, 80.0, 80.0, 40.0, 10.0, 10.0])


def test_an_arm_process_holds_its_arm_and_closes_it_when_the_agent_goes_away() -> None:
    remote, robot, thread = _served()
    sys0 = remote.sys0_motors(10.0)
    sys0.start()
    sys0.follow(_plans(0.0)["left"])
    _wait_for(lambda: robot._commands.kp[0] == 0.0)
    remote._requests.close()
    thread.join(2.0)
    assert not thread.is_alive() and robot.closed
    assert robot._commands.kp[0] == 80.0  # a PD hold, not the tracker's last torque


def test_a_write_refused_in_the_arm_process_is_raised_by_the_next_request() -> None:
    remote, robot, _ = _served()
    robot.write_error = RuntimeError("bus off")
    remote.write(types.ArmCommand(positions_rad=np.zeros(6), gripper=0.5))
    with pytest.raises(RuntimeError, match="bus off"):
        remote.read()
    remote.read()
    with pytest.raises(errors.ArmError, match="outside"):
        remote.write(types.ArmCommand(positions_rad=np.zeros(6), gripper=2.0))  # refused here, before the pipe
    remote.close()


def test_an_arm_in_its_process_answers_the_stations_arm_checks_as_a_yam_does() -> None:
    """The alarm, dead-hold and reopen checks the station and loop make of every arm, answered for a YAM."""
    remote, _, thread = _served()
    assert not remote.drops_when_limp and remote.alarm() is None and not remote.holds_dead()
    assert "motor chain" in remote.why_stopped()  # the YAM driver's reason, asked in the arm's process
    remote.close_for_reopen()
    thread.join(2.0)
    assert not thread.is_alive() and not remote.is_alive() and remote.why_stopped() == "it was closed"


def test_a_motor_path_over_arms_in_this_process_needs_the_network() -> None:
    arms = {arm: conftest.yam_arm(arm, FakeMotorRobot()) for arm in motors.ARMS}
    with pytest.raises(errors.ConfigError, match="no network"):
        motors.MotorPath(_Writer(), dict(arms), motors.IN_ARM_PROCESSES, max_ee_speed_m_s=10.0, clock=time.monotonic)


# --- the follower, the loop and the switch ----------------------------------------------------------------


def test_write_motor_command_swaps_the_block_and_remembers_the_target() -> None:
    robot = FakeMotorRobot()
    arm = conftest.yam_arm("left", robot)
    arm.write_motor_command(np.append(np.full(6, 0.3), 0.7), np.zeros(6), np.ones(6), np.full(6, 2.0))
    np.testing.assert_array_equal(robot._commands.torques, [2.0] * 6 + [0.0])
    np.testing.assert_array_equal(robot._commands.kd, [1.0] * 6 + [0.5])
    state = arm.read()
    assert state.commanded_gripper == pytest.approx(0.7)
    np.testing.assert_allclose(state.commanded_rad, 0.3)
    with pytest.raises(errors.ArmError, match="no MIT command block"):
        conftest.yam_arm("right", conftest.FakeRobot()).write_motor_command(np.zeros(7), *np.zeros((3, 6)))


def test_a_loop_with_system_0_drives_the_motors_after_the_ramp(
    build: Any,
) -> None:
    robots: dict[str, conftest.FakeRobot] = {arm: FakeMotorRobot() for arm in motors.ARMS}
    harness = build(robots=robots, action_horizon=8, sys0=_models())

    async def ticks() -> None:
        async with harness.loop:
            for _ in range(4):
                await harness.loop.step(harness.clock())
                harness.clock.now_s += 1 / 30

    asyncio.run(ticks())
    assert harness.loop._commands.written >= 3
    assert not harness.loop._commands.is_driving  # stop() handed the arms to their PD
    assert isinstance(robots["left"], FakeMotorRobot) and robots["left"]._commands.kp[0] == 80.0


def test_tracker_plans_time_the_buffered_rows_from_now() -> None:
    harness = conftest.make_loop(action_horizon=8)
    loop = harness.loop
    loop._ramp = None
    loop._buffer.extend([_action(0.1), _action(0.2)])
    rows = {arm: np.array([[0.0] * 7, [0.1] * 7, [0.2] * 7]) for arm in motors.ARMS}
    loop._chunk_plan = control._ChunkPlan(rows=rows, ee_speed_m_s={}, due_s=np.zeros(3))
    plans = loop._tracker_plans(_action(0.0), 5.0)
    assert plans is not None
    np.testing.assert_allclose(plans["left"].t_s, 5.0 + np.arange(3) / 30)
    np.testing.assert_allclose(plans["left"].rows[:, 0], [0.0, 0.1, 0.2])
    # The buffered rows carry the policy's raw gripper (0.1, 0.2); the plan holds the submitted, capped one.
    np.testing.assert_allclose(plans["left"].rows[:, 6], 0.5)
    loop._ramp = object()  # type: ignore[assignment]
    assert loop._tracker_plans(_action(0.0), 5.0) is None


def test_tracker_plans_ramp_the_gripper_over_the_tick_from_last_ticks_command() -> None:
    """The gripper's target moves from the command submitted last tick to this tick's over one period, so the tracker's
    4 ms samples make a ramp of it, not a 20 Hz staircase; unchanged, it is held as it was."""
    harness = conftest.make_loop(action_horizon=8)
    loop = harness.loop
    loop._ramp = None
    loop._last_action = _action(0.0, gripper=0.6)
    rows = {arm: np.array([[0.0] * 7, [0.1] * 7]) for arm in motors.ARMS}
    loop._chunk_plan = control._ChunkPlan(rows=rows, ee_speed_m_s={}, due_s=np.zeros(2))
    loop._buffer.extend([_action(0.1)])
    plans = loop._tracker_plans(_action(0.0, gripper=0.5), 5.0)
    assert plans is not None
    plan, period = plans["left"], 1 / 30
    np.testing.assert_allclose(plan.rows[:, 6], [0.6, 0.5])
    np.testing.assert_allclose(
        tracker.sample(plan, np.array([5.0, 5.0 + period / 2, 5.0 + period]))[:, 6], [0.6, 0.55, 0.5]
    )

    loop._buffer.clear()  # nothing buffered: a second row one period on carries the ramp's end
    plans = loop._tracker_plans(_action(0.0, gripper=0.4), 5.0)
    assert plans is not None
    np.testing.assert_allclose(plans["right"].t_s, [5.0, 5.0 + period])
    np.testing.assert_allclose(plans["right"].rows[:, 6], [0.6, 0.4])

    loop._last_action = _action(0.0, gripper=0.4)  # unchanged: one row, held
    plans = loop._tracker_plans(_action(0.0, gripper=0.4), 5.0)
    assert plans is not None and len(plans["left"].rows) == 1 and plans["left"].rows[0, 6] == 0.4


def test_system_0_refuses_a_body_other_than_the_yam() -> None:
    """The tracker and the fast motor loop are the i2rt YAM's: another body fails before any arm is opened."""
    from robots import station

    with pytest.raises(errors.ConfigError, match="System 0 drives the i2rt YAM only"):
        station.StationController.from_config(
            conftest.make_app_config(body="makers"), inference_log_path=None, sys0=True
        )


@pytest.mark.parametrize(
    ("body", "flags", "is_sys0"),
    [
        ("yam", [], True),
        ("yam", ["--no-sys0"], False),
        ("yam", ["--sys0"], True),
        ("makers", [], False),
        ("makers", ["--sys0"], True),
    ],
    ids=["yam-default", "yam-off", "yam-on", "makers-default", "makers-asked"],
)
def test_station_runs_system_0_by_default_on_a_yam_only(
    body: str, flags: list[str], is_sys0: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`python -m robots.station` serves a YAM with System 0 unless `--no-sys0` (`dreamcli robot up nosys0`); another
    body runs without it, and one asking for it with `--sys0` is left to `from_config` to refuse."""
    from robots import run, station

    seen: list[bool] = []

    async def fake_serve(cfg: Any, **kwargs: Any) -> None:
        seen.append(kwargs["sys0"])

    monkeypatch.setattr(config, "load", lambda path: conftest.make_app_config(body=body))
    monkeypatch.setattr(run, "with_task", lambda cfg, task: cfg)
    monkeypatch.setattr(station, "serve", fake_serve)
    station.main(["--config", "station.yaml", *flags])
    assert seen == [is_sys0]


def test_station_runs_the_eef_head_unless_no_eef(monkeypatch: pytest.MonkeyPatch) -> None:
    """`python -m robots.station` runs the model's EEF head, with System 0 on a YAM; `--no-eef` (`dreamcli robot up
    noeef`) runs its joint head, the action space `joint`, System 0 unchanged."""
    from robots import run, station

    seen: list[tuple[str, bool]] = []

    async def fake_serve(cfg: Any, **kwargs: Any) -> None:
        seen.append((cfg.action_space, kwargs["sys0"]))

    monkeypatch.setattr(config, "load", lambda path: conftest.make_app_config())
    monkeypatch.setattr(run, "with_task", lambda cfg, task: cfg)
    monkeypatch.setattr(station, "serve", fake_serve)
    station.main(["--config", "station.yaml"])
    station.main(["--config", "station.yaml", "--no-eef"])
    assert seen == [("eef", True), ("joint", True)]


# --- the fast motor loop (needs i2rt) ---------------------------------------------------------------------


class _Recorder:
    """i2rt's rate recorder, which the driver loop enters and ticks."""

    def __enter__(self) -> "_Recorder":
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def track(self) -> None:
        pass


class _Bus:
    """A motor chain for the fast driver loop: each cycle holds the command lock for 1.8 ms, as the pipelined bus."""

    def __init__(self) -> None:
        self.running = True
        self.command_lock = threading.RLock()
        self.state_lock = threading.Lock()
        self.commands: list[Any] = []
        self.same_bus_device_driver = None
        self.cycles = 0
        self._rate_recorder = _Recorder()

    def _set_commands(self, commands: list[Any]) -> list[Any]:
        time.sleep(0.0018)  # the bus round trip, with the interpreter lock released as socket I/O releases it
        self.cycles += 1
        return [pytypes.SimpleNamespace(error_code="0x1")]

    def _update_absolute_positions(self, feedback: list[Any]) -> None:
        pass


def test_fast_driver_loop_lets_the_robot_loop_take_the_lock_between_cycles() -> None:
    """On Python 3.10, i2rt's sleep(0) keeps the lock from the robot loop (3.12 hands it over anyway)."""
    bus = _Bus()
    driver = threading.Thread(target=fast_loop._driver_loop, args=(bus,), daemon=True)
    driver.start()
    taken, deadline = 0, time.monotonic() + 0.5
    while time.monotonic() < deadline:
        with bus.command_lock:  # what the robot loop's update() waits for
            taken += 1
        time.sleep(fast_loop.ROBOT_LOOP_SLEEP_S)
    bus.running = False
    driver.join(1.0)
    assert taken >= 0.6 * bus.cycles, f"robot loop took the lock {taken} times in {bus.cycles} driver cycles"


@pytest.mark.parametrize("motor_type", ["DM4340", "DM4310"])
def test_fast_loop_frames_match_i2rts_set_control(motor_type: str) -> None:
    dm = pytest.importorskip("i2rt.motor_drivers.dm_driver")
    iface = object.__new__(dm.DMSingleMotorCanInterface)
    iface.control_mode = dm.ControlMode.MIT
    iface.cmd_idoffset = dm.ControlMode.get_id_offset(dm.ControlMode.MIT)
    iface.receive_mode = dm.ReceiveMode.p16
    sent: dict[str, bytes] = {}

    def capture(id: int, motor_id: int, data: bytes, **_: Any) -> None:
        sent["data"] = bytes(data)
        raise StopIteration

    iface._send_message_get_response = capture
    rng = np.random.default_rng(0)
    for _ in range(20):
        pos, vel, kp, kd, tau = (
            rng.uniform(-2, 2),
            rng.uniform(-3, 3),
            rng.uniform(0, 80),
            rng.uniform(0, 5),
            rng.uniform(-5, 5),
        )
        with pytest.raises(StopIteration):
            iface.set_control(motor_id=3, motor_type=motor_type, pos=pos, vel=vel, kp=kp, kd=kd, torque=tau)
        assert fast_loop.mit_frame(motor_type, pos, vel, kp, kd, tau) == sent["data"]


## The loop tests' factory: make_loop on a clock that lets the capture threads publish, readers stopped after.
build = test_control.build

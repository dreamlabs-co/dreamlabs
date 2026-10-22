"""Each arm in a process of its own, with System 0: i2rt's motor threads and the tracker, and nothing else.

With System 0 the tracker writes each motor's torque every 4 ms, and i2rt's fast motor loop
(#robots.sys0.fast_loop) carries it to the bus. In the agent's process those threads would share one interpreter
lock with the cameras, the control loop and the server, where the tracker waits 50 to 70 ms for it, so every
run stops on the stale-block watchdog (the robot loop at 41 to 67 Hz where it runs at hundreds, the
20 Hz tick overrunning by up to 370 ms). So each arm gets a process of its own.

open_arms() spawns one process per arm and returns a #RemoteArm for each, which the station uses as it uses a
`FollowerArm` (#robots.followers.Arm): read(), hold(), is_alive() and the rest are requests answered by the arm's
process, write() is sent without waiting for an answer (a refusal is raised by the next request), and
sys0_motors() is the arm's System 0 for #robots.sys0.motors.MotorPath. In the arm's process #_Server owns the
`FollowerArm` (opened with the fast motor loop), the network and the arm's kinematics, and, between a start and a
stop, an #robots.sys0.motors.ArmMotors and the thread that ticks it every 4 ms; plans arrive on the request pipe
without an answer, and a safety hold goes back on a pipe of its own, which the agent's
#robots.sys0.motors.MotorPath reads every tick.

While the tracker drives, a PD write or a hold from the agent is dropped: the writer's last periods can arrive
after the plan that took the motors, and they would put a stale PD target under the tracker for a cycle. If the
agent's process goes away (its pipe closes) or the arm's process is told to terminate, the arm is held on its PD at
the last target and its driver closed, so a tracker torque is never left on the motors.

@see open_arms(), RemoteArm, RemoteMotors
"""

import dataclasses
import logging
import multiprocessing
import multiprocessing.connection
import pickle
import signal
import sys
import threading
import time
from typing import Any

import numpy as np
import numpy.typing as npt

from robots import config, errors, followers, types, yam
from robots.sys0 import motors, tracker

_log = logging.getLogger(__name__)

## Seconds an arm's process has to open its driver and load the tracker.
READY_TIMEOUT_S: float = 120.0
## Seconds a request waits for the arm's process to answer.
CALL_TIMEOUT_S: float = 5.0
## Seconds close() waits for the arm's process to exit after its driver closed, before it is terminated.
EXIT_TIMEOUT_S: float = 5.0
## The tracker's tick, seconds.
TICK_S: float = 1.0 / tracker.MOTOR_HZ
## The requests that end an arm's process once answered.
_FINAL_OPS: frozenset[str] = frozenset({"close", "torque_off"})


def open_arms(arm_configs: dict[str, config.ArmConfig]) -> dict[str, "RemoteArm"]:
    """Spawns one process per arm and waits until each has opened its driver and loaded the tracker.

    The processes start together, so the arms open side by side; if one fails, the others are closed.

    @param arm_configs Each arm's config, keyed `"left"` and `"right"`.
    @return Each arm, keyed the same.
    @throws ArmError An arm's process did not get ready in #READY_TIMEOUT_S or exited.
    @throws Exception Whatever opening the driver or loading the tracker raised in an arm's process (an i2rt
        error, a ConfigError).
    """
    context = multiprocessing.get_context("spawn")
    started: dict[str, RemoteArm] = {}
    try:
        for name, cfg in arm_configs.items():
            started[name] = RemoteArm.spawn(context, cfg, name)
        for arm in started.values():
            arm.wait_ready()
    except BaseException:
        for name, arm in started.items():
            try:
                arm.close()
            except Exception:
                _log.exception("arm '%s': closing its process after a failed open failed", name)
        raise
    return started


class RemoteArm:
    """An arm served from its own process: the `FollowerArm` surface (#robots.followers.Arm), plus System 0.

    Thread-safe: one lock covers each request and its answer, so the loop, the command thread and the station may
    share it.

    Attributes:
    - name: `"left"` or `"right"`.
    - home_rad: The arm's home joints, `[6]`, radians, from its config (`robots.home` moves the arm there).
    - dof: The arm's joints, as many as its home lists (a YAM's six; System 0 drives only a YAM).

    @see open_arms(), sys0_motors()
    """

    def __init__(
        self,
        name: str,
        process: Any,
        requests: multiprocessing.connection.Connection,
        events: multiprocessing.connection.Connection,
        *,
        home_rad: npt.ArrayLike,
    ) -> None:
        """Binds a started process and its two pipes; open_arms() builds these.

        @param name `"left"` or `"right"`.
        @param process The arm's process, started.
        @param requests This end of the request pipe.
        @param events This end of the events pipe, which the process writes a safety hold to.
        @param home_rad The arm's home joints, six values, radians.
        """
        self.name = name
        self.home_rad: npt.NDArray[np.float64] = np.array(home_rad, dtype=np.float64)
        self.dof: int = self.home_rad.shape[0]
        self._process = process
        self._requests = requests
        self._events = events
        self._lock = threading.Lock()
        self._is_closed = False

    @classmethod
    def spawn(cls, context: Any, cfg: config.ArmConfig, name: str) -> "RemoteArm":
        """Starts the arm's process; wait_ready() waits for it to open the arm.

        @param context The multiprocessing context (spawn: nothing of this process's threads is inherited).
        @param cfg The arm's config.
        @param name `"left"` or `"right"`.
        @return The arm, not ready yet.
        """
        requests, child_requests = context.Pipe(duplex=True)
        events, child_events = context.Pipe(duplex=False)
        process = context.Process(
            target=_serve, args=(cfg, name, child_requests, child_events), name=f"arm-{name}", daemon=True
        )
        process.start()
        child_requests.close()
        child_events.close()
        _log.info("arm '%s': process %d opening the arm on %s", name, process.pid, cfg.can_channel)
        return cls(name, process, requests, events, home_rad=cfg.home)

    def wait_ready(self) -> None:
        """Waits for the process to say the arm is open and the tracker loaded.

        @throws ArmError The process did not answer in #READY_TIMEOUT_S, or exited.
        @throws Exception What opening the arm raised in the process.
        """
        self._answer(READY_TIMEOUT_S, "open the arm")

    def read(self) -> types.ArmState:
        """Reads the arm's state in its process (`FollowerArm.read()`).

        @return The measured state and the last commanded targets.
        @throws ArmError The arm's process refused or did not answer, or a write sent before was refused.
        """
        state: types.ArmState = self._call("read")
        return state

    def write(self, command: types.ArmCommand) -> None:
        """Sends joint targets and the gripper without waiting; dropped in the arm's process while the tracker drives.

        @param command Six joint targets in radians and a gripper value in `[0, 1]`.
        @throws ArmError The command is malformed (checked here, as `FollowerArm.write()` checks it), or the
            arm's process is gone.
        """
        self._send("write", followers.validated(self.name, self.dof, command))

    def hold(self) -> None:
        """Re-commands the last target (`FollowerArm.hold()`); while the tracker drives, the tracker holds instead.

        @throws ArmError The arm's process refused or did not answer.
        """
        self._call("hold")

    def disengage(self) -> None:
        """Stops System 0 if it runs and puts the arm in its safest passive mode (`FollowerArm.disengage()`).

        @throws ArmError The arm's process refused or did not answer.
        """
        self._call("disengage")

    def is_alive(self) -> bool:
        """Whether the arm's process runs and its driver still answers.

        @return False once closed, once the process has exited, or when it does not answer.
        """
        if self._is_closed or not self._process.is_alive():
            return False
        try:
            alive: bool = self._call("is_alive")
        except errors.ArmError:
            return False
        return alive

    def close(self) -> None:
        """Closes the driver in the arm's process (`FollowerArm.close()`) and waits for the process to exit."""
        self._finish("close")

    def torque_off(self) -> None:
        """Switches the arm's motors off in its process (`FollowerArm.torque_off()`) and waits for it to exit."""
        self._finish("torque_off")

    @property
    def drops_when_limp(self) -> bool:
        """False: System 0 drives the YAM only, which floats when limp (`Driver.drops_when_limp`)."""
        return False

    def alarm(self) -> str | None:
        """None: the YAM's driver raises no alarm (`Driver.alarm()`), so the loop's per-tick check costs no round trip.

        @return None.
        """
        return None

    def holds_dead(self) -> bool:
        """False: a YAM does not drop when limp, so a dead one is never left holding (`FollowerArm.holds_dead()`).

        @return False.
        """
        return False

    def why_stopped(self) -> str:
        """Why the arm reads dead (`FollowerArm.why_stopped()`), for messages.

        @return That it was closed, that its process exited, or its driver's own reason.
        """
        if self._is_closed:
            return "it was closed"
        if not self._process.is_alive():
            return f"its process exited (exit code {self._process.exitcode})"
        try:
            reason: str = self._call("why_stopped")
        except errors.ArmError as exc:
            return f"its process does not answer: {exc}"
        return reason

    def close_for_reopen(self) -> None:
        """Closes the arm (`FollowerArm.close_for_reopen()`: a YAM is closed, never switched off, to reopen it)."""
        self.close()

    def sys0_motors(self, max_ee_speed_m_s: float) -> "RemoteMotors":
        """The arm's System 0 for one run, driven in its process.

        @param max_ee_speed_m_s The loop's end-effector speed cap, which the tracker puts on the plan too.
        @return The handle #robots.sys0.motors.MotorPath drives.
        """
        return RemoteMotors(self, max_ee_speed_m_s)

    def call(self, op: str, *args: Any) -> Any:
        """Sends a request and waits for the answer; for #RemoteMotors.

        @param op The request.
        @param args Its arguments.
        @return The answer.
        @throws ArmError The process did not answer in #CALL_TIMEOUT_S or is gone.
        """
        return self._call(op, *args)

    def send(self, op: str, *args: Any) -> None:
        """Sends a request without waiting for an answer; for #RemoteMotors.

        @param op The request.
        @param args Its arguments.
        @throws ArmError The process is gone.
        """
        self._send(op, *args)

    def next_event(self) -> tuple[str, str] | None:
        """The next event the arm's process sent (`("error", why)`), or None when there is none waiting.

        @return The event, or None.
        """
        try:
            if not self._events.poll():
                return None
            event: tuple[str, str] = self._events.recv()
        except (EOFError, OSError):
            return None
        return event

    def _call(self, op: str, *args: Any) -> Any:
        """Sends `op` and waits for its answer under the lock."""
        with self._lock:
            self._require_open()
            try:
                self._requests.send((op, args, True))
            except (BrokenPipeError, OSError) as exc:
                raise errors.ArmError(f"arm '{self.name}': its process is gone ({exc})") from exc
            return self._answer(CALL_TIMEOUT_S, op)

    def _send(self, op: str, *args: Any) -> None:
        """Sends `op` without waiting, under the lock, so it never lands between another request and its answer."""
        with self._lock:
            self._require_open()
            try:
                self._requests.send((op, args, False))
            except (BrokenPipeError, OSError) as exc:
                raise errors.ArmError(f"arm '{self.name}': its process is gone ({exc})") from exc

    def _answer(self, timeout_s: float, what: str) -> Any:
        """Receives one answer: its value, or what the process raised.

        @param timeout_s Seconds to wait.
        @param what What was asked, for the message.
        @return The value.
        @throws ArmError No answer in `timeout_s`, or the process exited.
        @throws Exception What the process raised.
        """
        try:
            if not self._requests.poll(timeout_s):
                raise errors.ArmError(f"arm '{self.name}': its process did not {what} in {timeout_s:g} s")
            kind, value = self._requests.recv()
        except (EOFError, OSError) as exc:
            raise errors.ArmError(f"arm '{self.name}': its process exited (code {self._process.exitcode})") from exc
        if kind == "err":
            raise value
        return value

    def _finish(self, op: str) -> None:
        """Sends a final request, waits for the process to exit, and terminates it if it does not.

        A process that exits by itself meanwhile (one whose arm failed to open) has nothing left to close, so its
        missing answer is not raised; one that has to be terminated is (its SIGTERM still holds the arm first).

        @param op `"close"` or `"torque_off"`.
        @throws ArmError The process did not answer and had to be terminated.
        @throws Exception What the final request raised in the process.
        """
        if self._is_closed:
            return
        unanswered: errors.ArmError | None = None
        try:
            if self._process.is_alive():
                self._call(op)
        except errors.ArmError as exc:
            unanswered = exc
        finally:
            self._is_closed = True
            self._process.join(EXIT_TIMEOUT_S)
            is_stuck = self._process.is_alive()
            if is_stuck:
                _log.warning("arm '%s': its process did not exit; terminating it", self.name)
                self._process.terminate()
                self._process.join(EXIT_TIMEOUT_S)
            self._requests.close()
            self._events.close()
        if unanswered is not None and is_stuck:
            raise unanswered

    def _require_open(self) -> None:
        """Raises once the arm was closed, as `FollowerArm` does."""
        if self._is_closed:
            raise errors.ArmError(f"arm '{self.name}' driver is closed; reset the arms to reopen it")


class RemoteMotors:
    """One arm's System 0 in the arm's process, as #robots.sys0.motors.MotorPath drives it (`motors.Handle`).

    The arm's process ticks the tracker itself, so once() only reads what the process reported.

    @see RemoteArm.sys0_motors()
    """

    def __init__(self, arm: RemoteArm, max_ee_speed_m_s: float) -> None:
        """Binds the arm; nothing is sent until start().

        @param arm The arm.
        @param max_ee_speed_m_s The loop's end-effector speed cap.
        """
        self._arm = arm
        self._max_ee_speed_m_s = max_ee_speed_m_s
        self._error: str | None = None

    @property
    def error(self) -> str | None:
        """Why the arm's process stopped the tracker (a safety hold, or what its tick raised), or None."""
        while (event := self._arm.next_event()) is not None:
            kind, why = event
            if kind == "error" and self._error is None:
                self._error = why
        return self._error

    def follow(self, plan: tracker.Plan | None) -> None:
        """Sends the tracker its plan (None hands the motors back to the PD path); no answer is awaited.

        @param plan What to follow, or None.
        @throws ArmError The arm's process is gone.
        """
        self._arm.send("follow", plan)

    def once(self, now_s: float) -> None:
        """Reads what the arm's process reported; its own thread ticks the tracker.

        @param now_s Unused: the arm's process keeps its own time on the same monotonic clock.
        """
        del now_s
        _ = self.error

    def start(self) -> None:
        """Starts the tracker's thread in the arm's process; an event left from an earlier run is dropped first.

        @throws ArmError The arm's process refused or did not answer.
        """
        while self._arm.next_event() is not None:
            pass
        self._arm.call("start", self._max_ee_speed_m_s)

    def stop(self) -> None:
        """Stops the tracker in the arm's process, which leaves the arm on its PD at the last target written.

        @throws ArmError The arm's process refused or did not answer.
        """
        self._arm.call("stop")


@dataclasses.dataclass
class _Run:
    """One run's System 0 in an arm's process: the arm's #robots.sys0.motors.ArmMotors and its thread.

    Attributes:
    - motors: The tracker and its check.
    - thread: The thread ticking it every #TICK_S.
    - is_running: Cleared to stop the thread.
    """

    motors: motors.ArmMotors
    thread: threading.Thread
    is_running: threading.Event


class _Server:
    """The arm's process: answers the agent's requests on the arm it owns.

    @see serve()
    """

    def __init__(
        self,
        name: str,
        arm: followers.FollowerArm,
        built: tuple[Any, Any],
        events: multiprocessing.connection.Connection,
    ) -> None:
        """Binds the open arm, the tracker's network and kinematics, and the events pipe.

        @param name `"left"` or `"right"`.
        @param arm The arm, open with the fast motor loop.
        @param built The network and the arm's kinematics (`motors.build_arm()`).
        @param events This end of the events pipe.
        """
        self._name = name
        self._arm = arm
        self._net, self._kinematics = built
        self._events = events
        self._run: _Run | None = None
        self._refused: Exception | None = None
        self._is_closed = False

    def serve(self, requests: multiprocessing.connection.Connection) -> None:
        """Answers requests until a final one, or until the agent's end of the pipe closes.

        @param requests This end of the request pipe.
        """
        while True:
            try:
                op, args, wants_answer = requests.recv()
            except EOFError:
                _log.warning("arm '%s': the agent's process went away; holding the arm and closing it", self._name)
                return
            try:
                value = getattr(self, f"_op_{op}")(*args)
            except Exception as exc:
                if not wants_answer:
                    # A write is not answered: its refusal is raised by the next request that is.
                    self._refused = self._refused or exc
                    continue
                requests.send(("err", _portable(exc)))
            else:
                if wants_answer:
                    requests.send(("ok", value))
            if op in _FINAL_OPS:
                return

    def shut_down(self) -> None:
        """Leaves the arm safe when the process ends without a close: tracker stopped, a PD hold, driver closed."""
        if self._is_closed:
            return
        try:
            self._stop_run()
            self._arm.hold()
        finally:
            self._arm.close()

    def _op_read(self) -> types.ArmState:
        self._raise_refused()
        return self._arm.read()

    def _op_write(self, command: types.ArmCommand) -> None:
        if self._is_driving():
            return
        self._arm.write(command)

    def _op_hold(self) -> None:
        self._raise_refused()
        if self._is_driving():
            return
        self._arm.hold()

    def _op_disengage(self) -> None:
        self._stop_run()
        self._arm.disengage()

    def _op_is_alive(self) -> bool:
        return self._arm.is_alive()

    def _op_why_stopped(self) -> str:
        return self._arm.why_stopped()

    def _op_close(self) -> None:
        self._is_closed = True
        self._stop_run()
        self._arm.close()

    def _op_torque_off(self) -> None:
        self._is_closed = True
        self._stop_run()
        self._arm.torque_off()

    def _op_start(self, max_ee_speed_m_s: float) -> None:
        self._raise_refused()
        self._stop_run()
        arm_motors = motors.ArmMotors(
            self._name,
            self._arm,
            self._net,
            self._kinematics.joint_range_rad,
            self._kinematics,
            max_ee_speed_m_s=max_ee_speed_m_s,
            clock=time.monotonic,
        )
        arm_motors.start()
        is_running = threading.Event()
        is_running.set()
        thread = threading.Thread(target=self._tick, args=(arm_motors, is_running), name="sys0-tick", daemon=True)
        self._run = _Run(motors=arm_motors, thread=thread, is_running=is_running)
        thread.start()

    def _op_follow(self, plan: tracker.Plan | None) -> None:
        if self._run is not None:
            self._run.motors.follow(plan)

    def _op_stop(self) -> None:
        self._stop_run()

    def _is_driving(self) -> bool:
        """Whether the tracker has the motors, so a PD write from the agent is stale."""
        return self._run is not None and self._run.motors.is_driving

    def _raise_refused(self) -> None:
        """Raises a write's refusal on the next request that is answered."""
        refused, self._refused = self._refused, None
        if refused is not None:
            raise refused

    def _stop_run(self) -> None:
        """Stops the tracker's thread and the tracker, which leaves the arm on its PD at the last target written."""
        run, self._run = self._run, None
        if run is None:
            return
        run.is_running.clear()
        run.thread.join(motors.STOP_TIMEOUT_S)
        if run.thread.is_alive():
            _log.warning("arm '%s': System 0 thread still ticking; holding the arm anyway", self._name)
        run.motors.stop()

    def _tick(self, arm_motors: motors.ArmMotors, is_running: threading.Event) -> None:
        """The tracker's thread: once() every #TICK_S; a safety hold or a failure is sent to the agent once."""
        is_reported = False
        try:
            next_s = time.monotonic()
            while is_running.is_set():
                now_s = time.monotonic()
                if now_s < next_s:
                    time.sleep(min(next_s - now_s, TICK_S))
                    continue
                next_s = max(next_s + TICK_S, now_s - TICK_S)  # never burst to catch up
                arm_motors.once(now_s)
                if arm_motors.error is not None and not is_reported:
                    self._events.send(("error", arm_motors.error))
                    is_reported = True
        except Exception as exc:
            # The thread boundary (checklist rule 4): the agent's next tick raises it, and its teardown stops the run.
            _log.exception("arm '%s': System 0 tick failed", self._name)
            self._events.send(("error", f"System 0 on the {self._name} arm failed: {type(exc).__name__}: {exc}"))


def _serve(
    cfg: config.ArmConfig,
    name: str,
    requests: multiprocessing.connection.Connection,
    events: multiprocessing.connection.Connection,
) -> None:
    """The arm's process: opens the arm and the tracker, says so, and serves until closed or orphaned.

    SIGINT is ignored, since `dreamcli robot down` signals the agent's whole process group and the agent closes its
    arms itself; SIGTERM ends the process the safe way (`_Server.shut_down()`).

    @param cfg The arm's config.
    @param name `"left"` or `"right"`.
    @param requests The arm's end of the request pipe.
    @param events The arm's end of the events pipe.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, _exit_on_term)
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s %(levelname)s [arm {name}] %(name)s: %(message)s")
    sys.setswitchinterval(motors.SWITCH_INTERVAL_S)
    try:
        arm = yam.open_arm(cfg, name, fast_motor_loop=True)
    except BaseException as exc:
        requests.send(("err", _portable(exc)))
        return
    try:
        server = _Server(name, arm, motors.build_arm(cfg), events)
    except BaseException as exc:
        arm.close()
        requests.send(("err", _portable(exc)))
        return
    requests.send(("ok", None))
    _log.info("arm '%s': open in process, System 0 ready", name)
    try:
        server.serve(requests)
    finally:
        server.shut_down()


def _exit_on_term(signum: int, frame: Any) -> None:
    """SIGTERM in an arm's process: unwinds through `_Server.shut_down()`, which holds the arm and closes it."""
    raise SystemExit(128 + signum)


def _portable(exc: BaseException) -> BaseException:
    """The exception itself when it pickles, else an ArmError carrying its type and message.

    @param exc What a request raised.
    @return Something the agent's process can unpickle and raise.
    """
    try:
        pickle.loads(pickle.dumps(exc))
    except Exception:
        return errors.ArmError(f"{type(exc).__name__}: {exc}")
    return exc

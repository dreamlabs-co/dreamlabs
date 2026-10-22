"""Station agent: a resident process that owns the arms and runs the loop on request.

`robots.run` is one shot: build everything, run until Ctrl-C, exit. On a station the operator
starts and stops runs many times a day, changes the prompt between them, and homes the arms
before each, so a station keeps a process up with Start / Stop / Home buttons. This
module is that process for `robots`: #StationController opens the two followers once and keeps
them for its lifetime, opens the cameras once at boot and keeps them publishing between runs (so
the live view shows the scene while the arms rest, and a prompt can be chosen with
the scene in view), and for every run builds a fresh dream client and a fresh
#robots.control.ControlLoop over those readers; a reader whose thread died is rebuilt at the
next start, and an arm whose driver died is reopened by reset(), which then homes both. `robots.station_api`
exposes the controller over HTTP and hands its calls to the controller on the event
loop, so every state transition happens on one thread. With a `live` section in the config, the station shows
itself on the platform's Live View through `dream.LiveViewClient`, the client any control loop uses (_feed_live()
streams it, _live_press() runs the page's buttons); every run is recorded by its own dream client.
Where DreamAPI and the platform are, and the key for both, are the credentials file (`dream.credentials`),
read once at start; the config holds the hardware and the loop.

```bash
python -m robots.station --config station12.yaml [--api-port 8091] [--task "..."] [--no-sys0] [--eef|--no-eef]
    [--ik-solve tick|traj]
```

The safety rules the controller enforces:

- Exactly one run at a time. A start while a run exists, or a home while a loop owns the arms,
  is refused with #robots.errors.StationError and nothing happens.
- Stop cancels the loop task and waits for its `__aexit__`, which stops the command thread and
  leaves both arms holding their last command; only then does the controller report idle.
- Home runs only when no loop exists, so the arms are never written from two places at once.
- Reset (let both arms go, reopen both drivers, then home) runs only when no loop exists, for the same reason. A
  brakeless arm that still answers parks and switches off before its reopen, so no arm is reopened under power
  (`FollowerArm.close_for_reopen()`).
- A home move or a reset, once started, runs to its end whatever happens to the caller that asked for it: the
  Live View link dropping or an API call timing out cancels the wait, never the move, so the state always
  lands in `idle` or `error` and never sticks in `homing` or `resetting`.
- A run DreamAPI ends by refusing the key as `disabled` (the account is out of credit or over its spending limit, or
  the key was revoked) is followed by the same home move unless the station file sets `home_on_key_refused: false`,
  only once the loop's `__aexit__` has left the arms holding, only when no stop was asked meanwhile, and only when
  both drivers answer; it ends in `error` with `error_code` #robots.errors.KEY_REFUSED. Every other failure leaves
  the arms holding.

@see StationController, serve(), main()
"""

import argparse
import asyncio
import datetime
import logging
import os
import pathlib
import signal
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import dream
from dream import credentials
from shared.contracts import envelope

from robots import (
    arms_lock,
    bodies,
    cameras,
    config,
    control,
    errors,
    followers,
    home,
    kinematics,
    rollouts,
    run,
    startup_failure,
    types,
)
from robots.sys0 import arm_process
from robots.sys0 import motors as sys0_motors

## Controller states, as `status()["state"]` reports them.
## Calls to the Live View client per second: twice the cameras' rate, so a new frame is shown within half a frame.
LIVE_FEED_HZ = 60

IDLE = "idle"
STARTING = "starting"
RUNNING = "running"
STOPPING = "stopping"
HOMING = "homing"
RESETTING = "resetting"
ERROR = "error"

## The inference log the station CLI writes when the config names none: one file per station, appended across runs.
DEFAULT_INFERENCE_LOG = "~/robots-inference.jsonl"

## Default port of the HTTP control API (`robots.station_api`).
DEFAULT_API_PORT = 8091

DEFAULT_HOST = "127.0.0.1"

SWITCHED_OFF_LINE = "station: exited with the arms switched off"

## Seconds between two looks at the camera readers while the station is idle (serve()'s watchdog).
CAMERA_WATCH_S = 2.0

ARM_WATCH_S = 1.0

## DreamAPI's `DA-1000` reason for a key the platform's export marks disabled: the account is out of credit or over
## its spending limit, or the key was revoked. The refusal after which the station brings the arms home unless the
## station file sets `home_on_key_refused: false` (refused_key()).
KEY_REFUSED_REASON = envelope.UnauthorizedReason.DISABLED.value

_log = logging.getLogger(__name__)


def warm_client_ik(dream_config: dict[str, Any]) -> None:
    """Compiles the dream client's IK once, while the arms are still closed.

    Every Start builds a fresh client, and its `ClientIK` compiles dream's solver on first use: about 9 s on a
    station's CPU that starve the arm thread, so a Metal arm's grippers miss their 500 ms CAN timeout (fault 0xD) and
    the run stops after 0 chunks. The compile is cached for the life of the process (a second `ClientIK` takes about
    0.01 s), so building one here, before the arms open, makes each Start's cheap. Any failure only logs: a Start then
    compiles as before.

    @param dream_config The station's config from `robots.run.policy_config`; without `ik` or `checkpoint` nothing
        is done.
    """
    ik, checkpoint = dream_config.get("ik"), dream_config.get("checkpoint")
    if ik is None or checkpoint is None:
        return
    from dream import client_ik

    started_s = time.monotonic()
    try:
        model = next((m for m in dream.DreamClient().get_models() if m.id == checkpoint), None)
        if model is None:
            _log.warning("IK warm-up skipped: %s is not among the account's models; a Start compiles it", checkpoint)
            return
        client_ik.ClientIK(dict(ik), dream_config["action_hz"], dream_config.get("embodiment"), model)
    except Exception as exc:
        _log.warning("IK warm-up skipped (%s); a Start compiles it", exc)
        return
    _log.info("IK warm-up for %s done in %.1f s, before the arms open", checkpoint, time.monotonic() - started_s)


class ArmsNotOpenedError(errors.ArmError):
    """An arm did not open, and the arm that did could not be switched off after it: it may still hold under power."""

    def __init__(self, cause: Exception, powered: dict[str, followers.Arm]) -> None:
        """Builds the error.

        @param cause Why an arm did not open.
        @param powered The opened arms whose switch-off failed, by name.
        """
        names = " and ".join(powered)
        super().__init__(
            f"{cause}. The {names} arm opened but could not be switched off after it, so it may still hold under "
            "power: support it, then switch its power off"
        )
        self.powered = powered


class StationController:
    """Owns the arms for the life of the process and runs one control loop at a time.

    Every method that changes state is a coroutine meant to run on one event loop; the HTTP
    API and the Live View callback call them through `asyncio.run_coroutine_threadsafe`. status() is a plain
    method that only reads attributes, so any thread may call it.

    Attributes:
    - state: One of #IDLE, #STARTING, #RUNNING, #STOPPING, #HOMING, #RESETTING, #ERROR.
    - prompt: The task prompt the next run starts with, or the running one's.
    - checkpoint: The checkpoint the next run's client is built with, or the running one's: the config's until a
      start() names another; None when the config names none (dream then refuses the build).
    - action_space: What the next run executes, or the running one: the config's `action_space` until a start()
      names another (`robots.config.ACTION_SPACES`); None on a body without the dream client's IK (a YAM).
    - steps: How many steps the next run takes, or the running one, as the last start() named it; 0 until one does,
      a run that goes on until it is stopped. A run that takes its steps ends on its own and the arms go home.
    - run_number: Runs started so far; the current run's ordinal while one exists.
    - loop: The current run's #robots.control.ControlLoop, kept after it ends until the next start (its
      `video`, `timing` and counters stay readable); None before the first run.
    - last_error: `{"type", "message", "code", "reason"}` of the last run's or home's failure, or None.
    - live: The Live View client's counters (`dream.LiveViewClient.stats()`), rebound by the station's feed and
      reported by status() under `live`; None without a `live` section.
    - cameras: The readers open_cameras() opened, publishing for the life of the process; empty before it.
    """

    def __init__(
        self,
        arms: dict[str, followers.Arm],
        *,
        task: str,
        make_cameras: Callable[[], list[cameras.CameraReader]],
        make_client: Callable[[str | None, str | None], Any],
        loop_settings: dict[str, Any],
        inference_log_path: str | None,
        checkpoint: str | None = None,
        action_space: str | None = None,
        make_arms: Callable[[], dict[str, followers.Arm]] | None = None,
        make_recorder: Callable[[str, int], rollouts.Recorder | None] | None = None,
        kinematics: dict[str, kinematics.Kinematics] | None = None,
        sys0: sys0_motors.Models | None = None,
        home_duration_s: float = home.HOME_DURATION_S,
        clock: Callable[[], float] = time.monotonic,
        home_on_key_refused: bool = True,
    ) -> None:
        """Binds the arms and the per-run factories; nothing runs until start().

        @param arms The two followers keyed `"left"` and `"right"`, opened by the caller and kept for the process.
        @param task The initial prompt.
        @param make_cameras Opens the camera readers; called by open_cameras() at boot and again at a start() that
            finds a reader dead (a reader's `stop()` releases its device, so a dead set is rebuilt whole).
        @param make_client Builds a fresh dream client for one run, its policy open on the checkpoint and the action
            space given (None: the config's); called on a worker thread per start().
        @param checkpoint The config's checkpoint, what status() reports until a start() names another.
        @param action_space The config's action space, what status() reports until a start() names another; None on
            a body without the dream client's IK, which start() then refuses one for.
        @param make_arms Reopens both followers, called by reset() on a worker thread after the old drivers are
            closed; None (tests without a factory) makes reset() refuse.
        @param make_recorder Builds the recorder that saves one run as an episode on disk, given the run's
            checkpoint and ordinal; returns None unless the config turns rollouts on. None (tests) saves nothing.
        @param loop_settings The remaining `ControlLoop` keyword arguments (`run.loop_settings(cfg)` in
            production; the test defaults plus a manual clock in tests).
        @param inference_log_path The JSON Lines file every run appends to, or None.
        @param kinematics The arms' forward kinematics keyed `"left"` and `"right"`, built once and handed to
            every run's loop for its end-effector speed clamp; None (tests) leaves the joints unclamped.
        @param sys0 System 0's models, handed to every run's loop: `robots.sys0.motors.IN_ARM_PROCESSES` when the
            arms are served from their own processes (`robots.sys0.arm_process`), or None when the agent runs
            without it.
        @param home_duration_s Seconds a home move takes.
        @param clock Monotonic clock, seconds, for `uptime_s` and `run_elapsed_s`.
        @param home_on_key_refused Whether a run DreamAPI ends by refusing the key as `disabled` is followed by a
            home move (the station file's `home_on_key_refused`); False holds the arms as for any other failure.
        """
        self.state = IDLE
        self.prompt = task
        self.checkpoint = checkpoint
        self.action_space = action_space
        self.steps = 0
        self.run_number = 0
        self.loop: control.ControlLoop | None = None
        self.last_error: dict[str, Any] | None = None
        self.live: dict[str, Any] | None = None
        self.cameras: list[cameras.CameraReader] = []
        self._arms = arms
        self._make_arms = make_arms
        self._make_cameras = make_cameras
        self._make_client = make_client
        self._make_recorder = make_recorder
        self._loop_settings = loop_settings
        self._inference_log_path = inference_log_path
        self._kinematics = kinematics
        self._sys0 = sys0
        self._home_duration_s = home_duration_s
        self._clock = clock
        self._home_on_key_refused = home_on_key_refused
        self._created_s = clock()
        self._run_started_s: float | None = None
        self._started_at: str | None = None
        self._run_task: asyncio.Task[None] | None = None
        # The last run's client and loop build, on a worker thread a stop cannot interrupt: close() waits it out.
        self._build: asyncio.Future[control.ControlLoop] | None = None
        self._home_task: asyncio.Task[None] | None = None
        # Held while the idle watchdog reopens the cameras, so a start that lands meanwhile waits for them.
        self._camera_lock = asyncio.Lock()
        self._reset_task: asyncio.Task[None] | None = None
        self._park_task: asyncio.Task[None] | None = None

    @classmethod
    def from_config(
        cls,
        cfg: config.AppConfig,
        *,
        inference_log_path: str | None,
        sys0: bool = False,
        skip_tcp_calibration: bool = False,
    ) -> "StationController":
        """Opens both arms from the config and wires the per-run factories from it.

        @param cfg The station config.
        @param inference_log_path The inference log every run appends to, or None.
        @param sys0 Drive every run's motors with System 0 (`robots.sys0`); each arm then opens in a process of
            its own with the fast motor loop and its tracker (`robots.sys0.arm_process`), which is why it is decided
            here, for the life of the process.
        @param skip_tcp_calibration Run an arm file whose tool point was never marked (`robots.run.SKIP_TCP_FLAG`); its
            check, `robots.run.check_tool_point()`, runs here and again at every Start, with the Start's action space.
        @return The controller, idle.
        @throws ConfigError The camera names are not serve roles, the credentials name a malformed gateway, i2rt
            or mujoco is not installed (a Makers or Metal arm: python-can or dream's IK), System 0 is asked of a
            body other than the YAM (its tracker and fast motor loop are the i2rt YAM's), an arm file's tool point
            was never marked (`robots.run.check_tool_point()`), or the config's checkpoint
            is a fine-tune of another embodiment (`robots.bodies.check_model`; a run's own checkpoint is checked
            when its client is built).
        """
        body = bodies.for_config(cfg)
        if sys0 and body.name != bodies.YAM.name:
            raise errors.ConfigError(
                f"System 0 drives the i2rt YAM only; this station's body is '{body.name}': run 'dreamcli robot up' "
                "without 'sys0', which runs the plain loop on this body"
            )
        run.check_tool_point(cfg, skip=skip_tcp_calibration)
        dream_config = run.policy_config(cfg, sys0=sys0)
        arm_kinematics = run.make_kinematics(cfg)
        open_arm = body.open_arm
        if body.ik is not None:
            # Before make_arms() powers the grippers: a Start's client then finds its IK compiled (warm_client_ik).
            warm_client_ik(dream_config)

        def make_arms() -> dict[str, followers.Arm]:
            if sys0:
                return dict(arm_process.open_arms({"left": cfg.follower_l, "right": cfg.follower_r}))
            arms: dict[str, followers.Arm] = {}
            try:
                for name, arm_cfg in (("left", cfg.follower_l), ("right", cfg.follower_r)):
                    arms[name] = open_arm(arm_cfg, name)
            except BaseException as exc:
                # The one that opened is switched off: closed, it would hold under power with nothing driving it.
                powered = {
                    name: arm
                    for name, arm in arms.items()
                    if not control.run_teardown(
                        f"switch {name} arm off after the other failed to open", arm.torque_off
                    )
                }
                if powered and isinstance(exc, Exception):
                    raise ArmsNotOpenedError(exc, powered) from exc
                raise
            return arms

        def make_client(checkpoint: str | None, action_space: str | None) -> dream.DreamClient:
            # Built again per run, so a model and an action space a Start names (the Live view's picks) are checked
            # as the file's were.
            started = run.with_action_space(cfg, action_space)
            run.check_tool_point(started, skip=skip_tcp_calibration)
            return run.open_client(run.policy_config(started, checkpoint, sys0=sys0))

        return cls(
            make_arms(),
            task=cfg.task,
            make_cameras=lambda: run.make_cameras(cfg),
            make_client=make_client,
            loop_settings=run.loop_settings(cfg),
            inference_log_path=inference_log_path,
            checkpoint=dream_config.get("checkpoint"),
            action_space=cfg.action_space if body.ik is not None else None,
            make_arms=make_arms,
            make_recorder=lambda checkpoint, run_number: run.make_recorder(cfg, checkpoint, run_number),
            kinematics=arm_kinematics,
            sys0=sys0_motors.IN_ARM_PROCESSES if sys0 else None,
            home_on_key_refused=cfg.home_on_key_refused,
        )

    def open_cameras(self) -> None:
        """Opens and starts the camera readers for the life of the process; blocks while the devices open.

        Called once at boot (`serve()`, the tests' station fixture) and by a start() whose readers
        include a dead one, in which case the old set is stopped first, since a stopped reader
        cannot be restarted.

        @throws CameraError A camera cannot be opened.
        """
        for reader in self.cameras:
            control.run_teardown(f"stop camera '{reader.name}'", reader.stop)
        self.cameras = self._make_cameras()
        for reader in self.cameras:
            reader.start()
        _log.info("station: cameras %s open and publishing", [reader.name for reader in self.cameras])

    async def reopen_dead_cameras(self) -> bool:
        """Reopens the cameras when one of them died while nothing runs; a no-op otherwise.

        A reader whose thread stopped keeps its last frame forever, so between runs the live view would show
        a frozen scene with nothing to say so; a start reopens a dead set on its own, but
        an idle station has no start coming. serve() calls this every #CAMERA_WATCH_S. The reopen runs on a
        worker thread under the camera lock, which start() takes before it accepts, so the two never open the
        devices at once.

        @return True when the cameras were reopened; False when nothing was dead, a run exists, or the reopen
            failed (logged; the next look tries again).
        """
        if self._run_task is not None or self.state not in (IDLE, ERROR):
            return False
        dead = [reader.name for reader in self.cameras if not reader.is_alive()]
        if not dead:
            return False
        _log.warning("station: camera %s died while idle; reopening the cameras", dead)
        async with self._camera_lock:
            try:
                await asyncio.to_thread(self.open_cameras)
            except errors.CameraError as exc:
                _log.error("station: reopening the cameras failed: %s", exc)
                return False
        return True

    def close_cameras(self) -> None:
        """Stops every reader; the devices are released. close() calls it after the run has ended."""
        for reader in self.cameras:
            control.run_teardown(f"stop camera '{reader.name}'", reader.stop)
        self.cameras = []

    @property
    def frames(self) -> dict[str, types.Frame]:
        """The latest frame per camera, running or idle; a camera that has published nothing yet is absent."""
        return {reader.name: reader.frame for reader in self.cameras if reader.frame is not None}

    async def start(
        self,
        prompt: str | None = None,
        checkpoint: str | None = None,
        action_space: str | None = None,
        steps: int | None = None,
    ) -> dict[str, Any]:
        """Starts a run; returns as soon as it is accepted, the build and the loop go on in a task.

        Whether the model may run the action space is the build's to decide (`robots.bodies.check_model`): a refusal
        fails the run with its words, before an arm moves.

        @param prompt A prompt for this run; None keeps the current one.
        @param checkpoint The checkpoint for this run and the ones after it; None keeps the current one.
        @param action_space What this run and the ones after it execute (`robots.config.ACTION_SPACES`); None keeps
            the current one.
        @param steps How many steps this run and the ones after it take, at least 1; None keeps the current count.
        @return status() right after the transition to #STARTING.
        @throws StationError A run exists, a home move or a reset is in progress, an arm's driver is dead (reset()
            reopens it), an action space is named on a body without the dream client's IK (a YAM), or there is no
            prompt: none given and none stored (a config without `task` and nothing typed yet), or no checkpoint:
            none given and the body has no default (a Makers or Metal arm).
        @throws ValueError `prompt` or `checkpoint` is not a non-empty string, `action_space` not one of
            `robots.config.ACTION_SPACES`, or `steps` not a whole number of at least 1.
        """
        async with self._camera_lock:
            pass
        if self._run_task is not None or self._reset_task is not None or self.state not in (IDLE, ERROR):
            raise errors.StationError(f"start refused: station is {self.state}")
        dead = self._dead_arms()
        if dead:
            raise errors.StationError(f"start refused: arm {dead[0]!r} driver is dead; reset the arms first")
        if action_space is not None and self.action_space is None:
            raise errors.StationError("start refused: this robot runs its model's joints and takes no action space")
        if action_space is not None and action_space not in config.ACTION_SPACES:
            raise ValueError(f"action_space must be one of {list(config.ACTION_SPACES)}, got {action_space!r}")
        if steps is not None and (isinstance(steps, bool) or not isinstance(steps, int) or steps < 1):
            raise ValueError(f"steps must be a whole number of at least 1, got {steps!r}")
        if prompt is not None:
            self.prompt = _checked_prompt(prompt)
        if checkpoint is not None:
            self.checkpoint = _checked_text(checkpoint, "checkpoint")
        if action_space is not None:
            self.action_space = action_space
        if steps is not None:
            self.steps = steps
        if not self.prompt:
            # A config written by `dreamcli setup` carries no task: the operator sends it with the Start.
            raise errors.StationError("start refused: no task prompt; send the task with the start")
        if self.checkpoint is None:
            raise errors.StationError(
                "start refused: no checkpoint; this body has no default model: send it with the start (the live "
                "view's Start does), or set 'checkpoint' in the station file"
            )
        self.last_error = None
        self.run_number += 1
        self._run_started_s = self._clock()
        self._started_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
        # Managed exposure moves only between runs: held from the request, before frame 0, so no brightness
        # change is written into an episode; released when the run ends, whichever way it ends.
        self._hold_exposure(True)
        self._set_state(STARTING)
        self._run_task = asyncio.create_task(self._run(), name=f"station-run-{self.run_number}")
        return self.status()

    def _hold_exposure(self, hold: bool) -> None:
        """Freezes or releases every reader's managed exposure (a no-op on readers without one)."""
        for reader in self.cameras:
            set_hold = getattr(reader, "set_exposure_hold", None)
            if set_hold is not None:
                set_hold(hold)

    async def stop(self) -> dict[str, Any]:
        """Cancels the run and waits until its loop has left the arms holding; a no-op with nothing running.

        @return status() once the run has ended.
        @throws StationError A home move is in progress (it cannot be interrupted; it ends within seconds), or a
            stop is already in progress.
        """
        if self.state == HOMING:
            raise errors.StationError("stop refused: a home move is in progress; it ends within seconds")
        if self.state == RESETTING:
            raise errors.StationError("stop refused: a reset is in progress; it ends within seconds")
        if self.state == STOPPING:
            raise errors.StationError("stop refused: already stopping")
        task = self._run_task
        if task is None:
            _log.info("station: stop with no run in progress; nothing to do")
            return self.status()
        self._set_state(STOPPING)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            # The run's own cancellation, delivered through the await; _run() has already reported it.
            pass
        return self.status()

    async def home(self) -> dict[str, Any]:
        """Moves both arms to the home pose (zeros, grippers open) off the event loop; refused while a loop exists.

        The move runs as its own task and this call only waits for it, shielded: a caller cancelled halfway (the
        Live View link dropping, an API call timing out) stops waiting, and the move still ends on its own
        and sets the state, so the station never sticks in #HOMING.

        @return status() once the move has ended; `state` is #IDLE when both arms made it, #ERROR otherwise.
        @throws StationError A run exists, a home move or a reset is already in progress, or an arm's driver is
            dead (reset() reopens it; a move would only fail on it).
        """
        if self._run_task is not None or self._reset_task is not None or self.state not in (IDLE, ERROR):
            raise errors.StationError(f"home refused: station is {self.state}; stop the run first")
        dead = self._dead_arms()
        if dead:
            raise errors.StationError(f"home refused: arm {dead[0]!r} driver is dead; reset the arms first")
        self.last_error = None
        self._home_task = asyncio.create_task(self._home_move(), name="station-home")
        await asyncio.shield(self._home_task)
        return self.status()

    async def _home_move(self, refused: dream.DreamAuthError | None = None) -> None:
        """The home move as one task: #HOMING, `move_home` off the event loop, then #IDLE or #ERROR.

        @param refused The refusal that ended the run this move follows (_home_after_key_refused()); the move then
            ends in #ERROR with `error_code` #robots.errors.KEY_REFUSED when both arms made it, never in #IDLE, so the
            page keeps saying why the robot stopped. None for a Home or a reset.
        """
        if self.state != HOMING:
            self._set_state(HOMING)
        try:
            missed = await asyncio.to_thread(home.move_home, self._arms, duration_s=self._home_duration_s)
        except Exception as exc:
            _log.exception("station: home move raised")
            self._record_error(exc)
            self._set_state(ERROR)
            return
        finally:
            self._home_task = None
        if missed:
            missed_words = "; ".join(f"arm '{name}' did not reach home: {why}" for name, why in sorted(missed.items()))
            after = "" if refused is None else f"the API key was refused ({refused}), and homing after it: "
            self.last_error = {
                "type": "HomeError",
                "message": f"{after}{missed_words}; press Reset to open it again",
                "code": None,
                "reason": None,
                "error_code": "HomeError",
            }
            self._set_state(ERROR)
        elif refused is None:
            self._set_state(IDLE)
        else:
            self.last_error = {
                "type": type(refused).__name__,
                "message": f"{refused}; the account is out of credit or over its spending limit, or the key was "
                "revoked, so the run stopped and the arms went home",
                "code": refused.code,
                "reason": refused.reason,
                "error_code": errors.KEY_REFUSED,
            }
            self._set_state(ERROR)

    async def reset(self) -> dict[str, Any]:
        """Lets both arms go, reopens their drivers and moves the arms home; returns as soon as it is accepted.

        The way back from an arm fault: i2rt's CAN thread stops when a motor stops answering (a bus glitch, a
        joint limit trip, a power dip), the run ends in #ERROR and `arms` reports the dead one, and nothing
        short of reopening the driver makes that arm answer again. Like start(), the work goes on in a task
        (a brakeless arm that answers parks and switches off first, _reopen_arms(); opening two CAN channels takes
        seconds, then the home move takes seconds more, longer than the live link waits for an ack): the state is
        #RESETTING while the arms are let go and reopened, #HOMING during the move, then #IDLE, or #ERROR with
        `last_error` saying what failed. A reopen that fails leaves the old, closed drivers in place, which read
        dead, so a start or a home is refused until a reset succeeds.

        @return status() right after the transition to #RESETTING.
        @throws StationError A run exists, a home move or a reset is in progress, or the controller was built
            without a `make_arms` factory.
        """
        if self._run_task is not None or self._reset_task is not None or self.state not in (IDLE, ERROR):
            raise errors.StationError(f"reset refused: station is {self.state}; stop the run first")
        if self._make_arms is None:
            raise errors.StationError("reset refused: this station cannot reopen its arms")
        self.last_error = None
        self._set_state(RESETTING)
        self._reset_task = asyncio.create_task(self._reset_arms(), name="station-reset")
        return self.status()

    async def _reset_arms(self) -> None:
        """The reset as one task: reopen the drivers off the event loop, then the home move; never cancelled."""
        held = sorted(name for name, arm in self._arms.items() if arm.holds_dead())
        try:
            self._arms = await asyncio.to_thread(self._reopen_arms)
        except Exception as exc:
            _log.exception("station: reopening the arms failed")
            if isinstance(exc, ArmsNotOpenedError):
                self._arms = {**self._arms, **exc.powered}
            self._record_error(exc)
            assert self.last_error is not None
            for name in held:
                self.last_error["message"] = (
                    f"{self.last_error['message'].rstrip('.')}. The {name} arm stopped answering and was let go where "
                    "it held; it may still hold under power so it does not drop: support it, then switch its power off"
                )
            self._reset_task = None
            self._set_state(ERROR)
            return
        self._reset_task = None
        await self._home_move()

    async def park_alarmed_arms(self) -> bool:
        """Parks and switches off every arm whose alarm is set while no run, home move or reset reads the arms.

        @return Whether an arm was parked.
        """
        busy = (self._run_task, self._home_task, self._reset_task, self._park_task)
        if any(task is not None for task in busy) or self.state not in (IDLE, ERROR):
            return False
        alarms = self._alarms()
        if not alarms:
            return False
        exc = errors.SafetyStop("; ".join(alarms.values()))
        _log.error("station: %s", exc)
        self._set_state(STOPPING)
        self._park_task = asyncio.create_task(self._park_move(alarms, exc), name="station-park")
        await asyncio.shield(self._park_task)
        return True

    async def _park_move(self, alarms: dict[str, str], exc: errors.SafetyStop) -> None:
        try:
            await asyncio.to_thread(self._park, alarms)
        finally:
            self._park_task = None
        self._record_error(exc)
        self._set_state(ERROR)

    def report_dead_arms(self) -> bool:
        """Puts an arm that stopped answering while the station idles in `last_error`, naming it and what to do.

        @return Whether an arm was reported.
        """
        busy = (self._run_task, self._home_task, self._reset_task, self._park_task)
        dead = self._dead_arms()
        if any(task is not None for task in busy) or self.state != IDLE or not dead:
            return False
        stopped = "; ".join(
            f"arm '{name}' stopped answering while the station was idle ({self._arms[name].why_stopped()})"
            for name in dead
        )
        exc = errors.ArmError(
            f"{stopped}. It may still hold under power with nothing driving it: support it, check its power and the "
            "cable to its adapter, then press Reset"
        )
        _log.error("station: %s", exc)
        self._record_error(exc)
        self._set_state(ERROR)
        return True

    def _alarms(self) -> dict[str, str]:
        alarms = {name: arm.alarm() for name, arm in self._arms.items() if arm.is_alive()}
        return {name: alarm for name, alarm in alarms.items() if alarm is not None}

    def _park(self, alarms: dict[str, str]) -> None:
        for name in alarms:
            _log.warning("station: parking the %s arm and switching it off; reset the arms to use it again", name)
            control.run_teardown(f"park {name} arm", self._arms[name].torque_off)

    def _dead_arms(self) -> list[str]:
        """The arms whose driver no longer answers, by name."""
        return [name for name, arm in self._arms.items() if not arm.is_alive()]

    def _reopen_arms(self) -> dict[str, followers.Arm]:
        """Lets every arm go and builds fresh ones.

        @return The new arms, keyed like the old.
        @throws Exception Whatever the driver raises on open (i2rt's own errors, `ConfigError` without i2rt).
        """
        assert self._make_arms is not None
        for name, arm in self._arms.items():
            control.run_teardown(f"let {name} arm go for the reopen", arm.close_for_reopen)
        arms = self._make_arms()
        _log.info("station: arms %s reopened", sorted(arms))
        return arms

    async def set_prompt(self, prompt: str) -> dict[str, Any]:
        """Stores the prompt; a running loop is told to start a fresh episode with it at its next safe point.

        @param prompt The new task prompt.
        @return status().
        @throws ValueError `prompt` is not a non-empty string.
        """
        self.prompt = _checked_prompt(prompt)
        if self._run_task is not None and self.loop is not None:
            self.loop.set_task(self.prompt)
            _log.info("station: prompt for the running loop set to %r", self.prompt)
        else:
            _log.info("station: prompt for the next run set to %r", self.prompt)
        return self.status()

    def arm_states(self) -> dict[str, types.ArmState]:
        """Each arm's joint state as its driver last reported it, by arm name; safe to call from any thread.

        One cached read per arm (i2rt keeps the newest observation under its own lock, so this never touches the
        bus); an arm whose driver is dead or closed is left out rather than raising, since the caller (the live
        agent's pose stream) wants whatever is readable.

        @return `{"left": ArmState, "right": ArmState}` minus the arms that cannot be read.
        """
        states: dict[str, types.ArmState] = {}
        for name, arm in self._arms.items():
            try:
                states[name] = arm.read()
            except errors.ArmError:
                continue
        return states

    def arms_alive(self) -> dict[str, bool]:
        """Whether each arm's driver still answers, by arm name; safe to call from any thread (it reads one flag).

        @return `{"left": True, "right": False}` and the like; an arm reads False once its motor chain stopped.
        """
        return {name: arm.is_alive() for name, arm in self._arms.items()}

    def status(self) -> dict[str, Any]:
        """A snapshot of the station for the API and the live view; safe to call from any thread.

        @return `state`, `prompt`, `checkpoint`, `action_space`, `steps`, `run`, `chunks`, `stalls`, `clamps`,
            `max_ee_speed_requested_m_s` (the largest end-effector speed the policy asked for this run, before the
            clamp), `max_ee_speed_m_s`
            (the configured cap), `arms` (whether each arm's driver still answers, by name), `cameras` (the age in
            seconds of each camera's latest frame, None before its first; the readers are open for the life of the
            process), `session_id`, `last_error`, `timing` (the loop's
            #robots.control.RefillTiming as a dict), `started_at` (ISO 8601 UTC of the last start, or None),
            `uptime_s` (seconds since the controller was created), `run_elapsed_s` (seconds since the last
            start, or None) and, with a `live` section, `live` (`dream.LiveViewClient.stats()`).
        """
        loop = self.loop
        timing = loop.timing.as_dict() if loop is not None else control.RefillTiming().as_dict()
        status = {
            "state": self.state,
            "prompt": self.prompt,
            "checkpoint": self.checkpoint,
            "action_space": self.action_space,
            "steps": self.steps,
            "run": self.run_number,
            "chunks": loop.chunk_count if loop is not None else 0,
            "stalls": loop.stall_count if loop is not None else 0,
            "clamps": loop.clamp_count if loop is not None else 0,
            "max_ee_speed_requested_m_s": loop.max_ee_speed_requested_m_s if loop is not None else 0.0,
            "max_ee_speed_m_s": self._loop_settings.get("max_ee_speed_m_s"),
            "arms": self.arms_alive(),
            "cameras": {
                reader.name: None if reader.frame is None else round(self._clock() - reader.frame.timestamp_s, 3)
                for reader in self.cameras
            },
            "exposure": {
                reader.name: exposure
                for reader in self.cameras
                if (exposure := getattr(reader, "exposure", None)) is not None
            },
            "capture_fps": {reader.name: getattr(reader, "capture_fps", None) for reader in self.cameras},
            "session_id": _session_id(loop),
            "last_error": self.last_error,
            "timing": timing,
            "started_at": self._started_at,
            "uptime_s": round(self._clock() - self._created_s, 1),
            "run_elapsed_s": None if self._run_started_s is None else round(self._clock() - self._run_started_s, 1),
        }
        if self.live is not None:
            status["live"] = self.live
        return status

    async def close(self) -> bool:
        """Ends whatever is in progress for process exit, then releases the cameras and switches the arms off.

        A run is stopped (the arms hold), or a reset or a home move is waited out, first; a run stopped while its
        client was still building (the client's IK compiles for seconds) has its build waited out too, so the build
        neither reopens the cameras this close stops nor leaves its client open (`_discard_built`); the readers are
        stopped next, so the live view shows the scene until the process is about to exit; the arms last.
        Closing a driver alone would leave the arms stiff: it stops its threads and its CAN channel and sends
        nothing, and a motor keeps the last MIT target it was given, so close_arms() also switches every motor
        off (`FollowerArm.torque_off()`) and the arms go limp. Without the close the driver threads, which are
        not daemons, keep the process alive after `serve()` returns and `dreamcli robot down` has to escalate
        to SIGTERM.

        @return close_arms()'s answer.
        """
        if self._reset_task is not None:
            await asyncio.shield(self._reset_task)
        if self._home_task is not None:
            await asyncio.shield(self._home_task)
        if self._park_task is not None:
            await asyncio.shield(self._park_task)
        if self._run_task is not None and self.state != STOPPING:
            await self.stop()
        elif self._run_task is not None:
            # A stop in progress ends its run cancelled, which is not this close's to raise.
            await asyncio.wait([self._run_task])
        if self._build is not None and not self._build.done():
            # A stopped run's build is still on its thread until it ends. Its `_discard_built` callback was added
            # when the run was cancelled, before this wait's own, so the client is closed once the wait returns.
            await asyncio.wait([self._build])
        self.close_cameras()
        return self.close_arms()

    def close_arms(self) -> bool:
        """Switches every arm off for process exit: its driver closed, then its motors disabled, so it goes limp.

        One arm's failure is logged and the other is still switched off; the arm's own log line says which
        motors may still be energized.

        @return True when every arm's switch-off finished.
        """
        # A list, not a generator: all() would stop at the first failure and leave the next arm on.
        return all(
            [control.run_teardown(f"switch {name} arm off", arm.torque_off) for name, arm in self._arms.items()]
        )

    async def _run(self) -> None:
        """The run task: build the client and the loop over the station's cameras, then tick until stopped or failed.

        A stop that arrives while the build is on its worker thread lets the build finish and
        then discards what it made, so no client is leaked (the cameras are the station's and stay
        up; close() waits such a build out before it stops them); a stop during the loop is the loop's ordinary
        cancellation, whose `__aexit__` leaves the arms holding.
        """
        try:
            await self._run_held()
        finally:
            self._hold_exposure(False)

    async def _run_held(self) -> None:
        """_run()'s body, with the exposure hold released around it whichever way it ends."""
        build = asyncio.ensure_future(asyncio.to_thread(self._build_loop))
        self._build = build
        try:
            loop = await asyncio.shield(build)
        except asyncio.CancelledError:
            build.add_done_callback(_discard_built)
            self._run_task = None
            self._set_state(IDLE)
            raise
        except Exception as exc:
            self._run_task = None
            self._fail(exc)
            return
        self.loop = loop
        entered = False
        try:
            async with loop:
                entered = True
                self._set_state(RUNNING)
                await loop.run()
        except asyncio.CancelledError:
            if not entered:
                # __aenter__ released its files on its way out; the client is ours to close.
                loop.discard()
            self._run_task = None
            self._set_state(IDLE)
            raise
        except Exception as exc:
            if not entered:
                loop.discard()
            if isinstance(exc, errors.SafetyStop):
                self._set_state(STOPPING)
                await asyncio.to_thread(self._park, self._alarms())
            self._run_task = None
            refused = refused_key(exc) if entered and self._home_on_key_refused else None
            if refused is not None and self.state == RUNNING:
                # RUNNING, not STOPPING: a stop (or close()) asked during the loop's exit keeps the arms holding.
                self._home_after_key_refused(refused)
            else:
                self._fail(exc)
                if errors.error_code(exc) == envelope.UnauthorizedReason.OVER_CAP:
                    await self._home_over_cap()
        else:
            # run() returns only once the run has taken its steps: it is over, and the arms go home as after a Stop.
            self._run_task = None
            self._set_state(IDLE)
            await self._home_after_steps()

    async def _home_after_steps(self) -> None:
        """Moves the arms home after a run that took its steps; a refused move leaves the arms holding, logged."""
        try:
            await self.home()
        except errors.StationError as exc:
            _log.warning("station: run %d took its steps but the arms cannot go home: %s", self.run_number, exc)

    def _build_loop(self) -> control.ControlLoop:
        """Builds the client and the loop for one run over the station's readers; on a worker thread, since a dead
        reader set is reopened here and opening cameras blocks.

        @return The loop, not yet entered.
        @throws CameraError A dead camera set cannot be reopened (the client built before it is closed).
        @throws DreamError The client cannot be built.
        """
        assert self.checkpoint is not None  # start() refuses a run without one
        client = self._make_client(self.checkpoint, self.action_space)
        try:
            if not self.cameras or any(not reader.is_alive() for reader in self.cameras):
                _log.warning("station: a camera reader is dead; reopening the cameras for run %d", self.run_number)
                self.open_cameras()
            return control.ControlLoop(
                followers=self._arms,
                cameras=self.cameras,
                client=client,
                task=self.prompt,
                kinematics=self._kinematics,
                sys0=self._sys0,
                inference_log_path=self._inference_log_path,
                run_number=self.run_number,
                recorder=self._make_recorder(self.checkpoint, self.run_number) if self._make_recorder else None,
                max_steps=self.steps or None,
                camera_owner=False,
                **self._loop_settings,
            )
        except BaseException:
            control.run_teardown("close dream client after a failed build", client.close)
            raise

    async def _home_over_cap(self) -> None:
        """Moves the arms home after DreamAPI refused the run `over_cap` (the owner's monthly budget cap), whether
        or not anyone is watching: no run can start again until the cap is raised, so the arms are not left holding
        the pose the run stopped in. The refusal stays in `last_error`, so the page and the log still say why.
        """
        refusal = self.last_error
        _log.warning("station: run %d stopped at the owner's budget cap; moving the arms home", self.run_number)
        try:
            await self.home()
        except errors.StationError as exc:
            _log.warning("station: could not home the arms after the budget cap stopped the run: %s", exc)
            return
        if self.state == IDLE:
            self.last_error = refusal

    def _home_after_key_refused(self, exc: dream.DreamAuthError) -> None:
        """Starts the home move after a run DreamAPI ended by refusing the key; the move goes on in its own task.

        Called once the loop's `__aexit__` has stopped the command thread and left the arms holding, so the move is
        the only writer, as a Home after a Stop is. The state is #HOMING from here, before the task first runs, so a
        start, a stop, a home or a reset that lands meanwhile is refused, and close() waits the move out like any
        home. Home is refused while an arm's driver is dead, so then neither arm moves: the run fails as any other,
        holding, and the message says so (reset() is the way back).

        @param exc The refusal that ended the run.
        """
        dead = self._dead_arms()
        if dead:
            _log.error("station: run %d's key was refused, but arm %r is dead; not homing", self.run_number, dead[0])
            self._record_error(exc)
            assert self.last_error is not None
            self.last_error["message"] = (
                f"{exc}; the arms were not homed and hold where the run stopped, because arm {dead[0]!r} stopped "
                "answering: support it, then press Reset"
            )
            self._set_state(ERROR)
            return
        _log.error("station: run %d ended: DreamAPI refused the key (%s); homing the arms", self.run_number, exc)
        self._set_state(HOMING)
        self._home_task = asyncio.create_task(self._home_move(refused=exc), name="station-home-key-refused")

    def _fail(self, exc: BaseException) -> None:
        """Records a run's failure and moves to #ERROR; the loop's `__aexit__` has already left the arms safe.

        @param exc What stopped the run.
        """
        _log.error("station: run %d failed with %s: %s", self.run_number, type(exc).__name__, exc)
        self._record_error(exc)
        self._set_state(ERROR)

    def _record_error(self, exc: BaseException) -> None:
        """Keeps the exception's type, message, DreamAPI envelope and `errors.error_code()` for status().

        @param exc The exception.
        """
        envelope = exc if isinstance(exc, dream.DreamServerError) else None
        self.last_error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "code": None if envelope is None else envelope.code,
            "reason": None if envelope is None else envelope.reason,
            "error_code": errors.error_code(exc),
        }

    def _set_state(self, state: str) -> None:
        """Moves to `state` and logs the transition at INFO.

        @param state The new state.
        """
        _log.info("station: %s -> %s (run %d, prompt %r)", self.state, state, self.run_number, self.prompt)
        self.state = state


def refused_key(exc: BaseException) -> dream.DreamAuthError | None:
    """The refusal when a run's failure is DreamAPI refusing the key as `disabled`, else None.

    That is `DA-1000 UNAUTHORIZED` for a key the platform's export marks disabled: an account out of credit or over its
    spending limit, or a key revoked. Another `DA-1000` reason (`missing`, `unknown_key`, `not_entitled`,
    `region_not_supported`) is a key or a model the station was set up with wrongly, met on the first request, and is
    not one.

    @param exc What ended the run.
    @return `exc` when it is a `dream.DreamAuthError` whose reason is #KEY_REFUSED_REASON; None for anything else.
    """
    return exc if isinstance(exc, dream.DreamAuthError) and exc.reason == KEY_REFUSED_REASON else None


def _discard_built(task: asyncio.Future[control.ControlLoop]) -> None:
    """Tears down a loop whose build finished after its run was cancelled: nobody will enter it.

    @param task The build future; its loop's client is closed when it succeeded (the cameras are the station's).
    """
    if task.cancelled() or task.exception() is not None:
        return
    loop = task.result()
    _log.info("station: run cancelled during the build; releasing what it opened")
    loop.discard()


def _session_id(loop: control.ControlLoop | None) -> str | None:
    """The DreamAPI session id of a loop's client, when it exposes one.

    @param loop The loop, or None.
    @return The id, or None.
    """
    return None if loop is None else loop.session_id


def _checked_prompt(prompt: Any) -> str:
    """Rejects anything but a non-empty string as a prompt.

    @param prompt The value handed in over the API or the live link.
    @return The prompt, stripped.
    @throws ValueError It is not a string, or is blank.
    """
    return _checked_text(prompt, "prompt")


def _checked_text(value: Any, what: str) -> str:
    """Rejects anything but a non-empty string.

    @param value The value handed in over the API or the live link.
    @param what Its name, for the message.
    @return The value, stripped.
    @throws ValueError It is not a string, or is blank.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} must be a non-empty string, got {value!r}")
    return value.strip()


async def serve(
    cfg: config.AppConfig,
    *,
    host: str = DEFAULT_HOST,
    api_port: int = DEFAULT_API_PORT,
    inference_log_path: str | None = None,
    arms_lock_path: pathlib.Path | None = None,
    sys0: bool = False,
    skip_tcp_calibration: bool = False,
    ready: threading.Event | None = None,
) -> None:
    """Runs the station until SIGINT, SIGTERM or SIGHUP: opens the arms, serves the API, then stops cleanly.

    Nothing moves at start: the operator starts a run through the API or the live view. On a signal a
    running loop is stopped, the arms are switched off (they go limp), the server is closed and the
    coroutine returns. A second SIGINT or SIGTERM while that runs only sets the same event again, so
    `dreamcli robot down`'s escalation cannot cut the switch-off short; its SIGKILL is the backstop.

    @param cfg The station config.
    @param host Bind address of the API.
    @param api_port Port of the HTTP API.
    @param inference_log_path The inference log; the config's when None, else #DEFAULT_INFERENCE_LOG.
    @param arms_lock_path The arms' lock file.
    @param sys0 Drive the arms with System 0 (`robots.sys0`); main() turns it on for a YAM unless `--no-sys0`
        (`dreamcli robot up nosys0`) asks otherwise.
    @param skip_tcp_calibration Run an arm file whose tool point was never marked (`robots.run.SKIP_TCP_FLAG`).
    @param ready Set once the arms, the cameras, the Live View client and the API are up, so main() can tell a failed
        start (`robots.startup_failure`) from a failure after it.
    @throws ConfigError The config is not what dream and the serve accept, i2rt is missing, or the config has a
        `live` section and the credentials have no key, the platform is `http://` off the box, or websockets is
        not installed.
    @throws CameraError A camera of the config cannot be opened.
    """
    account = credentials.load()
    if cfg.live is not None and not account.api_key:
        raise errors.ConfigError(
            f"the live section needs the api key, and neither the environment nor {account.path} sets "
            f"DREAM_API_KEY; the Setup API page on {account.platform_url} gives the lines that write that file"
        )

    event_loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        event_loop.add_signal_handler(signum, stop.set)
    log_path = inference_log_path or cfg.inference_log_path or os.path.expanduser(DEFAULT_INFERENCE_LOG)
    with arms_lock.holding(arms_lock_path, arms_lock.STATION):
        controller = await asyncio.to_thread(
            StationController.from_config,
            cfg,
            inference_log_path=log_path,
            sys0=sys0,
            skip_tcp_calibration=skip_tcp_calibration,
        )
        await _serve_opened(cfg, controller, stop, host, api_port, log_path, account, ready)


async def _serve_opened(
    cfg: config.AppConfig,
    controller: StationController,
    stop: asyncio.Event,
    host: str,
    api_port: int,
    log_path: str,
    account: credentials.Credentials,
    ready: threading.Event | None,
) -> None:
    from robots import station_api

    event_loop = asyncio.get_running_loop()
    api = None
    agent = None
    watches: list[asyncio.Task[None]] = []
    try:
        if stop.is_set():
            return
        await asyncio.to_thread(controller.open_cameras)
        if stop.is_set():
            return
        if cfg.live is not None:
            # The platform's Live View, through the same client any control loop uses; the station is one caller.
            agent = dream.LiveViewClient(
                cfg.live.robot_id,
                api_key=account.api_key,
                face_blur=cfg.face_blur,
                model_fixed=False,
                cameras=cfg.live.cameras,
            )
            agent.register_callback(_live_press(controller, agent, event_loop), home=True, reset=True, prompt=True)
            watches.append(asyncio.create_task(_feed_live(controller, agent), name="station-live-feed"))
        api = station_api.StationApi(controller, event_loop, host=host, port=api_port)
        api.start()
        watches.append(asyncio.create_task(_watch_cameras(controller), name="station-camera-watchdog"))
        watches.append(asyncio.create_task(_watch_arms(controller), name="station-arm-watchdog"))
        _log.info(
            "station ready: API on %s:%d, inference log %s, task %r; nothing moves until a start",
            host,
            api.port,
            log_path,
            controller.prompt,
        )
        if ready is not None:
            ready.set()
        await stop.wait()
    finally:
        _log.info("station: shutting down; stopping the run if one exists")
        for watch in watches:
            watch.cancel()
        switched_off = await controller.close()
        if agent is not None:
            await asyncio.to_thread(agent.close)
        if api is not None:
            api.stop()
        if switched_off:
            _log.info(SWITCHED_OFF_LINE)
        else:
            _log.error("station: exited, but an arm may still be powered (the lines above say which); cut its power")


async def _feed_live(controller: StationController, live: dream.LiveViewClient) -> None:
    """Streams the station to Live View #LIVE_FEED_HZ times a second.

    Each call carries the cameras' newest frames (the station's readers, open between runs), the state, the task, the
    run, the model and the step count, the arms' reads for the pose stream and their health, and why nothing moves.
    While a run is on the client reads the model's round trips and session from the run's own dream client.

    @param controller The station.
    @param live The station's Live View client.
    """
    period_s = 1.0 / LIVE_FEED_HZ
    while True:
        await asyncio.sleep(period_s)
        loop = controller.loop
        is_running = controller.state == RUNNING and loop is not None
        live.dream_client = loop.dream_client if is_running and loop is not None else None
        frames = {f"video.{name}": frame.data for name, frame in controller.frames.items()}
        try:
            live.stream(
                frames,
                controller.state,
                arms=_live_arms(controller, loop if is_running else None),
                error=_live_error(controller),
                prompt=controller.prompt,
                run=controller.run_number or None,
                model=(controller.checkpoint, controller.action_space),
                steps=controller.steps,
            )
        except dream.DreamObservationError as exc:
            _log.warning("live: a tick was not shown: %s", exc)
        controller.live = live.stats()


def _live_arms(controller: StationController, loop: Any) -> dict[str, Any]:
    """Each arm's joints for the page, `(q, gripper)`, from the running loop's last read or the drivers' cache; None
    for an arm whose driver no longer answers."""
    state = getattr(loop, "state", None)
    cached = controller.arm_states()
    arms: dict[str, Any] = {}
    for name, is_alive in controller.arms_alive().items():
        read = getattr(state, name, None) if state is not None else None
        if read is None or getattr(read, "positions_rad", None) is None:
            read = cached.get(name)
        arms[name] = (list(read.positions_rad), float(read.gripper)) if is_alive and read is not None else None
    return arms


def _live_error(controller: StationController) -> tuple[str, str | None] | None:
    """Why no actions reach the arms, `(message, code)`: in `error` what ended the run, while running the DreamAPI
    error the loop is still retrying; None otherwise."""
    if controller.state == ERROR and controller.last_error is not None:
        message, code = controller.last_error.get("message"), controller.last_error.get("error_code")
    elif controller.state == RUNNING and controller.loop is not None:
        message, code = controller.loop.refill_error, controller.loop.refill_error_code
    else:
        return None
    return (str(message), str(code) if code else None) if message else None


def _live_press(
    controller: StationController, live: dream.LiveViewClient, station_loop: asyncio.AbstractEventLoop
) -> Callable[[dream.EventChange], Any]:
    """The callback for the page's buttons: each runs the station's own coroutine on its loop.

    Start is `start()` with the task, model, actions and steps the viewer chose; Stop is `stop()` then `home()`, the
    page's Stop meaning "stop and go home"; Home is `home()`; Reset is `reset()`; a new task during a run is
    `set_prompt()`. A move that ended in `error` (an arm did not reach home) is a failure with the station's own words.

    @param controller The station.
    @param live The station's Live View client, whose getters say what the viewer chose.
    @param station_loop The loop the controller runs on.
    @return The callback, which returns the press's work as a future.
    """

    async def run(event: dream.EventChange) -> dict[str, Any]:
        try:
            if event is dream.EventChange.start:
                checkpoint, action_space = live.get_model()
                status = await controller.start(live.get_prompt(), checkpoint, action_space, live.get_steps())
            elif event is dream.EventChange.end:
                await controller.stop()
                status = await controller.home()
            elif event is dream.EventChange.home:
                status = await controller.home()
            elif event is dream.EventChange.reset:
                status = await controller.reset()
            else:
                status = await controller.set_prompt(live.get_prompt() or "")
        except (errors.StationError, ValueError) as exc:
            return {"state": controller.state, "error": str(exc)}
        state = str(status["state"])
        if event in (dream.EventChange.end, dream.EventChange.home) and state == ERROR:
            last_error = status.get("last_error") or {}
            return {"state": state, "error": str(last_error.get("message") or "an arm did not reach home")}
        return {"state": state}

    return lambda event: asyncio.run_coroutine_threadsafe(run(event), station_loop)


async def _watch_cameras(controller: StationController) -> None:
    """Looks at the readers every #CAMERA_WATCH_S and reopens a dead set while the station is idle."""
    while True:
        await asyncio.sleep(CAMERA_WATCH_S)
        await controller.reopen_dead_cameras()


async def _watch_arms(controller: StationController) -> None:
    while True:
        await asyncio.sleep(ARM_WATCH_S)
        await controller.park_alarmed_arms()
        controller.report_dead_arms()


def main(argv: list[str] | None = None) -> None:
    """The `python -m robots.station` entry point.

    @param argv Command-line arguments; `sys.argv[1:]` when None.
    """
    parser = argparse.ArgumentParser(description="Serve the station agent: Start/Stop/Home over HTTP.")
    parser.add_argument("--config", required=True, help="Path to the YAML application config.")
    parser.add_argument("--task", default=None, help="Override the config's task prompt.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind address of the API (default {DEFAULT_HOST}).")
    parser.add_argument(
        "--api-port", type=int, default=DEFAULT_API_PORT, help=f"HTTP API port (default {DEFAULT_API_PORT})."
    )
    parser.add_argument(
        "--inference-log",
        default=None,
        help=f"JSON Lines inference log; the config's inference_log_path, else {DEFAULT_INFERENCE_LOG}.",
    )
    parser.add_argument(
        "--arms-lock",
        default=None,
        help="The arms' lock file (robots.arms_lock); the station directory's, as dreamcli finds it, when left out.",
    )
    parser.add_argument(
        "--sys0",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Drive the arms with System 0, the learned motor tracker (robots.sys0); on by default for a YAM, the one "
        "body it drives. --no-sys0 runs the plain command writer.",
    )
    parser.add_argument(
        "--eef",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Which head of the model drives the arms, whatever the config says: --eef its tool poses, solved to "
        "joints with dream's IK (`dreamcli robot up eef`), --no-eef its joint head (`dreamcli robot up joint`, and "
        "the older `noeef`). Unset runs the config's `action_space`, which is 'eef' unless the file says 'joint'.",
    )
    parser.add_argument(
        "--ik-solve",
        choices=config.IK_SOLVES,
        default=None,
        help="How an eef run's tool poses become joints, for this agent only, whatever the config's `ik_solve`: tick "
        "(row by row) or traj (the whole chunk, continuing the arm's last three "
        "commands). Unset runs the config's ik_solve, else traj "
        "(robots.config.DEFAULT_IK_SOLVE).",
    )
    parser.add_argument(run.SKIP_TCP_FLAG, action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    from robots import cli

    lock_path = pathlib.Path(args.arms_lock) if args.arms_lock else cli.paths().lock
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = run.with_task(config.load(args.config), args.task)
        if args.eef is not None:
            cfg = run.with_action_space(cfg, "eef" if args.eef else "joint")
        cfg = run.with_ik_solve(cfg, args.ik_solve)
        sys0 = args.sys0 if args.sys0 is not None else bodies.for_config(cfg).name == bodies.YAM.name
    except errors.RobotsError as exc:
        _log.error("station: stopped: %s", exc)
        sys.exit(1)
    # A failure before the station is ready is a failed start: one line says which device, the error and the fix
    # after serve() has switched the arms off, for `dreamcli robot up` to show; the station still exits with
    # its own error.
    ready = threading.Event()
    try:
        asyncio.run(
            serve(
                cfg,
                host=args.host,
                api_port=args.api_port,
                inference_log_path=args.inference_log,
                arms_lock_path=lock_path,
                sys0=sys0,
                skip_tcp_calibration=args.skip_tcp_calibration,
                ready=ready,
            )
        )
    except errors.RobotsError as exc:
        _log.error("station: stopped: %s", exc)
        if not ready.is_set():
            startup_failure.log(exc, cfg)
        sys.exit(1)
    except Exception as exc:
        if not ready.is_set():
            startup_failure.log(exc, cfg)
        raise


if __name__ == "__main__":
    main()

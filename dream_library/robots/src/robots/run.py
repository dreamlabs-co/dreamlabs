"""Entry point: build the arms, cameras and dream client from an #robots.config.AppConfig and run the loop.

```bash
dreamcli robot run --config configs/yam.example.yaml [--task "..."] [--checkpoint "..."] [--dry-run]
dreamcli robot run --config configs/so101.example.yaml --checkpoint "dream-1-ft:..." --action-space joint
python -m robots.run --config configs/yam.example.yaml [--task "..."] [--checkpoint "..."] [--dry-run]
```

`--task` overrides the config's `task` for this run and `--checkpoint` the model, and `dreamcli robot run`'s
`--action-space` its `action_space` (with_action_space()); `--dry-run` runs the policy without writing a target to an
arm (#DryRun). A run starts from home; Ctrl-C stops it, homes the arms and switches
them off. `robots.station` is the resident alternative with Start/Stop/Home over HTTP; this needs no station agent
and registers nothing.

The dream config is built here (policy_config()): the model, #DEFAULT_CHECKPOINT on a YAM, and the
`embodiment` block filled in from what `robots.adapter` sends and receives, so dream's strict
mode and the adapter cannot disagree. Where DreamAPI is and the key are not in
the file: `dream.DreamClient()` reads them from the credentials file (`dream.credentials`), and open_client()
opens the run's policy on it. The camera names are checked here, before anything is built, because the serve
accepts only its body's camera roles. The arms' forward kinematics, which the loop's end-effector speed clamp
needs, are built here too (make_kinematics()). Everything that differs between the robots comes from the config's body
(`robots.bodies`). Every launch, here, `dreamcli robot run` and each start of `robots.station`, first checks that an
arm of your own's tool point was marked (check_tool_point()).

@see main(), policy_config(), open_client(), make_kinematics(), check_tool_point(), DryRun
"""

import argparse
import asyncio
import contextlib
import dataclasses
import logging
import math
import signal
import time
from typing import Any

import dream
import numpy as np
import numpy.typing as npt

from robots import adapter, bodies, cameras, config, control, errors, followers, home, kinematics, rollouts

## The model a YAM station runs unless a run names another: `robots.bodies.YAM_CHECKPOINT`, which says why.
DEFAULT_CHECKPOINT = bodies.YAM_CHECKPOINT

_log = logging.getLogger(__name__)

## Seconds between two dry-run log lines per arm.
DRY_RUN_LOG_S = 1.0

## Seconds a one-shot run gives managed exposure to converge before the loop starts and holds it: four
## controller steps (`robots.cameras.MANAGED_PERIOD_S`), each of up to 1.6x, from half the ceiling to the scene.
EXPOSURE_SETTLE_S = 2.0

## Round trip, milliseconds, past which every station's dream client paces synchronously (#policy_config), whatever
## the body: the client's own default (`dream.pacing.SYNC_ABOVE_MS`, 700) sent stations to sync at round trips they run
## async well. A YAM's 48 rows at 20 Hz carry exactly this past the 30-row execution horizon.
SYNC_ABOVE_MS = 900.0

## Milliseconds System 0 adds to the sync threshold (#policy_config): 1000 ms, where the plain loop has 900.
## In sync System 0 holds longer than the plain loop: its tracker looks 1 s ahead (`robots.sys0.tracker`), sees the
## plan end before the chunk does and slows early, where the plain loop's PD trails its target into the hold. So it
## stays async past what the chunk carries; the client then asks for the next chunk while rows remain
## (`dream.pacing`), and a reply may land as the last rows run.
SYS0_SYNC_EXTRA_MS = 100.0

## The flag that runs an arm file whose tool point nobody marked (check_tool_point()), not advised: on `dreamcli robot
## run`, `dreamcli robot up` (passed to `robots.station`) and `python -m robots.run`, and named only in the refusal and
## the customer docs' *Set it up yourself*.
SKIP_TCP_FLAG = "--skip-tcp-calibration"

## What each action space executes, for the line policy_config() logs per run.
HEADS = {"eef": "the model's tool poses, solved to joints with dream's IK", "joint": "the model's own joints"}


async def main(
    config: config.AppConfig,
    checkpoint: str | None = None,
    *,
    dry_run: bool = False,
    stop: asyncio.Event | None = None,
    home_duration_s: float = home.HOME_DURATION_S,
) -> None:
    """Homes the arms, runs the loop until SIGINT, SIGTERM or SIGHUP, homes them and switches them off.

    The run starts from each arm's home (#robots.home.return_to_home), as a station's Home then Start does. When it
    stops, the loop holds the arms, they move home again, and every arm is switched off as a station does when it exits
    (`FollowerArm.torque_off()`: a brakeless arm parks, a YAM goes limp at home). An arm's alarm (a motor too hot, a
    joint blocked too long: `errors.SafetyStop`) ends the run the same way. After an arm fault nothing homes: the loop
    disengages a YAM's arms, holds brakeless ones, and each is switched off (a brakeless arm that still answers parks
    first). A second signal while this runs is ignored, so the move and the switch-off are never cut short. A stop
    while the cameras and the client open ends the run before an arm opens, and one during the first move home
    switches the arms off there, without starting the loop; neither is recorded. A dry run
    opens everything and runs the policy, but no target reaches an arm (#DryRun): each arm's reading and the target it
    was asked for are logged once a second, nothing homes, and on exit the arms are only closed, so nothing moves at
    all. An arm on a motor bus (Makers, Metal, SO-101) is opened read only, with no motor switched on, so the dry run
    leaves it as it found it; a YAM's arms, which i2rt opens powered, stay powered where they are. Its last line says
    which.

    When the config's `rollouts.enabled` is true, the run is saved as an episode on this disk
    (#robots.rollouts); the run waits for the episode to be finished before it returns. A dry run saves none: its
    targets reach no arm, so as an episode its `action` column would be wrong.

    @param config Parsed application config; see #robots.config.AppConfig.
    @param checkpoint The model for this run.
    @param dry_run Run without writing a target to an arm.
    @param stop Ends the run when set.
    @param home_duration_s Seconds each home move takes.
    @throws ConfigError A camera is not named after a serve role, the body has no default model and none was
        given, or i2rt or mujoco (a Makers or Metal arm: python-can or dream's IK) is not installed.
    @throws ControlLoopError The loop stopped on an unrecoverable fault.
    @throws CameraError A camera never published or its thread died.
    @throws ConfigError The config has no `task` and none was given with `--task`; a one-shot run has no page to type
        one on.
    """
    if not config.task:
        raise errors.ConfigError("no task prompt: set 'task' in the config or pass --task")
    dream_config = policy_config(config, checkpoint)
    if "checkpoint" not in dream_config:
        raise errors.ConfigError(
            f"no checkpoint: a {config.body} body has no default model; pass --checkpoint, or set 'checkpoint' in "
            "the station file"
        )
    if stop is None:
        stop = asyncio.Event()
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            asyncio.get_running_loop().add_signal_handler(signum, stop.set)
    body = bodies.for_config(config)
    arms: dict[str, followers.Arm] = {}
    readers: list[cameras.CameraReader] = []
    recorder = None if dry_run else make_recorder(config, dream_config["checkpoint"], run_number=1)
    not_off: list[str] = []
    try:
        readers = await asyncio.to_thread(make_cameras, config)
        arm_kinematics = make_kinematics(config)
        client = await asyncio.to_thread(open_client, dream_config)
        if stop.is_set():
            _log.info("run: stopped before the arms opened; nothing moved and no run was recorded")
            control.run_teardown("close the dream client", client.close)
            return
        # The arms last, so a camera or a model that fails to set up never moves one.
        open_arm = body.open_arm_read_only if dry_run and body.open_arm_read_only is not None else body.open_arm
        for name, arm_cfg in (("left", config.follower_l), ("right", config.follower_r)):
            arm = open_arm(arm_cfg, name)
            arms[name] = followers.FollowerArm(name, DryRun(arm), home_rad=arm.home_rad) if dry_run else arm
        if not dry_run and not await asyncio.to_thread(home.return_to_home, arms, duration_s=home_duration_s):
            raise errors.ArmError("an arm did not reach home before the run; see the log")
        if stop.is_set():
            _log.info("run: stopped while the arms moved home; they switch off there, and no run was recorded")
            control.run_teardown("close the dream client", client.close)
        else:
            await _run_from_home(
                config, arms, readers, client, arm_kinematics, recorder, stop, dry_run, home_duration_s
            )
    finally:
        for reader in readers:
            reader.stop()
        for name, opened in arms.items():
            if not control.run_teardown(f"switch {name} arm off", opened.torque_off):
                not_off.append(name)
        if dry_run and arms:
            _log.info("run: dry run over: %s", _dry_run_end(body))
        if recorder is not None:
            recorder.wait()
    if not_off:
        raise errors.ArmError(
            f"arm {' and '.join(map(repr, not_off))} may still be powered: its switch-off failed (the log above says "
            "which motors); cut its power"
        )


async def _run_from_home(
    config: config.AppConfig,
    arms: dict[str, followers.Arm],
    readers: list[cameras.CameraReader],
    client: dream.DreamClient,
    arm_kinematics: dict[str, kinematics.Kinematics],
    recorder: rollouts.Recorder | None,
    stop: asyncio.Event,
    dry_run: bool,
    home_duration_s: float,
) -> None:
    for reader in readers:
        reader.start()
    if any(reader.exposure is not None for reader in readers):
        await asyncio.sleep(EXPOSURE_SETTLE_S)
    for reader in readers:
        reader.set_exposure_hold(True)
    loop = control.ControlLoop(
        followers=arms,
        cameras=readers,
        client=client,
        task=config.task,
        kinematics=arm_kinematics,
        inference_log_path=config.inference_log_path,
        camera_owner=False,
        recorder=recorder,
        **loop_settings(config),
    )
    fault: Exception | None = None
    try:
        async with loop:
            run = asyncio.create_task(loop.run(), name="robots-run-loop")
            stopped = asyncio.create_task(stop.wait(), name="robots-run-stop")
            await asyncio.wait({run, stopped}, return_when=asyncio.FIRST_COMPLETED)
            stopped.cancel()
            if run.done():
                run.result()
            if dry_run:
                _log.info("run: stopping; a dry run moved no arm")
            else:
                _log.info("run: stopping; the arms hold, move home, then switch off")
            run.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await run
    except Exception as exc:
        fault = exc
    if not dry_run and not isinstance(fault, errors.ArmError):
        if not await asyncio.to_thread(home.return_to_home, arms, duration_s=home_duration_s) and fault is None:
            fault = errors.ArmError("an arm did not reach home after the run; see the log")
    if fault is not None:
        raise fault


def make_recorder(app: config.AppConfig, checkpoint: str, run_number: int) -> rollouts.Recorder | None:
    """The recorder that saves a run as an episode on this disk, or None when the config turns rollouts off.

    @param app The parsed application config.
    @param checkpoint The model the run asks.
    @param run_number The run's ordinal, in the episode's folder name.
    @return The recorder, not yet begun.
    """
    if not app.rollouts.enabled:
        return None
    return rollouts.Recorder(
        app.rollouts,
        body=app.body,
        checkpoint=checkpoint,
        run_number=run_number,
        fps=app.control_hz,
        camera_names=[camera.name for camera in app.cameras],
        joints={"left": len(app.follower_l.home), "right": len(app.follower_r.home)},
        robot_id=None if app.live is None else app.live.robot_id,
        settings={key: getattr(app, key) for key in rollouts.SETTINGS},
    )


def _dry_run_end(body: bodies.Body) -> str:
    if body.open_arm_read_only is not None:
        return "no motor was switched on, so each arm is as it was found"
    if body is bodies.YAM:
        return (
            "the arms are only closed, and a YAM's stay powered and stiff where they are (i2rt opens them powered) "
            "until their power is cut: support each arm when you switch it off"
        )
    return "the arms are only closed, as your driver's close() leaves them"


class DryRun(followers.Driver):
    """An opened arm that takes no target: the dry run's driver over the arm the body opened.

    It reads the arm as it is and drops every write, logging the arm's reading beside the target it was asked for
    once a second; torque_off() only closes the arm, so a dry run never moves one, not even to park. An arm the body
    opened read only (`Body.open_arm_read_only`) was never switched on, so closing it leaves it as it was found.
    """

    def __init__(self, arm: followers.FollowerArm) -> None:
        """Wraps an opened arm.

        @param arm The arm as its body opened it.
        """
        self._arm = arm
        self._logged_at_s = -math.inf

    def read(self) -> followers.Reading:
        """The arm's own reading.

        @return The #robots.followers.Reading.
        """
        state = self._arm.read()
        return followers.Reading(state.positions_rad, state.velocities_rad_s, state.gripper, state.gripper_effort)

    def write(self, positions_rad: npt.NDArray[np.float64], gripper: float) -> None:
        """Drops the target, logging it beside the reading once a second.

        @param positions_rad Six joint targets, radians.
        @param gripper The gripper target.
        """
        now_s = time.monotonic()
        if now_s - self._logged_at_s < DRY_RUN_LOG_S:
            return
        self._logged_at_s = now_s
        state = self._arm.read()
        _log.info(
            "dry run: arm '%s' reads %s rad, gripper %.2f; asked %s rad, gripper %.2f (not written)",
            self._arm.name,
            np.array2string(state.positions_rad, precision=3, separator=", "),
            state.gripper,
            np.array2string(positions_rad, precision=3, separator=", "),
            gripper,
        )

    def is_alive(self) -> bool:
        """Whether the arm still answers.

        @return The arm's own answer.
        """
        return self._arm.is_alive()

    def close(self) -> None:
        """Closes the arm."""
        self._arm.close()

    def torque_off(self) -> None:
        """Closes the arm, so nothing moves; its motors keep the target they hold."""
        self._arm.close()


def loop_settings(app: config.AppConfig) -> dict[str, Any]:
    """The `ControlLoop` keyword arguments the config sets.

    Everything but the hardware, the kinematics, the dream client, the task, the inference log and the run number,
    which the caller supplies; `robots.run` and `robots.station` build their loops from the same dict.

    @param app The parsed application config.
    @return Keyword arguments for `control.ControlLoop`, one per timing and shaping key of #robots.config.AppConfig,
        and the wire of the config's #robots.bodies.Body.
    """
    return {
        "wire": bodies.for_config(app).wire(),
        "control_hz": app.control_hz,
        "action_horizon": app.action_horizon,
        "action_timeout_s": app.action_timeout_s,
        "stale_after_s": app.stale_after_s,
        "max_stall_ticks": app.max_stall_ticks,
        "first_chunk_timeout_s": app.first_chunk_timeout_s,
        "ramp_s": app.ramp_s,
        "ramp_max_rad_s": app.ramp_max_rad_s,
        "max_ee_speed_m_s": app.max_ee_speed_m_s,
        "max_gripper_step": app.max_gripper_step,
        "grip_force_cap": app.grip_force_cap,
        "trace_path": app.trace_path,
        "command_hz": app.command_hz,
        "command_interp_max_s": app.command_interp_max_s,
    }


def make_cameras(app: config.AppConfig) -> list[cameras.CameraReader]:
    """Opens one `CameraReader` per `cameras` entry, not yet started.

    A reader's `stop()` releases its capture device, so a loop's `__aexit__` leaves them unusable;
    the station calls this again for every run.

    @param app The parsed application config.
    @return The readers, in config order.
    @throws CameraError A camera cannot be opened.
    @throws RuntimeError librealsense refused a RealSense camera; the message starts with which one, `the <name> camera
        <serial> did not open: `, as setup words it, because librealsense's own words ("No device connected") do not
        say (`robots.startup_failure` names the camera from it for `dreamcli robot up`).
    """
    return [_open_camera(app, c) for c in app.cameras]


def _open_camera(app: config.AppConfig, camera: config.CameraConfig) -> cameras.CameraReader:
    """One camera of #make_cameras, a librealsense refusal prefixed with the camera it is about."""
    try:
        return cameras.CameraReader.from_config(
            camera,
            app.stale_after_s,
            obs_fps=app.obs_fps,
            policy_image_size=app.policy_image_size,
            face_blur=app.face_blur,
            exposure=app.exposure,
        )
    except RuntimeError as exc:
        if not camera.serial:
            raise
        raise RuntimeError(f"the {camera.name} camera {camera.serial} did not open: {exc}") from exc


def make_kinematics(app: config.AppConfig) -> dict[str, kinematics.Kinematics]:
    """Builds the forward kinematics of both arms for the config's body.

    A YAM gets one i2rt MuJoCo model per arm, from each arm's gripper type; every other body dream's own model of its
    arm, one side each (#robots.kinematics.DreamKinematics). Built once: the station keeps them for every run, since a
    model takes milliseconds (dream's, seconds) to load and the loop calls it several times per tick.

    @param app The parsed application config.
    @return The kinematics keyed `"left"` and `"right"`, as `ControlLoop` takes them.
    @throws ConfigError i2rt or mujoco is not installed (a Makers or Metal arm: dream's IK).
    """
    return bodies.for_config(app).kinematics(app)


def policy_config(app: config.AppConfig, checkpoint: str | None = None, *, sys0: bool = False) -> dict[str, Any]:
    """Returns the config of `dream.DreamClient.policy()` for the config's body and this run.

    It carries the model, the `embodiment`, the loop's rate as `action_hz`, the body's `ik`, and the frame window
    when the config's differs from the client's. The client keeps the policy's frame window, `obs_window` frames per
    camera at `obs_fps` (9 at 5 fps unless the config says otherwise, the defaults of both), so a config that changes
    either passes it on.

    The client's sync threshold (`dream.pacing`) is #SYNC_ABOVE_MS (900 ms) on every body, plus #SYS0_SYNC_EXTRA_MS
    with System 0 (1000 ms). The station file replaces either: `sync_above_ms` without System 0, `sys0_sync_above_ms`
    with it.

    @param app The parsed application config; never mutated.
    @param checkpoint The model for this run.
    @param sys0 Whether System 0 drives the motors (a YAM station, unless `--no-sys0`), which sizes the sync
        threshold.
    @return A new dict: `{checkpoint, embodiment, action_hz, sync_above_ms, ik}` on a YAM (`ik` arm `yam` with the
        config's `action_space` as its head; `embodiment` `robots.adapter.YAM_POSE_EMBODIMENT` on head `eef`, the
        default, and the YAM's own `robots.adapter.YAM_EMBODIMENT` on `joint`), `{embodiment, action_hz,
        sync_above_ms, ik}` (with `checkpoint` first when there is one) on a Makers or Metal arm, the `ik` head the
        config's `action_space`; that head is logged; plus `obs_window` and `obs_fps` when they are not the defaults.
    @throws ConfigError A camera name is not one of the body's camera roles, a body other than the YAM would run
        a model that answers in joints (#refuse_joint_space), or the model is a fine-tune of another embodiment than
        the one the config names, or `action_space` is `joint` and the model is not a fine-tune of that embodiment
        (`robots.bodies.check_model`), or an arm's home lists another count of joints than the wire's model of
        that arm (`robots.adapter.Wire.for_arms`).
    """
    body = bodies.for_config(app)
    refuse_joint_space(body)
    # A YAM's EEF head reads and answers each hand's pose beside the joints; its joint head, the YAM's own parts.
    wire = adapter.YAM_POSE_WIRE if body is bodies.YAM and app.action_space == "eef" else body.wire()
    # The loop binds the wire to its arms once they are home (control.ControlLoop); a home the wire cannot take is
    # refused here instead, before an arm opens or moves.
    wire.for_arms({"left": len(app.follower_l.home), "right": len(app.follower_r.home)})
    names = [camera.name for camera in app.cameras]
    not_roles = [name for name in names if name not in wire.camera_roles]
    if not_roles:
        raise errors.ConfigError(
            f"config key 'cameras' has names {not_roles} that are not serve camera roles; "
            f"name each camera one of {list(wire.camera_roles)}"
        )
    model = next((m for m in (checkpoint, app.checkpoint, body.checkpoint) if m is not None), None)
    if model is not None:
        bodies.check_model(app, model)
    dream_config: dict[str, Any] = {} if model is None else {"checkpoint": model}
    dream_config["embodiment"] = wire.embodiment(names, app.action_horizon)
    # The client paces the loop (dream.pacing): it must convert round trips to rows at the loop's own tick rate.
    dream_config["action_hz"] = app.control_hz
    dream_config["sync_above_ms"] = SYNC_ABOVE_MS + SYS0_SYNC_EXTRA_MS if sys0 else SYNC_ABOVE_MS
    if body is bodies.YAM:
        # The client adds each hand's pose from the YAM's joints (the serve computes none; a model that does not read
        # it ignores it). Head `joint`, the default, runs the joint answer of every model; `eef` (`dreamcli robot up
        # eef`) solves the model's tool poses to the joints, and runs the joint answer of a model with no EEF head
        # (dream.client_ik).
        dream_config["ik"] = {
            "arm": "yam",
            "head": app.action_space,
            "solve": app.ik_solve or config.DEFAULT_IK_SOLVE,
        }
        _log.info("dream client for %s: ik head %r (%s)", model, app.action_space, HEADS[app.action_space])
    if body.ik is not None:
        # The body's head is `eef`; the run's is the config's action space, which check_model() has allowed.
        dream_config["ik"] = {**body.ik, "head": app.action_space}
        _log.info(
            "dream client for %s: ik head %r (%s), solve %r",
            model,
            app.action_space,
            HEADS[app.action_space],
            app.ik_solve or config.DEFAULT_IK_SOLVE,
        )
        arms = {"left": app.follower_l, "right": app.follower_r}
        keep_out = {side: list(arm.keep_out) for side, arm in arms.items() if arm.keep_out is not None}
        if keep_out:
            dream_config["ik"]["keep_out"] = keep_out
        dream_config["ik"]["solve"] = app.ik_solve or config.DEFAULT_IK_SOLVE  # read by an eef run only
        region = {side: dict(arm.region) for side, arm in arms.items() if arm.region is not None}
        if region:
            dream_config["ik"]["region"] = region
        reach = {side: [list(point) for point in arm.reach] for side, arm in arms.items() if arm.reach is not None}
        if reach:
            dream_config["ik"]["reach"] = reach
            if app.action_space != "eef":
                _log.warning(
                    "the reach area applies only to eef runs; this %s run does not keep the hands in it",
                    app.action_space,
                )
    sync_above_ms = app.sys0_sync_above_ms if sys0 else app.sync_above_ms
    if sync_above_ms is not None:
        dream_config["sync_above_ms"] = sync_above_ms
    if app.obs_window != config.AppConfig.obs_window:
        dream_config["obs_window"] = app.obs_window
    if app.obs_fps != config.AppConfig.obs_fps:
        dream_config["obs_fps"] = app.obs_fps
    return dream_config


def open_client(dream_config: dict[str, Any]) -> dream.DreamClient:
    """A dream client with the run's policy open, on the credentials' key and URLs.

    It is `dream.DreamClient()`, which reads them (`dream.credentials`), then `client.policy(dream_config)`. Nothing is
    sent until the loop's first refill.

    @param dream_config The policy config (policy_config()).
    @return The client; the caller closes it (the loop does on its way out).
    @throws DreamConfigError The credentials cannot be read, or the config is not one the client takes.
    @throws DreamError Opening the policy failed otherwise (with `ik`, the platform refused or could not be reached).
    """
    client = dream.DreamClient()
    try:
        client.policy(dream_config)
    except BaseException:
        client.close()
        raise
    return client


def refuse_joint_space(body: bodies.Body) -> None:
    """Refuses joint-space actions on every robot but the YAM: such a body runs EEF checkpoints only.

    A model that answers in joints drives the motors with them as they come. The zero-shot Dream-1 answers in YAM
    joints, which on another arm are another robot's joint targets; only a fine-tune of that very robot answers in
    its joints, and running those is not supported yet. An EEF model answers with tool poses, which the dream
    client solves to this arm's joints with dream's IK of it (`ik` head `eef`).

    @param body The run's body.
    @throws ConfigError The body is not the YAM and its `ik` head is not `eef`.
    """
    if body.name != bodies.YAM.name and (body.ik is None or body.ik["head"] != "eef"):
        raise errors.ConfigError(
            f"a {body.name} body runs EEF checkpoints only (the dream client's ik head 'eef', got {body.ik}): a "
            "model that answers in joints would drive this arm with another robot's joint targets unless it is a "
            "fine-tune of this robot, which is not supported yet"
        )


def check_tool_point(app: config.AppConfig, *, skip: bool) -> None:
    """Refuses an eef run on an arm file whose tool point was never marked, and warns when it was skipped.

    An arm of your own from its URDF is an arm file (`custom.ik`, `dream.arm_file`), and an eef run puts the fingertips
    where the model wants them only as well as its tool point is right. The setup tools record what the operator did
    (`robots.kinematics.tcp_calibration`): `calibrated` runs; `skipped`, chosen and confirmed on the page, runs with a
    warning line at every launch; a file with no record, written by hand, is refused unless `skip` (#SKIP_TCP_FLAG),
    which warns the same way. dream's own arms, a YAM and a joint run are not checked: their tool point is dream's or
    unused.

    @param app The run's config, with the run's `action_space` (with_action_space()).
    @param skip The operator passed #SKIP_TCP_FLAG.
    @throws ConfigError The arm file records no tool point and `skip` is false, or the file or its record is
        malformed (`robots.kinematics.tcp_calibration`).
    """
    ik = bodies.for_config(app).ik
    if ik is None or app.action_space != "eef" or not ik["arm"].endswith(".json"):
        return
    arm = ik["arm"]
    status = kinematics.tcp_calibration(arm)
    if status == "calibrated":
        return
    if status is None and not skip:
        raise errors.ConfigError(
            f"the arm file {arm} (custom.ik) has no tool point: run 'dreamcli tcp', put the marker where your "
            f"gripper's fingertips close and press Set tool center point; or, not advised, pass {SKIP_TCP_FLAG}"
        )
    why = "was skipped" if status == "skipped" else f"was never marked ({SKIP_TCP_FLAG})"
    _log.warning("the tool point of %s %s, so hand positions may land off target; 'dreamcli tcp' marks it", arm, why)


def with_action_space(app: config.AppConfig, action_space: str | None) -> config.AppConfig:
    """The config with `action_space` replaced by a run's own, or unchanged when there is none.

    A run names its own with `dreamcli robot run --action-space` or a Start from the Live view.

    @param app The loaded config.
    @param action_space One of `robots.config.ACTION_SPACES`, or None.
    @return A new config when `action_space` is set, else `app` itself.
    @throws ConfigError `action_space` is not one of `robots.config.ACTION_SPACES`.
    """
    if action_space is None:
        return app
    if action_space not in config.ACTION_SPACES:
        raise errors.ConfigError(f"the action space must be one of {list(config.ACTION_SPACES)}, got {action_space!r}")
    return dataclasses.replace(app, action_space=action_space)


def with_ik_solve(app: config.AppConfig, ik_solve: str | None) -> config.AppConfig:
    """The config with `ik_solve` replaced for one agent run (`dreamcli robot up --ik-solve`), or unchanged when None.

    @param app The loaded config.
    @param ik_solve One of `robots.config.IK_SOLVES`, or None (the config's own, else
        `config.DEFAULT_IK_SOLVE`).
    @return A new config when `ik_solve` is set, else `app` itself.
    @throws ConfigError `ik_solve` is not one of `robots.config.IK_SOLVES`.
    """
    if ik_solve is None:
        return app
    if ik_solve not in config.IK_SOLVES:
        raise errors.ConfigError(f"the IK solve must be one of {list(config.IK_SOLVES)}, got {ik_solve!r}")
    return dataclasses.replace(app, ik_solve=ik_solve)


def with_task(app: config.AppConfig, task: str | None) -> config.AppConfig:
    """The config with `task` replaced by a command-line override, or unchanged when there is none.

    @param app The loaded config.
    @param task The `--task` value, or None.
    @return A new config when `task` is set, else `app` itself.
    @throws ConfigError `task` is an empty string.
    """
    if task is None:
        return app
    if not task:
        raise errors.ConfigError("--task must be a non-empty prompt")
    return dataclasses.replace(app, task=task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the dream inference control loop on a bimanual arm setup.")
    parser.add_argument("--config", required=True, help="Path to the YAML application config.")
    parser.add_argument("--task", default=None, help="Override the config's task prompt for this run.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="The model for this run; default: the config's checkpoint, else the YAM's default model.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run the policy without writing a target to an arm.")
    parser.add_argument(SKIP_TCP_FLAG, action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    loaded = with_task(config.load(args.config), args.task)
    check_tool_point(loaded, skip=args.skip_tcp_calibration)
    asyncio.run(main(loaded, args.checkpoint, dry_run=args.dry_run))

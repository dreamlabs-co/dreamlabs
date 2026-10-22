"""Command the simulated YAM with Dream's station observation contract."""

import argparse
import base64
import io
import json
import logging
import os
import re
import shlex
import time
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import urlsplit

import mujoco
import numpy as np
from PIL import Image
from yam_env import YamEnv, aim_viewer


def settings():
    """Load credential and gateway configuration for the next task.
    @return The settings by name (`DREAM_API_KEY`, `DREAM_GATEWAY_URL`, ...), the environment winning over the files.
    """
    result = {}
    paths = [Path(os.environ.get("DREAM_CREDENTIALS", "~/.dreamlabs/credentials")).expanduser()]
    for path in paths:
        if path.is_file():
            for line in path.read_text().splitlines():
                key, sep, value = line.removeprefix("export ").partition("=")
                if sep and not key.strip().startswith("#"):
                    result[key.strip()] = " ".join(shlex.split(value, comments=True))
    from credentials import saved_key

    # The dream library's order, with the /login key between: the environment, then /login, then the file.
    key = saved_key()
    if key:
        result["DREAM_API_KEY"] = key
    result.update(os.environ)
    return result


## The model the terminal names, and its shortcuts for it; any other model id is used as typed.
BETA_MODEL = "dreamlabs/dream-1-beta"
MODEL_SHORTCUTS = {"1": BETA_MODEL, "beta": BETA_MODEL}


def resolve_model(text):
    """The model a `/model` or `--model` value names: a shortcut's model, or the text itself as a model id.
    @param text What was typed: `beta`, `1`, or any model id (`dream-1-ft:...`, `<run>/checkpoint-<n>`).
    @return The model id to send as `checkpoint`.
    @throws ValueError When the text is empty or holds a space, which no model id does.
    """
    model = text.strip()
    if not model or any(c.isspace() for c in model):
        raise ValueError("A model is beta, or one model id with no spaces.")
    return MODEL_SHORTCUTS.get(model.lower(), model)


## The YAM's embodiment config: the parts a model that reads no hand poses reads and answers, and its camera roles.
YAM_PARTS = {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]}
YAM_EMBODIMENT = {
    "state": YAM_PARTS,
    "action": YAM_PARTS,
    "video": {role: f"observation.images.{role}" for role in ("overhead", "wrist_l", "wrist_r")},
}
## Models known to read each hand's pose; for any other a refusal naming a missing `_eef_` key switches the poses on.
POSE_MODELS = {BETA_MODEL}
## A model that reads hand poses is sent the YAM station's keys, so the serve computes the poses from the joints
## itself (it sets `ik.arm` from `robot.arm`). The model's own keys cannot carry them yet: the serve keeps only the
## state parts the checkpoint lists, which for an EEF-head checkpoint are the joints, so the poses the dream client
## adds (`ik` arm `yam`) are dropped and dream refuses with "EEF checkpoint: send ik.arm ... or [state.*_eef_*]".
## `robot.max_joint_vel_rad_s` 0 is no cap of the sim's own (the arm spec's caps); the gripper is already 0 to 1.
STATION_ROBOT = {
    "robot.arm": "yam",
    "robot.exec_head": "joint",
    "robot.max_joint_vel_rad_s": 0.0,
    "robot.gripper_open": 1.0,
}
## Each arm's slot in the station keys (`state.<slot>.joint_pos`, `state.<slot>.gripper_pos`).
STATION_SLOTS = {"left": "arm_l", "right": "arm_r"}


def policy_config(model, gateway, poses):
    """The dream client's session config for one model.
    @param model The model id.
    @param gateway The HTTPS gateway origin.
    @param poses Whether the model reads each hand's pose: the sim then speaks the station keys (#STATION_ROBOT),
        which the strict-mode embodiment does not name, so none is given.
    @return The config `dream.init` takes.
    """
    config = {
        "checkpoint": model,
        "timeout_ms": 45000,
        "strict": False,  # the sim checks every chunk itself (validate_chunk)
        "action_hz": 20,
        "_record": False,  # simulated episodes are not recorded to the account
    }
    if not poses:
        config["embodiment"] = YAM_EMBODIMENT
    return config


def open_policy(config, key):
    """Opens a dream client with the key, and its policy session for the model.

    The gateway and the platform are the credentials', the same environment and file settings() reads.
    @param config The session config (policy_config()).
    @param key The API key.
    @return The `dream.DreamClient`, its policy open: set_observation(), get_action(), close().
    """
    import dream

    # The terminal prints every round trip; the client's slow-network warning would only repeat it, at length.
    warnings.filterwarnings("ignore", category=dream.DreamSlowNetworkWarning)
    logging.getLogger("dream.slow_network").setLevel(logging.ERROR)
    client = dream.DreamClient(api_key=key)
    client.policy(config)
    return client


def feed(policy, env, windows, task, model, steps, every=1, station=False):
    """Hands the dream client the camera frames: the newest of each window, or with `every` the whole windows.
    @param policy The open dream client.
    @param env Simulation environment, for the state that goes with the frames.
    @param windows Chronological JPEG windows keyed by camera.
    @param task Natural-language task.
    @param model The model id.
    @param steps Control steps run so far: the newest frame's time is steps x 50 ms of simulated time.
    @param every When not 1, every frame of the windows, oldest first, 0.2 s apart (a new or reopened session).
    @param station Whether to speak the YAM station keys (a model that reads hand poses, #STATION_ROBOT).
    """
    sent = observation(env, windows, task, model, station)
    names = {frame_key(name, station): name for name in windows}
    state = {k: v for k, v in sent.items() if k not in names}
    count = len(next(iter(windows.values())))
    for i in [count - 1] if every == 1 else range(count):
        frames = {key: windows[name][i] for key, name in names.items()}
        seconds = (steps - (count - 1 - i) * FRAME_EVERY_STEPS) * 0.05
        policy.set_observation(frames, state, timestamp=max(seconds, 0.0) + i * 1e-6)


def needs_poses(error):
    """Whether a refusal says the model reads hand poses the request did not carry.
    @param error The exception get_action() raised.
    @return True for a refusal that names a missing `_eef_` key: `malformed` from the serve's own check, or `error`
        from dream's ("EEF checkpoint: send ik.arm ... or [...]").
    """
    return getattr(error, "reason", None) in ("malformed", "error") and "_eef_" in str(error)


def is_task(text):
    """Whether a typed line has a word in it (two letters or more), so it can be sent as a task.
    @param text The line as typed.
    @return False for digits, one letter or punctuation alone, which would still start the arms.
    """
    return re.search(r"[^\W\d_]{2,}", text) is not None


## Each camera's video role, the key the model reads it under (`video.<role>`), as on a YAM station.
VIDEO_ROLES = {"top": "overhead", "left": "wrist_l", "right": "wrist_r"}


def frame_key(name, station=False):
    """The request key a camera's frames go under.
    @param name The camera: `top`, `left` or `right`.
    @param station Whether the sim speaks the YAM station keys.
    @return `image_window.<name>` for the station keys, else `video.<role>`.
    """
    return f"image_window.{name}" if station else f"video.{VIDEO_ROLES[name]}"


## Frames in every camera window, and control steps (50 ms each, simulated time) between two of them: 0.2 s apart.
FRAMES, FRAME_EVERY_STEPS = 9, 4
## Width x height of every frame the model sees.
FRAME_SIZE = (320, 176)


def capture(env, windows, keep=True):
    """Render the cameras, and add a JPEG of each to its window when the frame is one the model should see.
    @param env Simulation environment.
    @param windows Chronological JPEG windows keyed by camera.
    @param keep Whether this step's frames join the windows (one step in FRAME_EVERY_STEPS); the viewer gets them anyway.
    @return The RGB images just rendered, by camera name.
    """
    images = env.images()
    if keep:
        for name, rgb in images.items():
            buffer = io.BytesIO()
            Image.fromarray(rgb).resize(FRAME_SIZE, Image.BILINEAR).save(buffer, format="JPEG", quality=85)
            windows[name].append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    return images


def observation(env, windows, task, model, station=False):
    """Construct the observation in the model's own keys, or in the YAM station's for a model that reads hand poses.

    The model's own keys are `state.<part>`, `video.<role>` and `prompt`; the station's are `state.<slot>.joint_pos`,
    `state.<slot>.gripper_pos`, `image_window.<camera>`, #STATION_ROBOT and `prompt`.
    @param env Simulation environment.
    @param windows Chronological JPEG windows keyed by camera.
    @param task Natural-language task.
    @param model Selected model or compiled MuJoCo model.
    @param station Whether to speak the station keys.
    @return The `observation` object of a `get_action` request.
    """
    obs = {"prompt": task}
    for side in ("left", "right"):
        state = env.state()[side]
        if station:
            obs[f"state.{STATION_SLOTS[side]}.joint_pos"] = state[:6].tolist()
            obs[f"state.{STATION_SLOTS[side]}.gripper_pos"] = state[6:].tolist()
        else:
            obs[f"state.{side}_arm"] = state[:6].tolist()
            obs[f"state.{side}_gripper"] = state[6:].tolist()
    if station:
        obs.update(STATION_ROBOT)
    for name, frames in windows.items():
        if len(frames) != FRAMES:
            raise ValueError(f"Each camera needs {FRAMES} frames.")
        obs[frame_key(name, station)] = list(frames)
    return obs


def validate_chunk(env, actions, action_limit=48, *, verbose=True, adjustments=None):
    """Validate the response; project model joint targets onto URDF bounds in the executed prefix.
    @param env Simulation environment.
    @param actions Returned action chunks.
    @param action_limit Maximum chunk rows to execute.
    @param verbose Whether to print detailed adjustments.
    @param adjustments Optional list collecting adjustment notices.
    @return The left and right rows to execute, each (rows, 7), clipped to the joint limits.
    """
    left, right = [np.array(actions[k], dtype=float, copy=True) for k in ("q_l_chunk", "q_r_chunk")]
    if left.ndim != 2 or left.shape != right.shape or left.shape[1] != 7 or not 1 <= len(left) <= 48:
        raise ValueError("Dream must return matching 1–48 x 7 joint chunks.")
    if not 1 <= action_limit <= 48:
        raise ValueError("Action limit must be 1–48.")
    notices = []
    prepared = []
    for side, full in [("left", left), ("right", right)]:
        if not np.isfinite(full).all():
            raise ValueError(f"{side} chunk contains non-finite values; no rows executed.")
        rows = full[:action_limit].copy()
        limits = np.vstack([*[env.model.joint(f"{side}_joint{i}").range for i in range(1, 7)], [0, 1]])
        bounded = np.clip(rows, limits[:, 0], limits[:, 1])
        overshoot = np.abs(rows - bounded)
        # Joint policy predictions are unbounded; project finite targets onto
        # actual URDF limits. Gripper
        # normalization is still checked to catch a wrong response contract.
        bad = np.argwhere(overshoot[:, 6] > 0.05)
        if len(bad):
            step = bad[0, 0]
            raise ValueError(
                f"{side} step {step + 1} gripper: requested {rows[step, 6]:.6f}; "
                "expected normalized opening 0..1. No rows executed."
            )
        for column in range(7):
            count = np.count_nonzero(overshoot[:, column])
            if count:
                label = f"joint{column + 1}" if column < 6 else "gripper"
                notices.append(
                    f"{side} {label}: bounded {count} values, max correction {overshoot[:, column].max():.6f}"
                )
        prepared.append(bounded)
    if adjustments is not None:
        adjustments.extend(notices)
    if verbose:
        for notice in notices:
            print(f"Limit adjustment: {notice}", flush=True)
    return tuple(prepared)


RESET_REFUSED = (
    "  Viewer reset (Backspace/Delete) is off during a run. Stop the run first: Ctrl-C, /reset_robot (arms home) "
    "or /reset (whole scene); click the terminal to type a task."
)


class RunViewerSync:
    """Sync the viewer during a run without letting MuJoCo's own reset (Backspace/Delete, or the side panel's Reset)
    put the scene back mid-run.

    The passive viewer only flags that reset when the key is pressed and applies it inside the next `viewer.sync()`,
    on this thread; the Python binding exposes neither the flag nor a way to cancel it. So the state from before each
    sync is kept, and put back when the simulated clock went backwards across the call, which only a reset does.
    """

    def __init__(self, env):
        """Initialize the object and its owned state.
        @param env Simulation environment.
        """
        self.env = env
        self.state = np.empty(mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_INTEGRATION))
        self.refused_at = -np.inf

    def __call__(self, viewer):
        """Sync the viewer; undo a viewer reset it applied and say so in the terminal, at most every 2 s.
        @param viewer Native MuJoCo viewer.
        @return Whether a reset was undone.
        """
        model, data, spec = self.env.model, self.env.data, mujoco.mjtState.mjSTATE_INTEGRATION
        before = data.time
        mujoco.mj_getState(model, data, self.state, spec)
        viewer.sync()
        if data.time >= before:
            return False
        mujoco.mj_setState(model, data, self.state, spec)
        mujoco.mj_forward(model, data)
        viewer.sync()  # draw the restored scene now, not the reset one until the next step
        if time.monotonic() - self.refused_at > 2:
            self.refused_at = time.monotonic()
            print(RESET_REFUSED, flush=True)
        return True


def main(argv=None, shared_env=None, shared_viewer=None, stop_event=None, collision_view=None):
    """Run this module entry point.
    @param argv Optional command-line arguments.
    @param shared_env Optional existing simulation environment.
    @param shared_viewer Optional existing viewer.
    @param stop_event Optional task cancellation event.
    @param collision_view Optional collision overlay.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="put apple into the bowl")
    parser.add_argument(
        "--model", type=resolve_model, default=BETA_MODEL, help="beta, or any model id, such as a fine-tune"
    )
    parser.add_argument("--actions", type=int, default=48, help="Execute 1–48 rows before observing again")
    parser.add_argument("--seconds", type=float, default=120, help="Wall-clock trial duration")
    parser.add_argument("--show-side-panel", action="store_true", help="Show native MuJoCo controls")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Render and validate observation without calling API")
    parser.add_argument("--quiet", action="store_true", help="Compact progress and one final summary")
    args = parser.parse_args(argv)
    if not 1 <= args.actions <= 48 or not np.isfinite(args.seconds) or args.seconds <= 0 or not args.task.strip():
        parser.error("Use 1–48 actions, positive finite seconds, and a nonempty task.")
    config = settings()
    key = config.get("DREAM_API_KEY", "")
    gateway = config.get("DREAM_GATEWAY_URL", "https://api.dreamlabs.ai")
    url = urlsplit(gateway)
    if (
        url.scheme != "https"
        or not url.hostname
        or url.username
        or url.password
        or url.path not in ("", "/")
        or url.query
        or url.fragment
    ):
        parser.error("DREAM_GATEWAY_URL must be an HTTPS origin.")
    if not key and not args.dry_run:
        parser.error("Use /login, set DREAM_API_KEY, or configure ~/.dreamlabs/credentials.")
    env = shared_env if shared_env is not None else YamEnv()
    policy = None
    request_times, adjustments = [], []
    executed_rows = 0
    try:
        windows = {name: deque(maxlen=FRAMES) for name in VIDEO_ROLES}
        steps = 0  # control steps run, simulated time: a frame joins the windows every FRAME_EVERY_STEPS
        while any(len(window) < FRAMES for window in windows.values()):
            state = env.state()
            env.step(state["left"], state["right"])
            capture(env, windows, keep=steps % FRAME_EVERY_STEPS == 0)
            steps += 1
        obs = observation(env, windows, args.task, args.model, args.model in POSE_MODELS)
        if args.dry_run:
            print(f"Dry run OK: task={args.task!r}, model={args.model}, {FRAMES} frames x 3 cameras. No API call.")
            return
        import mujoco.viewer

        if collision_view is None:
            from collision_view import CollisionView

            collision_view = CollisionView(env)
        context = (
            nullcontext(shared_viewer)
            if shared_viewer is not None or args.headless
            else mujoco.viewer.launch_passive(
                env.model,
                env.data,
                key_callback=collision_view.toggle,
                show_left_ui=args.show_side_panel,
                show_right_ui=args.show_side_panel,
            )
        )
        with context as viewer, ThreadPoolExecutor(max_workers=1) as pool:
            if viewer:
                aim_viewer(viewer)
            from camera_panel import CameraPanel

            cameras = CameraPanel(env, viewer) if viewer else None
            sync = RunViewerSync(env)
            deadline = time.monotonic() + args.seconds
            active = lambda: (
                (stop_event is None or not stop_event.is_set())
                and time.monotonic() < deadline
                and (viewer is None or viewer.is_running())
            )
            if args.quiet:
                print(f"  Running: {args.task} · {args.model.rsplit('/', 1)[-1]}", flush=True)
            else:
                print(f"Task: {args.task}\nModel: {args.model}\nGateway: {gateway}", flush=True)
            poses = args.model in POSE_MODELS
            if poses and not args.quiet:
                print("  This model reads each hand's pose; the serve adds it from the YAM station keys.")
            policy = open_policy(policy_config(args.model, gateway, poses), key)
            feed(policy, env, windows, args.task, args.model, steps, every=0, station=poses)
            restarts = 0
            while active():
                started = time.monotonic()
                future = pool.submit(policy.get_action)
                # Hold simulation time while inference is pending; keep the viewer responsive.
                stop_reported = False
                while not future.done():
                    if stop_event is not None and stop_event.is_set() and not stop_reported:
                        print("  Motion stopped. Waiting for the pending response to release its session.", flush=True)
                        stop_reported = True
                    if viewer and viewer.is_running():
                        if collision_view is not None:
                            collision_view.update(viewer)
                        sync(viewer)
                        cameras.update()
                    time.sleep(0.02)
                try:
                    reply = future.result()
                except Exception as error:
                    # The serve restarted the episode: the client already holds the new session, so ask again.
                    if type(error).__name__ == "DreamSessionExpired" and restarts < 3:
                        restarts += 1
                        print("  The model restarted the episode; asking again.", flush=True)
                        continue
                    if poses or not needs_poses(error):
                        raise
                    # The model reads each hand's pose: open it again with the poses, and ask again.
                    print("  This model reads each hand's pose; sending the YAM station keys.", flush=True)
                    policy.close()
                    poses = True
                    policy = open_policy(policy_config(args.model, gateway, poses), key)
                    feed(policy, env, windows, args.task, args.model, steps, every=0, station=poses)
                    continue
                restarts = 0
                if not active():
                    break
                request_times.append(time.monotonic() - started)
                print(
                    f"  Dream · {request_times[-1]:.2f}s RTT · {len(request_times)} requests · {executed_rows} actions",
                    flush=True,
                )
                if not active():
                    break
                try:
                    left, right = validate_chunk(
                        env, reply, args.actions, verbose=not args.quiet, adjustments=adjustments
                    )
                except ValueError:
                    from credentials import config_dir

                    diagnostic = config_dir() / "last_rejected_actions.json"
                    diagnostic.write_text(
                        json.dumps(
                            {"model": args.model, "actions": reply}, indent=2, default=lambda v: np.asarray(v).tolist()
                        )
                    )
                    print(f"Rejected action details saved to {diagnostic}")
                    raise
                for l, r in zip(left[: args.actions], right[: args.actions]):
                    if not active():
                        break
                    tick = time.monotonic()
                    env.step(l, r)
                    executed_rows += 1
                    kept = steps % FRAME_EVERY_STEPS == 0
                    images = capture(env, windows, keep=kept)
                    if kept:
                        feed(policy, env, windows, args.task, args.model, steps, station=poses)
                    steps += 1
                    if viewer:
                        if collision_view is not None:
                            collision_view.update(viewer)
                        sync(viewer)
                        cameras.update(images)
                    time.sleep(max(0, 0.05 - (time.monotonic() - tick)))
    finally:
        if policy is not None:
            try:
                policy.close()  # ends the session on the model
                if not args.quiet:
                    print("  Dream session released.", flush=True)
            except Exception:
                print("Session release unavailable; it will expire on the server.")
        if args.quiet and request_times:
            print(
                f"  Finished · {executed_rows} actions · {len(request_times)} requests · "
                f"average RTT {sum(request_times) / len(request_times):.2f}s"
                + (" · limits adjusted (details: --verbose)" if adjustments else ""),
                flush=True,
            )
        if shared_env is None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        print(f"Stopped: {error}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("Stopped.")

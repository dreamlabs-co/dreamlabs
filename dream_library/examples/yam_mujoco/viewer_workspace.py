"""Persistent viewer process. All MuJoCo rendering stays on its owning thread."""

import argparse
import json
import queue
import sys
import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
from camera_panel import CameraPanel
from dream_control import main as run_task
from yam_env import YamEnv, aim_viewer, home_path


def main():
    """Run this module entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-side-panel", action="store_true")
    args = parser.parse_args()
    commands = queue.Queue()
    task_tokens = []
    token_lock = threading.Lock()

    def read_commands():
        for line in sys.stdin:
            try:
                message = json.loads(line)
                with token_lock:
                    # A reset overrides like stop: the running task, and any not started, end first.
                    if message.get("command") in ("stop", "quit", "reset", "reset_robot"):
                        for token in task_tokens:
                            token.set()
                    if message.get("command") == "task":
                        # A new task replaces the one running and any not started yet: no queue.
                        for token in task_tokens:
                            token.set()
                        message["stop_event"] = threading.Event()
                        task_tokens.append(message["stop_event"])
                if message.get("command") != "stop":
                    commands.put(message)
            except ValueError:
                pass
        with token_lock:
            for token in task_tokens:
                token.set()
        commands.put({"command": "quit"})

    threading.Thread(target=read_commands, daemon=True).start()
    env = YamEnv()
    from collision_view import CollisionView

    collisions = CollisionView(env)
    try:
        with mujoco.viewer.launch_passive(
            env.model,
            env.data,
            key_callback=collisions.toggle,
            show_left_ui=args.show_side_panel,
            show_right_ui=args.show_side_panel,
        ) as viewer:
            aim_viewer(viewer)
            panel = CameraPanel(env, viewer)
            while viewer.is_running():
                try:
                    message = commands.get_nowait()
                except queue.Empty:
                    message = {}
                if message.get("command") == "quit":
                    break
                if message.get("command") == "collisions":
                    collisions.toggle()
                if message.get("command") == "reset":
                    env.reset()
                    print("  Scene reset.", flush=True)
                if message.get("command") == "reset_robot":
                    # Drive the arms home through the position servos; the objects stay where they are.
                    for left, right in home_path(env.state()):
                        if not viewer.is_running():
                            break
                        tick = time.monotonic()
                        env.step(left, right)
                        collisions.update(viewer)
                        viewer.sync()
                        panel.update()
                        time.sleep(max(0, 0.05 - (time.monotonic() - tick)))
                    print("  Arms at home.", flush=True)
                if message.get("command") == "task":
                    if message["stop_event"].is_set():
                        with token_lock:
                            task_tokens.remove(message["stop_event"])
                        continue
                    try:
                        run_task(
                            [
                                "--task",
                                message["task"],
                                "--model",
                                message["model"],
                                "--actions",
                                str(message["actions"]),
                                "--seconds",
                                str(message["seconds"]),
                            ]
                            + ([] if message.get("verbose") else ["--quiet"]),
                            shared_env=env,
                            shared_viewer=viewer,
                            stop_event=message["stop_event"],
                            collision_view=collisions,
                        )
                    except (Exception, SystemExit) as error:
                        detail = str(error)
                        if type(error).__name__ == "DreamAuthError":
                            print(
                                "  Authentication failed: Dream rejected the configured API key. "
                                "Use /login with a valid key for this gateway; /login saves but does not verify it.",
                                flush=True,
                            )
                        else:
                            print(f"  Task stopped: {error}", flush=True)
                    finally:
                        with token_lock:
                            task_tokens.remove(message["stop_event"])
                        # Hold measured positions, not the last unfinished policy target.
                        for i in range(env.model.nu):
                            joint = env.model.actuator_trnid[i, 0]
                            env.data.ctrl[i] = np.clip(
                                env.data.qpos[env.model.jnt_qposadr[joint]], *env.model.actuator_ctrlrange[i]
                            )
                        env.data.qvel[:] = 0
                    print("  Ready. Viewer remains open.", flush=True)
                mujoco.mj_step(env.model, env.data, nstep=10)
                collisions.update(viewer)
                viewer.sync()
                panel.update()
                time.sleep(0.02)
    finally:
        env.close()


if __name__ == "__main__":
    main()

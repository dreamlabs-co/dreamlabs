"""A small conversational terminal launcher for the MuJoCo Dream demo."""

import argparse
import atexit
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

from credentials import config_dir, login
from dream_control import BETA_MODEL, is_task, resolve_model, settings
from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.patch_stdout import patch_stdout
from terminal_input import DEFAULT_TASK, create_session, is_first_run

ROOT = Path(__file__).resolve().parent

## The longest trial `/seconds` accepts: a day, so a trial left running stops on its own.
MAX_SECONDS = 86400


def main():
    """Run this module entry point."""
    parser = argparse.ArgumentParser(description="Dream YAM Sim: enter a task to control simulated arms with Dream.")
    parser.add_argument("--show-side-panel", action="store_true", help="Show the native MuJoCo settings and controls")
    parser.add_argument("--verbose", action="store_true", help="Show every inference and limit adjustment")
    args = parser.parse_args()
    first_run = is_first_run()
    session = create_session()
    live_status = {"text": "Ready · Tab: tasks · ↑/↓: history", "running": False}
    session.bottom_toolbar = lambda: FormattedText([("bg:#252a33 #61afef", " " + live_status["text"] + " ")])
    python = Path(sys.executable).with_name("mjpython") if sys.platform == "darwin" else Path(sys.executable)
    viewer_args = [str(python), str(ROOT / "viewer_workspace.py")]
    if args.show_side_panel:
        viewer_args.append("--show-side-panel")
    worker = subprocess.Popen(
        viewer_args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=config_dir(),  # a writable place for MuJoCo's own log, whether ROOT is a checkout or an installed wheel
        start_new_session=True,
    )

    def relay_output():
        for line in worker.stdout:
            text = line.rstrip()
            if text.strip().startswith("Dream ·") and not args.verbose:
                live_status["text"] = text.strip()
                session.app.invalidate()
                continue
            if text.strip().startswith("Running:"):
                live_status["running"] = True
                live_status["text"] = "Running · waiting for first response…"
                session.app.invalidate()
            elif "Ready. Viewer remains open." in text:
                live_status["running"] = False
                live_status["text"] = "Ready · Tab: tasks · ↑/↓: history"
                session.app.invalidate()
            color = (
                "#e06c75"
                if "stopped:" in text.lower() or "error" in text.lower() or "failed:" in text.lower()
                else "#98c379"
                if "Ready." in text
                else "#61afef"
                if "Dream ·" in text
                else "#aab2bf"
            )
            print_formatted_text(FormattedText([(color, text)]))

    threading.Thread(target=relay_output, daemon=True).start()

    def stop_viewer():
        if worker.poll() is None:
            try:
                worker.stdin.write(json.dumps({"command": "quit"}) + "\n")
                worker.stdin.flush()
                worker.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                if os.name == "posix":
                    os.killpg(worker.pid, signal.SIGTERM)
                else:
                    worker.terminate()
                try:
                    worker.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(worker.pid, signal.SIGKILL)
                    else:
                        worker.kill()
                    worker.wait()

    atexit.register(stop_viewer)
    interrupted = False

    def interrupt():
        nonlocal interrupted
        if interrupted or worker.poll() is not None:
            print("\n  Closing workspace.")
            stop_viewer()
            return True
        interrupted = True
        worker.stdin.write(json.dumps({"command": "stop"}) + "\n")
        worker.stdin.flush()
        print("\n  Stopping task. Press Ctrl-C again to close the workspace.")
        return False

    model, actions, seconds = BETA_MODEL, 48, 120
    print_formatted_text(
        FormattedText([("#d7875f bold", "\n╭─ Dream YAM Sim ─────────────────────────────────────────╮")])
    )
    print("  Two arms · three cameras · MuJoCo + Dream")
    print("  Dream-1-Beta · joint control · 48 steps · 2 minute trials")
    print("  Tab for recommended tasks · type + Tab to complete · ↑/↓ history")
    print("  /help commands · /login API key · /reset scene · /reset_robot arms home · /quit exit")
    print("  Task submissions call Dream and may incur API usage.\n")
    if settings().get("DREAM_API_KEY"):
        print("  API key configured. Use /login to replace it.\n")
    else:
        print("  First-time setup — configure your API key to run tasks.")
        login()
    while True:
        try:
            # A first-time user's prompt starts filled in with a task they can run with Enter.
            text = session.prompt(
                FormattedText([("class:prompt", "❯ ")]), default=DEFAULT_TASK if first_run else ""
            ).strip()
            first_run = False
        except KeyboardInterrupt:
            if interrupt():
                return
            continue
        except EOFError:
            print("\nGoodbye.")
            return
        if not text:
            continue
        interrupted = False  # A new command begins a fresh interrupt sequence.
        command, _, arg = text.partition(" ")
        if command in ("/quit", "/exit"):
            return
        if command == "/login":
            login()
            continue
        if command == "/help":
            print(
                f"  /model (choose interactively)  /actions 1–48  /seconds 1–{MAX_SECONDS}\n"
                "  Shortcut: /model beta; /model <id> runs any other model, such as a fine-tune\n"
                "  /login (save/replace API key)  /view  /reset  /reset_robot  /collisions  /status  /quit\n"
                "  Otherwise enter a natural-language robot task. First Ctrl-C stops a run; second Ctrl-C exits."
            )
            continue
        if command == "/status":
            print(f"  {model} · {actions} actions per request · {seconds}s per trial")
            continue
        if command == "/model" and not arg.strip():
            print("  Choose a model:")
            print(f"  1. Beta     {'(current)' if model == BETA_MODEL else ''}")
            if model != BETA_MODEL:
                print(f"     Current: {model}")
            print("  Or type a model id, such as one of your fine-tunes.")
            try:
                arg = input("  Model [1/<id>, Enter to keep current]: ").strip()
            except KeyboardInterrupt:
                if interrupt():
                    return
                continue
            except EOFError:
                print("  Selection cancelled.")
                continue
            if not arg:
                continue
        if command in ("/model", "/actions", "/seconds"):
            try:
                if command == "/model":
                    model = resolve_model(arg)
                elif command == "/actions":
                    value = int(arg)
                    if not 1 <= value <= 48:
                        raise ValueError()
                    actions = value
                else:
                    value = int(arg)
                    if not 1 <= value <= MAX_SECONDS:
                        raise ValueError()
                    seconds = value
                next_task = " (from the next task)" if live_status["running"] else ""
                print(f"  Model: {model}{next_task}" if command == "/model" else "  Updated.")
            except ValueError:
                print(
                    "  Choose 1 or beta, or type one model id (no spaces)."
                    if command == "/model"
                    else "  Invalid setting. Type /help for options."
                )
            continue
        if command.startswith("/") and command not in ("/view", "/reset", "/reset_robot", "/collisions"):
            print("  Unknown command. Type /help.")
            continue
        if worker.poll() is not None:
            print("  Viewer closed. Restart yam_mujoco.py to reopen it.")
            return
        message = (
            {"command": command[1:]}
            if command in ("/reset", "/reset_robot", "/collisions")
            else {
                "command": "task",
                "task": text,
                "model": model,
                "actions": actions,
                "seconds": seconds,
                "verbose": args.verbose,
            }
        )
        if command == "/view":
            print("  MuJoCo is already open, with all three live camera views.")
            continue
        if message["command"] == "task" and not is_task(text):
            print("  Type a task in words, such as: put apple into the bowl")
            continue
        if message["command"] == "task" and not settings().get("DREAM_API_KEY"):
            print("  Run /login to configure your API key first.")
            continue
        worker.stdin.write(json.dumps(message) + "\n")
        worker.stdin.flush()
        if message["command"] in ("reset", "reset_robot") and live_status["running"]:
            print("  Stopping the current task.")
        elif message["command"] != "task":
            print("  Sent.")
        elif live_status["running"]:
            print("  Replacing the current task.")


if __name__ == "__main__":
    with patch_stdout():
        main()

"""`python -m dream.check -- LOOP`: when the robot does the wrong thing, is it the loop or the model?

Give it your loop's command after `--` (`python -m dream.check -- python my_loop.py`): once you type "the workspace is
clear", it runs your loop once, as you would with the model, with `DREAM_GATEWAY_URL` set to its own address, passes
the loop's output through and waits for it to end. Without `--`, it prints the address: run your loop once with
`DREAM_GATEWAY_URL` set to it (in the loop's environment only). Open the page it prints. Three checks follow, each with
a side taken out:

- LOOP, no model in it: the address answers with a scripted motion of known size and timing (#dream.check_loop),
  and the state your loop reports back is set against it: every joint and gripper at once, each hand round a circle
  8 cm across, twice, while each joint the circle hardly moves turns 10 degrees out and back, about 15 s, the same
  whether your loop runs the model's joints or its tool poses (`dreamcli robot run` among them); then, only if that
  finds a PROBLEM, each part at fault alone, to name it. The check reads your arm from the loop's first call
  (#dream.check_arm): the hand poses the client's `ik` config key sends, or a YAM's parts; a loop that sends neither
  is refused before anything moves. It passes when nothing moves more than 1 degree while the robot is told to hold,
  each command moves its own joint (any other under 2 degrees), the right way and x0.8 to x1.2 as far, at most 0.25 s
  late, and the loop plays rows within 5% of `--rate`.
- INPUTS, what the model sees: every call your loop makes during LOOP shows on the page (#dream.DebugView, on this
  computer only): each camera live, the state and the prompt, as the model gets them. The check counts how fast your
  loop fills each camera's history, the frames new to each call a second (#dream.check_loop.camera_rate): more than 10%
  from the 5 the model reads is a PROBLEM, and calls too far apart to count leave a reminder. Above them the page draws
  the arms in 3D: a ghost where the check commands each arm, and a solid arm at the pose your loop reported at its last
  call, with each hand's commanded circle (on an arm the check does not know, each hand's tool point only); live while
  the arms move, then replayed. Once the arms stop, the person who looked types yes when each camera is on its own key,
  upright and in natural colours, the state is in radians and the prompt is the task's; that records the check the
  client reminds you of (#dream.input_check).
- MODEL, no loop in it, only with `--recording`: with nothing moving, #MOMENTS moments of one episode of that
  recording (one session of your recordings, in the layout fine-tuning takes) go through the checkpoint your loop
  named, each with what your loop would send then (#dream.check_model), each hand's pose too when your loop sent it
  (the client adds it with the `ik` config key).
  Each 48-row chunk is set against what you commanded over the same rows. It passes when the model's squared error,
  summed over every moment, row and number, is at most #MODEL_LIMIT of holding still's (repeating what you commanded
  at each moment). Without `--recording` the MODEL line is a SKIP (#UNRECORDED).

The last four lines are the verdict:

    LOOP:   PASS | PROBLEM <what the loop got wrong>
    INPUTS: PASS | PROBLEM | SKIP (no answer)
    MODEL:  PASS | PROBLEM <how far the model is from your demos> | SKIP (no --recording)
    FAULT:  your control loop | your loop's inputs | the model (DreamLabs) | none found

A LOOP or INPUTS PROBLEM is yours: that motion had no model in it, and the page shows what your loop sent. A MODEL
PROBLEM after both pass is ours: your recordings reached the model with no loop in them; unless the model refused what
it was sent (#REFUSED), which is your loop's inputs, not the model. The arms move during LOOP, so
the command first asks you to type "the workspace is clear", with a hand on the e-stop. An agent running it for you
types those words only after the person at the robot has typed them to it; the INPUTS answer is that person's, never
an agent's.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Mapping, Sequence

import numpy as np

from dream import _window, check_arm, check_loop, check_model, input_check
from dream.debug_view import DebugView
from dream.errors import DreamAuthError, DreamBadRequest, DreamConfigError, DreamError

__all__ = ["MODEL_LIMIT", "ask_inputs", "loop_verdict", "main", "model_verdict"]

## The command, as its lines name it.
COMMAND = "python -m dream.check"
## The model's error at most this share of holding still's passes; a fine-tune that has not learned its robot
## scores near 1.
MODEL_LIMIT = 0.85
## Moments of the episode the model is asked at.
MOMENTS = 8
## The MODEL line's start when the model could not be asked at all (#model_line).
UNASKED = "MODEL:  SKIP    could not ask the model:"
## The MODEL line when no recording was given: the model is not asked.
UNRECORDED = "MODEL:  SKIP    no recording given; add --recording <folder> to check the model too"
## The MODEL line's start when the recording is not in the layout the check reads (#model_line).
UNREAD = "MODEL:  SKIP    could not read your recording:"
## The MODEL line's start when the model refused what it was sent (`DA-4000`): no verdict on the model (#fault).
REFUSED = "MODEL:  PROBLEM the model refused what your loop sends:"
## The port of the page of what the model sees.
VIEW_PORT = 8643
## A YAM's two arms in a recording's `observation.state` and `action`, as fine-tuning cuts them.
YAM_PARTS = {"right_arm": [0, 6], "right_gripper": [6, 7], "left_arm": [7, 13], "left_gripper": [13, 14]}


def loop_verdict(lines: Sequence[check_loop.Line]) -> str:
    """The LOOP line: PASS when every measurement passed, else the first that did not, in its own words.

    @param lines #dream.check_loop.judge()'s lines.
    @return The line.
    """
    bad = [line for line in lines if line.status != "PASS"]
    if bad:
        return f"LOOP:   PROBLEM {bad[0].check}: {bad[0].detail}"
    return "LOOP:   PASS    your loop moved the arms as told"


def model_verdict(
    chunks: Sequence[tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]], error: DreamError | None
) -> str:
    """The MODEL line: the model's squared error over every moment against holding still's.

    @param chunks Per moment answered, `(model's rows, your rows)`, each per part `[48, width]`.
    @param error Why the model stopped answering, or None.
    @return The line: #REFUSED with the model's words when it refused the request; #UNASKED when it refused the key
        (`DA-1000`), no verdict on the model.
    """
    if isinstance(error, DreamBadRequest):
        return f"{REFUSED} {error}"
    if isinstance(error, DreamAuthError):
        return f"{UNASKED} {error}"
    if error is not None:
        return f"MODEL:  PROBLEM the model answered {len(chunks)} of {MOMENTS} calls, then: {error}"
    miss: dict[str, float] = {}
    hold: dict[str, float] = {}
    for got, want in chunks:
        for part in got:
            if check_loop._kind(part) == "rot":
                continue
            miss[part] = miss.get(part, 0.0) + float(np.sum((got[part] - want[part]) ** 2))
            hold[part] = hold.get(part, 0.0) + float(np.sum((want[part][0] - want[part]) ** 2))
    if not miss:
        return "MODEL:  PROBLEM the model's chunks carry none of the recording's action parts"
    if sum(hold.values()) == 0:
        return "MODEL:  SKIP    the arms hold still at the moments asked; give an episode where they move (--episode)"
    ratio = sum(miss.values()) / sum(hold.values())
    worst = max(miss, key=lambda part: miss[part] / max(hold[part], 1e-9))
    said = f"the model's error is x{ratio:.2f} of holding still's (limit x{MODEL_LIMIT:.2f})"
    if ratio > MODEL_LIMIT:
        return f"MODEL:  PROBLEM {said}; worst: {worst} (x{miss[worst] / max(hold[worst], 1e-9):.2f})"
    return f"MODEL:  PASS    {said}, over {len(chunks)} moments"


def model_line(
    session: pathlib.Path, checkpoint: str, cameras: Sequence[str], poses: bool, yam: bool, episode: int
) -> str:
    """The MODEL line: asks the model (#ask_model) and judges its answers (#model_verdict).

    A failure before the first call (a key from another site, the platform's model listing unreachable) is no verdict
    on the model: the line is #UNASKED with why, and the check exits 1. So is a recording it cannot read
    (#dream.check_model.Unreadable), found before any request: the line is #UNREAD with why.

    @param session The recording (`--recording`).
    @param checkpoint The model the loop named.
    @param cameras The camera roles the loop sent.
    @param poses Whether the loop sent each hand's pose (#dream.check_loop.StandIn.poses).
    @param yam True when the arms are YAMs (#ask_model).
    @param episode The episode_index.
    @return The line.
    """
    try:
        check_model.readable(session, episode)
        chunks, error = ask_model(session, checkpoint, cameras, poses, yam, episode)
    except DreamError as failure:
        return f"{UNASKED} {failure}"
    except check_model.Unreadable as failure:
        return f"{UNREAD} {failure}"
    return model_verdict(chunks, error)


def ask_inputs(inputs: str | None, url: str, rate: check_loop.Line) -> str:
    """The INPUTS line, from the camera rate and the person who looked at the page.

    A camera history at the wrong rate is a PROBLEM, with no question; otherwise the person's yes records the check
    (#dream.input_check).

    @param inputs The inputs of the loop's first call (#dream.check_loop.StandIn.inputs); None when it made none.
    @param url The page of what the model sees.
    @param rate How fast the loop filled each camera's history (#dream.check_loop.camera_rate).
    @return The line.
    """
    if inputs is None:
        return "INPUTS: SKIP    your loop made no call"
    if rate.status == "PROBLEM":
        return f"INPUTS: PROBLEM {rate.detail}"
    print(
        f"\nLook at {url}: is each camera under its own name, upright and in natural colours, the state in radians, "
        "and the prompt the task's?\nThe person who looked answers, never an agent.",
        flush=True,
    )
    try:
        answer = input("Type yes or no: ").strip().lower()
    except EOFError:  # no terminal to type in
        answer = ""
    if answer == "yes":
        input_check.record(inputs)
        return f"INPUTS: PASS    you checked what the model sees (recorded on this computer as {inputs})"
    if answer == "no":
        return "INPUTS: PROBLEM you saw a camera, the state or the prompt wrong on the page"
    return f"INPUTS: SKIP    no answer; the person who looked records it: {input_check.COMMAND} {inputs}"


def fault(loop: str, inputs: str, model: str) -> str:
    """The FAULT line, from the LOOP, INPUTS and MODEL lines.

    @param loop The LOOP line.
    @param inputs The INPUTS line.
    @param model The MODEL line.
    @return The FAULT line: whose fault a PROBLEM is, or none found.
    """
    if loop.startswith("LOOP:   PROBLEM"):
        return "FAULT:  your control loop (fix the LOOP line, then run this again)"
    if inputs.startswith("INPUTS: PROBLEM"):
        return "FAULT:  your control loop (fix the INPUTS line, then run this again)"
    if model.startswith(REFUSED):
        return "FAULT:  your loop's inputs: the model refused them (see the MODEL line)"
    if model.startswith("MODEL:  PROBLEM"):
        return "FAULT:  the model (DreamLabs): send us these lines"
    if model.startswith("MODEL:  SKIP"):
        return "FAULT:  not your loop; the model was not checked"
    return "FAULT:  none found"


def ask_model(
    session: pathlib.Path, checkpoint: str, cameras: Sequence[str], poses: bool, yam: bool, episode: int
) -> tuple[list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]], DreamError | None]:
    """Asks the model at #MOMENTS moments of one recorded episode, as check_model does.

    @param session The folder of recordings.
    @param checkpoint The model the loop named.
    @param cameras The camera roles the loop sent; each is the recording's `observation.images.<role>`.
    @param poses Whether the loop sent each hand's pose: the client adds them here too, as the docs' YAM loop does.
    @param yam True when the arms are YAMs: a checkpoint the platform lists no body for is then read as a YAM pair.
    @param episode The episode_index.
    @return Per moment answered, `(model's rows, your rows)`; and why the model stopped answering, or None.
    """
    import dream

    # The reminder to look at what the model sees is for a live loop; these frames are your recordings.
    warnings.filterwarnings("ignore", message=input_check.REMINDER)
    with dream.DreamClient() as client:
        listed = client._embodiment_config_for(checkpoint)
        if listed is None and not yam:
            raise DreamConfigError(
                f"the platform lists no body for {checkpoint!r}, and on an arm that is not a YAM the check reads your "
                "recording's parts from the body of a fine-tune of that arm: name one in your loop"
            )
        embodiment = listed or {
            "state": YAM_PARTS,
            "action": YAM_PARTS,
            "video": {role: f"observation.images.{role}" for role in cameras},
        }
        recording = check_model.read_episode(session, episode, embodiment["video"])
        rows = check_model.moments(len(recording.state), recording.fps, MOMENTS)
        step = 1 / _window.OBS_FPS
        ticks = {row: [row / recording.fps - k * step for k in range(_window.OBS_WINDOW - 1, -1, -1)] for row in rows}
        needed = {max(0, round(tick * recording.fps)) for times in ticks.values() for tick in times}
        frames = {role: check_model._frames(path, needed) for role, path in recording.videos.items()}
        config = {"checkpoint": checkpoint, "embodiment": embodiment}
        if poses and yam:
            config["ik"] = {"arm": "yam", "head": "joint"}
        client.policy(config)
        chunks: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = []
        for row in rows:
            try:
                action = check_model.ask(client, recording, embodiment["state"], row, ticks[row], frames)
            except DreamError as failure:
                return chunks, failure
            got = check_model.predicted(action, list(embodiment["action"]))
            chunks.append(
                (got, {part: check_model.recorded(recording, embodiment["action"][part], row) for part in got})
            )
    return chunks, None


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m dream.check [--recording FOLDER] -- YOUR LOOP`: the loop and inputs checks, then the model's.

    @param argv The arguments; `sys.argv[1:]` when None. What follows `--` is your loop's command.
    @return 0 when none is a PROBLEM, 1 when one is, the model could not be asked or your loop ended before the motion
        did, 2 for a bad argument or no "the workspace is clear".
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[argv.index("--") + 1 :] if "--" in argv else None
    argv = argv[: argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser(
        prog=COMMAND,
        usage=f"{COMMAND} [options] [-- YOUR LOOP'S COMMAND]",
        description=__doc__.split("\n\n")[0],
        epilog=f'-- YOUR LOOP\'S COMMAND: run it once you type "{check_loop.START}", with DREAM_GATEWAY_URL set to '
        "the check's address (-- python my_loop.py); without it, the check prints the address for you to run your "
        "loop with.",
    )
    parser.add_argument(
        "--recording",
        type=pathlib.Path,
        help="one session of your recordings (meta/, data/, videos/); only MODEL reads it, a SKIP without it",
    )
    parser.add_argument("--episode", type=int, default=0, help="the episode_index the model is asked about (0)")
    parser.add_argument("--port", type=int, default=check_loop.DEFAULT_PORT, help="the port to serve on (8642)")
    parser.add_argument(
        "--view-port", type=int, default=VIEW_PORT, help="the port of the page of what the model sees (8643)"
    )
    parser.add_argument(
        "--rate", type=float, default=check_loop.DEFAULT_RATE, help="rows a second your loop plays (20)"
    )
    args = parser.parse_args(argv)
    if args.recording is not None:
        if not (args.recording / "meta" / "info.json").is_file():
            print(
                f"{COMMAND}: {args.recording}: no meta/info.json; give one session of your recordings", file=sys.stderr
            )
            return 2
        try:
            check_model.readable(args.recording, args.episode)
        except check_model.Unreadable as error:
            print(
                f"{COMMAND}: {args.recording}: {error}\nMODEL will read SKIP; LOOP and INPUTS still run.",
                file=sys.stderr,
            )
    try:
        check_arm.ik_specs()
    except DreamConfigError as error:
        print(f"{COMMAND}: {error}", file=sys.stderr)
        return 2
    if command is not None and (not command or shutil.which(command[0]) is None):
        print(
            f"{COMMAND}: after -- give your loop's command, a program it can run; got {shlex.join(command)!r}",
            file=sys.stderr,
        )
        return 2
    moving = (check_loop.HORIZON + check_loop.TOGETHER_ROWS + check_loop.REST_ROWS) / args.rate
    if not check_loop.confirm(
        f"The arms will move for about {moving:.0f} s (longer if a part is at fault), with no model: each hand "
        f"circles twice, {200 * check_loop.CIRCLE_R:.0f} cm across, while every joint turns."
    ):
        return 2
    with DebugView(None, args.view_port) as view:
        stand_in = check_loop.StandIn(rate=args.rate, view=view)
        view.show_arms(stand_in.arms)
        url = stand_in.serve(args.port)
        if command is None:
            print(
                f"\nRun your loop once, as with the model, with DREAM_GATEWAY_URL={url}\n"
                f"Watch the arms and what the model is sent: {view.url}",
                flush=True,
            )
            loop_run = None
        else:
            print(
                f"\nRunning your loop with DREAM_GATEWAY_URL={url}: {shlex.join(command)}\n"
                f"Watch the arms and what the model is sent: {view.url}",
                flush=True,
            )
            # The client's reminder to run this check would print inside the check itself.
            quiet = ",".join(filter(None, (os.environ.get("PYTHONWARNINGS"), f"ignore:{input_check.REMINDER}")))
            loop_run = subprocess.Popen(command, env={**os.environ, "DREAM_GATEWAY_URL": url, "PYTHONWARNINGS": quiet})
        threading.Thread(target=check_loop.watch, args=(stand_in,), daemon=True).start()
        while not stand_in.done.wait(0.25):
            if loop_run is not None and loop_run.poll() is not None:  # it ended before the motion did
                stand_in.close()
                loop = (
                    f"LOOP:   PROBLEM your loop exited ({loop_run.returncode}) before the motion ended; see its lines"
                )
                skip = "SKIP    your loop ended before the motion did"
                inputs, model = f"INPUTS: {skip}", f"MODEL:  {skip}"
                print(f"\n{loop}\n{inputs}\n{model}\n{fault(loop, inputs, model)}")
                return 1
        time.sleep(0.5)
        stand_in.close()
        if loop_run is not None:
            loop_run.wait()
        lines = check_loop.judge(stand_in)
        rate = check_loop.camera_rate(stand_in)
        for line in [*lines, rate]:
            print(f"  {line}")
        if stand_in.plan is not None:
            trace = check_loop.trace_path()
            check_loop.save_trace(stand_in, [*lines, rate], trace)
            print(f"  The run's trace (the check's targets and every call): {trace}")
        loop = loop_verdict(lines)
        inputs = ask_inputs(stand_in.inputs, view.url, rate)
    if args.recording is None:
        model = UNRECORDED
    elif stand_in.checkpoint is None:
        model = "MODEL:  SKIP    the loop's calls named no checkpoint"
    else:
        print(f"Asking {stand_in.checkpoint} about {MOMENTS} moments of episode {args.episode}; nothing moves.")
        yam = stand_in.arm == "yam"
        model = model_line(args.recording, stand_in.checkpoint, stand_in.cameras, stand_in.poses, yam, args.episode)
    print(f"\n{loop}\n{inputs}\n{model}\n{fault(loop, inputs, model)}")
    return 1 if "PROBLEM" in loop + inputs + model or model.startswith((UNASKED, UNREAD)) else 0


if __name__ == "__main__":
    sys.exit(main())

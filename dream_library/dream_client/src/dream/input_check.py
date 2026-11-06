"""The record that a person checked what the model sees on this computer, and the reminder until they have.

A loop's inputs are right when a person has looked at them: each camera on its own key, upright, in natural colours;
the state in radians; the prompt (`python -m dream.check` shows them and asks; so does #dream.DebugView). Until someone
has, the client prints one line per run, at the run's first request: it names `python -m dream.check`, which records
the check when the person who looked types yes, and the command that records a check made another way:

    python -m dream.input_check <id>

`<id>` is what this robot sends the model, without the values: each camera's key and frame size, each state key's
width and the robot the request names. A camera added, renamed or resized, a part added, or another robot (`ik` sends
the hands' poses too) is another id, so the reminder comes back for it. `python -m dream.check` records the same id
when the person who looked types yes. The record is a file beside the credentials file
(`~/.dreamlabs/inputs_checked`, one id a line), so it is this computer's. Nothing is sent anywhere for it.
"""

from __future__ import annotations

import base64
import binascii
import datetime
import hashlib
import json
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dream import credentials
from dream.arm_file import _warn

__all__ = ["checked", "inputs_id", "main", "record", "remind"]

## The record's name, in the credentials file's folder.
FILE_NAME = "inputs_checked"
## The request observation's key that names the robot.
ROBOT_KEY = "embodiment"
## The command that records the check, before its id.
COMMAND = "python -m dream.input_check"
## The reminder's first words (#remind), which a warnings filter names: no comma or colon.
REMINDER = "no one has checked what the model sees"


def inputs_id(observation: Mapping[str, Any]) -> str:
    """This robot's inputs as an id: each camera's key and frame size, each state key's width, the robot named.

    The values (the frames' pixels, the joints, the prompt) are left out, so every request of a loop has the same id.

    @param observation A `/v1/get_action` request's `observation`: `video.<role>` as one base64 JPEG or a list of them,
        oldest first; `state.<part>` as numbers; the robot under #ROBOT_KEY when the request names one.
    @return 12 hex digits.
    """
    cameras, state = {}, {}
    for key, value in observation.items():
        if not isinstance(key, str):  # not a key a request can carry: sending it raises the client's own error
            continue
        if key.startswith("video."):
            last = (value[-1] if value else None) if isinstance(value, list) else value
            cameras[key] = _jpeg_size(last) if isinstance(last, str) else None
        elif key.startswith("state."):
            state[key] = int(np.size(value))
    robot = observation[ROBOT_KEY] if ROBOT_KEY in observation else None
    text = json.dumps({"cameras": cameras, "state": state, "robot": robot}, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def checked(inputs: str) -> bool:
    """Whether a person recorded the check of these inputs on this computer.

    @param inputs The id, inputs_id().
    @return True when the record holds it.
    """
    path = _path()
    return path.is_file() and any(line.split(" ", 1)[0] == inputs for line in path.read_text().splitlines())


def record(inputs: str) -> pathlib.Path:
    """Records that a person checked these inputs: one line, the id and when.

    @param inputs The id the reminder named.
    @return The record's file.
    @throws ValueError `inputs` is not an id: 12 hex digits.
    """
    if len(inputs) != 12 or any(c not in "0123456789abcdef" for c in inputs):
        raise ValueError(f"{inputs!r} is not an inputs id: copy the 12 hex digits the reminder names")
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as out:
        out.write(f"{inputs} {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}\n")
    return path


def remind(observation: Mapping[str, Any]) -> None:
    """Prints the reminder, one `UserWarning` line, when no one has recorded the check of these inputs here.

    @param observation The `observation` of a policy's first request, as sent.
    """
    inputs = inputs_id(observation)
    if checked(inputs):
        return
    _warn(
        f"{REMINDER} for these cameras and this robot: run python -m dream.check before the arms move for "
        f"real. Checked another way? The person who looked (never an agent) records it: {COMMAND} {inputs}",
        every_time=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """`python -m dream.input_check <id>`: records the check the reminder asked for.

    @param argv The arguments; `sys.argv[1:]` when None.
    @return The exit status: 0 recorded, 2 when the id is missing or malformed.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(f"usage: {COMMAND} <id>  (the id the reminder names)", file=sys.stderr)
        return 2
    try:
        path = record(args[0])
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    print(f"recorded in {path}: no reminder for these inputs on this computer until a camera or the robot changes")
    return 0


def _path() -> pathlib.Path:
    """The record: #FILE_NAME beside the credentials file."""
    return credentials.file_path().parent / FILE_NAME


def _jpeg_size(frame: str) -> list[int] | None:
    """A base64 JPEG's width and height from its frame header, without decoding it; None when it has none."""
    try:
        data = base64.b64decode(frame, validate=False)
    except (binascii.Error, ValueError):
        return None
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            return None
        marker, length = data[i + 1], int.from_bytes(data[i + 2 : i + 4], "big")
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):  # a start of frame: height, then width
            return [int.from_bytes(data[i + 7 : i + 9], "big"), int.from_bytes(data[i + 5 : i + 7], "big")]
        i += 2 + length
    return None


if __name__ == "__main__":
    sys.exit(main())

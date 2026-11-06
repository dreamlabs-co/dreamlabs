"""What a fine-tune was trained on, held against what a run sends it: each camera's aspect, and the prompt.

A fine-tune learned from the frames and tasks of its dataset. The model resizes every frame to one input size, so a
camera sent at another aspect than its training videos (a square crop, a letterbox) shows the model shapes it never
saw; a prompt in other words than the dataset's tasks is one it was never trained to follow. A frame's size alone is
not a line: at the training aspect, a larger or smaller frame becomes the same input.

DreamClient.policy() reads, once per checkpoint, what one of your fine-tunes was trained on (fetch(), from
`GET /v1/fine_tuning/models/<id>/training_data`); each episode's first get_action() holds its frames and prompt to it
(notes()), prints each line once a policy as a `UserWarning` that never refuses, and hands the episode's lines to
#dream.DebugView with each of its round trips (`RoundTrip.trained`). Only a fine-tune's id (#FINE_TUNE_PREFIX) is
looked up, so a public model costs no request, and one that is not yours answers nothing to compare.
"""

from __future__ import annotations

import difflib
import re
import urllib.parse
from collections.abc import Mapping, Sequence
from typing import Any

from dream import datasets
from dream.errors import DreamServerError

__all__ = ["FINE_TUNE_PREFIX", "fetch", "notes"]

## What every fine-tune's id starts with (`fine_tuned_model`, `dream-1-ft:<uid8>-<suffix>`), as the gateway tells a
## fine-tune from a public model: only such a checkpoint is looked up.
FINE_TUNE_PREFIX = "dream-1-ft:"

## How far a frame's aspect (width over height) may be from a training camera's, as a share of it, before it is a
## line: 640 x 360 against 320 x 176 is within it.
ASPECT_TOLERANCE = 0.05

## How alike a training prompt must be to the one sent (difflib's ratio) to be named as its closest.
CLOSE_PROMPT = 0.6


def fetch(checkpoint: str, api_key: str | None) -> dict[str, Any] | None:
    """What one of your fine-tunes was trained on: each camera role's frame sizes and its dataset's tasks.

    One request, not repeated: policy() is not held up by a data service that does not answer.

    @param checkpoint The policy's checkpoint.
    @param api_key Your API key; the credentials' when None.
    @return `{"cameras": {role: [[width, height], ...]}, "prompts": [task, ...]}`; None when the checkpoint is not
        one of your fine-tunes (a public model), or the data service predates the two fields.
    @throws DreamError The data service could not be reached, or refused for another reason.
    """
    api_key, host = datasets._resolve(api_key)
    path = f"/v1/fine_tuning/models/{urllib.parse.quote(checkpoint, safe='')}/training_data"
    try:
        answer = datasets._call("GET", host, path, api_key)
    except DreamServerError as error:
        if error.code == "not_found":
            return None
        raise
    if not isinstance(answer, dict) or "cameras" not in answer or "prompts" not in answer:
        return None
    return {"cameras": answer["cameras"], "prompts": answer["prompts"]}


def notes(
    trained: Mapping[str, Any], checkpoint: str, frames: Mapping[str, Sequence[str]], prompt: object
) -> list[str]:
    """The lines for one call's frames and prompt against what the checkpoint was trained on; none when they match.

    @param trained fetch()'s answer for the checkpoint.
    @param checkpoint The checkpoint, named in each line.
    @param frames Per frame key (`video.<role>`), the base64 JPEGs the call sends; the newest one's size is read.
    @param prompt The call's `prompt`, or None when it sends none.
    @return One line per camera whose aspect is off every size its role was trained at, then one for a prompt that
        is none of the training tasks (case, spacing and end punctuation aside).
    """
    # Deferred: `python -m dream.input_check` runs that module after the package loads; the package never imports it.
    from dream.input_check import _jpeg_size

    out = []
    for key, window in frames.items():
        role = key.removeprefix("video.")
        sizes = trained["cameras"].get(role)
        size = _jpeg_size(window[-1]) if window else None
        if not key.startswith("video.") or not sizes or size is None:
            continue
        width, height = size
        if all(abs((width / height) / (w / h) - 1) > ASPECT_TOLERANCE for w, h in sizes):
            shown = " or ".join(f"{w} x {h} (aspect {w / h:.2f})" for w, h in sizes)
            out.append(
                f"{key} frames are {width} x {height} (aspect {width / height:.2f}), but {checkpoint} was trained on "
                f"{shown}: the model resizes each frame to one input size, so it sees shapes it never learned; send "
                "this camera at its training aspect, cropped from the full frame, not padded or letterboxed"
            )
    tasks = {_plain(task): task for task in trained["prompts"]}
    if isinstance(prompt, str) and tasks and _plain(prompt) not in tasks:
        close = difflib.get_close_matches(_plain(prompt), list(tasks), n=1, cutoff=CLOSE_PROMPT)
        some = next(iter(tasks.values()))
        hint = f"the closest is {tasks[close[0]]!r}" if close else f"{len(tasks)} tasks, such as {some!r}"
        out.append(
            f"the prompt {prompt!r} is none of the tasks {checkpoint} was trained on ({hint}): a fine-tune follows "
            "the words of its training tasks best"
        )
    return out


def _plain(text: str) -> str:
    """A prompt as compared: lower case, single spaces, no punctuation at its ends."""
    return re.sub(r"\s+", " ", text).strip().strip(".!?,;:").strip().casefold()

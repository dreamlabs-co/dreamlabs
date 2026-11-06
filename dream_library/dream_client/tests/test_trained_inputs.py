"""Tests of `dream.trained_inputs`: a run's frames and prompt held to what its fine-tune was trained on.

A camera sent square or letterboxed to a fine-tune trained on 4:3 frames, and a prompt in other words than the
training tasks, each print one line at an episode's first call, once a policy, never refusing, and ride on every
round trip of the episode to #dream.DebugView. A public model, a matching run and a lookup that fails print nothing
more than that.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from test_input_check import jpeg

import dream
from dream import datasets, trained_inputs
from dream.debug_view import DebugView, RoundTrip
from dream.errors import DreamProtocolError, DreamServerError

## What the fine-tune in these tests was trained on: 4:3 cameras, one task.
TRAINED: dict[str, Any] = {"cameras": {"top": [[640, 480]], "wrist": [[640, 480]]}, "prompts": ["stack the cups"]}

## The fine-tune's id.
CHECKPOINT = "dream-1-ft:0123abcd-cups"

## The policy config: no model lookup and no recording, so only the training data is looked up.
CONFIG = {"checkpoint": CHECKPOINT, "strict": False, "_record": False}


def test_a_camera_off_its_training_aspect_and_a_prompt_outside_the_tasks_are_lines() -> None:
    frames = {"video.top": [jpeg(224, 224)], "video.wrist": [jpeg(1280, 960)]}
    lines = trained_inputs.notes(TRAINED, CHECKPOINT, frames, "stack-cups")
    assert lines == [
        "video.top frames are 224 x 224 (aspect 1.00), but dream-1-ft:0123abcd-cups was trained on 640 x 480 "
        "(aspect 1.33): the model resizes each frame to one input size, so it sees shapes it never learned; send this "
        "camera at its training aspect, cropped from the full frame, not padded or letterboxed",
        "the prompt 'stack-cups' is none of the tasks dream-1-ft:0123abcd-cups was trained on (the closest is "
        "'stack the cups'): a fine-tune follows the words of its training tasks best",
    ]


def test_a_matching_run_has_no_line() -> None:
    # Another size at the training aspect, within 5% of it, and the task in other case and punctuation.
    frames = {"video.top": [jpeg(320, 240)], "video.wrist": [jpeg(656, 480)], "video.extra": [jpeg(10, 90)]}
    assert trained_inputs.notes(TRAINED, CHECKPOINT, frames, "Stack the cups.") == []
    assert trained_inputs.notes(TRAINED, CHECKPOINT, frames, None) == [], "no prompt sent: nothing to compare"
    far = trained_inputs.notes(TRAINED, CHECKPOINT, {}, "open the drawer")
    assert far == [
        "the prompt 'open the drawer' is none of the tasks dream-1-ft:0123abcd-cups was trained on (1 tasks, such "
        "as 'stack the cups'): a fine-tune follows the words of its training tasks best"
    ]


def test_fetch_reads_the_training_data_and_a_public_model_has_none(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def call(method: str, host: str, path: str, api_key: str, body: Any = None, *, retry: bool = False) -> Any:
        calls.append((path, str(retry)))
        if "dreamlabs" in path:
            raise DreamServerError("No fine-tuning job of yours made it.", code="not_found")
        return {"url": "https://x", "video": {"top": "observation.images.top"}, **TRAINED}

    monkeypatch.setattr(datasets, "_call", call)
    assert trained_inputs.fetch(CHECKPOINT, "k") == TRAINED
    assert trained_inputs.fetch("dreamlabs/dream-1-beta", "k") is None
    assert calls == [
        ("/v1/fine_tuning/models/dream-1-ft%3A0123abcd-cups/training_data", "False"),
        ("/v1/fine_tuning/models/dreamlabs%2Fdream-1-beta/training_data", "False"),
    ], "one request each, never retried: policy() is not held up"
    monkeypatch.setattr(datasets, "_call", lambda *args, **kwargs: {"url": "https://x", "video": {}})
    assert trained_inputs.fetch(CHECKPOINT, "k") is None, "a service that predates the fields has nothing to compare"


class FakePolicy:
    """A stand-in policy client: answers one chunk."""

    def __init__(self) -> None:
        self.last_timing: dict[str, Any] | None = None

    def get_action(self, observation: dict[str, Any], *, commanded: Any = None) -> dict[str, Any]:
        return {"q_r_chunk": np.zeros((48, 7))}

    def reset_episode(self) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A DreamClient whose policies answer from #FakePolicy, with the reminder of `dream.input_check` recorded."""
    monkeypatch.setattr(dream, "_open_policy", lambda config, api_key=None: FakePolicy())
    monkeypatch.setattr("dream.input_check.checked", lambda inputs: True)
    opened = dream.DreamClient(api_key="k")
    yield opened
    opened.close()


def _calls(client: Any, n: int, prompt: str) -> list[str]:
    """`n` get_action() calls on a square top camera; the lines they printed."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in range(n):
            client.set_observation({"video.top": jpeg(224, 224)}, {"prompt": prompt, "state.arm": [0.0]}, timestamp=i)
            client.get_action()
    return [str(w.message) for w in caught]


def test_each_line_prints_once_a_policy_and_rides_on_every_round_trip(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    looked_up: list[str] = []
    monkeypatch.setattr(trained_inputs, "fetch", lambda checkpoint, api_key: looked_up.append(checkpoint) or TRAINED)
    trips: list[RoundTrip] = []
    client.on_round_trip(trips.append)
    client.policy(CONFIG)
    printed = _calls(client, 3, "stack-cups")
    assert len(printed) == 2 and printed[0].startswith("video.top frames are 224 x 224")
    assert [len(trip.trained) for trip in trips] == [2, 2, 2]
    client.reset_episode()
    assert _calls(client, 1, "stack the cups") == [], "a line already printed this policy is not printed again"
    assert len(trips[-1].trained) == 1, "the new episode's own lines: the camera alone"
    client.policy(CONFIG)
    assert len(_calls(client, 1, "stack-cups")) == 2, "a new policy prints again"
    assert looked_up == [CHECKPOINT], "looked up once per checkpoint"


def test_a_public_model_has_no_line_and_a_failed_lookup_says_so_once(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    looked_up: list[str] = []
    monkeypatch.setattr(trained_inputs, "fetch", lambda checkpoint, api_key: looked_up.append(checkpoint))
    client.policy({**CONFIG, "checkpoint": "dreamlabs/dream-1-beta"})
    assert _calls(client, 2, "anything") == []
    assert looked_up == [], "a public model's id is not looked up"
    client.policy(CONFIG)
    assert _calls(client, 1, "anything") == [] and looked_up == [CHECKPOINT], "not yours: nothing to compare"

    def down(checkpoint: str, api_key: str | None) -> Any:
        raise DreamProtocolError("cannot reach the Dream service")

    monkeypatch.setattr(trained_inputs, "fetch", down)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        client.policy({**CONFIG, "checkpoint": "dream-1-ft:0123abcd-other"})
    assert [str(w.message) for w in caught] == [
        "could not read what dream-1-ft:0123abcd-other was trained on, so this policy's cameras and prompt are not "
        "compared with it: cannot reach the Dream service"
    ]
    assert _calls(client, 1, "stack-cups") == [], "the policy runs, uncompared"


def test_debug_view_shows_the_lines() -> None:
    view = DebugView(None)
    try:
        trip = RoundTrip(
            sent_s=1.0, frames={}, observation={}, action=None, timing=None, error=None, trained=["a line"]
        )
        view._add(trip)
        assert view._one(1)["trained"] == ["a line"]
        old = RoundTrip(1.0, {}, {}, None, None, None)
        assert old.trained == [], "a RoundTrip built by position before the field still builds"
    finally:
        view.close()

"""Tests of `dream.input_check`: the reminder a policy's first request prints until a person records that they
checked what the model sees, the id it names, and `python -m dream.input_check <id>`, which records the check.

The record sits beside the credentials file, which conftest points at a temporary folder, so no test reads or writes
the developer's own `~/.dreamlabs`.
"""

from __future__ import annotations

import base64
import subprocess
import sys
import warnings
from typing import Any

import pytest
from conftest import FakeDreamAPI

import dream
from dream import credentials, input_check

## The policy config every test opens: no model lookup (`strict` off) and no recording, so nothing is dialed.
CONFIG = {"checkpoint": "c", "strict": False, "_record": False}


def jpeg(width: int, height: int) -> str:
    """A base64 JPEG head that says `width` x `height`: what a frame's size is read from (no picture data needed)."""
    app0 = b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof0 = b"\xff\xc0\x00\x11\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big") + b"\x03" + b"\x00" * 9
    return base64.b64encode(b"\xff\xd8" + app0 + sof0 + b"\xff\xd9").decode()


@pytest.fixture
def client(server: FakeDreamAPI) -> Any:
    """A DreamClient whose policies call the fake gateway over HTTP."""
    opened = dream.DreamClient(api_key="k")
    yield opened
    opened.close()


def first_calls(client: Any, frame: str) -> list[str]:
    """Opens a policy and calls get_action() twice on `frame`; the reminder lines those calls printed."""
    client.policy(CONFIG)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for timestamp in (1.0, 1.1):
            client.set_observation({"video.top": frame}, {"prompt": "pick", "state.arm": [0.0]}, timestamp=timestamp)
            client.get_action()
    return [str(w.message) for w in caught if "what the model sees" in str(w.message)]


def test_the_first_call_of_each_policy_reminds_until_the_check_is_recorded(client: Any) -> None:
    """One line at a policy's first call, none at its second; a new policy (a station's next run) reminds again;
    once the id it names is recorded, no policy on these inputs reminds."""
    [line] = first_calls(client, jpeg(320, 176))
    inputs = input_check.inputs_id({"video.top": [jpeg(320, 176)], "prompt": "pick", "state.arm": [0.0]})
    assert line.endswith(f"The person who looked (never an agent) records it: python -m dream.input_check {inputs}")
    assert "run python -m dream.check before the arms move for real" in line
    assert len(first_calls(client, jpeg(320, 176))) == 1
    assert input_check.main([inputs]) == 0
    assert input_check.checked(inputs)
    assert first_calls(client, jpeg(320, 176)) == []


def test_every_run_reminds_under_pythons_default_filter(client: Any) -> None:
    """Python's default filter shows one text from one line once a program; a station's runs share one program, so the
    reminder bypasses that: each policy's first call prints it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for timestamp in (1.0, 2.0):
            client.policy(CONFIG)
            client.set_observation({"video.top": jpeg(320, 176)}, {"prompt": "pick"}, timestamp=timestamp)
            client.get_action()
    assert len([w for w in caught if "what the model sees" in str(w.message)]) == 2


def test_the_command_prints_only_its_own_lines() -> None:
    """The package never imports the module, so `python -m dream.input_check` adds no warning of Python's own."""
    done = subprocess.run([sys.executable, "-m", "dream.input_check"], capture_output=True, text=True, check=False)
    usage = "usage: python -m dream.input_check <id>  (the id the reminder names)\n"
    assert (done.returncode, done.stderr) == (2, usage)


def test_a_resized_or_renamed_camera_another_part_or_robot_is_another_id() -> None:
    """The id follows the camera keys and sizes, the state keys and widths and the robot named, and not the values."""
    sent = {"video.top": [jpeg(320, 176)], "state.arm": [0.0] * 6, "prompt": "pick"}
    base = input_check.inputs_id(sent)
    assert base == input_check.inputs_id({**sent, "state.arm": [0.5] * 6, "prompt": "stack"})
    assert base == input_check.inputs_id({**sent, "video.top": jpeg(320, 176)})
    assert base == input_check.inputs_id({**sent, "video.top": [jpeg(640, 360), jpeg(320, 176)]})
    assert base != input_check.inputs_id({**sent, "video.top": [jpeg(640, 360)]})
    assert base != input_check.inputs_id({"video.overhead": [jpeg(320, 176)], "state.arm": [0.0] * 6})
    assert base != input_check.inputs_id({**sent, "state.arm": [0.0] * 7})
    assert base != input_check.inputs_id({**sent, "state.right_pose": [0.0] * 9})
    assert base != input_check.inputs_id({**sent, "embodiment": "yam"})
    assert input_check._jpeg_size(jpeg(320, 176)) == [320, 176]
    assert input_check._jpeg_size("not a jpeg") is None


def test_the_command_records_only_an_id_and_says_how_to_use_it(capsys: pytest.CaptureFixture[str]) -> None:
    """`python -m dream.input_check` takes exactly one 12-hex-digit id; anything else exits 2 and records nothing."""
    assert input_check.main([]) == 2
    assert input_check.main(["xyz"]) == 2
    assert input_check.main(["0123456789ab", "extra"]) == 2
    assert "usage: python -m dream.input_check <id>" in capsys.readouterr().err
    assert not input_check.checked("0123456789ab")
    assert input_check.main(["0123456789ab"]) == 0
    assert "recorded in" in capsys.readouterr().out
    path = input_check._path()
    assert path.parent == credentials.file_path().parent
    [line] = path.read_text().splitlines()
    assert line.startswith("0123456789ab ")

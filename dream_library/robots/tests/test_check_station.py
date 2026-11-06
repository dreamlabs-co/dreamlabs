"""`python -m dream.check` on `dreamcli robot run`, the stations' own loop, on fake motors: a Metal pair and a
YAM pair, each under the check's stand-in with the client's real IK. The check reads each arm from the run's first
call and answers its joints and its hand poses alike, so a run on the eef head (the stations' default) and one on the
joint head pass every check with nothing set. Each run plays the whole motion in real time, about 20 s."""

import asyncio
import dataclasses
import time
from typing import Any

import conftest
import numpy as np
import pytest
import test_end_to_end
import test_makers_station
from dream import check_loop
from test_makers_station import CHECKPOINT, EEFDreamAPI
from test_metal_station import bus, metal_config, solver  # noqa: F401  (fixtures)

from robots import cameras, config, errors, followers, run, yam

## A YAM pose clear of every joint limit, where each fake arm starts and homes.
POSE = np.array([0.0, 1.6, 1.3, 0.0, 0.3, 0.0])

## The motion at 20 rows a second (about 17 s), with room for the start and the IK's first solve.
RUN_S = 150.0

## The station's own `CameraReader.from_config`, kept before a fixture swaps in its fake (#_cameras).
FROM_CONFIG = cameras.CameraReader.__dict__["from_config"]


def _expect(stand_in: check_loop.StandIn, arm: str) -> None:
    """The check read the run's arms as `arm`, and every line passes, the camera rate among them."""
    lines = [*check_loop.judge(stand_in), check_loop.camera_rate(stand_in)]
    said = "\n".join([stand_in.read_as(), *(str(line) for line in lines)])
    assert stand_in.done.is_set() and stand_in.stopped is None, said
    assert stand_in.arm == arm, said
    assert {line.status for line in lines} == {"PASS"}, said


class Counting:
    """A camera class of your own (a camera's `source`) whose every frame differs, one per period of its `fps`.

    Frame n shows n's 24 low bits as black and white 8-pixel blocks.
    """

    def __init__(self, cfg: config.CameraConfig) -> None:
        self.period_s, self.n = 1 / cfg.fps, 0

    def read(self) -> tuple[bool, np.ndarray]:
        """The next frame, a frame period after the last, as a camera delivers it."""
        time.sleep(self.period_s)
        frame, self.n = conftest.make_bgr_frame(), self.n + 1
        for bit in range(24):
            row, col = 8 * (bit // 8), 8 * (bit % 8)
            frame[row : row + 8, col : col + 8] = 255 * (self.n >> bit & 1)
        return True, frame

    def release(self) -> None:
        """Nothing to let go."""


def _cameras(app: config.AppConfig, monkeypatch: pytest.MonkeyPatch) -> config.AppConfig:
    """The app with every camera a #Counting camera, opened by the station's own `CameraReader.from_config`.

    It encodes one capture per period of the 5 frames a second the model reads, as on a robot. The fixtures' fake
    encodes every millisecond read, and three of those take the cores a 2-core runner's control loop needs.
    """
    monkeypatch.setattr(cameras.CameraReader, "from_config", FROM_CONFIG)
    source = f"{__name__}:{Counting.__name__}"
    return dataclasses.replace(app, cameras=[dataclasses.replace(c, device=None, source=source) for c in app.cameras])


def test_dreamcli_robot_run_on_a_metal_pair_passes_the_check(
    bus: Any,  # noqa: F811  (the Metal station's fixture)
    solver: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run drives the arms from the eef head through the client's IK; its state carries each hand's pose, which
    names the Metal, and each gripper in millimetres beside the 0-to-1 grip the client drives, which the check
    judges."""
    platform = EEFDreamAPI(test_makers_station.eef_per_key(solver))
    platform.start()
    app = _cameras(metal_config(), monkeypatch)
    stand_in = check_loop.StandIn(rate=app.control_hz)
    monkeypatch.setenv("DREAM_GATEWAY_URL", stand_in.serve(0))
    monkeypatch.setenv("DREAM_API_URL", f"http://{platform.host}:{platform.port}")
    with pytest.raises(errors.ControlLoopError, match="DA-5000"):
        asyncio.run(asyncio.wait_for(run.main(app, CHECKPOINT, home_duration_s=0.0), timeout=RUN_S))
    stand_in.close()
    platform.close()
    _expect(stand_in, "metal")


class FollowingRobot(conftest.FakeRobot):
    """A fake YAM whose joints and gripper go where they are told at once."""

    def command_joint_pos(self, positions: np.ndarray[Any, Any]) -> None:
        super().command_joint_pos(positions)
        self.joint_pos = np.asarray(positions[:6], dtype=np.float64).copy()
        self.gripper_pos = np.asarray(positions[6:7], dtype=np.float64).copy()


@pytest.mark.parametrize("space", ["eef", "joint"])
def test_a_yam_run_passes_the_check_on_either_head(space: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """`dreamcli robot run` on a YAM pair, which ends on the stand-in's refusal once the motion is played; its cameras
    send a new picture on every read, and the station fills each camera's history at the 5 frames a second the model
    reads."""
    robots: dict[str, conftest.FakeRobot] = {
        side: FollowingRobot(joint_pos=POSE.copy(), gripper_pos=0.5) for side in ("left", "right")
    }
    test_end_to_end._patch_hardware(monkeypatch, robots)
    monkeypatch.setattr(
        yam,
        "open_arm",
        lambda cfg, name: followers.FollowerArm(name, yam.I2rtDriver(name, robots[name]), home_rad=POSE.copy()),
    )
    app = dataclasses.replace(
        _cameras(test_end_to_end._make_app_config(), monkeypatch),
        action_space=space,
        action_horizon=check_loop.HORIZON,
        control_hz=20.0,
        action_timeout_s=3.0,
        max_stall_ticks=400,
    )
    stand_in = check_loop.StandIn(rate=app.control_hz)
    platform = test_end_to_end.FakeDreamAPI(test_end_to_end.HORIZON)
    platform.start()
    monkeypatch.setenv("DREAM_API_URL", f"http://{platform.host}:{platform.port}")
    monkeypatch.setenv("DREAM_GATEWAY_URL", stand_in.serve(0))
    with pytest.raises(errors.ControlLoopError, match="DA-5000"):
        asyncio.run(asyncio.wait_for(run.main(app, home_duration_s=0.0), timeout=RUN_S))
    stand_in.close()
    platform.close()
    _expect(stand_in, "yam")

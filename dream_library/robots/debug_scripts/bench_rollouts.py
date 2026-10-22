"""What recording a rollout costs the control tick: the same fake run with and without the recorder.

Builds a real `ControlLoop` over fake arms, N fake cameras at a real frame rate and a fake dream client that answers
after a delay, runs it on the wall clock for `--seconds`, and prints the tick time (how long `step()` took), the tick
interval and the process's CPU use, with `--record off` and `--record on`. The recorder is the product's own
(`robots.rollouts.Recorder`) writing into `--dir`; nothing else is faked about it. Run it on a CPU box, never on a
station: it uses no hardware but it does use cores.

    .venv/bin/python debug_scripts/bench_rollouts.py --record on --seconds 120 --dir /tmp/bench_rollouts
    taskset -c 0-3 .venv/bin/python debug_scripts/bench_rollouts.py --record off    # a four-core PC

The fake cameras return moving textured frames with sensor-like noise, so the encoder sees about what a real scene
costs; a flat frame would encode for nothing.
"""

import argparse
import asyncio
import json
import pathlib
import resource
import threading
import time
from typing import Any

import cv2
import numpy as np

from robots import cameras, config, control, followers, rollouts

## Joints of one fake arm.
ARM_DOF = 6

## Distinct frames each fake camera cycles through.
FRAME_CYCLE = 90


class BenchRobot:
    """An i2rt-shaped arm that sits still and counts its commands."""

    def __init__(self) -> None:
        """Starts with no commands counted."""
        self.commands = 0

    def get_observations(self) -> dict[str, np.ndarray]:
        """The i2rt observation of an arm at rest.

        @return `joint_pos`, `joint_vel` and `gripper_pos`.
        """
        return {"joint_pos": np.zeros(ARM_DOF), "joint_vel": np.zeros(ARM_DOF), "gripper_pos": np.array([0.5])}

    def command_joint_pos(self, positions: np.ndarray) -> None:
        """Counts a command.

        @param positions The 7-vector.
        """
        self.commands += 1

    def close(self) -> None:
        """Nothing to release."""


class BenchSource:
    """A capture source of moving, noisy frames at a fixed rate, in the `cv2.VideoCapture` shape."""

    def __init__(self, width: int, height: int, fps: float, seed: int) -> None:
        """Pre-renders the frames so the capture thread spends its time waiting, as a camera does.

        @param width Frame width.
        @param height Frame height.
        @param fps Frames per second the source delivers.
        @param seed Random seed, so cameras differ.
        """
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:height, 0:width]
        self._frames = []
        for i in range(FRAME_CYCLE):
            base = 128 + 90 * np.sin((xx + 6 * i) / 40.0 + yy / 90.0)
            noise = rng.normal(0, 4, (height, width, 3))
            frame = np.clip(base[..., None] * np.array([1.0, 0.8, 0.6]) + noise, 0, 255).astype(np.uint8)
            cv2.circle(frame, ((i * 9) % width, height // 2), height // 6, (30, 200, 30), -1)
            self._frames.append(frame)
        self._period_s = 1.0 / fps
        self._next_s = time.monotonic()
        self._index = 0

    def read(self) -> tuple[bool, np.ndarray]:
        """Waits for the next frame time and returns the next frame.

        @return `(True, frame_bgr)`.
        """
        self._next_s += self._period_s
        time.sleep(max(0.0, self._next_s - time.monotonic()))
        self._index += 1
        return True, self._frames[self._index % FRAME_CYCLE]

    def release(self) -> None:
        """Nothing to release."""


class BenchClient:
    """A dream client whose policy answers a chunk after a fixed delay, like a round trip to the gateway."""

    def __init__(self, horizon: int, delay_s: float) -> None:
        """Binds the chunk length and the delay.

        @param horizon Rows per chunk.
        @param delay_s Seconds `get_action` takes.
        """
        self._horizon = horizon
        self._delay_s = delay_s
        self.mode = "async"
        self.last_timing = None
        self.session_id = None

    def set_observation(self, frames: dict[str, str], observation: dict[str, Any], *, timestamp: Any = None) -> None:
        """Takes a tick and keeps nothing.

        @param frames Ignored.
        @param observation Ignored.
        @param timestamp Ignored.
        """

    def get_action(self, *, commanded: Any = None) -> dict[str, Any]:
        """Waits, then returns a slowly swinging chunk.

        @param commanded Ignored.
        @return `q_r_chunk` and `q_l_chunk`, `[horizon, 7]`.
        """
        time.sleep(self._delay_s)
        phase = np.linspace(0, 1, self._horizon)[:, None]
        rows = np.concatenate([0.2 * np.sin(phase * 3 + np.arange(ARM_DOF)), np.full((self._horizon, 1), 0.5)], axis=1)
        return {"q_r_chunk": rows.tolist(), "q_l_chunk": rows.tolist(), "action.right_eef_xyz": rows[:, :3]}

    def refill_due(self, executed: int, remaining: int) -> bool:
        """Asks for the next chunk when a third of this one is left.

        @param executed Rows executed since the install.
        @param remaining Rows left.
        @return Whether to refill.
        """
        return remaining <= self._horizon // 3

    def reset_episode(self) -> None:
        """Nothing to reset."""

    def close(self) -> None:
        """Nothing to close."""


def percentile(values: list[float], q: float) -> float:
    """The q-th percentile (0 to 100) by nearest rank.

    @param values The samples.
    @param q The percentile.
    @return The value.
    """
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q / 100.0))]


async def run_bench(args: argparse.Namespace) -> dict[str, Any]:
    """Runs the fake loop for the requested time and collects tick times.

    @param args The parsed command line.
    @return The measurements.
    """
    names = ["top", "left", "right"][: args.cameras]
    readers = [
        cameras.CameraReader(
            name,
            BenchSource(args.width, args.height, args.camera_fps, seed=i),
            stale_after_s=0.5,
            obs_period_s=0.2,
            policy_image_size=(640, 360),
            face_blur=False,
        )
        for i, name in enumerate(names)
    ]
    recorder = None
    if args.record == "on":
        recorder = rollouts.Recorder(
            config.RolloutsConfig(dir=args.dir, robot_name="bench"),
            body="makers",
            checkpoint="bench",
            run_number=1,
            fps=args.hz,
            camera_names=names,
            robot_id=None,
        )
    robots = {"left": BenchRobot(), "right": BenchRobot()}
    loop = control.ControlLoop(
        followers={name: followers.FollowerArm(name, robot) for name, robot in robots.items()},
        cameras=readers,
        client=BenchClient(horizon=args.horizon, delay_s=0.3),
        task="bench",
        control_hz=args.hz,
        action_horizon=args.horizon,
        action_timeout_s=3.0,
        stale_after_s=0.5,
        max_stall_ticks=10_000,
        ramp_s=0.5,
        max_ee_speed_m_s=1e9,
        recorder=recorder,
    )
    step_ms: list[float] = []
    starts: list[float] = []
    real_step = loop.step

    async def timed_step(now_s: float) -> None:
        started = time.perf_counter()
        await real_step(now_s)
        step_ms.append((time.perf_counter() - started) * 1e3)
        starts.append(started)

    loop.step = timed_step  # type: ignore[method-assign]
    cpu_before = resource.getrusage(resource.RUSAGE_SELF)
    wall_before = time.perf_counter()
    async with loop:
        task = asyncio.create_task(loop.run())
        await asyncio.sleep(args.seconds)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    wall_s = time.perf_counter() - wall_before
    cpu_after = resource.getrusage(resource.RUSAGE_SELF)
    finish_started = time.perf_counter()
    if recorder is not None:
        recorder.wait()
    finish_s = time.perf_counter() - finish_started
    cpu_run = (cpu_after.ru_utime - cpu_before.ru_utime) + (cpu_after.ru_stime - cpu_before.ru_stime)
    cpu_end = resource.getrusage(resource.RUSAGE_SELF)
    intervals_ms = [(b - a) * 1e3 for a, b in zip(starts, starts[1:], strict=False)]
    result: dict[str, Any] = {
        "record": args.record,
        "ticks": len(step_ms),
        "tick_ms_p50": round(percentile(step_ms, 50), 3),
        "tick_ms_p95": round(percentile(step_ms, 95), 3),
        "tick_ms_p99": round(percentile(step_ms, 99), 3),
        "tick_ms_max": round(max(step_ms), 3),
        "interval_ms_p50": round(percentile(intervals_ms, 50), 3),
        "interval_ms_p99": round(percentile(intervals_ms, 99), 3),
        "interval_ms_max": round(max(intervals_ms), 3),
        "overruns": sum(1 for v in intervals_ms if v > 1.5 * 1e3 / args.hz),
        "cpu_cores_during_run": round(cpu_run / wall_s, 3),
        "cpu_s_finishing": round((cpu_end.ru_utime + cpu_end.ru_stime) - (cpu_after.ru_utime + cpu_after.ru_stime), 2),
        "finish_wall_s": round(finish_s, 2),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "threads": threading.active_count(),
    }
    if recorder is not None:
        folder = recorder.episodes[0]
        info = json.loads((folder / "rollout.json").read_text())
        size = sum(path.stat().st_size for path in folder.rglob("*") if path.is_file())
        result.update(
            status=info["status"],
            frames=info["frames"],
            episode_mb=round(size / 1e6, 1),
            mb_per_minute=round(size / 1e6 / (info["frames"] / args.hz / 60.0), 1),
            video_mb_per_camera_minute=round(
                sum(p.stat().st_size for p in folder.glob("videos/**/*.mp4"))
                / 1e6
                / args.cameras
                / (info["frames"] / args.hz / 60.0),
                1,
            ),
        )
    return result


def main() -> None:
    """Parses the command line, runs the bench and prints the result as JSON."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", choices=("on", "off"), required=True)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--cameras", type=int, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--horizon", type=int, default=48)
    parser.add_argument("--dir", default="/tmp/bench_rollouts", type=str)
    args = parser.parse_args()
    pathlib.Path(args.dir).mkdir(parents=True, exist_ok=True)
    print(json.dumps(asyncio.run(run_bench(args)), indent=2))


if __name__ == "__main__":
    main()

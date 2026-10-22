"""The rollout recorder: saves every run on this disk as a LeRobot v2.1 episode, off the control tick.

A #Recorder belongs to one `ControlLoop`, which calls begin(), tick(), reply() and end(). Each only puts references on
queues; a journal thread and one VP9 encoder thread per camera (PyAV releases the GIL) do the work. Recording never
stops a run: when a write fails or an encoder falls behind (a camera queue holds #MAX_QUEUE_BYTES) the episode stops
recording, the reason is logged and written to `rollout.json`, and what was saved finishes as a `partial` episode.
What an episode holds is `robots.rollout_episode`.

@see Recorder
"""

import datetime
import errno
import fractions
import logging
import os
import pathlib
import queue
import shutil
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import av
import numpy as np
import numpy.typing as npt
from dream import client_ik

from robots import config, rollout_episode, types
from robots.rollout_episode import RolloutError

_log = logging.getLogger(__name__)

## Raw frame bytes a camera queue may hold before its encoder counts as overloaded; free bytes a start needs.
MAX_QUEUE_BYTES = 128 * 1024 * 1024
MIN_FREE_START_BYTES = 2 * 1024**3

SETTINGS = ("max_ee_speed_m_s", "max_gripper_step", "grip_force_cap", "max_stall_ticks", "action_timeout_s")

## The video encoder: VP9 through PyAV's `libvpx-vp9`, BSD licensed (the wheel's H.264 encoder, x264, is GPL and is not
## used). `deadline=realtime` with `lag-in-frames=0` and no alternate reference frames writes every frame as soon as it
## is handed in, in order, so a killed run keeps everything encoded and a cut after any packet is a valid stream;
## `cpu-used` 6 encodes a 640x480 stream at about 240 fps on two threads (measured); constant quality
## `crf` 32 (`b:v` 0 turns the bitrate cap off) and a key frame every 10 ticks bound a killed run's loss.
VIDEO_CODEC = "libvpx-vp9"
VIDEO_OPTIONS = {
    "deadline": "realtime",
    "cpu-used": "6",
    "lag-in-frames": "0",
    "auto-alt-ref": "0",
    "row-mt": "1",
    "crf": "32",
    "b:v": "0",
    "g": "10",
    "threads": "2",
}


def _vector(arm: types.ArmState | types.ArmCommand) -> list[float]:
    """The arm's joints then its gripper, as the dataset stores one arm."""
    return [*map(float, arm.positions_rad), float(arm.gripper)]


def _reason_of(exc: BaseException) -> str:
    """The failure code of a worker's exception: `disk_full`, `io_error` or `encoder_error`."""
    if getattr(exc, "errno", None) in (errno.ENOSPC, errno.EDQUOT):
        return "disk_full"
    return "io_error" if isinstance(exc, OSError) else "encoder_error"


def _write_frame(container: Any, stream: Any, rgb: npt.NDArray[np.uint8], pts: int) -> None:
    """Encodes one frame into the camera's file."""
    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
    frame.pts = pts
    container.mux(stream.encode(frame))


class _Episode:
    """The threads and queues of one episode being recorded; `failure` and `error` say why it stopped early."""

    def __init__(self, root: pathlib.Path, camera_names: Sequence[str], fps: float, journal: pathlib.Path) -> None:
        """Starts the journal thread, writing `journal` (`rollout_episode.journal_file()`), and one encoder thread per
        camera in the episode folder `root`."""
        self.root = root
        self._journal_path = journal
        self.failure: str | None = None
        self.error: str | None = None
        self._fps = fps
        self._ticks = 0
        self._replies = 0
        self._max_frames = 0
        self._journal: queue.Queue[tuple[Any, ...] | None] = queue.Queue()
        self._cameras: dict[str, queue.Queue[Any]] = {name: queue.Queue() for name in camera_names}
        self._threads = [threading.Thread(target=self._journal_worker, daemon=True)]
        self._threads += [threading.Thread(target=self._camera_worker, args=(n,), daemon=True) for n in camera_names]
        for thread in self._threads:
            thread.start()

    def _queue_tick(self, tick: tuple[Any, ...], frames: Mapping[str, types.Frame]) -> None:
        """Queues one commanded tick: its frames for the encoders, its numbers for the journal."""
        if self.failure is not None:
            return
        try:
            if not self._max_frames:
                self._max_frames = max(2, MAX_QUEUE_BYTES // next(iter(frames.values())).data.nbytes)
            for name, frame_queue in self._cameras.items():
                if frame_queue.qsize() >= self._max_frames:
                    self._fail("encoder_overloaded", f"camera '{name}' encoder is {self._max_frames} frames behind")
                    return
                frame_queue.put_nowait(frames[name].data)
            self._journal.put_nowait(("tick", *tick))
            self._ticks += 1
        except Exception as exc:
            self._fail("recorder_error", f"{type(exc).__name__}: {exc}")

    def _queue_reply(self, raw: Mapping[str, Any], observed_at_s: float) -> None:
        """Queues one model reply, copied now since the client may reuse its arrays."""
        if self.failure is not None:
            return
        try:
            arrays = {k: np.array(repr(v)) if (a := np.array(v)).dtype == object else a for k, v in raw.items()}
            arrays |= {"_observed_at_s": np.array(observed_at_s), "_after_tick": np.array(self._ticks)}
            self._journal.put_nowait(("reply", self._replies, arrays))
            self._replies += 1
        except Exception as exc:
            self._fail("recorder_error", f"{type(exc).__name__}: {exc}")

    def _fail(self, reason: str, message: str) -> None:
        """Stops recording at this point and says why, once."""
        if self.failure is not None:
            return
        self.failure, self.error = reason, message
        _log.error("rollouts: %s stopped recording (%s): %s. The run goes on.", self.root, reason, message)

    def _close(self) -> None:
        """Tells every thread to finish what is queued, then waits for them."""
        self._journal.put(None)
        for frame_queue in self._cameras.values():
            frame_queue.put(None)
        for thread in self._threads:
            thread.join()

    def _journal_worker(self) -> None:
        """Writes journal rows and raw replies until told to stop."""
        flush_every = max(1, round(self._fps))
        try:
            (self.root / rollout_episode.CHUNKS_DIR).mkdir(exist_ok=True)
            with open(self._journal_path, "wb") as journal:
                written = 0
                while (item := self._journal.get()) is not None:
                    if item[0] == "reply":
                        np.savez(self.root / rollout_episode.CHUNKS_DIR / f"reply_{item[1]:06d}.npz", **item[2])
                        continue
                    _, t_s, chunk, row, state, action = item
                    values = [t_s, chunk, row, *_vector(state.left), *_vector(state.right)]
                    values += [*_vector(action.left), *_vector(action.right)]
                    journal.write(np.array(values, dtype=np.float64).tobytes())
                    written += 1
                    if written % flush_every == 0:
                        journal.flush()
        except Exception as exc:
            self._fail(_reason_of(exc), f"journal: {type(exc).__name__}: {exc}")

    def _camera_worker(self, name: str) -> None:
        """Encodes one camera's frames into its fragmented MP4 until told to stop."""
        frame_queue = self._cameras[name]
        container: Any = None
        stream: Any = None
        try:
            while (rgb := frame_queue.get()) is not None:
                if container is None:
                    path = rollout_episode.video_file(self.root, name, partial=True)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    # flush_packets makes every fragment reach the file, so a killed run keeps what was encoded.
                    movflags = "frag_keyframe+empty_moov+default_base_moof"
                    container = av.open(
                        str(path), "w", format="mp4", options={"movflags": movflags, "flush_packets": "1"}
                    )
                    stream = container.add_stream(
                        VIDEO_CODEC, rate=fractions.Fraction(self._fps).limit_denominator(1001), options=VIDEO_OPTIONS
                    )
                    stream.height, stream.width = rgb.shape[0] // 2 * 2, rgb.shape[1] // 2 * 2
                    stream.pix_fmt = "yuv420p"
                    count = 0
                _write_frame(container, stream, np.ascontiguousarray(rgb[: stream.height, : stream.width]), count)
                count += 1
        except Exception as exc:
            self._fail(_reason_of(exc), f"camera '{name}': {type(exc).__name__}: {exc}")
        finally:
            if container is not None:
                try:
                    container.mux(stream.encode(None))
                    container.close()
                except Exception as exc:
                    self._fail(_reason_of(exc), f"camera '{name}' close: {type(exc).__name__}: {exc}")


class Recorder:
    """Saves the episodes of one `ControlLoop`; no method raises once begin() has returned.

    A task change mid-run is end() then begin(): each prompt gets its own episode. `episodes` lists the folders begun.
    """

    def __init__(
        self,
        cfg: config.RolloutsConfig,
        *,
        body: str,
        checkpoint: str,
        run_number: int,
        fps: float,
        camera_names: Sequence[str],
        joints: Mapping[str, int],
        robot_id: str | None,
        settings: Mapping[str, Any],
    ) -> None:
        """Binds what every episode of the run records about itself; nothing is written until begin().

        @param cfg The `rollouts` section.
        @param body The config's body (`yam`, `makers`, `metal`).
        @param checkpoint The model the run asks.
        @param run_number The run's ordinal, in the folder name.
        @param fps The tick rate.
        @param camera_names The configured cameras, which every episode records.
        @param joints Per arm, `"left"` and `"right"`, its joints (`robots.followers.FollowerArm.dof`): the width of
            its part of every recorded vector.
        @param robot_id The `live.robot_id`, or None without a live section.
        @param settings The run's value of each of #SETTINGS.
        """
        self.episodes: list[pathlib.Path] = []
        self._dir = pathlib.Path(cfg.dir).expanduser()
        self._fixed: dict[str, Any] = {
            "robot_name": cfg.robot_name,
            "robot_id": robot_id,
            "body": body,
            "checkpoint": checkpoint,
            "run": run_number,
            "fps": fps,
            "cameras": list(camera_names),
            "joints": dict(joints),
            **{key: settings[key] for key in SETTINGS},
            # Proprio test switch, carried for the record while the test runs.
            "proprio_test": os.environ.get(client_ik.PROPRIO_TEST_ENV) or None,
        }
        self._episode: _Episode | None = None
        self._finishers: list[threading.Thread] = []

    def begin(self, task: str) -> pathlib.Path:
        """Creates the episode folder and starts recording into it; also finishes episodes a killed run left.

        @param task The prompt of the episode.
        @return The episode folder.
        @throws RolloutError The folder cannot be made, has under #MIN_FREE_START_BYTES free, or the encoder is
            missing.
        """
        try:
            av.codec.Codec(VIDEO_CODEC, "w")
            self._dir.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(self._dir).free
        except Exception as exc:
            raise RolloutError(f"rollouts cannot start in {self._dir}: {exc}; set rollouts.enabled: false") from exc
        if free < MIN_FREE_START_BYTES:
            raise RolloutError(
                f"rollouts folder {self._dir} has {free >> 20} MB free, under {MIN_FREE_START_BYTES >> 20} MB; "
                "free some space, set rollouts.dir to another disk, or set rollouts.enabled: false"
            )
        if folders := rollout_episode.unfinished(self._dir):
            self._finishers.append(threading.Thread(target=_recover_all, args=(folders,)))
            self._finishers[-1].start()
        started = datetime.datetime.now(datetime.timezone.utc)
        suffix = f"_{len(self.episodes) + 1}" if self.episodes else ""
        folder = self._dir / f"{started.astimezone():%Y-%m-%dT%H-%M-%S}_run{self._fixed['run']}{suffix}"
        info = {"status": rollout_episode.STATUS_RECORDING, "task": task, **self._fixed}
        info |= {"started_at_utc": started.isoformat(timespec="seconds"), "stop_reason": None, "error": None}
        try:
            folder.mkdir()
            rollout_episode.write_json(folder / rollout_episode.INFO_NAME, info)
            (folder / rollout_episode.MARKER_NAME).write_text(f"pid {os.getpid()}\n", encoding="utf-8")
        except OSError as exc:
            raise RolloutError(f"rollout folder {folder} cannot be written: {exc}") from exc
        journal = rollout_episode.journal_file(folder, self._fixed["joints"])
        self._episode = _Episode(folder, self._fixed["cameras"], self._fixed["fps"], journal)
        self.episodes.append(folder)
        _log.info("rollouts: saving this run to %s", folder)
        return folder

    def tick(
        self,
        t_s: float,
        chunk: int,
        row: int,
        state: types.RobotState,
        action: types.Action,
        frames: Mapping[str, types.Frame],
    ) -> None:
        """Records one commanded tick: what the arms measured, what was commanded, and the cameras' latest frames.

        @param t_s The tick time on the loop's clock.
        @param chunk Chunks installed so far.
        @param row The row of the chunk this tick commanded.
        @param state The measured state read this tick.
        @param action The command submitted this tick, after the ramp and the clamp.
        @param frames The latest frame per camera.
        """
        if self._episode is not None:
            self._episode._queue_tick((t_s, chunk, row, state, action), frames)

    def reply(self, raw: Mapping[str, Any], observed_at_s: float) -> None:
        """Saves one raw model reply as the client returned it.

        @param raw The dict `get_action` returned.
        @param observed_at_s The monotonic time of the observation it answers.
        """
        if self._episode is not None:
            self._episode._queue_reply(raw, observed_at_s)

    def end(
        self, reason: str, error: str | None = None, loop_counts: Mapping[str, Any] | None = None
    ) -> pathlib.Path | None:
        """Stops the current episode and finishes it on a background thread; returns at once.

        @param reason Why the run ended: `"stopped"`, `"cancelled"` or an exception's class name.
        @param error The message of the exception that ended the run.
        @param loop_counts What the loop did to the plan over the episode.
        @return The episode folder, or None when none was recording.
        """
        episode, self._episode = self._episode, None
        if episode is None:
            return None
        self._finishers.append(threading.Thread(target=self._finish, args=(episode, reason, error, loop_counts or {})))
        self._finishers[-1].start()
        return episode.root

    def wait(self) -> None:
        """Blocks until every ended episode is finished on disk."""
        for thread in self._finishers:
            thread.join()

    def _finish(self, episode: _Episode, reason: str, error: str | None, loop_counts: Mapping[str, Any]) -> None:
        """Joins an episode's threads and turns what they wrote into the dataset."""
        episode._close()
        status = rollout_episode.STATUS_PARTIAL if episode.failure else rollout_episode.STATUS_COMPLETE
        try:
            info = rollout_episode.finish(
                episode.root,
                status=status,
                stop_reason=episode.failure or reason,
                error=episode.error if episode.failure else error,
                loop_counts=loop_counts,
            )
        except Exception:
            _log.exception("rollouts: could not finish %s; its raw files are kept", episode.root)
            return
        _log.info("rollouts: saved %s: %s, %d ticks", episode.root, info["status"], info["frames"])


def _recover_all(folders: list[pathlib.Path]) -> None:
    """Recovers each folder; one that cannot be is logged and left as it is."""
    for folder in folders:
        try:
            _log.info(
                "rollouts: recovered %s: %d ticks",
                folder,
                rollout_episode.finish(folder, status="partial", stop_reason="recovered_after_crash")["frames"],
            )
        except Exception:
            _log.exception("rollouts: could not recover %s", folder)

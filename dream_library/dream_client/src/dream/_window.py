"""The frame window of `dream.DreamClient`: which camera ticks it keeps, and the last few of each camera.

A caller of set_observation() hands the client every camera tick, at whatever rate its cameras run (30 Hz on a YAM
station), while a checkpoint reads each camera's window at the rate it was trained on, `obs_fps`. So the window keeps
one tick every `1 / obs_fps` seconds and drops the rest, deciding on the caller's timestamps, never on a count of
calls: a count drifts as soon as a tick is late or missed.

The rule, per episode:

- The first tick is kept, and the next one is due a period after it.
- A later tick is kept when it is at or past the due time less a slack: half the median interval between the last
  #SLACK_CALLS distinct timestamps (this one included), at most half a period, and none before the second. A tick that
  repeats the last timestamp (a loop that runs faster than its cameras and hands the same frame again) is not counted,
  so it never shrinks the slack. The tick kept is
  then the one nearest the due time, so 30 Hz ticks keep exactly every sixth at 5 fps even with a few milliseconds of
  jitter, and a tick one interval early is never kept.
- After a kept tick the next one is due a period later, anchored on the schedule so jitter never drifts it; a tick
  kept half a period or more after its due time (the caller paused) starts the schedule again from itself.

Every tick, kept or not, replaces the latest observation (the state and the prompt), so what is sent is always the
newest state, with the frames of the ticks kept so far.
"""

import collections
import itertools
import math
import statistics
import threading
from collections.abc import Mapping
from typing import Any

from dream.errors import DreamObservationError

## Frames per window key once the window is full, unless the `obs_window` config key says otherwise.
OBS_WINDOW = 9

## Frames kept per second, unless the `obs_fps` config key says otherwise: one every 0.2 seconds.
OBS_FPS = 5.0

## Ticks whose intervals size the slack: one late or early tick among them does not move the median, and a camera
## that changes rate is followed within a few of its ticks.
SLACK_CALLS = 8


class Window:
    """One episode's frames, a ring per frame key, and its latest observation; safe to use from several threads.

    One lock guards all of it, held only while a tick goes in or the observation is copied out, never across a
    request, so a tick never waits for the model.

    Attributes:
    - size: Frames per ring once it is full (`obs_window`).
    - period_s: Seconds between kept ticks (`1 / obs_fps`).
    """

    def __init__(self, size: int, period_s: float) -> None:
        """Starts empty: no frame key, no observation, nothing due.

        @param size Frames per ring once it is full; at least 1.
        @param period_s Seconds between kept ticks; positive.
        """
        self.size = size
        self.period_s = period_s
        self._lock = threading.Lock()
        ## Frame key to its last `size` kept frames, oldest first; every ring gets a frame on the same ticks.
        self._rings: dict[str, collections.deque[str]] = {}
        ## The last tick's observation, a copy; None before the first tick of the episode.
        self._observation: dict[str, Any] | None = None
        ## The last #SLACK_CALLS distinct tick timestamps, oldest first; the slack is sized from their intervals.
        self._recent_s: collections.deque[float] = collections.deque(maxlen=SLACK_CALLS)
        ## When the next tick is due to be kept; None before the first tick of the episode.
        self._due_s: float | None = None

    def add(self, frames: Mapping[str, str], observation: Mapping[str, Any], timestamp_s: float) -> bool:
        """Takes one camera tick: keeps its frames when they are due, and its observation in any case.

        Nothing changes when a check fails.

        @param frames Frame key to one frame, a non-empty base64 JPEG `str`; the same keys on every tick of the
            episode.
        @param observation Every other key the checkpoint reads; replaces the last tick's, copied (not the values).
        @param timestamp_s When the frames were captured, seconds on a monotonic clock; never earlier than the last
            tick's.
        @return Whether the frames were kept.
        @throws DreamObservationError The tick is not valid (_check_tick() says how), its frame keys are not the
            episode's, or its timestamp is earlier than the last tick's; the message names the key.
        """
        _check_tick(frames, observation, timestamp_s)
        ticked = dict(frames)
        latest = dict(observation)
        with self._lock:
            if self._rings and ticked.keys() != self._rings.keys():
                missing = sorted(self._rings.keys() - ticked.keys())
                extra = sorted(ticked.keys() - self._rings.keys())
                raise DreamObservationError(
                    f"frame keys changed within the episode: missing {missing}, extra {extra}; send the same frame "
                    "keys on every set_observation(), or call reset_episode() first"
                )
            if self._recent_s and timestamp_s < self._recent_s[-1]:
                raise DreamObservationError(
                    f"set_observation() timestamp {timestamp_s!r} is earlier than the previous call's, "
                    f"{self._recent_s[-1]!r}; timestamps never go back within an episode"
                )
            if not self._recent_s or timestamp_s > self._recent_s[-1]:
                self._recent_s.append(timestamp_s)
            self._observation = latest
            gaps_s = [later - earlier for earlier, later in itertools.pairwise(self._recent_s)]
            slack_s = min(statistics.median(gaps_s) / 2, self.period_s / 2) if gaps_s else 0.0
            due_s = self._due_s
            if due_s is not None and timestamp_s < due_s - slack_s:
                return False
            # Anchored on the schedule, so jitter never drifts it; a tick half a period late or more means the caller
            # paused, and the schedule starts again from it.
            anchor_s = due_s if due_s is not None and timestamp_s < due_s + self.period_s / 2 else timestamp_s
            self._due_s = anchor_s + self.period_s
            if not self._rings:
                self._rings = {key: collections.deque(maxlen=self.size) for key in ticked}
            for key, frame in ticked.items():
                self._rings[key].append(frame)
            return True

    def build_observation(self) -> dict[str, Any]:
        """The observation to send: the latest tick's, plus each frame key's window, oldest frame first.

        @return A new dict holding new lists, so a later tick changes nothing in it. Each window is as long as the
            ticks kept so far, up to `size`, and never empty.
        @throws DreamObservationError No tick since the episode started.
        """
        with self._lock:
            if self._observation is None:
                raise DreamObservationError(
                    "no observation since the episode started: call set_observation(...) before get_action()"
                )
            return {**self._observation, **{key: list(ring) for key, ring in self._rings.items()}}

    def clear(self) -> None:
        """Forgets the episode: the frames, the latest observation and the schedule; the next tick is a first one."""
        with self._lock:
            self._rings = {}
            self._observation = None
            self._recent_s.clear()
            self._due_s = None


def _check_tick(frames: Mapping[str, str], observation: Mapping[str, Any], timestamp_s: float) -> None:
    """Refuses a tick whose arguments cannot be kept, before the window is touched.

    @param frames The tick's frames.
    @param observation The tick's observation.
    @param timestamp_s The tick's timestamp.
    @throws DreamObservationError `frames` is not a mapping or is empty, a frame key is not a non-empty `str`, a frame
        is not one non-empty `str` (bytes get a hint to send base64), `observation` is not a mapping or carries a
        frame key, or the timestamp is not a finite number; the message names the key.
    """
    if not isinstance(frames, Mapping):
        raise DreamObservationError(
            f"set_observation() frames must be a mapping of observation key to one frame, got {type(frames).__name__}"
        )
    if not frames:
        raise DreamObservationError("set_observation() frames is empty: pass one frame per window key")
    for key, frame in frames.items():
        if not isinstance(key, str) or not key:
            raise DreamObservationError(
                f"frame key {key!r} must be a non-empty str, the observation key the checkpoint reads the window under"
            )
        if isinstance(frame, (bytes, bytearray, memoryview)):
            raise DreamObservationError(f"frame {key!r} is bytes; send each frame as a base64 str, not bytes")
        if not isinstance(frame, str) or not frame:
            got = "an empty str" if isinstance(frame, str) else type(frame).__name__
            raise DreamObservationError(
                f"frame {key!r} must be one frame, a non-empty base64 JPEG str (the client keeps the window), "
                f"got {got}"
            )
    if not isinstance(observation, Mapping):
        raise DreamObservationError(
            f"set_observation() observation must be a mapping, got {type(observation).__name__}"
        )
    for key in frames:
        if key in observation:
            raise DreamObservationError(
                f"observation key {key!r} is also a frame key; leave it out of the observation, the client sends the "
                "window it keeps under that key"
            )
    if isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float)) or not math.isfinite(timestamp_s):
        raise DreamObservationError(
            f"set_observation() timestamp must be a finite number of seconds on a monotonic clock, got {timestamp_s!r}"
        )

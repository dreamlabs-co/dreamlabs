"""Seam smoothing: shapes each returned chunk so it takes over from the motion the arm is on; internal.

The model computes every chunk from one observation and knows nothing of the chunk before it, so where a control
loop swaps the old chunk for the new one the target can step in position and in velocity: the hitch a policy shows
at every refill. This module acts on the chunk before the loop sees it, from what the client holds at that moment:
the observation the chunk answers (the arms' measured pose, which says which row of the previous chunk the arms are
on), the chunk it returned last (the plan the arms follow until this one lands), the round trip, and the loop's row
rate (`action_hz`), which together say which row of the new chunk the loop installs it at.

It is always on and has no setting: the numbers are the library's, so every chunk a control loop receives is as
smooth as the library can make it. Per arm, in order: the anchor is the row of the previous chunk the loop runs
next, the one after the joints it last commanded when the caller passes them (`commanded`), else the row nearest the
measured joint pose (a real arm trails its command, so that row is a few behind and the join steps back); the clock
row is the row of the new chunk the loop installs it at, the rows the loop executes during the round trip (`skipped`
in `robots.control`), or every row the previous plan had left when the arm ran out of them and is holding; a
Savitzky-Golay filter, run at twice the row rate, removes per-row jitter from the clock row on, with half a window
of the new chunk's earlier rows as left context; then the seam join (`dream.seam_join`) replaces the
`seam_join.RAMP_ROWS` rows from the clock row with the velocity blend from the old motion into the new plan, landing
exactly on the plan, in end-effector space when the caller gives the arm's kinematics and in joint space otherwise.
Rows before the clock row continue the previous plan, so a loop that installs a row or two early still meets the
motion it is on. The first chunk of an episode, and an arm off its plan, are filtered only: there is no commanded
motion to continue. See #SeamSmoother.
"""

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from dream import seam_join
from dream.errors import DreamActionError

## Each chunk key with the observation key holding the measured pose of the arm it drives, per the wire's naming
## (README, "Observation and action keys"). A chunk's leading columns are those joints; the rest is the gripper.
ARM_POSE_KEYS = {"q_r_chunk": "state.arm_r.joint_pos", "q_l_chunk": "state.arm_l.joint_pos"}

## Deprecated: the joints of a six-joint arm's chunk, for code written when every arm had six; an arm's own count is
## its measured pose's (SeamSmoother.arm_joints()).
ARM_JOINTS = 6

## Samples of the Savitzky-Golay window and the polynomial order it fits, at twice the row rate (#UPSAMPLE): about
## five rows wide; order 3 leaves any motion a cubic can follow unchanged.
SAVGOL_WINDOW = 11
SAVGOL_POLYORDER = 3

## The filter runs on the chunk resampled at this many samples per row (linear in between), then read back per row.
UPSAMPLE = 2

## Rows of the new chunk before the clock row the filter takes as left context, so the first commanded row is a
## centred fit rather than a one-sided filter edge.
FILTER_CONTEXT_ROWS = 5

## Largest joint-space distance (L2 over the joints, radians) between the measured pose and the nearest row of the
## previous chunk (or the joints last commanded) for the arm to count as on that plan; beyond it (a reset, a stop, a
## hand on the arm) no join.
MAX_OFFSET_RAD = 0.5


class SeamSmoother:
    """Joins each chunk onto the motion the arm is on; keeps the previous chunk per arm."""

    def __init__(
        self,
        action_hz: float,
        kinematics: Mapping[str, seam_join.Kinematics] | None = None,
        pose_keys: Mapping[str, str] = ARM_POSE_KEYS,
        fence: Callable[[str, npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None = None,
    ) -> None:
        """No chunk is known yet, so the first chunk is filtered only.

        @param action_hz The loop's row rate, rows per second: with the round trip, the clock row.
        @param kinematics Per chunk key, the kinematics of the arm it drives; an arm with an entry is joined in
            end-effector space, every other arm in joint space.
        @param pose_keys Per chunk key, the observation key of the measured joints of the arm it drives: the YAM
            wire's (#ARM_POSE_KEYS), or a body's own (`dream.client_ik`).
        @param fence Takes a chunk key and that arm's joined chunk and gives the rows the arm may run (the reach
            area's hold, `dream.client_ik.ClientIK.fence`), or None. It runs on the joined rows, before they are kept
            as the previous chunk, since the join carries the old motion on into the new chunk.
        """
        self._action_hz = action_hz
        self._kinematics = dict(kinematics or {})
        self._pose_keys = dict(pose_keys)
        self._fence = fence
        # The chunk last handed to the caller, per chunk key: the plan the arm follows until the next one lands.
        self._previous: dict[str, npt.NDArray[np.float64]] = {}

    def reset(self) -> None:
        """Forgets the previous chunks: a fresh episode has no plan to continue; its first chunk is filtered only."""
        self._previous.clear()

    def smooth(
        self,
        observation: dict[str, Any],
        action: dict[str, Any],
        sent_s: float,
        rtt_s: float,
        commanded: Mapping[str, npt.NDArray[np.float64]] | None = None,
    ) -> dict[str, Any]:
        """Returns `action` with each arm's chunk filtered and joined onto the motion that arm is on.

        An arm whose chunk key is absent, or whose chunk is not two-dimensional, is left alone. An arm whose
        measured pose the observation does not carry (or carries as something other than numbers) gets the filter
        but no join, and so does an arm with no previous chunk of the same shape (the first chunk of an episode) or
        whose measured pose is farther than #MAX_OFFSET_RAD from every row of it (an arm off its plan): there is no
        commanded motion to continue.

        @param observation The observation the chunk answers; its measured joint poses (the pose keys) anchor the join.
        @param action The model's action dict, as validated.
        @param sent_s When the request was sent, seconds on a monotonic clock; unused, kept for the caller's call.
        @param rtt_s The request's round trip, seconds: with the row rate, the rows the loop executed meanwhile.
        @param commanded Per chunk key, the joints the loop last commanded that arm before the observation, or None;
            an arm with an entry is anchored on them (#anchor_row).
        @return A shallow copy of `action` with `q_r_chunk` and `q_l_chunk` rewritten as lists of rows; every other
            key is the model's value, untouched.
        @throws DreamActionError A chunk is not rows of numbers (ragged, or holding strings); the message names it.
        """
        del sent_s
        result = dict(action)
        for key, pose_key in self._pose_keys.items():
            if key not in action:
                continue
            try:
                chunk = np.asarray(action[key], dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise DreamActionError(f"action key '{key}' is not rows of numbers: {error}") from None
            if chunk.ndim != 2 or chunk.shape[0] == 0:
                continue  # strict mode refused this already; without it, a shape we cannot join is left alone
            last = None if commanded is None else commanded.get(key)
            smoothed = self._join(key, chunk, _pose(observation.get(pose_key)), last, rtt_s)
            if self._fence is not None:
                smoothed = self._fence(key, smoothed)
            self._previous[key] = smoothed
            result[key] = smoothed.tolist()
        return result

    def arm_joints(self, observation: Mapping[str, Any]) -> dict[str, int | None]:
        """Per chunk key, how many joints the arm's measured pose in the observation holds: the arm's own count.

        @param observation The observation a chunk will answer.
        @return Per chunk key, the count, or None when the observation carries no flat pose of that arm that is
            numbers (then `commanded` may have any length). A non-dict
            observation carries none; sending it fails.
        """
        out: dict[str, int | None] = {}
        for key, pose_key in self._pose_keys.items():
            value = observation.get(pose_key) if isinstance(observation, Mapping) else None
            pose = _pose(value)
            out[key] = pose.shape[0] if pose is not None and np.asarray(value).ndim == 1 else None
        return out

    def takeover(
        self,
        key: str,
        shape: tuple[int, int],
        pose: npt.NDArray[np.float64],
        rtt_s: float,
        commanded: npt.NDArray[np.float64] | None,
    ) -> tuple[int, npt.NDArray[np.float64]] | None:
        """Where a new chunk takes over one arm: its clock row, and the joints the arm is commanded to there.

        The same anchor and clock row the join of that chunk uses (#nearest_row, #old_motion): the previous chunk's
        row at the clock row is the command the arm is at when the loop installs the new chunk.

        @param key The chunk key.
        @param shape The new chunk's `[rows, columns]`.
        @param pose `[joints]` the arm's measured joint pose.
        @param rtt_s The round trip, seconds.
        @param commanded `[joints]` the joints the loop last commanded the arm, or None.
        @return `(clock, joints)`, or None when the chunk will not be joined: no previous chunk of that shape (the
            first of an episode) or an arm off its plan.
        """
        history = self.takeover_history(key, shape, pose, rtt_s, commanded)
        return None if history is None else history[:2]

    def takeover_history(
        self,
        key: str,
        shape: tuple[int, int],
        pose: npt.NDArray[np.float64],
        rtt_s: float,
        commanded: npt.NDArray[np.float64] | None,
    ) -> tuple[int, npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
        """#takeover, plus the previous chunk's two rows before the takeover row.

        Those rows are the motion the arm is on as it gets there, which the `traj` solve continues.

        @param key The chunk key.
        @param shape The new chunk's `[rows, columns]`.
        @param pose `[joints]` the arm's measured joint pose.
        @param rtt_s The round trip, seconds.
        @param commanded `[joints]` the joints the loop last commanded the arm, or None.
        @return `(clock, joints, before, before2)`, `before` and `before2` the previous chunk's rows one and two
            before `joints` (its first row where there is none), or None as #takeover.
        """
        previous = self._previous.get(key)
        if previous is None or previous.shape != shape:
            return None
        joints = previous[:, : pose.shape[0]]
        anchor = anchor_row(joints, pose, commanded)
        if anchor is None:
            return None
        clock, _, _ = old_motion(joints, anchor, rtt_s * self._action_hz)
        at = anchor + clock
        return clock, joints[at], joints[max(at - 1, 0)], joints[max(at - 2, 0)]

    def _join(
        self,
        key: str,
        chunk: npt.NDArray[np.float64],
        pose: npt.NDArray[np.float64] | None,
        commanded: npt.NDArray[np.float64] | None,
        rtt_s: float,
    ) -> npt.NDArray[np.float64]:
        """One arm's chunk, filtered from its clock row and joined onto the old motion there.

        @param key The chunk key, for the previous chunk and the kinematics.
        @param chunk `[rows, columns]`: the joints, then the gripper.
        @param pose The arm's measured joint pose, or None when the observation carries none.
        @param commanded The joints the loop last commanded the arm, or None.
        @param rtt_s The round trip, seconds.
        @return `[rows, columns]`.
        """
        joint_count = chunk.shape[1] - 1
        if pose is None:
            return filter_from(chunk, 0)
        joint_count = min(pose.shape[0], joint_count)
        previous = self._previous.get(key)
        if previous is not None and previous.shape != chunk.shape:
            previous = None
        anchor = None
        if previous is not None:
            anchor = anchor_row(previous[:, :joint_count], pose[:joint_count], commanded)
        if previous is None or anchor is None:
            # No plan the arm is on, so no commanded motion to continue: the loop's own start ramp, from the measured
            # pose, is what joins the first chunk of a run, and an arm off its plan is not where any row put it.
            return filter_from(chunk, 0)
        clock, old_rows, times = old_motion(previous[:, :joint_count], anchor, rtt_s * self._action_hz)
        plan = filter_from(chunk, clock)
        count = min(seam_join.RAMP_ROWS, chunk.shape[0] - 1 - clock)
        if count < 1:
            return plan
        window = plan[clock : clock + count + 1, :joint_count]
        now, velocity = seam_join.fit_motion(old_rows, times)
        joints = seam_join.velocity_blend(now, velocity, window)
        kinematics = self._kinematics.get(key)
        if kinematics is not None:
            eef = seam_join.eef_window(old_rows, times, window, joints, kinematics)
            if eef is not None:
                joints = eef
        out = plan.copy()
        out[clock : clock + count + 1, :joint_count] = joints
        # Rows before the clock row are never commanded when the loop installs where the round trip says; if it
        # installs earlier, they are the previous plan's, the motion the arm is on.
        past = np.minimum(anchor + np.arange(clock), previous.shape[0] - 1)
        out[:clock, :joint_count] = previous[past, :joint_count]
        return out


def old_motion(
    previous: npt.NDArray[np.float64], anchor: int, rows_in_flight: float
) -> tuple[int, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """The clock row and the old-motion rows the join continues, from where the arm was on the previous plan.

    The loop kept executing the previous plan from the anchor while the chunk was in flight: the clock row is the
    rows it executed, the round trip in rows, rounded. The last of them it commanded is row `anchor + clock - 1`,
    and the next ones it would have are the old motion the join continues (`seam_join.OLD_PATH_TIMES`). An arm that
    ran out of rows before the chunk landed holds the previous plan's last row: the clock row is the rows it had,
    and the old motion is that row alone, at rest.

    @param previous `[rows, joints]` the plan the arm is on.
    @param anchor Row of `previous` the arm is on at the observation the new chunk answers.
    @param rows_in_flight The round trip in rows (seconds times the row rate).
    @return `(clock, rows, times)`: the clock row, `[count, joints]` old-motion rows and their `[count]` times.
    """
    last = previous.shape[0] - 1
    remaining = last - anchor
    clock = int(round(rows_in_flight))
    if clock >= remaining:
        return remaining, previous[last : last + 1], np.zeros(1)
    indices = anchor + clock + np.array(seam_join.OLD_PATH_TIMES, dtype=np.int64)
    valid = (indices >= 0) & (indices <= last)
    return clock, previous[indices[valid]], np.array(seam_join.OLD_PATH_TIMES)[valid]


def filter_from(chunk: npt.NDArray[np.float64], clock: int) -> npt.NDArray[np.float64]:
    """The chunk with every column Savitzky-Golay filtered from row `clock` on, at #UPSAMPLE samples per row.

    The filter runs over `chunk[clock - FILTER_CONTEXT_ROWS:]` resampled linearly, and only rows from `clock` on are
    written back: the rows before it are context, never commanded.

    @param chunk `[rows, columns]`.
    @param clock First row written.
    @return A filtered copy; `chunk` itself when the rows from the context on are too few for the window.
    """
    start = max(0, clock - FILTER_CONTEXT_ROWS)
    segment = chunk[start:]
    count = segment.shape[0]
    fine = np.arange((count - 1) * UPSAMPLE + 1, dtype=np.float64) / UPSAMPLE
    upsampled = np.column_stack([np.interp(fine, np.arange(count), column) for column in segment.T])
    filtered = savgol_filter(upsampled, SAVGOL_WINDOW, SAVGOL_POLYORDER)
    if filtered is upsampled:
        return chunk
    out = chunk.copy()
    out[clock:] = filtered[::UPSAMPLE][clock - start :]
    return out


def anchor_row(
    previous: npt.NDArray[np.float64], pose: npt.NDArray[np.float64], commanded: npt.NDArray[np.float64] | None
) -> int | None:
    """Row of the previous chunk the loop runs next at the observation, or None for an arm off that plan.

    With the joints the loop last commanded, the row after the one they are (the last row when they are the last
    row: the arm holds there); the arm is off the plan when those joints are on no row or the measured pose is far
    from them. Without, the row nearest the measured pose, which trails the command by the arm's lag.

    @param previous `[rows, joints]` the previous chunk.
    @param pose `[joints]` the measured pose.
    @param commanded The joints last commanded, or None; a whole row as the loop wrote it (the gripper after the
        joints) is fine, only its first `joints` values are read.
    @return The row index, or None.
    """
    if commanded is None:
        return nearest_row(previous, pose, MAX_OFFSET_RAD)
    commanded = commanded[: previous.shape[1]]
    if np.linalg.norm(pose - commanded) > MAX_OFFSET_RAD:
        return None
    row = nearest_row(previous, commanded, MAX_OFFSET_RAD)
    return None if row is None else min(row + 1, previous.shape[0] - 1)


def nearest_row(chunk: npt.NDArray[np.float64], pose: npt.NDArray[np.float64], max_offset_rad: float) -> int | None:
    """Index of the row of `chunk` nearest `pose`, or None when even that row is farther than `max_offset_rad`.

    Of rows equally near, the first, unless they run to the plan's last row: a plan that ends at rest repeats its last
    row and the arm holding there is at its end, while a hold mid-plan is left from its first row on.

    @param chunk `[rows, joints]` targets.
    @param pose `[joints]` measured pose.
    @param max_offset_rad Largest L2 distance, radians, that still counts as being on the plan.
    @return The row index, or None.
    """
    distances = np.linalg.norm(chunk - pose, axis=1)
    index = int(np.argmin(distances))
    if (distances[index:] == distances[index]).all():
        index = len(distances) - 1
    return index if distances[index] <= max_offset_rad else None


def savgol_filter(rows: npt.NDArray[np.float64], window: int, polyorder: int) -> npt.NDArray[np.float64]:
    """Savitzky-Golay filter along the rows, each column on its own.

    Each row becomes the centre value of the least-squares polynomial of order `polyorder` through the `window`
    rows around it; the first and last half-windows take the polynomial fitted through the edge window, as scipy's
    `interp` mode does. So per-row jitter goes, while a trajectory the polynomial can follow (a cubic, for order 3)
    passes unchanged. A numpy-only implementation, because scipy is not a dependency of this package.

    @param rows `[count, columns]`.
    @param window Odd row count of the window.
    @param polyorder Polynomial order, below `window`.
    @return `[count, columns]`; `rows` itself when there are fewer rows than the window.
    """
    count = rows.shape[0]
    if count < window:
        return rows
    half = window // 2
    offsets = np.arange(-half, half + 1, dtype=np.float64)
    basis = np.vander(offsets, polyorder + 1, increasing=True)
    fit = np.linalg.pinv(basis)  # maps a window of values to the polynomial's coefficients
    windows = np.lib.stride_tricks.sliding_window_view(rows, window, axis=0)
    out = np.empty_like(rows)
    out[half : count - half] = windows @ fit[0]  # the constant coefficient is the value at the centre
    out[:half] = basis[:half] @ (fit @ rows[:window])
    out[count - half :] = basis[half + 1 :] @ (fit @ rows[count - window :])
    return out


def _pose(value: Any) -> npt.NDArray[np.float64] | None:
    """The measured joint pose as a flat array, or None when the observation carries none that is numbers."""
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None

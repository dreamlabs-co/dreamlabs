"""`dreamcli robot report`: checks the last run, a line per check, PASS or PROBLEM with the number behind it
(README *Run report*).

It reads what the loop writes for each run (`trace_path`, on by default): the trace, one row per commanded tick
(`robots.control.trace_columns()`), and the model inputs kept beside it (`robots.control.trace_inputs_dir()`), one set
a second of the frames the client was handed, with the state and the prompt handed with them. Nothing is sent
anywhere.

The checks: each camera's frames as sent (black, frozen, stretched), the state sent (numbers), the loop (pauses),
and each joint, commanded against achieved (diverging, oscillating). A camera or joint with a problem gets a PROBLEM
line of its own; the cameras that pass share one PASS line, as do each arm's joints that pass. Joints are counted
from 0, as the trace's columns are; times are from the run's first commanded tick. The constants below are what
counts as a problem, and #FIXES what to do about each, the worst first. plot() draws each joint, commanded against
achieved, to one PNG.

@see report(), worst(), plot()
"""

import csv
import dataclasses
import json
import pathlib
import re
from typing import Any

import numpy as np
import numpy.typing as npt

from robots import cameras, control, errors

## A joint whose achieved position is off its command by more than this, radians, for #DIVERGE_S or longer diverges.
DIVERGE_RAD = 0.1
DIVERGE_S = 0.5
## A position that turns back at this rate or faster, Hz, by more than #OSCILLATION_RAD peak to peak around its own
## one-second average, for #OSCILLATION_S or longer, oscillates.
OSCILLATION_HZ = 3.0
OSCILLATION_RAD = 0.01
OSCILLATION_S = 1.0
## A step smaller than this, radians per tick, is noise: it neither starts nor reverses a direction.
STEP_NOISE_RAD = 0.002
## A gap between two commanded ticks this much longer than a tick, seconds, is a pause: nothing was commanded.
HOLD_S = 0.25
## The most ticks the achieved position is searched to trail its command by (the lag).
LAG_MAX_TICKS = 20
## A joint whose command spans less than this over the run, radians, is still: it has no lag.
STILL_RAD = 0.02
## A frame whose mean grey level, 0 to 255, is under this is black.
BLACK_LEVEL = 12.0
## Two frames whose grey levels differ by less than this on average are the same picture (a live camera's noise alone
## differs by more); a camera that sends the same picture for #FROZEN_S or longer is frozen.
FROZEN_LEVEL = 0.1
FROZEN_S = 3.0

## What to do about each kind of problem, the worst kind first: the report names the worst problem a run has.
FIXES = {
    "nan": "a value is not a number: fix what your driver's read() returns",
    "black": "the camera sends black frames: check its exposure, lens and cable",
    "frozen": "the camera repeats one picture: check its cable and driver",
    "diverging": "the joint does not reach its commands: check what your driver or bridge does with each step "
    "(speed limits, dropped or refused steps)",
    "stretched": "the model sees this camera stretched: capture at the shape it is sent at",
    "oscillating": "the joint oscillates around its command: lower its controller's gains",
    "pause": "the loop stopped commanding: check the network round trip and that every camera keeps sending",
}

## Each panel of plot(), pixels, and the colours of its two lines (BGR): commanded blue, achieved orange.
_PANEL = (320, 150)
_CMD = (180, 100, 25)
_MEAS = (20, 110, 240)

## The trace's per-tick columns the report reads besides the joints.
_COLUMNS = ("t_s", "chunk_id")


@dataclasses.dataclass(frozen=True)
class Check:
    """One line of the report.

    Attributes:
    - name: What was checked: `cameras top, left, right`, `camera top`, `state`, `loop`, `right arm`,
      `right arm joint 3`.
    - kind: The worst problem found, a key of #FIXES, or None when it passed.
    - text: The numbers when it passed; what is wrong, with the numbers, when not.
    """

    name: str
    kind: str | None
    text: str

    def line(self) -> str:
        """The check as one line.

        @return `PASS     name: text` or `PROBLEM  name: text`.
        """
        return f"{'PASS' if self.kind is None else 'PROBLEM':8} {self.name}: {self.text}"


@dataclasses.dataclass(frozen=True)
class Trace:
    """The last run of a trace file, one entry per commanded tick.

    Attributes:
    - start_s: The run's first commanded tick on the loop's clock, seconds.
    - t_s: Each tick from the first, `[ticks]` seconds.
    - measured: Per arm, `"left"` and `"right"`, the joints it read, `[ticks, joints]` radians.
    - commanded: Per arm, the joints it was commanded, `[ticks, joints]` radians.
    """

    start_s: float
    t_s: npt.NDArray[np.float64]
    measured: dict[str, npt.NDArray[np.float64]]
    commanded: dict[str, npt.NDArray[np.float64]]


@dataclasses.dataclass(frozen=True)
class Report:
    """What report() found.

    Attributes:
    - run: The run in one line: its length, rate and prompt.
    - checks: Every line, the cameras first, then the state, the loop and each arm: a camera or joint with a
      problem has its own, and the rest of the cameras, or of an arm's joints, share one PASS line after them.
    - trace: The run's trace, for plot().
    """

    run: str
    checks: list[Check]
    trace: Trace


def report(trace_path: pathlib.Path) -> Report:
    """Reads the trace's last run and its model inputs, and checks them.

    @param trace_path The trace's CSV file.
    @return The #Report.
    @throws ConfigError A file is missing or does not belong to the trace's last run.
    """
    trace = read_trace(trace_path)
    start, *sets = read_inputs(trace_path, trace)
    if not sets:
        raise errors.ConfigError("the trace's last run kept no model inputs: it ended before handing any frames")
    hz = float(start["control_hz"])
    checks = _cameras(trace_path, sets, trace.start_s) + [_state(sets), _pauses(trace, hz)]
    for arm in control.ARMS:
        checks += _arm(arm, trace, hz)
    prompts = " then ".join(dict.fromkeys(f'"{s["state"]["prompt"]}"' for s in sets))
    run = f"{trace.t_s[-1]:.0f} s at {hz:g} Hz, {prompts}"
    return Report(run, checks, trace)


def worst(checks: list[Check]) -> Check | None:
    """The check whose problem comes first in #FIXES, the first such one.

    @param checks report()'s checks.
    @return That check; None when every check passed.
    """
    failed = [check for check in checks if check.kind is not None]
    return min(failed, key=lambda check: list(FIXES).index(str(check.kind)), default=None)


def plot(trace: Trace, checks: list[Check], out: pathlib.Path) -> None:
    """Draws one panel per joint, commanded against achieved over the run, a row per arm, to one PNG.

    @param trace The run.
    @param checks report()'s checks; a joint with a problem has its name in red.
    @param out The PNG to write; replaced when it exists.
    """
    import cv2

    failed = {check.name for check in checks if check.kind is not None}
    width, height = _PANEL
    cols = max(trace.commanded[arm].shape[1] for arm in control.ARMS)
    image = np.full((height * len(control.ARMS) + 30, width * cols, 3), 255, np.uint8)  # [H, W, 3]
    key = f"blue: commanded   orange: achieved   (rad, each panel t=0 to {trace.t_s[-1]:.0f} s)"
    cv2.putText(image, key, (8, 20), 0, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    for row, arm in enumerate(control.ARMS):
        for j in range(trace.commanded[arm].shape[1]):
            name = f"{arm} arm joint {j}"
            x0, y0 = j * width, 30 + row * height
            series = (trace.commanded[arm][:, j], trace.measured[arm][:, j])
            low, high = np.nanmin(np.concatenate(series)), np.nanmax(np.concatenate(series))
            span = max(high - low, 1e-3)
            cv2.rectangle(image, (x0 + 4, y0 + 4), (x0 + width - 4, y0 + height - 4), (200, 200, 200), 1)
            colour = (0, 0, 220) if name in failed else (0, 0, 0)
            label = f"{name}  {low:.2f} to {high:.2f}"
            cv2.putText(image, label, (x0 + 10, y0 + 20), 0, 0.42, colour, 1, cv2.LINE_AA)
            xs = x0 + 10 + trace.t_s / max(float(trace.t_s[-1]), 1e-3) * (width - 20)  # [ticks]
            for values, line in zip(series, (_CMD, _MEAS), strict=True):
                ys = y0 + height - 10 - (values - low) / span * (height - 40)  # [ticks]
                points = np.stack([xs, ys], axis=1)  # [ticks, 2]
                ok = np.isfinite(ys)  # [ticks]
                cuts = np.flatnonzero(np.diff(ok)) + 1  # where a run of numbers starts or ends
                pieces = zip(np.split(points, cuts), np.split(ok, cuts), strict=True)
                runs = [p.round().astype(np.int32) for p, k in pieces if k[0]]
                cv2.polylines(image, runs, False, line, 1, cv2.LINE_AA)  # a gap where values are not numbers
    if not cv2.imwrite(str(out), image):
        raise errors.ConfigError(f"could not write {out}")


def read_trace(path: pathlib.Path) -> Trace:
    """Reads the last run of a trace: the rows after the chunk count last went back.

    @param path The trace's CSV file.
    @return The run.
    @throws ConfigError The file is missing, has no rows, or is not a trace of the loop.
    """
    if not path.is_file():
        raise errors.ConfigError(f"no trace at {path}: run the robot once, then report")
    with open(path, newline="", encoding="utf-8") as file:
        header, *rows = list(csv.reader(file))
    if not set(_COLUMNS) <= set(header) or not rows:
        raise errors.ConfigError(f"{path} is not a trace with rows: its header is {header[:6]}..., {len(rows)} rows")
    values = np.asarray(rows, dtype=np.float64)  # [rows, columns]
    first = np.flatnonzero(np.diff(values[:, header.index("chunk_id")]) < 0)
    run = values[int(first[-1]) + 1 if len(first) else 0 :]  # [ticks, columns]
    column = {name: run[:, i] for i, name in enumerate(header)}

    def joints(prefix: str) -> npt.NDArray[np.float64]:
        names = [name for name in header if re.fullmatch(rf"{prefix}\d+", name)]
        return np.stack([column[name] for name in names], axis=1)  # [ticks, joints]

    return Trace(
        start_s=float(column["t_s"][0]),
        t_s=column["t_s"] - column["t_s"][0],
        measured={arm: joints(f"measured_{arm[0]}") for arm in control.ARMS},
        commanded={arm: joints(f"commanded_{arm[0]}") for arm in control.ARMS},
    )


def read_inputs(trace_path: pathlib.Path, trace: Trace) -> list[dict[str, Any]]:
    """Reads the model inputs the loop kept beside the trace for its last run.

    @param trace_path The trace's CSV file.
    @param trace Its last run.
    @return The run's `run_start` record, then its `inputs` records in order.
    @throws ConfigError There are none, or the last run they record started after the trace's.
    """
    path = control.trace_inputs_dir(str(trace_path)) / control.TRACE_INPUTS_NAME
    if not path.is_file():
        raise errors.ConfigError(
            f"no model inputs beside the trace at {path}: this release's loop keeps them; run again"
        )
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    starts = [i for i, record in enumerate(records) if record["kind"] == "run_start"]
    if not starts or records[starts[-1]]["t_s"] > trace.start_s:
        raise errors.ConfigError(f"{path} does not hold the trace's last run; run again, then report")
    return records[starts[-1] :]


def _arm(arm: str, trace: Trace, hz: float) -> list[Check]:
    """A PROBLEM line per joint of the arm that has one, then one PASS line for the joints that follow their commands,
    with their worst error and lag."""
    cmd, meas = trace.commanded[arm], trace.measured[arm]  # [ticks, joints]
    joints = [_joint(f"{arm} arm joint {j}", trace.t_s, cmd[:, j], meas[:, j], hz) for j in range(cmd.shape[1])]
    found = [check for check in joints if check is not None]
    ok = [j for j, check in enumerate(joints) if check is None]
    if not ok:
        return found
    err = float(np.max(np.abs(cmd[:, ok] - meas[:, ok])))
    ticks = sorted({_lag_ticks(cmd[:, j], meas[:, j]) for j in ok if np.ptp(cmd[:, j]) >= STILL_RAD})
    ms = [f"{tick * 1000 / hz:.0f}" for tick in (ticks[0], ticks[-1])] if ticks else []
    lag = "still" if not ms else f"lag {ms[0]} ms" if ms[0] == ms[1] else f"lag {ms[0]} to {ms[1]} ms"
    if len(ok) == 1:
        which = "the other joint follows its command" if found else "its one joint follows its command"
    else:
        which = f"{'other' if found else 'all'} {len(ok)} joints follow their commands"
    return found + [Check(f"{arm} arm", None, f"{which}, worst error {err:.3f} rad, {lag}")]


def _joint(
    name: str, t: npt.NDArray[np.float64], cmd: npt.NDArray[np.float64], meas: npt.NDArray[np.float64], hz: float
) -> Check | None:
    """One joint, commanded against achieved: what is wrong, or None when it follows its command."""
    period = 1.0 / hz
    bad = ~np.isfinite(cmd) | ~np.isfinite(meas)
    if bad.any():
        return Check(name, "nan", f"{int(bad.sum())} ticks are not a number, first at t={t[np.argmax(bad)]:.1f} s")
    err = cmd - meas  # [ticks]
    lag = "still" if np.ptp(cmd) < STILL_RAD else f"lag {_lag_ticks(cmd, meas) * period * 1000:.0f} ms"
    diverge = _lasting(np.abs(err) > DIVERGE_RAD, t, DIVERGE_S, period)
    if diverge:
        a, b = max(diverge, key=lambda span: t[span[1] - 1] - t[span[0]])
        peak = float(np.max(np.abs(err[a:b])))
        return Check(
            name, "diverging", f"off its command by {peak:.2f} rad {_when(t, a, b, period, len(diverge))}, {lag}"
        )
    for kind, series in (("achieved", meas), ("commanded", cmd)):
        spans = _oscillating(series, t, hz)
        if spans:
            a, b, freq, swing = spans[0]
            when = _when(t, a, b, period, len(spans))
            return Check(
                name, "oscillating", f"{kind} oscillates at {freq:.0f} Hz, {swing:.3f} rad peak to peak {when}"
            )
    return None


def _lasting(
    mask: npt.NDArray[np.bool_], t: npt.NDArray[np.float64], min_s: float, period: float
) -> list[tuple[int, int]]:
    """The `[start, end)` index runs where `mask` holds for at least `min_s` seconds, a tick counting one period."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return [
        (int(a), int(b)) for a, b in zip(edges[0::2], edges[1::2], strict=True) if t[b - 1] - t[a] + period >= min_s
    ]


def _when(t: npt.NDArray[np.float64], a: int, b: int, period: float, count: int) -> str:
    """`for 2.1 s at t=14.0 s`, with how many times in all when it happened more than once."""
    again = f", {count} times in all" if count > 1 else ""
    return f"for {t[b - 1] - t[a] + period:.1f} s at t={t[a]:.1f} s{again}"


def _lag_ticks(cmd: npt.NDArray[np.float64], meas: npt.NDArray[np.float64]) -> int:
    """How many ticks the achieved position trails the command by: the shift with the smallest RMS difference."""
    shifts = range(min(LAG_MAX_TICKS, len(cmd) - 2) + 1)
    rms = [np.sqrt(np.mean((cmd[: len(cmd) - k] - meas[k:]) ** 2)) for k in shifts]
    return int(np.argmin(rms))


def _oscillating(
    x: npt.NDArray[np.float64], t: npt.NDArray[np.float64], hz: float
) -> list[tuple[int, int, float, float]]:
    """The runs where `x` turns back at #OSCILLATION_HZ or faster with a swing over #OSCILLATION_RAD.

    @return Per run, its `[start, end)` ticks, its rate in Hz and its swing, peak to peak around the one-second
        average, radians.
    """
    step = np.diff(x)  # [ticks - 1]
    moving = np.flatnonzero(np.abs(step) > STEP_NOISE_RAD)
    turns = np.zeros(len(x))
    turned = moving[1:][np.sign(step[moving[1:]]) != np.sign(step[moving[:-1]])]
    turns[turned + 1] = 1.0
    width = max(2, round(hz))
    rate = np.convolve(turns, np.ones(width), "same") * hz / width  # [ticks] turns per second
    window = np.ones(width)
    detrended = x - np.convolve(x, window, "same") / np.convolve(np.ones(len(x)), window, "same")  # [ticks]
    runs = []
    for a, b in _lasting(rate >= 2 * OSCILLATION_HZ, t, OSCILLATION_S, 1.0 / hz):
        swing = float(np.ptp(detrended[a:b]))
        if swing > OSCILLATION_RAD:
            runs.append((a, b, float(turns[a:b].sum()) / 2 / max(t[b - 1] - t[a], 1.0 / hz), swing))
    return runs


def _pauses(trace: Trace, hz: float) -> Check:
    """The loop: whether it commanded nothing for #HOLD_S or longer, the arms standing while it waited."""
    gaps = np.diff(trace.t_s) - 1.0 / hz  # [ticks - 1] seconds past one tick
    holds = np.flatnonzero(gaps >= HOLD_S)
    if not len(holds):
        return Check("loop", None, f"commanded every tick, longest gap {gaps.max(initial=0.0) + 1.0 / hz:.2f} s")
    worst_gap = int(holds[np.argmax(gaps[holds])])
    what = f"commanded nothing {len(holds)} times for {HOLD_S:g} s or longer"
    return Check("loop", "pause", f"{what}, longest {gaps[worst_gap]:.1f} s at t={trace.t_s[worst_gap]:.1f} s")


def _cameras(trace_path: pathlib.Path, sets: list[dict[str, Any]], start_s: float) -> list[Check]:
    """A PROBLEM line per camera whose frames as sent are black, frozen or stretched against the capture, then one
    PASS line for the cameras that are not."""
    folder = control.trace_inputs_dir(str(trace_path))
    times = np.asarray([s["t_s"] for s in sets], dtype=np.float64) - start_s  # [sets]
    checks = []
    passed: dict[str, str] = {}  # camera: the size it sent
    for name in sets[0]["frames"]:
        levels, same, (width, height) = _scan([folder / s["frames"][name]["file"] for s in sets])
        captured = sets[0]["frames"][name]["captured"]
        found = []
        black = levels < BLACK_LEVEL
        if black.any():
            found.append(
                (
                    "black",
                    f"{int(black.sum())} of {len(sets)} frames black, first at t={times[np.argmax(black)]:.1f} s",
                )
            )
        frozen = [(a - 1, b) for a, b in _lasting(same, times, FROZEN_S, control.TRACE_INPUTS_EVERY_S)]
        if frozen:
            a, b = max(frozen, key=lambda span: times[span[1] - 1] - times[span[0]])
            found.append(("frozen", f"the same picture {_when(times, a, b, 0.0, len(frozen))}"))
        stretch = (width / height) / (captured[0] / captured[1])
        if abs(stretch - 1) >= cameras.SHAPE_TOLERANCE:
            look = f"{stretch:.2f}x wider" if stretch > 1 else f"{1 / stretch:.2f}x taller"
            sizes = f"captured {captured[0]}x{captured[1]}, sent {width}x{height}"
            found.append(("stretched", f"{sizes}: the model sees it {look}"))
        if not found:
            passed[name] = f"{width}x{height}"
            continue
        kind = min((k for k, _ in found), key=list(FIXES).index)
        checks.append(Check(f"camera {name}", kind, "; ".join(text for _, text in found)))
    if not passed:
        return checks
    names = ", ".join(passed)
    shapes = set(passed.values())
    sent = f"sent {shapes.pop()}" if len(shapes) == 1 else "sent " + ", ".join(f"{s} ({n})" for n, s in passed.items())
    name = f"camera {names}" if len(passed) == 1 else f"cameras {names}"
    return checks + [Check(name, None, f"{sent}, no black or frozen frames")]


def _scan(files: list[pathlib.Path]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_], tuple[int, int]]:
    """Each frame's mean grey level, whether it is the same picture as the one before, and the first one's size.

    One frame is decoded at a time, so a long run's frames are never all in memory.
    """
    import cv2

    def grey(file: pathlib.Path) -> npt.NDArray[np.float64]:
        image = cv2.imdecode(np.frombuffer(file.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise errors.ConfigError(f"{file} is not a JPEG")
        return image.astype(np.float64)

    last = grey(files[0])  # [height, width]
    size = (last.shape[1], last.shape[0])
    levels, same = [float(last.mean())], [False]
    for file in files[1:]:
        frame = grey(file)  # [height, width]
        same.append(frame.shape == last.shape and float(np.abs(frame - last).mean()) < FROZEN_LEVEL)
        levels.append(float(frame.mean()))
        last = frame
    return np.asarray(levels), np.asarray(same), size


def _state(sets: list[dict[str, Any]]) -> Check:
    """The state sent each second: whether every value was a number."""
    keys = [k for k in sets[0]["state"] if k.startswith("state.")]
    count = 0
    for key in keys:
        values = np.asarray([s["state"][key] for s in sets], dtype=np.float64)  # [sets, dims]
        bad = ~np.isfinite(values).all(axis=1)  # [sets]
        if bad.any():
            return Check("state", "nan", f"{key} not a number in {int(bad.sum())} of {len(sets)} sets sent")
        count += values.shape[1]
    return Check("state", None, f"{count} values in {len(keys)} parts, all numbers")

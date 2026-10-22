"""The slow-network warning: one `DreamSlowNetworkWarning` when a client's recent round trips stay slow.

A control loop on a slow link still gets every action, only later, and the robot then moves less smoothly: in sync
pacing (`dream.pacing`) the arms hold through each round trip. Nothing fails, so nothing else would say so. After every
successful `get_action` the client hands its `last_timing` to its #Watch, which keeps the last #WINDOW of them; when
their median round trip is at or above #SLOW_MS the watch words one warning, saying where the time goes: the
network, when it is the largest part of the round trip (with the upload rate when the kernel reports one), else the
model. The client issues it as a Python warning of category `dream.DreamSlowNetworkWarning` and logs it on the
`dream` logger (warn()). The watch then stays quiet for #QUIET_S while the round trips stay slow, unless their median
first recovers under #RECOVERED_MS and then degrades again, which warns at once.

The warning is diagnostics: it never raises out of `get_action` and never delays it, and its text carries only
milliseconds and megabits per second. A caller silences it with the standard filter,
`warnings.filterwarnings("ignore", category=dream.DreamSlowNetworkWarning)`; there is no environment variable.
"""

import collections
import inspect
import logging
import os
import statistics
import warnings
from collections.abc import Iterable
from typing import Any

from dream import errors

## Median round trip, milliseconds, at or above which the recent round trips are slow: past a second the arms visibly
## pause between actions.
SLOW_MS: float = 1000.0

## Median round trip, milliseconds, under which slow round trips count as recovered, so the next slowdown warns at
## once; below #SLOW_MS so a median hovering at the line does not warn over and over.
RECOVERED_MS: float = 800.0

## Round trips the median is taken over: one slow reply among quick ones never warns.
WINDOW: int = 5

## Seconds a watch stays quiet after a warning while the round trips stay slow; a reminder follows after that.
QUIET_S: float = 300.0

## The logger the warning is also written to; a child of `dream`, so the client's log file gets it.
_log = logging.getLogger("dream.slow_network")

## The `dream` package's directory: the warning names the first line outside it, the caller's own `get_action` call,
## whether that went through `DreamClient` or straight to the inference client.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


class Watch:
    """One client's last #WINDOW round trips, and whether a slow median among them is worth a warning now.

    Attributes:
    - is_armed: Whether the next slow median warns at once: true at the start and after a recovery, false after a
      warning until then.
    - warned_at_s: When the last warning was worded, on the caller's monotonic clock, seconds; None before the first.
    """

    def __init__(self) -> None:
        """Starts armed, with no round trip seen."""
        self.is_armed = True
        self.warned_at_s: float | None = None
        self._recent: collections.deque[dict[str, Any]] = collections.deque(maxlen=WINDOW)

    def record(self, timing: dict[str, Any] | None, now_s: float) -> str | None:
        """Takes one successful `get_action`'s `last_timing`, and words a warning when one is due now.

        @param timing The client's `last_timing` for the call; None (nothing reached the wire) is ignored.
        @param now_s A monotonic clock, seconds.
        @return The warning's text when the median of the last #WINDOW round trips is at or above #SLOW_MS and the
            watch is armed or has been quiet for #QUIET_S; None otherwise.
        """
        if timing is None:
            return None
        self._recent.append(timing)
        if len(self._recent) < WINDOW:
            return None
        rtt_ms = statistics.median(t["rtt_ms"] for t in self._recent)
        if rtt_ms < RECOVERED_MS:
            self.is_armed = True
            return None
        if rtt_ms < SLOW_MS:
            return None
        if not self.is_armed and self.warned_at_s is not None and now_s - self.warned_at_s < QUIET_S:
            return None
        self.is_armed = False
        self.warned_at_s = now_s
        return message(list(self._recent))


def message(timings: list[dict[str, Any]]) -> str:
    """Words the warning for slow round trips: the network or the model, from the median of each part.

    The parts of a round trip are the network (`network_ms`), our servers (`gateway_ms` less `serve_infer_ms`) and the
    model (`serve_infer_ms`); the network is to blame when its median is larger than each other part's, the model
    otherwise. Without a network figure (no answer carried timing) the text names no cause.

    @param timings The `last_timing` dicts of the round trips judged, at least one.
    @return One line, with the median round trip, its network part and the upload rate when known.
    """
    rtt_ms = statistics.median(t["rtt_ms"] for t in timings)
    network_ms = _median_of(t.get("network_ms") for t in timings)
    ours_ms = _median_of(t.get("gateway_ms") for t in timings)
    model_ms = _median_of(t.get("serve_infer_ms") for t in timings)
    servers_ms = _median_of(_servers_ms(t) for t in timings)
    up_mbps = _median_of(t.get("up_mbps") for t in timings)
    smooth = "the robot's motion may be less smooth. See client.last_timing."
    if network_ms is None:
        return f"get_action round trips are slow: about {_about(rtt_ms)} ms; {smooth}"
    if all(part is None or network_ms > part for part in (servers_ms, model_ms)):
        upload = f", upload {up_mbps:.1f} Mb/s" if up_mbps is not None else ""
        return (
            f"Network is slow: get_action round trips are about {_about(rtt_ms)} ms "
            f"(network {_about(network_ms)} ms{upload}); {smooth}"
        )
    ours = f"the model and our servers {_about(ours_ms)} ms, " if ours_ms is not None else ""
    return (
        f"The model is responding slowly: get_action round trips are about {_about(rtt_ms)} ms "
        f"({ours}network {_about(network_ms)} ms); {smooth}"
    )


def warn(text: str) -> None:
    """Issues the warning as `DreamSlowNetworkWarning` at the caller's line and logs it on `dream`.

    Never raises: a filter that turns warnings into errors (`-W error`) would otherwise throw away an action the call
    already holds, so an escalated warning is dropped here; the log line still records it.

    @param text The warning's text, from message().
    """
    _log.warning(text)
    # Level 1 is this frame; walking out of the package one frame per level ends on the caller's line and its level.
    level, frame = 1, inspect.currentframe()
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE_DIR):
        level, frame = level + 1, frame.f_back
    del frame  # a frame held in a local keeps its whole stack alive
    try:
        warnings.warn(text, errors.DreamSlowNetworkWarning, stacklevel=level)
    except Warning:
        pass


def _servers_ms(timing: dict[str, Any]) -> float | None:
    """The time on our side outside the model's inference for one round trip, or None without both figures."""
    gateway_ms, model_ms = timing.get("gateway_ms"), timing.get("serve_infer_ms")
    return gateway_ms - model_ms if gateway_ms is not None and model_ms is not None else None


def _median_of(values: Iterable[Any]) -> float | None:
    """The median of the values that are numbers, or None when none is."""
    known = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return statistics.median(known) if known else None


def _about(ms: float) -> int:
    """A typical time to the nearest 10 milliseconds, as the warning says "about" it."""
    return int(round(ms / 10.0) * 10)

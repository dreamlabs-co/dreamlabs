"""Pacing: when the next chunk is asked for, and sync or async by latency.

A control loop plays a chunk one row per tick and asks for the next one before or after the rows run out; that
choice decides whether the arms keep moving through an inference or hold through it, and it depends on one number
the client measures anyway: how long `get_action` takes. So the choice lives here, beside the measurement, and the
loop asks (`Pacer.refill_due`) instead of sizing anything itself.

Three numbers, each a config key of `DreamClient.policy()` with the default below:

- `execution_horizon` (30): in async, rows of each chunk the arms execute before the next chunk takes over. A
  checkpoint returns more (its action horizon, 48 for the Dream-1 base model); the rows past this are runway, played
  only when the next chunk is late. Sync does not use it: there the whole chunk runs.
- `action_hz` (20): the loop's row rate, rows per second; it turns a round trip into rows.
- `sync_above_ms` (700): the round trip past which the client paces synchronously.

Two modes, chosen by the median of the last #RTT_WINDOW round trips (each the whole `get_action`, validation and
smoothing included), so one reply that waited behind another robot's inference on a shared serve does not flip it:

- async (median round trip up to `sync_above_ms`): the next request goes out while the current chunk is still
  running, timed so its reply lands as the arms finish `execution_horizon` rows of it. The reply is installed past
  the rows the arms ran during the round trip, so a chunk must hold `round trip x action_hz + execution_horizon`
  rows; at 700 ms and 20 Hz that is 14 + 30 = 44 of 48, and past it a prefetched reply would arrive with no runway
  left.
- sync (median round trip above `sync_above_ms`): the arms run the whole chunk, every row the serve returned (the
  action horizon, 48 for the Dream-1 base model), and the next request goes out only once they have; the arms hold
  through the inference, so there is nothing to gain by dropping rows of the plan and paying for another round trip
  sooner. A sync request is sent while the arms hold, so its reply installs whole. The one exception is the first
  sync chunk after a switch: its request went out in async while the arms were moving, so the loop installs it past
  the rows they ran during the round trip (about 15 at 750 ms and 20 Hz), and the arms run the remaining 33 or so.

The switch is checked on every reply, so a link that slows down is paced synchronously once most recent replies are
slow (the third slow one of five) and one that recovers is back to async as soon as most are fast again.

The async lead is sized from the shortest recent round trip, the one no queue added to. On a serve shared by two
robots a reply that waited behind the other's inference comes back late; sizing the lead from it (or from the
largest, as the client did before) sends every following request early, onto the other robot's next inference,
which makes that reply late too. Sized from the shortest, a late reply only moves this robot's next cycle later by
the wait, away from the other robot. A link that really got slower is still covered: the rows past the execution
horizon are runway (18 of 48 at 20 Hz, 0.9 s), and the median moves the mode to sync once most replies are slow.
See #Pacer.
"""

import collections
import logging
import math
import statistics

## Rows of each chunk the arms execute before the next one takes over.
EXECUTION_HORIZON = 30

## The loop's row rate, rows per second: the station's control rate.
ACTION_HZ = 20.0

## Round trip, milliseconds, past which the client paces synchronously.
SYNC_ABOVE_MS = 700.0

## Round trips the mode and the async lead are taken from: the median decides the mode, the shortest sizes the lead.
## Five is a few chunks: one slow reply moves neither, and a link that changed is believed within seconds.
RTT_WINDOW = 5

## Seconds of queued rows the async path never lets the buffer drop below without a request in flight, whatever
## the execution horizon says: a chunk shorter than the round trip (a sync chunk on the way back to async) would
## otherwise run dry before its successor lands.
RUNWAY_MARGIN_S = 0.25

ASYNC = "async"
SYNC = "sync"

_log = logging.getLogger("dream.pacing")


class Pacer:
    """Measures round trips, picks the mode from them, and answers the question a control loop asks: refill now?

    Attributes:
    - execution_horizon: In async, rows of each chunk the arms execute before the next one takes over.
    - action_hz: The loop's row rate, rows per second.
    - sync_above_ms: Round trip, milliseconds, past which the mode is sync.
    """

    def __init__(
        self,
        execution_horizon: int = EXECUTION_HORIZON,
        action_hz: float = ACTION_HZ,
        sync_above_ms: float = SYNC_ABOVE_MS,
    ) -> None:
        """Starts in async with no round trip measured; the first request is the one an empty buffer makes anyway.

        @param execution_horizon Rows of each chunk to execute; at least 1.
        @param action_hz Row rate, rows per second; positive.
        @param sync_above_ms Sync threshold, milliseconds; positive.
        """
        self.execution_horizon = execution_horizon
        self.action_hz = action_hz
        self.sync_above_ms = sync_above_ms
        self._mode = ASYNC
        self._recent_rtt_s: collections.deque[float] = collections.deque(maxlen=RTT_WINDOW)

    @property
    def mode(self) -> str:
        """`"async"` or `"sync"`, as the median of the recent round trips decided; `"async"` before the first."""
        return self._mode

    def record(self, rtt_s: float) -> None:
        """Takes one successful `get_action`'s round trip and sets the mode from the recent median, logging a change.

        @param rtt_s The round trip, seconds.
        """
        self._recent_rtt_s.append(rtt_s)
        median_s = statistics.median(self._recent_rtt_s)
        mode = SYNC if median_s * 1e3 > self.sync_above_ms else ASYNC
        if mode != self._mode:
            _log.info(
                "pacing %s: get_action's median round trip is %.0f ms (last %.0f ms), %s the %.0f ms threshold",
                mode,
                median_s * 1e3,
                rtt_s * 1e3,
                "over" if mode == SYNC else "within",
                self.sync_above_ms,
            )
        self._mode = mode

    def refill_due(self, executed: int, remaining: int) -> bool:
        """Whether the next request should go out now, given none is in flight.

        Sync: once no row of the chunk is left; the whole chunk runs. Async: once the rows executed of the current
        chunk plus the rows the round trip will take reach `execution_horizon`, so the reply lands as the last of
        those rows runs; or once the rows left are worth less than the round trip plus #RUNWAY_MARGIN_S, so the
        buffer never runs dry first. The round trip is the shortest of the last #RTT_WINDOW, the one no queue at the
        serve added to: a reply that waited behind another robot then moves this robot's next request later instead
        of earlier, and a late reply eats into the runway rows rather than holding the arms. Before any round trip is
        measured the lead is zero.

        @param executed Rows of the current chunk the arms have executed since it was installed.
        @param remaining Rows of it still queued.
        @return True when a request should start.
        """
        if remaining <= 0:
            return True
        if self._mode == SYNC:
            return False
        rtt_s = min(self._recent_rtt_s, default=0.0)
        lead_rows = math.ceil(rtt_s * self.action_hz)
        return executed + lead_rows >= self.execution_horizon or remaining / self.action_hz <= rtt_s + RUNWAY_MARGIN_S

"""A page on this computer of every round trip: the frames the model saw, the state, the chunk, timing and refusals.

#DebugView serves it on 127.0.0.1 from what DreamClient.on_round_trip() hands it, and from nothing else: no frame or
value leaves this computer for it. Open its `url` in a browser before the arms move, or show it in a Jupyter cell by
leaving the view as the cell's last line. Each #RoundTrip is what one get_action() sent and got back, so a camera on
the wrong key, a frame of the wrong size or colours, a stale camera or a state in the wrong units shows on the page
as the model gets it.

@par Example
```python
import dream

with dream.DreamClient() as client:
    client.policy(config)
    view = dream.DebugView(client)
    print(view.url)  # open it: every get_action() shows there
    while running:
        client.set_observation(frames, observation)
        action = client.get_action()
```

`python -m dream.check` also hands the view its scripted motion (DebugView.show_arms()), and the page then draws
both arms in 3D above what the model sees.

@see DebugView, RoundTrip, DreamClient.on_round_trip()
"""

from __future__ import annotations

import collections
import dataclasses
import importlib.resources
import json
import math
import threading
import time
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import TYPE_CHECKING, Any

import numpy as np

from dream import _window
from dream.errors import DreamError, DreamServerError

if TYPE_CHECKING:
    from dream.facade import DreamClient

__all__ = ["DebugView", "RoundTrip"]

## Round trips the page keeps, newest last; an older one is dropped as a new one arrives.
KEEP = 20

## The files the page draws the arms from (DebugView.show_arms()), in the package's `arms` folder: per path, the file
## and its type. yam.glb and yam.joints.json are I2RT's YAM (MIT License, arms/LICENSE).
ARMS = importlib.resources.files("dream") / "arms"
ARM_FILES = {
    "/arms.js": ("arms.js", "text/javascript; charset=utf-8"),
    "/yam.glb": ("yam.glb", "model/gltf-binary"),
    "/yam.joints.json": ("yam.joints.json", "application/json"),
}

## The status codes of a reply that carries an answer, as the client reads them: OK, and OK on a new session.
ANSWERED = ("DA-0000", "DA-0001")

## Past this, in either direction, a state value is not radians or metres: degrees or raw motor counts, most likely.
RADIANS_MAX = 2 * math.pi
## A gripper's state is 0 closed to 1 open; past this margin either side, it is in other units or the wrong way round.
GRIP_MARGIN = 0.05

## The fewest round trips the checks across calls look at; fewer say too little about what moves.
MIN_CALLS = 4
## A state part whose values span less than this over the calls has not moved (radians, metres, 0 to 1).
STILL = 1e-3
## The model's targets for a part moved when they span this much over those calls' chunks, every row included.
TARGETS_MOVED = 0.05
## An arm moved when some part of its state spans this much over the calls.
ARM_MOVED = 0.01
## A still arm's state matches the other arm's on a call when every shared part is within this of it (2 cm, about a
## degree of rotation, 2% of a grip): a pose copied from one arm to the other, give or take a reading's drift.
SAME_POSE = 0.02


@dataclasses.dataclass(frozen=True)
class RoundTrip:
    """One get_action(): what the model was sent and what came back; DreamClient.on_round_trip() hands it over.

    Attributes:
    - sent_s: When the request went out, seconds since the epoch (`time.time()`).
    - frames: Per frame key (`video.<role>`), the base64 JPEGs sent, oldest first: the frames the model saw.
    - observation: Every other key sent: the state and the prompt, as set_observation() was given them.
    - action: The dict get_action() returned; None when it raised.
    - timing: DreamClient.last_timing after the call; None when nothing reached the network, or when a reply given to
      add() carries none (a refusal).
    - error: What get_action() raised, a #DreamError (a refusal is a #DreamServerError, with `code` and `reason`);
      None when it returned.
    - trained: Where the episode's frames or prompt differ from what the fine-tune was trained on, one line each
      (#dream.trained_inputs); empty when they match, for a model that is not one of your fine-tunes, and for add().
    """

    sent_s: float
    frames: dict[str, list[str]]
    observation: dict[str, Any]
    action: dict[str, Any] | None
    timing: dict[str, Any] | None
    error: DreamError | None
    # Last and empty by default, so a RoundTrip built by position before the field existed still builds.
    trained: list[str] = dataclasses.field(default_factory=list)


class DebugView:
    """Serves the page of a client's round trips on 127.0.0.1, from now until close() or the program's exit.

    It registers itself with DreamClient.on_round_trip(), or, for a loop that calls the HTTP API itself, takes each
    call from add(); it keeps the last #KEEP round trips in memory; the page
    asks for them twice a second and draws the newest, or the one picked. Nothing is uploaded, and the page is not
    reachable from another computer (forward the port, `ssh -L <port>:localhost:<port>`, to see it from one).

    Attributes:
    - url: The page, `http://127.0.0.1:<port>/`.
    """

    def __init__(self, client: DreamClient | None, port: int = 0) -> None:
        """Starts the page's server on a thread of its own and registers with `client`.

        @param client The client whose round trips the page shows, any policy it opens, now or later; None for a loop
            that calls the HTTP API itself and hands each call to add().
        @param port The port on 127.0.0.1; 0 lets the system pick a free one, which `url` names.
        @throws OSError The port cannot be listened on (another program holds it).
        """
        self._lock = threading.Lock()
        self._trips: collections.deque[tuple[int, RoundTrip]] = collections.deque(maxlen=KEEP)
        self._count = 0
        self._arms: Callable[[], Mapping[str, Any] | None] | None = None
        self._server = ThreadingHTTPServer(("127.0.0.1", port), _handler(self))
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/"
        threading.Thread(target=self._server.serve_forever, name="dream-debug-view", daemon=True).start()
        if client is not None:
            self.watch(client)

    def watch(self, client: DreamClient) -> None:
        """Shows `client`'s round trips too: a station that opens a client per run hands each one here.

        @param client A client whose every get_action() the page shows from now on.
        """
        client.on_round_trip(self._add)

    def add(self, sent: Mapping[str, Any], reply: Mapping[str, Any] | None) -> None:
        """Shows one call a loop made to the HTTP API itself, without a DreamClient: what it sent and what came back.

        @param sent The JSON body of the loop's inference request, as a dict. Its `observation` is shown: each
            `video.<role>` as that camera's frames (one base64 JPEG, or a list of them oldest first), the rest as the
            state and the prompt.
        @param reply The JSON reply, as a dict: its `actions` are the chunk and its `timing` the timing; a reply whose
            status is not an answer shows as a refusal with its code and reason. None when the call got no reply.
        @throws TypeError `sent` has no `observation` dict.
        """
        observation = sent.get("observation")
        if not isinstance(observation, Mapping):
            raise TypeError("sent is the JSON body of the /v1/get_action request: a dict with an 'observation' dict")
        frames = {
            key: [value] if isinstance(value, str) else list(value)
            for key, value in observation.items()
            if key.startswith("video.")
        }
        rest = {key: value for key, value in observation.items() if not key.startswith("video.")}
        action, error = _answer(reply)
        timing = reply.get("timing") if reply is not None else None
        self._add(
            RoundTrip(sent_s=time.time(), frames=frames, observation=rest, action=action, timing=timing, error=error)
        )

    def show_arms(self, motion: Callable[[], Mapping[str, Any] | None]) -> None:
        """Draws the arms in 3D at the top of the page too, from a scripted motion (python -m dream.check).

        A ghost of each arm moves where the motion commands it, and a solid arm stands at the pose the loop reported at
        its last call, with each hand's commanded path and a dot where each call's pose put the hand; live while the
        arms move, then replayed. A YAM is drawn whole, another arm the check knows as its chain of joints, and an arm
        it does not know as its hand's tool point. For this the page loads three.js from cdn.jsdelivr.net; the frames
        and values still never leave this computer.

        @param motion Returns #dream.check_loop.StandIn.arms(): the targets per row, each call's state and the clock;
            None before the loop's first call.
        """
        self._arms = motion

    def __enter__(self) -> DebugView:
        """Returns the view; leaving the block calls close()."""
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """Calls close().

        @param exc_type The exception's type, when the block raised.
        @param exc The exception, when the block raised.
        @param tb Its traceback.
        """
        self.close()

    def close(self) -> None:
        """Stops the server; the page then says it lost its program."""
        self._server.shutdown()
        self._server.server_close()

    def _repr_html_(self) -> str:
        """The page in a Jupyter cell, when the notebook's browser runs on this computer."""
        return f'<iframe src="{self.url}" width="100%" height="760" style="border:0"></iframe>'

    def _add(self, trip: RoundTrip) -> None:
        """Keeps one round trip, numbered from 1."""
        with self._lock:
            self._count += 1
            self._trips.append((self._count, trip))

    def _list(self) -> list[dict[str, Any]]:
        """The round trips kept, oldest first, without their frames or values: what the page's list shows."""
        with self._lock:
            kept = list(self._trips)
        return [
            {
                "n": n,
                "sent_s": trip.sent_s,
                "error": None if trip.error is None else type(trip.error).__name__,
                # add() keeps the HTTP reply's own timing, which has no round trip time
                "rtt_ms": None if trip.timing is None else _jsonable(trip.timing.get("rtt_ms")),
            }
            for n, trip in kept
        ]

    def _one(self, n: int) -> dict[str, Any] | None:
        """Round trip `n` as the page draws it, or None when it is no longer kept."""
        with self._lock:
            upto = [(kept, trip) for kept, trip in self._trips if kept <= n]
        if not upto or upto[-1][0] != n:
            return None
        trip = upto[-1][1]
        return {
            "n": n,
            "sent_s": trip.sent_s,
            "frames": trip.frames,
            "fps": _window.OBS_FPS,
            "observation": _jsonable(trip.observation),
            "units": _units(trip.observation),
            "session": _still_parts(upto) + _still_arm(upto),
            "action": _jsonable(trip.action),
            "timing": _jsonable(trip.timing),
            "error": _error(trip.error),
            "trained": list(trip.trained),
        }


def _answer(reply: Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, DreamError | None]:
    """What a raw HTTP reply holds for the page: its `actions`, or the error it stands for.

    @param reply The JSON reply of `/v1/get_action`; None when none came.
    @return (actions, None) for an answer; (None, the refusal as a #DreamServerError with its code and reason) for any
        other status; (None, a #DreamError) when no reply came or it carries no status.
    """
    if reply is None:
        return None, DreamError("no reply: the call failed before DreamAPI answered")
    status = reply.get("status")
    if not isinstance(status, Mapping) or not isinstance(status.get("code"), str):
        return None, DreamError("the reply carries no status envelope")
    if status["code"] in ANSWERED:
        return dict(reply["actions"]), None
    reason = status.get("reason") if isinstance(status.get("reason"), str) else None
    message = f"DreamAPI refused the inference request: {status['code']} {status.get('name') or ''}".rstrip()
    return None, DreamServerError(message + (f" ({reason})" if reason else ""), code=status["code"], reason=reason)


def _jsonable(value: Any) -> Any:
    """`value` as JSON carries it: arrays as lists, and a NaN or an infinity as its name, so the page shows it."""
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _units(observation: Mapping[str, Any]) -> list[str]:
    """What looks like a state in the wrong units: the page shows each line above the state.

    @param observation The keys sent besides the frames.
    @return One line per `state.` key with a gripper value outside 0 to 1 (a key naming `grip`), or any other value
        beyond #RADIANS_MAX: joints are radians and hand positions metres, so a larger one is degrees or motor counts.
    """
    lines = []
    for key, value in observation.items():
        if not key.startswith("state."):
            continue
        numbers = np.asarray(value, dtype=float).ravel() if _numeric(value) else np.zeros(0)
        numbers = numbers[np.isfinite(numbers)]
        if not numbers.size:
            continue
        if "grip" in key:
            if numbers.min() < -GRIP_MARGIN or numbers.max() > 1 + GRIP_MARGIN:
                lines.append(f"{key} is {_range(numbers)}: a gripper is 0 closed to 1 open")
        elif np.abs(numbers).max() > RADIANS_MAX:
            lines.append(f"{key} reaches {_range(numbers)}: beyond 2 pi, so not radians; degrees or motor counts?")
    return lines


def _still_parts(kept: list[tuple[int, RoundTrip]]) -> list[str]:
    """State parts that stayed still over the answered calls while the model's targets for them moved.

    In a run that moves the arms, that is a loop not applying the targets (a gripper left open while the model closes
    it) or not reading the part back live; in a dry run nothing moves, so there it is expected.

    @param kept The round trips up to the one shown, numbered, oldest first.
    @return One line naming the parts, or none when fewer than #MIN_CALLS calls were answered or nothing is still.
    """
    answered = [(trip, trip.action) for _, trip in kept if trip.action is not None]
    if len(answered) < MIN_CALLS:
        return []
    trips = [trip for trip, _ in answered]
    still = []
    for key in trips[-1].observation:
        states = _series(trips, key) if key.startswith("state.") else None  # [calls, values]
        if states is None:
            continue
        targets = [_targets(action, key, states.shape[1]) for _, action in answered]  # each [rows, values]
        found = [rows for rows in targets if rows is not None]
        if len(found) < len(targets):
            continue
        # [calls * rows, values]: every target of every call, so a move within one chunk counts too
        moved = np.ptp(np.concatenate(found), axis=0).max()
        if np.ptp(states, axis=0).max() < STILL and moved >= TARGETS_MOVED:
            still.append(key)
    if not still:
        return []
    named = ", ".join(still[:3]) + (f" and {len(still) - 3} more" if len(still) > 3 else "")
    return [
        f"{named} stayed the same over the last {len(answered)} calls while the model's targets for "
        f"{'it' if len(still) == 1 else 'them'} moved: in a run that moves the arms, your loop is not applying those "
        "targets or not reading the part back live (in a dry run nothing moves, so there this is expected)"
    ]


def _still_arm(kept: list[tuple[int, RoundTrip]]) -> list[str]:
    """One arm whose state never changed while the other's did, or two arms sending the same moving state.

    @param kept The round trips up to the one shown, numbered, oldest first.
    @return A line for each; none with fewer than #MIN_CALLS calls or without both a `right_` and a `left_` state part.
    """
    if len(kept) < MIN_CALLS:
        return []
    trips = [trip for _, trip in kept]
    parts: dict[str, dict[str, np.ndarray]] = {}
    for side in ("right", "left"):
        prefix = f"state.{side}_"
        series = {
            key.removeprefix(prefix): _series(trips, key) for key in trips[-1].observation if key.startswith(prefix)
        }
        parts[side] = {part: values for part, values in series.items() if values is not None}  # each [calls, values]
    if not parts["right"] or not parts["left"]:
        return []
    shared = sorted(set(parts["right"]) & set(parts["left"]))
    span = {side: max(float(np.ptp(values, axis=0).max()) for values in parts[side].values()) for side in parts}
    lines = []
    for frozen, moving in (("right", "left"), ("left", "right")):
        if span[frozen] != 0.0 or span[moving] < ARM_MOVED:
            continue
        same = [
            n
            for i, (n, _) in enumerate(kept)
            if shared
            and all(np.abs(parts[frozen][part][0] - parts[moving][part][i]).max() < SAME_POSE for part in shared)
        ]
        hint = f", and it matches the {moving} arm's state on call {same[0]}" if same else ""
        lines.append(
            f"the {frozen} arm's state has not changed at all over the last {len(kept)} calls while the {moving} "
            f"arm's has{hint}: is the {frozen} arm read live? One arm sent as two, or a placeholder for a missing "
            "arm, looks like this"
        )
    if (
        shared
        and min(span.values()) >= ARM_MOVED
        and all(np.array_equal(parts["right"][part], parts["left"][part]) for part in shared)
    ):
        lines.append(
            f"the right and left arms sent the same state on each of the last {len(kept)} calls while moving: is one "
            "arm's reading copied to the other?"
        )
    return lines


def _series(trips: list[RoundTrip], key: str) -> np.ndarray | None:
    """State part `key` over the calls, [calls, values]; None when a call lacks it as finite numbers or it resizes.

    @param trips The round trips, oldest first.
    @param key The state part.
    @return The stacked values; a NaN or an infinity is named apart, so a part holding one is left out here.
    """
    rows: list[np.ndarray] = []
    for trip in trips:
        value = trip.observation.get(key)
        numbers = np.asarray(value, dtype=float).ravel() if _numeric(value) else None
        if (
            numbers is None
            or not numbers.size
            or not np.isfinite(numbers).all()
            or (rows and numbers.size != rows[0].size)
        ):
            return None
        rows.append(numbers)
    return np.stack(rows)


def _targets(action: Mapping[str, Any], key: str, size: int) -> np.ndarray | None:
    """The model's targets for state part `key` in one answer, [rows, size], or None when the answer has none.

    @param action What get_action() returned.
    @param key The state part, `state.<part>`.
    @param size How many values the part holds.
    @return `per_key`'s `action.<part>` (for a gripper part, also `action.<side>_grip`); else, for a side's gripper or
        its `<side>_arm`, the last column or the joint columns of `q_r_chunk` / `q_l_chunk`.
    """
    part = key.removeprefix("state.")
    side = part.split("_", 1)[0]
    per_key = action.get("per_key")
    names = [f"action.{part}", f"action.{side}_grip"] if "grip" in part else [f"action.{part}"]
    for name in names:
        rows = _rows(per_key.get(name)) if isinstance(per_key, Mapping) else None
        if rows is not None and rows.shape[1] == size:
            return rows
    chunk = _rows(action.get({"right": "q_r_chunk", "left": "q_l_chunk"}.get(side, "")))
    if chunk is None:
        return None
    if "grip" in part and size == 1:
        return chunk[:, -1:]
    if part == f"{side}_arm" and chunk.shape[1] == size + 1:
        return chunk[:, :size]
    return None


def _rows(value: Any) -> np.ndarray | None:
    """A chunk as [rows, values], one value a row when it is flat; None when it is not a chunk of numbers."""
    if not _numeric(value) or np.ndim(value) not in (1, 2):
        return None
    rows = np.asarray(value, dtype=float)
    return rows if rows.ndim == 2 else rows.reshape(len(rows), 1)  # [rows] -> [rows, 1]


def _numeric(value: Any) -> bool:
    """Whether a value is a number or an array of numbers (a prompt or a nested dict is not)."""
    try:
        return np.asarray(value).dtype.kind in "biuf"
    except (TypeError, ValueError):
        return False


def _range(numbers: np.ndarray) -> str:
    """`numbers` as the page names them: the one value, or its smallest and largest."""
    lo, hi = float(numbers.min()), float(numbers.max())
    return f"{lo:.3g}" if lo == hi else f"{lo:.3g} to {hi:.3g}"


def _error(error: DreamError | None) -> dict[str, Any] | None:
    """What the page shows of an error: its class, a refusal's code and reason, and its message."""
    if error is None:
        return None
    if isinstance(error, DreamServerError):
        return {"type": type(error).__name__, "code": error.code, "reason": error.reason, "message": str(error)}
    return {"type": type(error).__name__, "code": None, "reason": None, "message": str(error)}


def _handler(view: DebugView) -> type[BaseHTTPRequestHandler]:
    """The request handler of `view`'s server: the page, the list of round trips, and one round trip."""

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, kind: str, code: int) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            number = path.removeprefix("/trip/")
            if path == "/":
                self._send(_PAGE.encode(), "text/html; charset=utf-8", 200)
            elif path == "/trips":
                self._send(json.dumps(view._list()).encode(), "application/json", 200)
            elif path.startswith("/trip/") and number.isdigit() and (trip := view._one(int(number))) is not None:
                self._send(json.dumps(trip).encode(), "application/json", 200)
            elif path == "/arms" and view._arms is not None:
                self._send(json.dumps(view._arms()).encode(), "application/json", 200)
            elif path in ARM_FILES:
                name, kind = ARM_FILES[path]
                self._send((ARMS / name).read_bytes(), kind, 200)
            else:
                self._send(b"not found", "text/plain", 404)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return Handler


## The page: plain HTML and script, nothing fetched from anywhere else; only once show_arms() is called does it load
## arms.js, which imports three.js from cdn.jsdelivr.net (the import map) to draw the arms.
_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dream: what the model sees</title>
<style>
:root{--bg:#fafaf9;--fg:#1c1c1e;--mut:#6a6a70;--line:#e3e3e0;--card:#fff;--acc:#2864b4;--bad:#b3261e;--badbg:#fdecea;
--meas:#d0642c;--stage:#eceeef;--arm-dark:#1c1c1f;--arm-light:#e0e0db}
@media (prefers-color-scheme: dark){:root{--bg:#131315;--fg:#e9e9eb;--mut:#9d9da5;--line:#2d2d32;--card:#1b1b1f;
--acc:#86b4ff;--bad:#ff8a80;--badbg:#3a1d1b;--meas:#ff8b52;--stage:#2e333b;--arm-dark:#17181b;--arm-light:#d8d8d2}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1200px;margin:0 auto;padding:12px 16px 40px}h1{font-size:18px;margin:4px 0}
h2{font-size:14px;margin:18px 0 6px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
#status{color:var(--mut)}#list{display:flex;flex-wrap:wrap;gap:4px;margin:8px 0}
#list button{font:inherit;font-size:12px;border:1px solid var(--line);background:var(--card);color:var(--fg);
border-radius:5px;padding:2px 7px;cursor:pointer}#list button.on{background:var(--acc);color:var(--bg)}
#list button.err{border-color:var(--bad);color:var(--bad)}#list button.on.err{background:var(--bad);color:var(--bg)}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;margin:6px 0}
.cams{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:8px}
.cams img{display:block;width:100%;height:auto;border-radius:4px;margin-top:6px}
.head{font-weight:600}.warn{color:var(--bad);background:var(--badbg);border-radius:5px;padding:2px 8px;margin:4px 0}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{border-bottom:1px solid var(--line);padding:4px 6px;
text-align:left;vertical-align:top}th{color:var(--mut);font-weight:600}td.v{font-family:ui-monospace,monospace;
font-size:12px;word-break:break-all}svg{display:block;width:100%;max-width:520px;height:110px}
.plots{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:8px}
.stage3d{position:relative;height:440px;background:var(--stage);border-radius:6px;overflow:hidden}
.side{position:absolute;transform:translate(-50%,-100%);font-size:12px;font-weight:600;color:var(--fg);
background:var(--card);border-radius:4px;padding:0 5px;pointer-events:none}.stage3d canvas{display:block;
width:100%;height:100%;touch-action:none}.armsbar{display:flex;align-items:center;gap:8px;margin:8px 0 4px}
.armsbar input{flex:1;min-width:0;accent-color:var(--acc)}.armsbar button{font:inherit;font-size:12px;
border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:5px;padding:2px 9px}
.clock{color:var(--mut);font-variant-numeric:tabular-nums}.key{color:var(--mut);font-size:12.5px}
.key .g{color:var(--acc)}.key .o{color:var(--meas)}
@media (max-width:640px){.stage3d{height:300px}.armsbar{flex-wrap:wrap}.clock{flex-basis:100%}}
</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
</head><body><main>
<div id="arms"></div>
<h1>What the model sees</h1>
<div id="status">Waiting for the first get_action().</div>
<div id="list"></div>
<div id="error"></div>
<div id="trained"></div>
<h2>Cameras, live as sent</h2><div id="cameras"></div>
<h2>State and prompt</h2><div id="state"></div>
<h2>Chunk returned</h2><div id="chunk"></div>
<h2>Timing</h2><div id="timing"></div>
</main><script>
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text; return e; };
const warn = (box, text) => box.append(el('div', 'warn', text));
const colours = ['#2864b4', '#d0642c', '#2e8b57', '#9c4dcc', '#c0392b', '#7f8c8d', '#b8860b', '#16a085'];
let shown = null, follow = true;

function shape(v) { const s = []; while (Array.isArray(v)) { s.push(v.length); v = v[0]; } return s; }
function fmt(v) { return JSON.stringify(v, (k, x) => typeof x === 'number' ? Number(x.toFixed(4)) : x); }
function bad(v) { return JSON.stringify(v).match(/"(nan|inf|-inf)"/) !== null; }

// Each camera live: the frames each call adds to its window, played at the rate the model reads them; a call picked
// from the list plays its whole window.
const live = {};
let playing = null;
function cameras(frames, fps) {
  const box = $('cameras'), keys = Object.keys(frames), newest = [];
  if (Object.keys(live).join() !== keys.join()) {
    for (const key of Object.keys(live)) delete live[key];
    const grid = el('div', 'cams');
    box.replaceChildren(el('div', 'key', `Each camera plays the frames your loop sent, ${fps} a second as the model ` +
      'reads them: moving slower or faster than the robot means the loop hands the client its frames at the wrong ' +
      'rate.'), el('div'), grid);
    if (!keys.length) grid.append(el('div', 'card', 'No camera frames in this call.'));
    for (const key of keys) {
      const cam = live[key] = { head: el('div', 'head'), notes: el('div'), img: new Image(), queue: [], last: null };
      const card = el('div', 'card'); card.append(cam.head, cam.notes, cam.img); grid.append(card);
    }
  }
  const summary = box.children[1]; summary.replaceChildren();
  for (const key of keys) {
    const cam = live[key], list = frames[key], at = list.lastIndexOf(cam.last);
    cam.queue = (at >= 0 ? cam.queue.concat(list.slice(at + 1)) : list.slice()).slice(-list.length);
    cam.last = list[list.length - 1]; cam.head.textContent = `${key}: ${list.length} frame(s) a call`;
    cam.notes.replaceChildren();
    const imgs = list.map((b64) => { const im = new Image(); im.src = 'data:image/jpeg;base64,' + b64; return im; });
    newest.push(Promise.all(imgs.map((im) => im.decode().then(() => [im.naturalWidth, im.naturalHeight], () => null)))
      .then((sizes) => {
        const ok = sizes.filter((s) => s !== null);
        if (ok.length < sizes.length)
          warn(cam.notes, 'a frame is not a picture the browser can open: send base64 JPEG');
        if (!ok.length) return null;
        const [w, h] = ok[ok.length - 1];
        cam.head.textContent += `, ${w} x ${h} px (aspect ${(w / h).toFixed(2)})`;
        if (new Set(ok.map((s) => s.join('x'))).size > 1) warn(cam.notes, 'frames of different sizes in one window');
        return `${key} ${w} x ${h}`;
      }));
    if (list.length > 1 && list.every((f) => f === list[0]))
      warn(cam.notes, 'every frame in the window is the same picture: is this camera stale?');
  }
  Promise.all(newest).then((sizes) => {
    const shown = sizes.filter((s) => s !== null);
    if (new Set(shown.map((s) => s.split(' ').slice(1).join(' '))).size > 1)
      warn(summary, `the cameras are not one size: ${shown.join(', ')}; is each resized as the model expects?`);
  });
  playing = playing || setInterval(() => {
    for (const cam of Object.values(live))
      if (cam.queue.length) cam.img.src = 'data:image/jpeg;base64,' + cam.queue.shift();
  }, 1000 / fps);
}

function state(obs, units, session) {
  const box = $('state'); box.replaceChildren();
  const card = el('div', 'card'), table = el('table');
  if (!('prompt' in obs)) warn(card, 'no prompt was sent');
  for (const line of units) warn(card, line);
  for (const line of session) warn(card, line);
  table.append(el('tr')); table.rows[0].append(el('th', '', 'key'), el('th', '', 'shape'), el('th', '', 'value'));
  for (const [key, value] of Object.entries(obs)) {
    const row = table.insertRow();
    row.append(el('td', '', key), el('td', '', JSON.stringify(shape(value))), el('td', 'v', fmt(value)));
    if (bad(value)) warn(card, `${key} holds a NaN or an infinity`);
  }
  card.append(table); box.append(card);
}

function plot(rows) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  const cols = Array.isArray(rows[0]) ? rows[0].length : 1, series = [];
  for (let c = 0; c < cols; c++) series.push(rows.map((r) => Number(Array.isArray(r) ? r[c] : r)));
  const all = series.flat().filter(Number.isFinite), lo = Math.min(...all), hi = Math.max(...all), span = hi - lo || 1;
  svg.setAttribute('viewBox', '0 0 520 110'); svg.setAttribute('preserveAspectRatio', 'none');
  series.forEach((ys, c) => {
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    const x = (i) => (i / Math.max(ys.length - 1, 1)) * 516 + 2, y = (v) => 106 - ((v - lo) / span) * 102;
    line.setAttribute('points', ys.map((v, i) => `${x(i)},${y(v)}`).join(' '));
    line.setAttribute('fill', 'none'); line.setAttribute('stroke', colours[c % colours.length]);
    line.setAttribute('stroke-width', '1.5'); line.setAttribute('vector-effect', 'non-scaling-stroke');
    svg.append(line);
  });
  return [svg, `min ${lo.toFixed(3)}, max ${hi.toFixed(3)}`];
}

function chunk(action) {
  const box = $('chunk'); box.replaceChildren();
  if (action === null) { box.append(el('div', 'card', 'No chunk: the call raised (above).')); return; }
  const flat = {};
  for (const [key, value] of Object.entries(action)) {
    if (value !== null && typeof value === 'object' && !Array.isArray(value))
      for (const [inner, v] of Object.entries(value)) flat[`${key}.${inner}`] = v;
    else flat[key] = value;
  }
  const plots = el('div', 'plots'), rest = el('table');
  for (const [key, value] of Object.entries(flat)) {
    const s = shape(value);
    if ((s.length === 1 || s.length === 2) && s[0] > 1 && !bad(value)) {
      const card = el('div', 'card'), [svg, range] = plot(value);
      card.append(el('div', 'head', `${key}: ${JSON.stringify(s)}`), svg, el('div', '', range)); plots.append(card);
    } else {
      const row = rest.insertRow(); row.append(el('td', '', key), el('td', 'v', fmt(value).slice(0, 400)));
      if (bad(value)) warn(box, `${key} holds a NaN or an infinity`);
    }
  }
  box.append(plots); if (rest.rows.length) { const card = el('div', 'card'); card.append(rest); box.append(card); }
}

function render(trip) {
  const when = new Date(trip.sent_s * 1000).toLocaleTimeString();
  $('status').textContent = `Round trip ${trip.n}, sent at ${when}.` + (follow ? ' Following the newest.' : '');
  const err = $('error'); err.replaceChildren();
  if (trip.error) { const e = trip.error;
    warn(err, `${e.type}${e.code ? ' ' + e.code : ''}${e.reason ? ' (' + e.reason + ')' : ''}: ${e.message}`); }
  const trained = $('trained'); trained.replaceChildren();
  for (const line of trip.trained) warn(trained, 'Not as trained: ' + line);
  cameras(trip.frames, trip.fps); state(trip.observation, trip.units, trip.session); chunk(trip.action);
  const t = $('timing'); t.replaceChildren(); const card = el('div', 'card'), table = el('table');
  for (const [key, value] of Object.entries(trip.timing || {})) {
    const row = table.insertRow(); row.append(el('td', '', key), el('td', 'v', fmt(value)));
  }
  card.append(trip.timing ? table : el('div', '', 'No timing for this call.')); t.append(card);
}

function list(trips) {
  const box = $('list'); box.replaceChildren();
  for (const t of trips) {
    const label = `${t.n}${t.error ? ' ' + t.error : ''}`;
    const b = el('button', (t.n === shown ? 'on' : '') + (t.error ? ' err' : ''), label);
    b.title = t.rtt_ms === null ? 'no round trip time' : `${Math.round(t.rtt_ms)} ms`;
    b.onclick = () => { follow = t.n === trips[trips.length - 1].n; shown = null; pick = t.n; };
    box.append(b);
  }
}

let pick = null;
async function poll() {
  try {
    const trips = await (await fetch('trips')).json();
    if (trips.length) {
      const want = pick !== null && !follow ? pick : trips[trips.length - 1].n;
      if (want !== shown) { const r = await fetch('trip/' + want);
        if (r.ok) { shown = want; render(await r.json()); } else { follow = true; pick = null; } }
    }
    list(trips);
  } catch (e) { $('status').textContent = 'The page lost its program: the client or the view was closed.'; }
  setTimeout(poll, 500);
}
poll();
// The arms, only when the view was handed a motion (show_arms(); 404 otherwise).
fetch('arms').then((r) => r.ok && import('./arms.js').then((arms) => arms.start($('arms'), warn)))
  .catch((e) => warn($('arms'),
    `The arms could not be drawn: ${e.message}. They need three.js from cdn.jsdelivr.net.`));
</script></body></html>
"""

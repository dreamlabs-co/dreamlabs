"""`dreamcli robot run --view`: a live 3D page of each policy chunk while the run runs (README *Run*)."""

import ast
import glob
import importlib.resources
import json
import logging
import os
import pathlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import numpy.typing as npt

from robots import bodies, config, errors, setup_model, setup_view

_log = logging.getLogger(__name__)

DEFAULT_PORT = 8093
SETTLE_S = 0.3
FRAME_EVERY = 8
APPROACH_M, CROSS_M = 0.03, 0.015


class RunView:
    """What the page draws for one run: the arm, its keep-out shapes, and the newest saved chunk."""

    def __init__(self, app: config.AppConfig, since_s: float) -> None:
        """Builds dream's model of the config's arm and reads what the IK keeps each arm off.

        @param app The parsed config of the run.
        @param since_s Wall-clock time.
        """
        body = bodies.for_config(app)
        if body.ik is None:
            raise errors.ConfigError(
                f"--view draws the hand path an EEF model answers with; a {body.name} model answers in joints. "
                "Run without --view"
            )
        if not app.rollouts.enabled:
            raise errors.ConfigError(
                "--view reads the chunks the run saves, and the config saves none: add "
                "'rollouts: {enabled: true}' to the config, or run without --view"
            )
        self._chunks = os.path.join(os.path.expanduser(app.rollouts.dir), "*", "chunks", "reply_*.npz")
        self._since_s = since_s
        self.model = setup_model.ArmModel(body.ik["arm"])
        self.drawing = setup_view.drawing(body.name, self.model)
        from dream_ik.specs import EMBODIMENTS

        own = EMBODIMENTS[body.ik["arm"]].obstacles
        arms = {"left": app.follower_l, "right": app.follower_r}
        self.keep_out = {
            side: [o.model_dump() for o in own[side]] if arm.keep_out is None else [dict(o) for o in arm.keep_out]
            for side, arm in arms.items()
        }
        self._driven = [i for i in range(len(self.model.joint_names)) if i not in self.model.held]
        self._lock = threading.Lock()
        self._path: str | None = None
        self._payload: dict[str, Any] = {"chunk": None}

    def view(self) -> dict[str, Any]:
        """What the page loads once.

        @return The drawing, the arm at rest and, per side, the keep-out shapes.
        """
        return {"drawing": self.drawing, "rest": self.model.frame(self.model.rest), "keep_out": self.keep_out}

    def latest(self) -> dict[str, Any]:
        """The newest chunk saved since the view started, decoded once per file.

        @return `{"chunk": None}` before the first.
        """
        now = time.time()
        files = [
            f
            for f in glob.glob(self._chunks)
            if self._since_s <= os.path.getmtime(f) and now - os.path.getmtime(f) > SETTLE_S
        ]
        path = max(files, key=os.path.getmtime, default=None)
        with self._lock:
            if path is not None and path != self._path:
                with np.load(path, allow_pickle=False) as z:
                    per_key = ast.literal_eval(str(z["per_key"]))
                    chunks = {
                        side: np.asarray(z[f"q_{side[0]}_chunk"], dtype=np.float64) for side in ("left", "right")
                    }
                self._payload = {
                    "episode": pathlib.Path(path).parent.parent.name,
                    "chunk": int(pathlib.Path(path).stem.split("_")[-1]),
                    "mtime": os.path.getmtime(path),
                    "sides": {side: self._side(per_key, side, q) for side, q in chunks.items()},
                }
                self._path = path
            if self._payload["chunk"] is None:
                return self._payload
            return {**self._payload, "age_s": round(now - self._payload["mtime"], 1)}

    def _side(self, per_key: dict[str, Any], side: str, q: npt.NDArray[np.float64]) -> dict[str, Any]:
        from dream_ik.decode import rot6d_to_mat

        xyz = np.asarray(per_key[f"action.{side}_eef_xyz"], dtype=np.float64)
        rot = rot6d_to_mat(np.asarray(per_key[f"action.{side}_eef_rot6"], dtype=np.float64))
        grip = np.asarray(per_key[f"action.{side}_grip"], dtype=np.float64).reshape(-1)
        if q.shape != (len(xyz), len(self._driven) + 1):
            raise errors.ConfigError(
                f"chunk {side} joints are {q.shape}, not {len(xyz)} rows of {len(self._driven)} joints and the grip"
            )
        joints = np.zeros((len(q), len(self.model.joint_names)))
        joints[:, self._driven] = q[:, :-1]
        tool = np.stack([self.model.tool(row, side)[:3, 3] for row in joints])
        held = int(np.argmin(np.all(q == q[0], axis=1))) if not np.all(q == q[0]) else len(q)
        miss = np.linalg.norm(xyz[held:] - tool[held:], axis=-1)
        rows = range(0, len(xyz), FRAME_EVERY)
        return {
            "model": xyz.round(4).tolist(),
            "ik": tool.round(4).tolist(),
            "approach": [[xyz[i].tolist(), (xyz[i] + APPROACH_M * rot[i, :, 2]).tolist()] for i in rows],
            "cross": [[xyz[i].tolist(), (xyz[i] + CROSS_M * rot[i, :, 0]).tolist()] for i in rows],
            "start": self.model.frame(joints[0]),
            "end": self.model.frame(joints[-1]),
            "grip_start": round(float(q[0, -1]), 3),
            "grip_end": round(float(q[-1, -1]), 3),
            "grip_asked": [round(float(grip[0]), 3), round(float(grip[-1]), 3)],
            "miss_mm": round(float(miss.max() * 1000), 1) if len(miss) else 0.0,
            "travel_mm": round(float(np.linalg.norm(xyz[-1] - xyz[0]) * 1000), 1),
        }


def _handler(view: RunView) -> type[BaseHTTPRequestHandler]:
    package = importlib.resources.files("robots")
    page = package.joinpath("run_view.html").read_bytes()
    viewer = package.joinpath("setup_viewer.js").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, kind: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?")[0]
            name = path.removeprefix(setup_view.URL)
            if path == "/":
                self._send(page, "text/html; charset=utf-8")
            elif path == "/setup_viewer.js":
                self._send(viewer, "text/javascript")
            elif path == "/api/view":
                self._send(json.dumps(view.view()).encode(), "application/json")
            elif path == "/api/latest":
                self._send(json.dumps(view.latest()).encode(), "application/json")
            elif path.startswith(setup_view.URL) and name in setup_view.files():
                kind = setup_view.CONTENT_TYPES[os.path.splitext(name)[1]]
                self._send(setup_view.files()[name].read_bytes(), kind)
            else:
                self._send(b"not found", "text/plain", 404)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return Handler


def start(app: config.AppConfig, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Serves the page for a run of `app` on 127.0.0.1:`port`, on a daemon thread, from now on.

    @param app The parsed config of the run.
    @param port The page's port.
    @return The running server.
    """
    view = RunView(app, since_s=time.time())
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), _handler(view))
    except OSError as exc:
        raise errors.ConfigError(
            f"--view cannot listen on port {port} ({exc}); pass another with --view-port"
        ) from None
    threading.Thread(target=server.serve_forever, name="robots-run-view", daemon=True).start()
    _log.info("run view on http://127.0.0.1:%d", port)
    return server

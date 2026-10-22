"""HTTP control API for the station: Start / Stop / Home / prompt / status as JSON over the stdlib server.

A dashboard or the operator's shell drives a station with curl-sized requests, so
the API is six routes on `http.server.ThreadingHTTPServer`, no new dependency. Every request
is handled on the server's thread and handed to the #robots.station.StationController through
`asyncio.run_coroutine_threadsafe`, so the transition itself runs on the controller's event
loop, serialised with the live view's requests and the loop's ticks.

| Route | Body | Reply |
| --- | --- | --- |
| `GET /status` | | 200, the controller's `status()` dict |
| `POST /start` | `{"prompt": "..."}` optional | 200 status, or 409 `{"error": ...}` when a run exists |
| `POST /stop` | | 200 status once the arms hold; 409 during a home move |
| `POST /home` | | 200 status once the arms are home; 409 while a run exists |
| `POST /prompt` | `{"prompt": "..."}` | 200 status; 400 without a non-empty prompt |

Bad JSON or a bad prompt is 400, a refused transition is 409, an unknown route 404, a wrong
method 405, a controller call that does not return within #REQUEST_TIMEOUT_S 504, and any
other failure 500; every error body is `{"error": message}`.

@see StationApi
"""

import asyncio
import http.server
import json
import logging
import threading
from collections.abc import Coroutine
from typing import Any

from robots import errors, station

## Seconds a request waits for the controller: a stop waits for the loop's exit, a home for the move (3 s).
REQUEST_TIMEOUT_S: float = 60.0

## Largest request body accepted; a prompt is a sentence.
MAX_BODY_BYTES: int = 64 * 1024

_log = logging.getLogger(__name__)


class StationApi:
    """The HTTP server on its own thread, bound to a #robots.station.StationController and its event loop.

    Attributes:
    - host: Bind address.
    - port: Bound port; the actual one once start() has bound (0 asks the OS for a free port).

    @see start(), stop()
    """

    def __init__(
        self,
        controller: station.StationController,
        event_loop: asyncio.AbstractEventLoop,
        *,
        host: str = station.DEFAULT_HOST,
        port: int = station.DEFAULT_API_PORT,
    ) -> None:
        """Binds the controller and the event loop its coroutines run on; nothing listens until start().

        @param controller The station controller.
        @param event_loop The loop the controller's coroutines run on (the one `station.serve` runs in).
        @param host Bind address.
        @param port Port; 0 for an ephemeral one (tests).
        """
        self.host = host
        self.port = port
        self._controller = controller
        self._event_loop = event_loop
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Binds the socket and serves on a daemon thread.

        @throws OSError The port is taken.
        """
        self._server = _Server((self.host, self.port), _Handler, self)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, name="station-api", daemon=True)
        self._thread.start()
        _log.info("station API listening on %s:%d", self.host, self.port)

    def stop(self) -> None:
        """Stops serving and closes the socket; a no-op before start()."""
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(REQUEST_TIMEOUT_S)
        self._server = None
        self._thread = None

    def call(self, coroutine: Coroutine[Any, Any, dict[str, Any]]) -> dict[str, Any]:
        """Runs one controller coroutine on the event loop from the server thread and waits for its result.

        @param coroutine The controller call, e.g. `controller.start(prompt)`.
        @return The status dict the call returned.
        @throws StationError The controller refused the transition.
        @throws ValueError The prompt was not a non-empty string.
        @throws TimeoutError The call did not return within #REQUEST_TIMEOUT_S.
        """
        future = asyncio.run_coroutine_threadsafe(coroutine, self._event_loop)
        try:
            return future.result(timeout=REQUEST_TIMEOUT_S)
        except TimeoutError:
            future.cancel()
            raise

    @property
    def controller(self) -> station.StationController:
        """The controller the routes act on."""
        return self._controller


class _Server(http.server.ThreadingHTTPServer):
    """The stdlib server plus a reference to its #StationApi, which the handler reads through `self.server`."""

    ## Lets the station restart within TIME_WAIT of its last run.
    allow_reuse_address = True
    ## Request threads die with the process rather than block its exit.
    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: type[http.server.BaseHTTPRequestHandler], api: StationApi):
        """Binds the address and keeps the API for the handler.

        @param address `(host, port)`.
        @param handler The request handler class.
        @param api The API the routes act through.
        """
        super().__init__(address, handler)
        self.api = api


class _Handler(http.server.BaseHTTPRequestHandler):
    """Routes one request to the controller and writes the JSON reply."""

    ## The routes and the controller call each makes; GET /status is handled apart since it needs no coroutine.
    server: _Server

    def do_GET(self) -> None:
        """`GET /status` returns the controller's status; anything else is 404."""
        if self.path.split("?", 1)[0] != "/status":
            self._reply(404, {"error": f"no such route: GET {self.path}"})
            return
        self._reply(200, self.server.api.controller.status())

    def do_POST(self) -> None:
        """`POST /start`, `/stop`, `/home`, `/reset` and `/prompt` hand their call to the controller; else 404."""
        route = self.path.split("?", 1)[0]
        controller = self.server.api.controller
        try:
            body = self._read_json()
            if route == "/start":
                result = self.server.api.call(controller.start(body.get("prompt"), body.get("checkpoint")))
            elif route == "/stop":
                result = self.server.api.call(controller.stop())
            elif route == "/home":
                result = self.server.api.call(controller.home())
            elif route == "/reset":
                result = self.server.api.call(controller.reset())
            elif route == "/prompt":
                if "prompt" not in body:
                    raise ValueError("body must be a JSON object with a 'prompt' key")
                result = self.server.api.call(controller.set_prompt(body["prompt"]))
            else:
                self._reply(404, {"error": f"no such route: POST {route}"})
                return
        except errors.StationError as exc:
            self._reply(409, {"error": str(exc)})
        except ValueError as exc:
            self._reply(400, {"error": str(exc)})
        except TimeoutError:
            self._reply(504, {"error": f"the station did not answer within {REQUEST_TIMEOUT_S:g} s"})
        except Exception as exc:
            # The request boundary: a fault in the controller is reported to the caller, never left as a
            # half-written reply, and logged with its traceback here.
            _log.exception("station API: %s %s failed", self.command, route)
            self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})
        else:
            self._reply(200, result)

    def _read_json(self) -> dict[str, Any]:
        """Reads the request body as a JSON object; an empty body is an empty object.

        @return The parsed object.
        @throws ValueError The body is not valid JSON, not an object, or larger than #MAX_BODY_BYTES.
        """
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise ValueError(f"request body of {length} bytes is larger than {MAX_BODY_BYTES}")
        raw = self.rfile.read(length) if length else b""
        if not raw.strip():
            return {}
        try:
            body = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"request body is not valid JSON: {exc.msg}") from exc
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        return body

    def _reply(self, status: int, payload: dict[str, Any]) -> None:
        """Writes one JSON reply.

        @param status The HTTP status code.
        @param payload The body, JSON-encoded.
        """
        data = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (the stdlib's signature)
        """Routes the server's access lines to the module logger at DEBUG instead of stderr.

        @param format The stdlib's format string.
        @param args Its arguments.
        """
        _log.debug("station API: " + format, *args)

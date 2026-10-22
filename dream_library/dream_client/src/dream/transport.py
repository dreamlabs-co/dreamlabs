"""HTTP channel to the model's host, plus the JSON wire encoding; internal, with no compatibility promise.

This is the only module that knows how bytes reach the model. #Transport owns one keep-alive
HTTP connection and posts JSON to `/v1/get_action`; encode() and decode() are the wire format.
Everything above this module talks to "a request/reply channel" and never imports `http.client`.

Every failure path drops the connection before raising, so the next request opens a fresh
one and the failure is confined to the request that caused it. Every request, failed or not,
leaves its cost on the wire in `Transport.last_exchange` (#Exchange), which is what the client
turns into `last_timing`.
"""

import dataclasses
import http.client
import json
import secrets
import socket
import struct
import time
from typing import Any

import numpy as np

from dream import credentials
from dream.errors import DreamConfigError, DreamError, DreamProtocolError, DreamTimeout

## The one robot-facing endpoint of DreamAPI.
GET_ACTION_PATH = "/v1/get_action"

## The gateway's session release, `POST /v1/end_session`: frees the seat now instead of on idle eviction.
END_SESSION_PATH = "/v1/end_session"

## The request header every request carries with this library's own version, `dream.__version__`, and nothing else.
## The gateway never compares the commit a build names; it refuses the request as `DA-4000 client_outdated` in two
## cases: on a checkpoint whose robot has parts, whatever its settings, when the version is older than the first
## library that converts parts; and, with its version check on, when the major is below the oldest one it serves,
## which is 0, so no major is refused yet.
CLIENT_VERSION_HEADER = "Dream-Client-Version"

## The product every request's `User-Agent` names, `dream-client/<dream.__version__> <8 hex>`: the hex is drawn once
## per #Transport, so each robot's channel is one caller to the gateway, which tells robots on one key apart by it when
## a pool is full and their requests carry no session yet (the gateway reads this form).
USER_AGENT_PRODUCT = "dream-client"


class _StaleConnectionError(Exception):
    """Internal: the peer closed a reused keep-alive connection before reading our request."""


class _Connection(http.client.HTTPConnection):
    """An HTTP connection with Nagle's algorithm off.

    `http.client` writes headers and body as two segments; with Nagle on, the second waits for
    the peer's delayed ACK, which costs a 40 ms floor per request on Linux loopback.
    """

    def connect(self) -> None:
        """Opens the socket as `HTTPConnection` does, then disables Nagle on it."""
        super().connect()
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


class _SecureConnection(http.client.HTTPSConnection):
    """The `_Connection` for a TLS host: HTTPS with Nagle off, for a host fronted by a proxy that
    terminates TLS. Certificates are verified against the system trust store, the default of
    `HTTPSConnection`; the disabled Nagle carries over unchanged.
    """

    def connect(self) -> None:
        """Opens the TLS socket as `HTTPSConnection` does, then disables Nagle on it."""
        super().connect()
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


@dataclasses.dataclass(frozen=True)
class Exchange:
    """What one request cost on the wire, whether or not a reply came back.

    Attributes:
    - rtt_ms: Wall time of the whole exchange, from the first byte sent to the last byte read or to the
      failure: connect (on a fresh connection), the one stale-connection retry, and the reply.
    - request_bytes: Size of the encoded request body.
    - response_bytes: Size of the reply body; `0` when none was read.
    - up_mbps: The kernel's own delivery rate for the request's socket, megabits per second, read from
      TCP_INFO right after the reply. This is the achieved UPLOAD throughput and the first thing to look at
      when a round trip drifts: the request body is ~1.4 MB, so a low `up_mbps` here (with a small
      `tcp_cwnd_kb` or a rising `tcp_retrans`) means the client's own send path, not DreamAPI, is the limit.
    - tcp_cwnd_kb: The sender's congestion window in KB (how much it may have in flight); small = window-limited.
    - tcp_retrans: Cumulative retransmits on the connection; a rising count is packet loss on the send path.
    - tcp_rtt_ms: The kernel's smoothed RTT for the connection, milliseconds.
    - tcp_app_limited: True when the kernel says the last delivery-rate sample was limited by the app not
      having bytes ready (host-side pacing) rather than by the network; a low `up_mbps` with this set is
      NOT a network problem.
    All five TCP fields are `None` where TCP_INFO is unavailable (only Linux exposes it) or the read failed.
    """

    rtt_ms: float
    request_bytes: int
    response_bytes: int
    up_mbps: float | None = None
    tcp_cwnd_kb: float | None = None
    tcp_retrans: int | None = None
    tcp_rtt_ms: float | None = None
    tcp_app_limited: bool | None = None


_TCP_INFO_MIN_LEN = 192  # struct tcp_info through tcpi_delivery_rate; older/other kernels return less


def _read_tcp_info(sock: Any) -> dict[str, Any]:
    """Best-effort sender-side TCP health for one socket, or `{}` when unavailable; never raises.

    Reads the kernel's `TCP_INFO` (a read-only getsockopt, no I/O and no effect on the connection) and
    pulls the few fields that answer "is the upload the bottleneck": the delivery rate, congestion window,
    retransmits, RTT and the app-limited flag. `{}` on any platform without `TCP_INFO` (only Linux has it),
    on a closed socket, or on a short/odd struct, so the caller falls back to `None` fields.
    """
    opt = getattr(socket, "TCP_INFO", None)
    if opt is None or sock is None:
        return {}
    try:
        raw = sock.getsockopt(socket.IPPROTO_TCP, opt, 256)
    except (OSError, AttributeError, ValueError):
        return {}
    if len(raw) < _TCP_INFO_MIN_LEN:
        return {}

    def cell(fmt: str, off: int) -> int:
        return int(struct.unpack_from(fmt, raw, off)[0])

    try:
        mss = cell("<I", 16) or 1448  # tcpi_snd_mss
        return {
            "up_mbps": round(cell("<Q", 160) * 8 / 1e6, 1),  # tcpi_delivery_rate, bytes/s -> Mbit/s
            "tcp_cwnd_kb": round(cell("<I", 80) * mss / 1024, 1),  # tcpi_snd_cwnd, in MSS -> KB
            "tcp_retrans": cell("<I", 100),  # tcpi_total_retrans
            "tcp_rtt_ms": round(cell("<I", 68) / 1000, 1),  # tcpi_rtt, microseconds -> ms
            "tcp_app_limited": bool(cell("<B", 7) & 0x01),  # tcpi_delivery_rate_app_limited (low bit)
        }
    except struct.error:
        return {}


def encode(obj: Any) -> bytes:
    """Serializes a request body to JSON, turning numpy arrays into lists and numpy scalars into numbers.

    @param obj Any JSON-compatible structure; `np.ndarray` and numpy scalars are allowed anywhere inside it.
    @return The UTF-8 bytes.
    @throws TypeError An object-dtype array or another unsupported type is present.
    """
    return json.dumps(obj, default=_json_default, separators=(",", ":")).encode("utf-8")


def decode(data: bytes) -> Any:
    """Deserializes a JSON reply.

    @param data Bytes as DreamAPI sent them.
    @return The decoded structure.
    @throws ValueError The bytes are not a complete JSON document.
    """
    return json.loads(data)


class Transport:
    """Request/reply channel to one DreamAPI gateway over HTTP.

    One connection, reused across requests; any fault closes it so the next request starts
    clean. One outstanding request at a time: the client serialises callers.

    Attributes:
    - host: Gateway host.
    - port: Gateway port.
    - timeout_ms: Per-request connect, send and receive timeout.
    - last_exchange: The #Exchange of the most recent request(), failed or not; `None` before the first.
    - instance: 8 hex digits drawn once for this channel, the last word of every request's `User-Agent`
      (#USER_AGENT_PRODUCT); random, so it names no robot or person.
    """

    def __init__(self, host: str, port: int, timeout_ms: int, scheme: str = "http") -> None:
        """Binds the endpoint; no connection is opened until the first request.

        @param host Gateway host.
        @param port Gateway port.
        @param timeout_ms Per-request timeout.
        @param scheme `http` (plain, the default, for a gateway reached directly on its listener) or
            `https` (TLS, for a gateway behind a proxy that terminates it). Decides the connection class.
        """
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.scheme = scheme
        self.last_exchange: Exchange | None = None
        self.instance = secrets.token_hex(4)
        self._closed = False
        self._connection: http.client.HTTPConnection | None = None
        # True once a response was read on the current connection: a failure to send on it is
        # then a stale keep-alive (the gateway closed it idle, or restarted), not a dead gateway.
        self._reused = False

    def request(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
        *,
        path: str = GET_ACTION_PATH,
        timeout_ms: int | None = None,
    ) -> Any:
        """Posts one JSON body to `path` (`/v1/get_action` by default) and returns the decoded reply body.

        The HTTP status is not interpreted here: DreamAPI puts the outcome in the body's status
        envelope on every response, success and refusal alike, and the client reads that.
        `last_exchange` is set on every path out of here that sent bytes, the failures included.

        @param payload The request body; packed with encode().
        @param headers Extra request headers (the credential, when there is one); #CLIENT_VERSION_HEADER and the
            `User-Agent` (#USER_AGENT_PRODUCT) are always added.
        @param path The gateway route; #GET_ACTION_PATH, or #END_SESSION_PATH for a session release.
        @param timeout_ms A timeout for this request alone, in place of the channel's; None keeps the channel's.
        @return The decoded reply body, whatever shape the gateway sent.
        @throws DreamError The channel is closed.
        @throws DreamTimeout No reply within `timeout_ms`; the connection has been dropped.
        @throws DreamProtocolError The gateway cannot be reached, closed the connection, or answered
            something that is not JSON (also dropped).
        @throws DreamConfigError The host is not a name a connection can be opened to (a space, an empty
            label), or a header holds a character HTTP cannot carry.
        @throws TypeError The payload contains something encode() refuses (ValueError for a circular one).
        """
        if self._closed:
            raise DreamError(f"inference request on a closed channel to {self._endpoint()}")
        body = encode(payload)
        all_headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            CLIENT_VERSION_HEADER: _client_version(),
            "User-Agent": f"{USER_AGENT_PRODUCT}/{_client_version()} {self.instance}",
            **headers,
        }
        started_s = time.perf_counter()
        raw = b""
        tcp: dict[str, Any] = {}
        try:
            try:
                status, raw, tcp = self._exchange(body, all_headers, path, timeout_ms)
            except _StaleConnectionError:
                # The one transport retry: the peer dropped an idle keep-alive connection before our
                # request was read (a gateway restart looks exactly like this). Nothing was processed,
                # so one fresh connection is safe; a second failure is reported.
                status, raw, tcp = self._exchange(body, all_headers, path, timeout_ms)
        finally:
            rtt_ms = (time.perf_counter() - started_s) * 1000.0
            self.last_exchange = Exchange(
                rtt_ms=rtt_ms,
                request_bytes=len(body),
                response_bytes=len(raw),
                up_mbps=tcp.get("up_mbps"),
                tcp_cwnd_kb=tcp.get("tcp_cwnd_kb"),
                tcp_retrans=tcp.get("tcp_retrans"),
                tcp_rtt_ms=tcp.get("tcp_rtt_ms"),
                tcp_app_limited=tcp.get("tcp_app_limited"),
            )
        try:
            return decode(raw)
        except ValueError as exc:
            self._drop()
            raise DreamProtocolError(
                f"inference reply from {self._endpoint()} is not JSON (HTTP {status}); "
                "is the correct DreamAPI gateway running there?"
            ) from exc

    def _exchange(
        self, body: bytes, headers: dict[str, str], path: str = GET_ACTION_PATH, timeout_ms: int | None = None
    ) -> tuple[int, bytes, dict[str, Any]]:
        """One request/response on the current connection, as `(http_status, raw_body, tcp_info)`; every
        failure drops the connection first. `tcp_info` is the sender-side TCP health of this request's
        socket (empty on any platform without TCP_INFO). A `timeout_ms` applies to this exchange only; the
        connection goes back to the channel's timeout after it."""
        reused = self._reused
        answered = False  # Once the reply's status line is in, the gateway has processed the request
        timeout_s = (timeout_ms if timeout_ms is not None else self.timeout_ms) / 1000.0
        try:
            connection = self._connect()
            connection.timeout = timeout_s
            if connection.sock is not None:
                connection.sock.settimeout(timeout_s)
            connection.request("POST", path, body=body, headers=headers)
            response = connection.getresponse()
            answered = True
            raw = response.read()
            tcp = _read_tcp_info(getattr(connection, "sock", None))
        except TimeoutError as exc:
            # The connection may still deliver the late reply; dropping it is what keeps that
            # reply from being read as the answer to the next request.
            self._drop()
            raise DreamTimeout(
                f"inference request to {self._endpoint()} timed out after {int(timeout_s * 1000)} ms"
            ) from exc
        except (BrokenPipeError, ConnectionResetError, http.client.RemoteDisconnected) as exc:
            self._drop()
            if reused and not answered:
                raise _StaleConnectionError() from exc
            raise DreamProtocolError(_unreachable(self._endpoint(), exc)) from exc
        except (ValueError, http.client.InvalidURL) as exc:
            # A host name the resolver cannot encode (`UnicodeError`, a ValueError) or a connection refuses to name
            # (InvalidURL): the config, not the network. The client checks its one header before sending, so the
            # text here never holds the key.
            self._drop()
            raise DreamConfigError(
                f"inference request to {self._endpoint()} cannot be sent: {exc}; check the host in the config or "
                "DREAM_GATEWAY_URL"
            ) from exc
        except (OSError, http.client.HTTPException) as exc:
            self._drop()
            raise DreamProtocolError(_unreachable(self._endpoint(), exc)) from exc
        self._reused = True
        connection.timeout = self.timeout_ms / 1000.0
        if connection.sock is not None:
            connection.sock.settimeout(connection.timeout)
        return response.status, raw, tcp

    def close(self) -> None:
        """Drops the connection. Idempotent and exception-free."""
        if self._closed:
            return
        self._closed = True
        self._drop()

    def _connect(self) -> http.client.HTTPConnection:
        """Returns the live connection, opening one lazily; connecting itself is deferred to the request."""
        if self._connection is None:
            connection_class = _SecureConnection if self.scheme == "https" else _Connection
            self._connection = connection_class(self.host, self.port, timeout=self.timeout_ms / 1000.0)
        return self._connection

    def _drop(self) -> None:
        """Closes the connection so the next request opens a fresh one."""
        self._reused = False
        if self._connection is None:
            return
        try:
            self._connection.close()
        except Exception:
            # Teardown must never mask the error that caused it; the one broad except here.
            pass
        self._connection = None

    def _endpoint(self) -> str:
        """Returns the `<scheme>://host:port` address; also used in error messages."""
        return f"{self.scheme}://{self.host}:{self.port}"


def _client_version() -> str:
    """This library's version, `dream.__version__`, for #CLIENT_VERSION_HEADER.

    Imported when a request is sent rather than at the top: the `dream` package imports this module before it sets
    its version.

    @return The version, such as `0.0.3+g<commit>` for the library the platform serves.
    """
    import dream

    return dream.__version__


def _unreachable(url: str, exc: Exception) -> str:
    why = exc.strerror if isinstance(exc, OSError) and exc.strerror else str(exc) or type(exc).__name__
    return (
        f"could not reach Dream Labs at {url}: {why}; check this computer's internet connection, or "
        f"{credentials.where_set(url)}"
    )


def _json_default(obj: Any) -> Any:
    """`json.dumps` hook: arrays become nested lists, numpy scalars become Python numbers."""
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "O":
            raise TypeError(
                f"cannot encode an object-dtype array (shape {obj.shape}); "
                "convert it to a concrete numeric dtype before sending"
            )
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"cannot encode object of type {type(obj).__name__} for the wire")

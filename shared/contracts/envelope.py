"""The status envelope and the seven response codes.

Every DreamAPI response, success included, carries one #Status. A code exists only when a caller must behave
differently because of it; finer diagnosis is the per-code reason enum plus a `trace_id` that points at the
log line with full context. A non-OK response never carries actions.

The `name` and `retryable` fields of the envelope are derived from `code` and `reason` (`Status.retryable` is the
rule), so a caller constructs a #Status from `code`, `reason` and `trace_id` only and cannot contradict it.

#CODE_HELP and #REASON_HELP are what each code and reason tells a customer of the `dream` client, and what to do about
it; the customer docs' error-code page is generated from them. They are words only:
nothing that answers or reads an envelope consults them.
"""

import dataclasses
import enum

import pydantic


class StatusCode(enum.StrEnum):
    """The seven permanent response codes; the member name is the envelope's `name` field."""

    OK = "DA-0000"
    OK_NEW_SESSION = "DA-0001"
    UNAUTHORIZED = "DA-1000"
    NO_CAPACITY = "DA-2000"
    BAD_REQUEST = "DA-4000"
    SERVE_FAULT = "DA-5000"
    GATEWAY_FAULT = "DA-6000"


class NewSessionReason(enum.StrEnum):
    """Why the returned `session_id` differs from the one sent (`DA-0001`); a harness fails the row unless `NEW`."""

    NEW = "new"
    EXPIRED = "expired"
    UNKNOWN = "unknown"
    SERVE_DIED = "serve_died"


class UnauthorizedReason(enum.StrEnum):
    """Why an operator call was refused (`DA-1000`); never retryable.

    `REGION_NOT_SUPPORTED` is answered by the edge in front of the gateway, for a
    robot whose address is outside `ROBOT_ALLOWED_COUNTRIES`; the gateway itself never produces it.
    """

    MISSING = "missing"
    UNKNOWN_KEY = "unknown_key"
    DISABLED = "disabled"
    OVER_CAP = "over_cap"
    NOT_ENTITLED = "not_entitled"
    REGION_NOT_SUPPORTED = "region_not_supported"


class NoCapacityReason(enum.StrEnum):
    """Why a request was not placed or forwarded (`DA-2000`); all retryable except `MAX_SERVES`."""

    POOL_FULL = "pool_full"
    POOL_WARMING = "pool_warming"
    RATE_LIMITED = "rate_limited"
    QUEUE_TIMEOUT = "queue_timeout"
    MAX_SERVES = "max_serves"


class BadRequestReason(enum.StrEnum):
    """Why the request itself was rejected (`DA-4000`); never retryable.

    `CLIENT_OUTDATED` is answered by two checks of the `dream` library's version header (`Dream-Client-Version`).
    With `DREAMAPI_CLIENT_VERSION_CHECK` on, a `get_action` or `start_session` whose header is `MAJOR.MINOR.PATCH` with
    a major below the oldest one the gateway serves is refused; a request without the header, or whose version is of
    another shape, passes this check. Whatever that flag says, a call on a checkpoint whose robot has parts is refused
    unless its header is a `MAJOR.MINOR.PATCH` version no older than the first library that converts parts, so on such
    a checkpoint a request without the header, or whose version is of another shape, is refused too.
    """

    MALFORMED = "malformed"
    TOO_LARGE = "too_large"
    UNKNOWN_OP = "unknown_op"
    INVALID_SESSION_ID = "invalid_session_id"
    GLOBAL_OP_REJECTED = "global_op_rejected"
    UNKNOWN_CHECKPOINT = "unknown_checkpoint"
    CLIENT_OUTDATED = "client_outdated"


class ServeFaultReason(enum.StrEnum):
    """Why the serve failed the forwarded request (`DA-5000`); retryable. `CAPACITY` is a fault on DreamAPI's side."""

    TIMEOUT = "timeout"
    ERROR = "error"
    UNREACHABLE = "unreachable"
    CAPACITY = "capacity"


class GatewayFaultReason(enum.StrEnum):
    """Why DreamAPI itself failed the request (`DA-6000`); retryable and alertable."""

    INTERNAL = "internal"
    STORE_WRITE_FAILED = "store_write_failed"
    CANNOT_SCALE = "cannot_scale"
    STEERING_UNAVAILABLE = "steering_unavailable"


## Any reason value; each member type belongs to exactly one code (see #REASONS_BY_CODE).
Reason = (
    NewSessionReason | UnauthorizedReason | NoCapacityReason | BadRequestReason | ServeFaultReason | GatewayFaultReason
)

## The reason enum each non-OK code carries; `DA-0000` carries none and is absent here.
REASONS_BY_CODE: dict[StatusCode, type[enum.StrEnum]] = {
    StatusCode.OK_NEW_SESSION: NewSessionReason,
    StatusCode.UNAUTHORIZED: UnauthorizedReason,
    StatusCode.NO_CAPACITY: NoCapacityReason,
    StatusCode.BAD_REQUEST: BadRequestReason,
    StatusCode.SERVE_FAULT: ServeFaultReason,
    StatusCode.GATEWAY_FAULT: GatewayFaultReason,
}


class Status(pydantic.BaseModel):
    """The status envelope on every response.

    Its shape is `{code, name, reason?, retryable, retry_after_s?, trace_id, detail?}`.

    Attributes:
    - code: One of the seven #StatusCode values.
    - trace_id: Id of the structured log line with the full context of this response.
    - reason: The code's reason; required for every code but `OK`, which carries none.
    - retry_after_s: How long the caller should wait before asking again, seconds; set on `pool_full`,
      `pool_warming`, `rate_limited` and `queue_timeout`, and on `max_serves` with `DREAMAPI_SEAT_FIRST` on,
      where the request is not `retryable` as it stands.
    - detail: Free-text diagnosis for humans; never something a caller branches on.
    - name: Derived: the `code` member name, e.g. `NO_CAPACITY`.
    - retryable: Derived: `DA-5000` and `DA-6000` always, `DA-2000` unless `max_serves`.
    """

    code: StatusCode
    trace_id: str
    reason: Reason | None = None
    retry_after_s: float | None = None
    detail: str | None = None

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @property
    def name(self) -> str:
        """The `code` member name, so a reader never has to decode `DA-2000`."""
        return self.code.name

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @property
    def retryable(self) -> bool:
        """Whether the caller may retry the same request."""
        if self.code in (StatusCode.SERVE_FAULT, StatusCode.GATEWAY_FAULT):
            return True
        if self.code is StatusCode.NO_CAPACITY:
            return self.reason is not NoCapacityReason.MAX_SERVES
        return False

    @pydantic.model_validator(mode="after")
    def check_reason_matches_code(self) -> "Status":
        """Rejects an envelope whose reason is missing, or belongs to a different code (#REASONS_BY_CODE).

        @return The validated envelope, unchanged.
        @throws ValueError If `code` is `OK` with a reason, or any other code without a reason of its own enum.
        """
        expected = REASONS_BY_CODE.get(self.code)
        if expected is None:
            if self.reason is not None:
                raise ValueError(f"{self.code.name} carries no reason, got {self.reason!r}")
            return self
        if not isinstance(self.reason, expected):
            raise ValueError(f"{self.code.name} needs a {expected.__name__} reason, got {self.reason!r}")
        return self


@dataclasses.dataclass(frozen=True)
class CustomerHelp:
    """What one reason tells a customer of the `dream` client, and what they should do about it.

    Customer words: no internal names, hosts or requirement ids, ASCII, one line each. Whether the reason is
    retryable is not repeated here; `Status.retryable` says it.

    Attributes:
    - meaning: What happened, in one sentence.
    - action: What the customer should do about it.
    """

    meaning: str
    action: str


## What each code means for a customer, one sentence; the error-code page's table of codes.
CODE_HELP: dict[StatusCode, str] = {
    StatusCode.OK: "The request succeeded; the reply carries the action chunk.",
    StatusCode.OK_NEW_SESSION: "The request succeeded on a new session; the reason says why the session is new.",
    StatusCode.UNAUTHORIZED: "The API key was refused, so the request was not run.",
    StatusCode.NO_CAPACITY: "No model server could take the request right now.",
    StatusCode.BAD_REQUEST: "The request itself was rejected; sending it again gets the same answer.",
    StatusCode.SERVE_FAULT: "The model server running the checkpoint failed the request.",
    StatusCode.GATEWAY_FAULT: "The inference gateway itself failed the request.",
}

## A new session after a lost one: what every reason of `DA-0001` but `new` asks of the customer.
_NEW_EPISODE = (
    "Treat it as the start of a new episode: the client has adopted the new session but does not return the action, "
    "so bring the robot to a safe pose and start the task again, or stop."
)

## What each reason means for a customer and what to do, one entry per member of every reason enum.
REASON_HELP: dict[Reason, CustomerHelp] = {
    NewSessionReason.NEW: CustomerHelp(
        "The request carried no session (the first of an episode, or the first after `reset_episode()`), so the "
        "gateway started one.",
        "Nothing: the client returns the action and sends the new session with the episode's next requests.",
    ),
    NewSessionReason.EXPIRED: CustomerHelp(
        "The session sent was ended after it sat idle too long, so the gateway started a new one and the model's "
        "memory of the episode starts over.",
        _NEW_EPISODE + " Keep requests flowing while an episode runs.",
    ),
    NewSessionReason.UNKNOWN: CustomerHelp(
        "The gateway no longer holds the session sent, so it started a new one and the model's memory of the "
        "episode starts over.",
        _NEW_EPISODE,
    ),
    NewSessionReason.SERVE_DIED: CustomerHelp(
        "The model server holding the session stopped, so the gateway started a new session on another one and the "
        "model's memory of the episode starts over.",
        _NEW_EPISODE,
    ),
    UnauthorizedReason.MISSING: CustomerHelp(
        "The request carried no API key.",
        "Pass `api_key`, or set `DREAM_API_KEY` in the environment or in `~/.dreamlabs/credentials`.",
    ),
    UnauthorizedReason.UNKNOWN_KEY: CustomerHelp(
        "The API key is not one the platform issued; it may be mistyped or cut short.",
        "Copy the key again from the platform's API keys page, whole.",
    ),
    UnauthorizedReason.DISABLED: CustomerHelp(
        "The API key was revoked, or the account is suspended.",
        "Create a new key on the platform, or contact Dream Labs about the account.",
    ),
    UnauthorizedReason.OVER_CAP: CustomerHelp(
        "The account's spending this month has passed its monthly budget cap, so its API keys are paused.",
        "Raise the budget cap on the platform's Billing page, or wait for the next month; the keys work again within "
        "about two minutes.",
    ),
    UnauthorizedReason.NOT_ENTITLED: CustomerHelp(
        "The account may not run the checkpoint named.",
        "Pick a checkpoint that `DreamClient.get_models()` lists, or contact Dream Labs about access.",
    ),
    UnauthorizedReason.REGION_NOT_SUPPORTED: CustomerHelp(
        "Inference is not offered in the country the request came from yet; the error message names the countries "
        "it is offered in.",
        "Run from a country it is offered in, or contact Dream Labs.",
    ),
    NoCapacityReason.POOL_FULL: CustomerHelp(
        "Dream-1 is at max capacity right now.",
        "Wait `retry_after_s` seconds, then send the same request again.",
    ),
    NoCapacityReason.POOL_WARMING: CustomerHelp(
        "No model server for this checkpoint is up yet, so one is starting: the first request to a checkpoint, or "
        "the first after it sat idle, starts it cold.",
        "Wait `retry_after_s` seconds and send the same request again; keep retrying until a server answers.",
    ),
    NoCapacityReason.RATE_LIMITED: CustomerHelp(
        "The session sent requests faster than the gateway allows for one session.",
        "Send requests less often; wait a moment, then send the same request again.",
    ),
    NoCapacityReason.QUEUE_TIMEOUT: CustomerHelp(
        "The request waited too long for its model server to be free, and was dropped.",
        "Send the same request again.",
    ),
    NoCapacityReason.MAX_SERVES: CustomerHelp(
        "Dream-1 is at max capacity right now, and no more model servers may start for this checkpoint.",
        "Sending the same request again at once gets the same answer: try again later, or contact Dream Labs about "
        "the limit.",
    ),
    BadRequestReason.MALFORMED: CustomerHelp(
        "The request is not one the gateway can read, or its observation does not fit what the checkpoint reads: a "
        "camera or state key missing, a state of the wrong width, or frames that do not decode.",
        "Fix the observation as the error message says; if the message is about the request itself, upgrade the "
        "`dream` client, and if a current client still gets it, contact Dream Labs with the error message.",
    ),
    BadRequestReason.TOO_LARGE: CustomerHelp(
        "The request is larger than the gateway accepts; the camera frames are most of it.",
        "Send smaller frames: lower the camera resolution or the JPEG quality.",
    ),
    BadRequestReason.UNKNOWN_OP: CustomerHelp(
        "The request asked for an operation the gateway does not know.",
        "The `dream` client never sends one; upgrade it, or contact Dream Labs if a current client gets this.",
    ),
    BadRequestReason.INVALID_SESSION_ID: CustomerHelp(
        "The session id sent is not of the form the gateway issues.",
        "The `dream` client sends only the ids the gateway gave it; upgrade it, or contact Dream Labs if a current "
        "client gets this.",
    ),
    BadRequestReason.GLOBAL_OP_REJECTED: CustomerHelp(
        "The request asked to change a setting of the whole model server, which a customer request may not do.",
        "The `dream` client never sends one; upgrade it, or contact Dream Labs if a current client gets this.",
    ),
    BadRequestReason.UNKNOWN_CHECKPOINT: CustomerHelp(
        "The gateway does not know the checkpoint named.",
        "Use an `id` that `DreamClient.get_models()` lists; a fine-tune's id is its job's `fine_tuned_model`, set "
        "once the checkpoint is published.",
    ),
    BadRequestReason.CLIENT_OUTDATED: CustomerHelp(
        "The `dream` library that sent the request is older than the oldest major version the inference gateway "
        "serves, and such a library is refused rather than run against an API it does not match.",
        'Install the current library with your API key, `curl -fOJ -H "Authorization: Bearer $DREAM_API_KEY" '
        "<platform>/downloads/latest && pip install ./dream-*.whl`, where `<platform>` is the platform your key is "
        "from (the error message gives the whole command), then start the run again.",
    ),
    ServeFaultReason.TIMEOUT: CustomerHelp(
        "The model server did not answer in time.",
        "Send the same request again.",
    ),
    ServeFaultReason.ERROR: CustomerHelp(
        "The model server failed while computing the action.",
        "Send the same request again; if the same observation fails every time, contact Dream Labs with the error "
        "message.",
    ),
    ServeFaultReason.UNREACHABLE: CustomerHelp(
        "The gateway could not reach the model server.",
        "Send the same request again.",
    ),
    ServeFaultReason.CAPACITY: CustomerHelp(
        "The model server refused the session as full: a fault on the Dream Labs side, and Dream Labs is alerted "
        "to it.",
        "Send the same request again.",
    ),
    GatewayFaultReason.INTERNAL: CustomerHelp(
        "The gateway hit an internal fault.",
        "Send the same request again; if it keeps happening, contact Dream Labs with the error message.",
    ),
    GatewayFaultReason.STORE_WRITE_FAILED: CustomerHelp(
        "The gateway could not record the session, so it did not run the request.",
        "Send the same request again.",
    ),
    GatewayFaultReason.STEERING_UNAVAILABLE: CustomerHelp(
        "This checkpoint needs a gateway feature that is not turned on, so the request was not run.",
        "Nothing on your side fixes it: contact Dream Labs with the checkpoint id and the error message.",
    ),
    GatewayFaultReason.CANNOT_SCALE: CustomerHelp(
        "No model server can be started for this checkpoint; for a fine-tune, its weights could not be prepared.",
        "Sending the request again tries once more; if it fails again, contact Dream Labs with the checkpoint id and "
        "the error message.",
    ),
}

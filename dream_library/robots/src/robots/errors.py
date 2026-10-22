"""Exception hierarchy raised by the `robots` control loop.

Everything the package raises on its own behalf derives from #RobotsError, so
`except robots.errors.RobotsError` catches all of it. Driver exceptions (i2rt, OpenCV) and
`dream.DreamError` are never wrapped: they propagate unchanged, and the loop boundary is the
one place that catches `dream.DreamError`.

Messages name the component and the fact, e.g.
`"camera 'top' frame is 0.412 s old, older than stale_after_s=0.200"`.
"""


class RobotsError(Exception):
    """Base of everything the package raises itself."""


class ControlLoopError(RobotsError):
    """The loop cannot continue: a stall past `max_stall_ticks`, or a chunk of the wrong shape.

    Attributes:
    - reason: For a stall, the `error_code()` of the refill failure that emptied the buffer (`pool_warming`,
      `max_serves`, `DreamActionError`, ...) when one stood, so `error_code()` of the stall names the cause a
      page can act on rather than the stall itself; None otherwise.
    """

    def __init__(self, message: str, reason: str | None = None) -> None:
        """Builds the error.

        @param message What the loop cannot do, and why.
        @param reason The refill failure's code behind a stall, or None.
        """
        super().__init__(message)
        self.reason = reason


class ArmError(RobotsError):
    """A follower arm returned an incomplete observation or was given an invalid command."""


class SafetyStop(RobotsError):
    """An arm's own guard ended the run: a motor too hot, a joint blocked for too long, a gripper that stopped."""


class ModelReady(RobotsError):
    """The model answered only after a long start (`robots.control.MOVE_AFTER_WAIT_LIMIT_S`): the run stopped before
    its first move, so the arms never start moving on their own after a wait the operator may have walked away from;
    starting the run again moves at once."""


class CameraError(RobotsError):
    """A camera could not be opened, its capture thread died, or it never published a frame."""


class ConfigError(RobotsError):
    """The config file is missing a key, carries an unknown one, or holds a value of the wrong type."""


class StationError(RobotsError):
    """The station controller refused a start, stop, home or prompt request because of its current state."""


def error_code(exc: BaseException) -> str:
    """The short, machine-readable name of a failure, what the live `status.error_code` and `last_error` carry.

    @param exc The exception that ended a run, a refill or a home move.
    @return DreamAPI's envelope reason (`pool_warming`, `unknown_checkpoint`, `not_entitled`, ...) when the
        exception carries one (`dream.DreamServerError`), the cause behind a stall (`ControlLoopError.reason`),
        else the exception's class name (`ArmError`, `CameraError`, `DreamTimeout`, ...), so a page can word the
        message from the code instead of the text.
    """
    reason = getattr(exc, "reason", None)
    return reason if isinstance(reason, str) and reason else type(exc).__name__


## The `error_code` of a run DreamAPI ended by refusing the station's key as `disabled` (the account is out of credit
## or over its spending limit, or the key was revoked), after which the station brought the arms home
## (`robots.station`, unless the station file sets `home_on_key_refused: false`). The station's own code rather than
## DreamAPI's reason, so a page can say both why the run stopped and that the arms are home.
KEY_REFUSED = "key_refused"

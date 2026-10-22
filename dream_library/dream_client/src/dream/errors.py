"""The errors the `dream` client raises.

Every error the client raises derives from #DreamError, so `except dream.DreamError` catches all of them, and each
also derives from a builtin (`ValueError`, `TimeoutError` or `RuntimeError`), so code that catches the builtin keeps
working. They fall in three families:

- The call was wrong, and making it again gets the same error: #DreamConfigError (the config, the key, a URL, a path,
  a missing dependency), #DreamObservationError (the observation), #DreamActionError (the model's reply did not
  match), #DreamDatasetError (the dataset). Fix what the message names first.
- No answer the client could read: #DreamTimeout and #DreamProtocolError. Check the URL and the network, then make the
  same call again.
- A refusal: #DreamServerError and its subclasses #DreamAuthError, #DreamBadRequest and #DreamSessionExpired, which
  carry the answer's `code`, `reason`, `retryable` and `retry_after_s`. When `retryable` is True, wait
  `retry_after_s` seconds (a moment when it is None) and make the same call again; otherwise change the request.

#DreamUploadError is an upload that did not finish; running the upload again retries it. It and #DreamDatasetError,
the two errors only the dataset upload raises, are not in the `dream` namespace:
`from dream.errors import DreamDatasetError, DreamUploadError`.

One more type here is a warning, not an error: #DreamSlowNetworkWarning, issued (never raised) when the inference
client's recent round trips stay slow. Its page says how to silence it.

@par Example
```python
import time

import dream


def act(client, observation):
    while True:
        try:
            return client.get_action(observation)
        except dream.DreamServerError as error:
            if not error.retryable:
                raise
            time.sleep(error.retry_after_s or 1.0)
```
"""


class DreamError(Exception):
    """Base of every error the client raises; also raised as itself when a client is used wrongly.

    Raised as itself for a call on a client after close(), and for a second call on one inference client while the
    first is still running: an inference client serves one thread, so use one client per thread.
    """


class DreamConfigError(DreamError, ValueError):
    """Something the caller passed or configured is wrong; the message says what, and the same call fails again.

    The causes a customer meets:

    - no API key: pass `api_key`, or set `DREAM_API_KEY` in the environment or in `~/.dreamlabs/credentials`;
    - an API key holding a character an HTTP header cannot carry (a line break, a typographic quote): copy the key
      again;
    - a URL that is not `http(s)://<host>[:<port>]` (in the credentials, or passed to `DreamClient`), a host name that
      is not valid (a space, an empty label), or a credentials file that cannot be read; the message names the URL or
      the file;
    - a config key of the wrong type or range, a missing `checkpoint`, or a key the client does not take
      (`DreamClient.policy()`);
    - `DreamClient.set_observation()`, `DreamClient.get_action()` or another inference call before
      `DreamClient.policy()`;
    - an embodiment config that is not valid (the message names the key or part);
    - a dataset path that is not a directory or cannot be listed, or a request value JSON cannot carry (a numpy
      number as a fine-tuning knob: pass a Python `int` or `float`);
    - the dataset dependencies missing; the message says how to install the library with them.

    The model refusing a request as malformed (`DA-4000`) is #DreamBadRequest, a subclass.
    """


class DreamObservationError(DreamError, ValueError):
    """An observation that cannot be sent; raised before anything is sent, with the key named in the message.

    Either it does not match the embodiment the client checks it against (a key missing or extra, a shape, a dtype, a
    camera window that is not a non-empty list of base64 JPEG strings), or a value cannot travel as JSON (bytes, a
    set, an object array, a key that is not a string: send each frame as a base64 `str`). `strict=False` turns off the
    embodiment check, not the JSON one.
    """


class DreamActionError(DreamError, ValueError):
    """The model's reply is not an action the client can return; nothing is returned, and the message names the key.

    Either the reply does not match the embodiment the client checks it against (strict mode), which usually means
    the `embodiment` given to `DreamClient.policy()` is not the body the checkpoint was trained on, so
    check that config; or it is no action at all (no action dict, or a chunk that is not rows of numbers).
    """


class DreamTimeout(DreamError, TimeoutError):
    """The model did not answer within `timeout_ms` (15000 ms unless the config sets it).

    The connection is dropped before this is raised, so the next call opens a fresh one and calling again is safe.
    """


class DreamProtocolError(DreamError, RuntimeError):
    """No answer the client could read: the host could not be reached, or what answered is not the Dream API.

    Raised by get_action() and by the embodiment, dataset, fine-tuning and model calls alike, when the connection
    fails or the answer is not JSON or not the shape the Dream API answers. Check the network, then make the same call
    again.
    """


class DreamServerError(DreamError, RuntimeError):
    """A refusal, carried with the answer's code and reason; whether to call again is `retryable`.

    The codes come in two families:

    - From get_action(), the model's status codes: `DA-2000` (the call cannot be taken right now: `reason`
      `pool_warming` means the model is starting, `rate_limited` that the calls came too fast, `pool_full` and
      `queue_timeout` that it is at max capacity, `max_serves` that it is at max capacity and no more copies of it may
      start: not `retryable`, but with `retry_after_s` set a call after that wait may find a seat freed), `DA-5000`
      (the model failed the call) and `DA-6000` (a fault on the Dream Labs side; `cannot_scale` means the model
      cannot be started, and the message says why).
    - From the embodiment, dataset, fine-tuning and model calls, the service's own lowercase codes, with `reason`
      None: `not_found` (no such embodiment, dataset or job on this account), `conflict` (a name in use, a dataset not
      `processed`, a job already finished, the account at its limit of active jobs), `invalid_request` (a name,
      suffix or fine-tuning knob out of range; the message names it), `validation` (a value of the wrong type), and
      the HTTP 5xx ones, which are `retryable`.

    When `retryable` is True, wait `retry_after_s` seconds (a moment when it is None) and make the same call again;
    otherwise the same call gets the same answer, so change it first.

    Attributes:
    - code: The code, e.g. `"DA-2000"` from get_action() or `"not_found"` from the other calls; None when the answer
      carried none.
    - reason: Why, e.g. `"pool_warming"`; None when the answer carried none (always None outside get_action()).
    - retryable: Whether the same call may be made again: as the model's answer says for get_action(), and True for an
      HTTP 5xx answer of the other calls.
    - retry_after_s: Seconds to wait before calling again, or None.
    - message: The message; also what `str(error)` returns.
    """

    def __init__(
        self,
        message: str,
        code: str | None = None,
        reason: str | None = None,
        retryable: bool = False,
        retry_after_s: float | None = None,
    ) -> None:
        """Stores the answer's fields as they came; the client raises it, a caller never needs to construct one.

        @param message Human-readable text, shown verbatim to the caller.
        @param code The answer's code.
        @param reason The answer's reason.
        @param retryable Whether the same call may be made again.
        @param retry_after_s Seconds to wait before calling again.
        """
        super().__init__(message)
        self.message = message
        self.code = code
        self.reason = reason
        self.retryable = retryable
        self.retry_after_s = retry_after_s


class DreamAuthError(DreamServerError):
    """The API key was refused; never retryable: check the key, or the account, before calling again.

    From get_action() it is `DA-1000`, with `reason` `missing` (no key sent), `unknown_key` (mistyped or cut short),
    `disabled` (revoked, or the account suspended), `over_cap` (the account's spending this month passed its monthly
    budget cap: raise it on the platform's Billing page) or `not_entitled` (the checkpoint is not available to run on
    this account). From the embodiment, dataset, fine-tuning and model calls it is HTTP 401 or 403, with `code`
    `invalid_api_key`, or `forbidden` when the account has no username yet: every embodiment, dataset and fine-tuning
    call needs one, so ask Dream Labs support to set it. Check `DREAM_API_KEY` (or the `api_key` argument) against the
    platform's API keys page.
    """


class DreamBadRequest(DreamConfigError, DreamServerError):
    """The model refused the request as malformed (`DA-4000`); never retryable, the request must change.

    `reason` says why: `unknown_checkpoint` (use an id `DreamClient.get_models()` lists, or a job's
    `fine_tuned_model`), `too_large` (the camera frames are too big: lower the resolution or the JPEG quality),
    `client_outdated` (this library is not the build the model's server runs, after a deploy: the message ends with
    the one command that installs the current library from your platform) or `malformed`. A #DreamConfigError,
    because the config or the observation must change, that also carries the answer's code and reason like every
    other refusal, so `reason` tells the cases apart without parsing the message.
    """


class DreamSessionExpired(DreamServerError):
    """The model replaced the session in the middle of an episode (`DA-0001` with a reason other than `new`).

    The session sat idle too long, or the model restarted, so the model started a new session and its memory of the
    episode starts over. The new session is already in use: the client has adopted it, and the action that came with
    it is not returned. Bring the robot to a safe pose, then call get_action() again to continue on the new session,
    or call reset_episode() first to start the episode over.
    """


class DreamDatasetError(DreamError, ValueError):
    """The local checks refused the dataset, and nothing was sent; `str(error)` is the one line to fix.

    The dataset is not a LeRobot v2.1 dataset of the embodiment named, or one of its files cannot be read.
    `validate_dataset()` reports the same line; fix it and run `upload_dataset()` again. Import it from `dream.errors`.
    """


class DreamUploadError(DreamError, RuntimeError):
    """An upload that did not finish; `str(error)` is the one line, and running the upload again retries it.

    The dataset zip could not be written, the transfer to S3 did not complete, or the service did not find the zip at
    the size declared. Once the dataset was registered it is `failed` with the same line and its name is free again:
    run `upload_dataset()` again, which reuses the same dataset. Import it from `dream.errors`.
    """


class DreamSlowNetworkWarning(UserWarning):
    """A Python warning, not an error: the inference client's recent round trips are slow, and the robot's motion may
    be less smooth.

    get_action() issues it when the median of the last five round trips is at or above 1000 ms, with where the time
    goes: the network, when it is the largest part (with the upload rate when it is known), or the model. The action
    is returned as usual; the warning is never raised out of get_action(), even when warnings are turned into errors.
    It is issued once, then not again for five minutes while the round trips stay slow, unless they first recover
    (their median under 800 ms) and then slow down again. The same line goes to the `dream` logger, so it is in the
    client's log file too. For the figures of each call, read `last_timing` on the inference client.

    @par Example
    ```python
    import warnings

    import dream

    # Turn the warning off for the whole program; the `dream` log still records it.
    warnings.filterwarnings("ignore", category=dream.DreamSlowNetworkWarning)
    ```
    """

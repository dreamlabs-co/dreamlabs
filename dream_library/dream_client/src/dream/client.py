"""PolicyClient, the inference client: one call per control step, one boundary per episode, a lifecycle.

The client checks an observation against the configured embodiment, sends it to the model, reads the
status of the model's answer, checks the returned action chunk, and hands it back. It keeps the
session (none before the first answer and after `reset_episode()`, then the one the model answered)
and turns every refusal into an exception; the channel underneath is #dream.transport.Transport.

Observation keys are the model's own names (`state.arm_r.joint_pos`, `image_window.top`, `prompt`);
the `embodiment` config block groups them as `{group: {key: {"shape", "dtype"}}}`, and the flattened
form `"<group>.<key>"` is what the checks compare against. One key is the client's own, never the caller's:
`embodiment`, the robot the model runs on (`yam` or `other_embodiment`), from the config's `robot` or the `ik` default,
added after the checks.

The `embodiment` block may instead be the embodiment config a body was registered with (the dict
`add_embodiment()` takes: `state` and `action` parts with their `[start, end)` spans, `video` roles),
for a checkpoint fine-tuned on that body. The observation is then keyed by the config: `state.<part>`
arrays as wide as the part's span, `video.<role>` frame windows, `prompt`; and the reply's `per_key`
must carry `action.<part>` rows for every action part. The two forms cannot be confused (every entry
of a spec is a dict, a config's are spans and video keys), so one key carries either; the client reads
it once, when it is constructed. A config with `parts` names a body whose fine-tune reads and answers the
slots its plan trains (`shared.datasets.conversion`): the caller still sends and gets the body's own parts,
and the client converts each way.
"""

import dataclasses
import logging
import threading
import time
from collections.abc import Mapping
from types import TracebackType
from typing import Any

import numpy as np
import numpy.typing as npt
from shared.datasets import conversion, embodiments

from dream import client_ik, credentials, datasets, pacing, seam_join, slow_network, smoothing
from dream.errors import (
    DreamActionError,
    DreamAuthError,
    DreamBadRequest,
    DreamConfigError,
    DreamError,
    DreamObservationError,
    DreamProtocolError,
    DreamServerError,
    DreamSessionExpired,
)
from dream.models import Model
from dream.transport import END_SESSION_PATH, Exchange, Transport, encode

## One record per session boundary, so a run log shows how many episodes the gateway saw and why each began.
_log = logging.getLogger("dream.client")

## Config keys consumed locally and never sent (see `DreamClient.policy()`).
_CONNECTION_KEYS = ("host", "port", "timeout_ms", "strict", "scheme")

## Config keys that pace the caller's loop (`dream.pacing`); consumed locally and never sent (`DreamClient.policy()`).
_PACING_KEYS = ("execution_horizon", "action_hz", "sync_above_ms")

## Config keys the client reads for the request or the checks; DreamAPI carries no other config.
_SESSION_KEYS = ("checkpoint", "embodiment", "robot")

## The config key that names the robot the model runs on (#ROBOTS); the client sends it on every request as the
## observation's #_EMBODIMENT_KEY.
_ROBOT_KEY = "robot"

## The robot a model trained on YAM arms runs on, as the serve reads it: `yam`, its own arm, or `other_embodiment`,
## any other, on which the serve withholds the model's state (all zeros) and answers no joint head.
YAM = "yam"
OTHER_EMBODIMENT = "other_embodiment"
ROBOTS = (YAM, OTHER_EMBODIMENT)

## The observation key the robot travels under, the one the serve reads; never the caller's, so the client refuses an
## observation that carries it.
_EMBODIMENT_KEY = "embodiment"

## The reply's joint head, which the serve answers as null on #OTHER_EMBODIMENT: another robot's joints are never
## this arm's.
_JOINT_HEAD_KEYS = ("action.right_arm", "action.left_arm")

## Observation group whose entries are camera windows on the wire: a non-empty list of base64 JPEG strings, oldest
## to newest. A bare string is a real mistake the serve reports only as "key video.<role> not found".
_FRAME_WINDOW_GROUP: str = "image_window"

## The observation group an embodiment config's cameras travel under, one key per camera role, each a frame window too.
_VIDEO_GROUP: str = "video"

## Embodiment group whose keys describe actions; every other group describes observations.
_ACTION_GROUP = "action"

## Reply key under which the serve returns every action part it computed, as rows; what a body named by an embodiment
## config reads its action from.
_PER_KEY = "per_key"

## Envelope codes, per `shared.contracts.envelope` of the DreamAPI product.
_OK = "DA-0000"
_OK_NEW_SESSION = "DA-0001"
_UNAUTHORIZED = "DA-1000"
_BAD_REQUEST = "DA-4000"
## The bad request's reason for a checkpoint the key may not run.
_UNKNOWN_CHECKPOINT = "unknown_checkpoint"
## The bad request's reason for a library the gateway refuses by the version it names
## (`dream.transport.CLIENT_VERSION_HEADER` says when).
_CLIENT_OUTDATED = "client_outdated"

## The `DA-0001` reason that means "you asked for a new session", as opposed to "you lost one".
_NEW_SESSION_REASON = "new"

## Timeout of the `/v1/end_session` request reset_episode() and close() send: a release, not an inference, and
## best effort, so a gateway that is slow or gone holds the caller for at most this long.
_END_SESSION_TIMEOUT_MS = 2000


@dataclasses.dataclass(frozen=True)
class _Checks:
    """What strict mode compares each observation and reply against, compiled once from the `embodiment` block.

    Attributes:
    - observation: Dotted observation key to its `{"shape", "dtype"}` entry (`{}` only requires the key).
    - frame_window_group: The observation group whose keys must each be a frame window: `image_window` for a spec,
      `video` for an embodiment config.
    - action: Bare action key (`q_r_chunk`) to its entry; empty for an embodiment config.
    - per_key_widths: `action.<part>` to the part's width, each of which the reply's `per_key` must carry as rows;
      empty for a spec.
    """

    observation: dict[str, dict[str, Any]]
    frame_window_group: str
    action: dict[str, dict[str, Any]]
    per_key_widths: dict[str, int]


class PolicyClient:
    """The inference client for one checkpoint: sends observations to the model and returns its action chunks.

    Internal: `DreamClient.policy()` opens it and builds every observation it sends from the ticks set_observation()
    was given; constructing it directly is not supported. One call at a time per client.

    Attributes:
    - host: The model's host name.
    - port: The model's port.
    - timeout_ms: How long one call waits for the model, milliseconds.
    - scheme: `http` or `https`, what the requests go over.
    - api_token: The API key sent with every request, or `None`.
    """

    ## Everything a user may touch; `__dir__` hides the rest.
    _PUBLIC = (
        "get_action",
        "reset_episode",
        "close",
        "last_timing",
        "mode",
        "refill_due",
        "execution_horizon",
        "action_hz",
    )

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        timeout_ms: int = 15000,
        api_token: str | None = None,
        session_config: dict[str, Any] | None = None,
        strict: bool = True,
        scheme: str = "http",
        execution_horizon: int = pacing.EXECUTION_HORIZON,
        action_hz: float = pacing.ACTION_HZ,
        sync_above_ms: float = pacing.SYNC_ABOVE_MS,
        kinematics: Mapping[str, seam_join.Kinematics] | None = None,
        ik: Any = None,
        model: Model | None = None,
    ) -> None:
        """Binds the connection settings, the checkpoint and the pacing; no request is sent.

        @param host The model's host name.
        @param port The model's port.
        @param timeout_ms How long one call waits for the model, milliseconds.
        @param api_token The API key, sent as `Authorization: Bearer`; omitted when falsy.
        @param session_config `checkpoint` (required) and optionally `embodiment`, a shape spec or the embodiment
            config of the body the checkpoint was fine-tuned on, and `robot`, `"yam"` or `"other_embodiment"`, which
            every request carries as its observation's `embodiment` (left out: `"other_embodiment"` when `ik` sends a
            model trained on YAM arms zeros for another arm's joints, else nothing); nothing else.
        @param strict Check observations and actions against `session_config["embodiment"]`.
        @param scheme `http` (default) or `https`; `https` speaks TLS to a host fronted by a proxy that
            terminates it.
        @param execution_horizon In async, rows of each chunk the caller's loop executes before the next takes over.
        @param action_hz The caller's row rate, rows per second.
        @param sync_above_ms Round trip, milliseconds, past which the client paces synchronously (`dream.pacing`).
        @param kinematics Per chunk key (`q_r_chunk`, `q_l_chunk`), the kinematics of the arm it drives; an arm with an
            entry has its seams joined on the end effector's pose (`dream.seam_join`), every other arm in joint space.
        @param ik Client-side IK for an EEF checkpoint (`dream.client_ik`): `{"arm", "head"}`, or None for none.
        @param model The platform's record of the checkpoint (`dream.models`), which `ik` needs: what the model was
            trained on decides how the client sends the arm. None without `ik`.
        @throws DreamConfigError `checkpoint` is missing or not a non-empty string, the config carries
            a key the request has no field for, `robot` is not `"yam"` or `"other_embodiment"`, `strict` is on and the
            embodiment block is a malformed spec or an embodiment config that is not valid, or `ik` is not one the
            client takes, names head `joint` for a model trained on YAM arms or with `robot` `"other_embodiment"`,
            names a beta robot whose gripper part the embodiment does not name, or cannot send this arm or its cameras
            the way a model trained on YAM arms reads them (`dream.client_ik`).
        """
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.scheme = scheme
        self.api_token = api_token
        config = dict(session_config or {})
        unknown = sorted(str(key) for key in config if key not in _SESSION_KEYS)
        if unknown:
            # DreamAPI's request body has no config field: a key we cannot deliver must fail here,
            # not vanish silently.
            raise DreamConfigError(
                f"config keys {unknown} have no field in DreamAPI's /v1/get_action request; "
                f"the client forwards only {list(_SESSION_KEYS)}"
            )
        checkpoint = config.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise DreamConfigError(
                f"config key 'checkpoint' must be a non-empty str naming the checkpoint, got {checkpoint!r}"
            )
        self._checkpoint: str = checkpoint
        robot = config.get(_ROBOT_KEY)
        if _ROBOT_KEY in config and not (isinstance(robot, str) and robot in ROBOTS):
            raise DreamConfigError(f"config key '{_ROBOT_KEY}' must be one of {list(ROBOTS)}, got {robot!r}")
        if robot == OTHER_EMBODIMENT and isinstance(ik, dict) and ik.get("head") == "joint":
            # The serve answers no joint head on another robot, so a joint head would run nothing.
            raise DreamConfigError(
                f"config key '{_ROBOT_KEY}' is {OTHER_EMBODIMENT!r}, on which the model answers no joints, so config "
                "key 'ik' with the head 'joint' has nothing to run: use the head 'eef'"
            )
        # None until the gateway assigns one on the first reply, and again after reset_episode().
        self._session_id: str | None = None
        # Validation is local and free, so a malformed spec is a config error now rather than
        # a confusing check failure on the first step.
        self._embodiment: dict[str, Any] | None = config.get("embodiment")
        # A fine-tune of a body with `parts` reads and answers the slots its plan trains; the client converts both
        # ways, whatever `strict` says, so the caller sends and gets the body's own parts.
        self._body = _planned(self._embodiment)
        if self._body is not None and ik is not None:
            raise DreamConfigError(
                "config key 'ik' runs a model that answers hand poses on another arm; a fine-tune of a robot whose "
                "embodiment config has parts answers in that robot's own parts, so it takes no ik"
            )
        self._checks: _Checks | None = None
        if strict and "embodiment" in config:
            self._checks = _compile_checks(self._embodiment)
        # Client-side IK for an EEF checkpoint (dream.client_ik): the tool poses out, the joints back. Built and
        # compiled here, so the first get_action() does not pay for it.
        self._ik = None
        client_ik.refuse_proprio_test_without_ik(ik)
        if ik is not None:
            if model is None or model.id != checkpoint:
                raise DreamConfigError(f"config key 'ik' needs the platform's record of {checkpoint!r}, got {model!r}")
            self._ik = client_ik.ClientIK(ik, action_hz, self._embodiment, model)
            if self._checks is not None:
                # The checks read what is sent and answered: the cameras under the model's roles, a YAM model's joints.
                observation = self._ik.sent_checks(self._checks.observation)
                widths = self._ik.reply_widths(self._checks.per_key_widths)
                self._checks = dataclasses.replace(self._checks, observation=observation, per_key_widths=widths)
        ## The robot every request names in its observation's #_EMBODIMENT_KEY, or None to name none: `robot` as given,
        ## else #OTHER_EMBODIMENT when the client sends a model trained on YAM arms zeros for another arm's joints
        ## (`client_ik.ClientIK.yam_joints`), so the serve withholds the state that is not this arm's.
        defaulted = OTHER_EMBODIMENT if self._ik is not None and self._ik.yam_joints is not None else None
        self._robot: str | None = robot if _ROBOT_KEY in config else defaulted
        if self._checks is not None and self._robot == OTHER_EMBODIMENT:
            # The serve answers the joint head as null there, so the checks do not require it.
            widths = {key: width for key, width in self._checks.per_key_widths.items() if key not in _JOINT_HEAD_KEYS}
            self._checks = dataclasses.replace(self._checks, per_key_widths=widths)
        # The seam join between consecutive chunks (dream.smoothing): always on, nothing in the config sets it; the
        # row rate and the arms' kinematics are inputs it needs, not settings. With IK, or on a model's own wire (an
        # embodiment config, what a YAM station sends), it anchors on the body's joints, `state.<side>_arm`.
        pose_keys = (
            client_ik.POSE_KEYS
            if self._ik is not None or _is_embodiment_config(self._embodiment)
            else smoothing.ARM_POSE_KEYS
        )
        fence = self._ik.fence if self._ik is not None and (self._ik.reach or self._ik.region) else None
        self._smoother = smoothing.SeamSmoother(action_hz, kinematics, pose_keys, fence)
        # Execution horizon, row rate and the sync/async switch (dream.pacing): what the caller's loop asks before
        # every request, and how many rows each reply carries.
        self._pacer = pacing.Pacer(execution_horizon, action_hz, sync_above_ms)
        # Non-blocking: a second thread must fail fast, never wait for a connection it may corrupt.
        self._lock = threading.Lock()
        self._transport = Transport(host, port, timeout_ms, scheme)
        # The cost of the last request that reached the wire, success or failure; see last_timing.
        self._last_timing: dict[str, Any] | None = None
        # The last few successful round trips, and whether a slow run of them is worth a warning (dream.slow_network).
        self._slow_watch = slow_network.Watch()
        self._closed = False

    def get_action(self, observation: dict[str, Any], *, commanded: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Sends one observation to the model and returns its action chunk.

        The observation is the only thing sent: the key was given when the client was opened, and the session is
        the client's own (none on the first call of an episode, so the model starts one; then the one it answered,
        until reset_episode()). The first call to a checkpoint that is not running yet is refused with `DA-2000`
        `pool_warming` while the model starts: wait `retry_after_s` and call again.

        When the median of the last five round trips is at or above 1000 ms, the call issues one
        `dream.DreamSlowNetworkWarning` (a Python warning, never raised) saying whether the network or the model is
        slow, and then stays quiet for five minutes unless the round trips recover first; the action is returned as
        usual. `warnings.filterwarnings("ignore", category=dream.DreamSlowNetworkWarning)` turns it off.

        With a robot (the config's `robot`, or `"other_embodiment"` when `ik` sends a model trained on YAM arms zeros
        for another arm's joints), the observation sent also carries it as `embodiment`. On `"other_embodiment"` the
        model reads none of the arm's state and answers no joints: `per_key`'s `action.right_arm` and
        `action.left_arm` come back as None, and the strict checks do not require them.

        @param observation The observation dict, keyed as the checkpoint reads it. A fine-tuned model: one
            `state.<part>` per part of its embodiment config's `state` (a flat list or array as wide as the part),
            one `video.<role>` per role of its `video` (that camera's last 9 frames, oldest first, each a base64
            JPEG string), and `prompt`; for a robot whose config has `parts`, still its own parts, which the client
            converts to the ones the model trained on before sending. `dreamlabs/dream-1-beta` reads its own keys the
            same way (`state.right_arm`, `state.right_gripper`, `state.left_arm`, `state.left_gripper`,
            `video.overhead`, `video.wrist_l`,
            `video.wrist_r`, `prompt`, and each hand's pose, which `ik` adds) or, deprecated, the YAM station keys:
            `state.arm_r.*` and `state.arm_l.*`
            arrays, `image_window.*` lists of base64 JPEG frames, and `prompt`. Values are what JSON carries
            (numbers, strings, lists, numpy arrays), never bytes. Never `embodiment`, which the client sends from
            `robot`.
        @param commanded Per chunk key (`q_r_chunk`, `q_l_chunk`), the joints the loop last commanded that arm before
            it took the observation (an arm not commanded yet left out; a whole written row, the gripper after the
            joints, is fine), or None. Never sent: the seam join and the
            IK use them to take over exactly where the loop is on the previous chunk. Without them the client finds
            that from the measured joints, which trail the command on a real arm, so each chunk starts a few rows
            back along the plan.
        @return The action dict. A fine-tuned model's actions are under `per_key`: one `action.<part>` per part of its
            config's `action`, as rows, one per step (48 for Dream-1). For a robot whose config has `parts`, one per
            action part the model trains, converted back from its answer: rows of the part's width, in its own units
            (a gripper between its closed and open, a rotation in its format, a part marked delta as the change each
            row makes: the first from the observed state, each later one from the row before); a pose part recorded
            but not trained is not among them. A YAM body's are `q_r_chunk` and `q_l_chunk`,
            `[H, joints + 1]` (the arm's joint targets, then the gripper), already blended into the previous chunk so
            the control loop meets no step where one chunk replaces the last. Any other key the model adds is passed
            through. The chunk comes back whole whatever the pacing; calls slower than `sync_above_ms` change only
            when refill_due() says to ask again (`mode`). With `ik` and head `eef`, `q_r_chunk` and `q_l_chunk` are the
            model's tool poses solved to joints, then blended the same way: the arm's joints in radians, then the
            grip; for a beta robot so is head `joint`, from its joint answer. Those replies drop the serve's `q_r`
            and `q_l`, the model's own first row of joints (a YAM-trained model's are YAM's), never this arm's;
            `per_key` is the model's answer as it came.
        @throws DreamObservationError Before anything is sent: the observation carries `embodiment` (set `robot`
            instead), it does not match the configured embodiment (strict mode), a value cannot be sent as JSON
            (bytes, a set, an object array; send each
            frame as a base64 `str`), or, with `ik`, it lacks `state.right_arm` or `state.left_arm`, carries a tool
            pose or a gripper drive the client computes, or (a beta robot) lacks a grip (`state.right_grip`,
            `state.left_grip`) of one number; for a robot whose config has `parts`, a part the model reads is not as
            many numbers as its span or a rotation part's value is not a rotation; or `commanded` names a key that is
            not a chunk key or holds anything but the arm's joints (as many as its measured pose in the observation
            holds, its spec's with `ik`) or its written row (one more) as finite numbers. The message names the key.
        @throws DreamActionError The model's reply is not a usable action: it does not match the configured
            embodiment (strict mode), it carries no action dict or a chunk that is not rows of numbers, or, with `ik`,
            its `per_key` lacks a part the client solves or converts from (a tool pose, a grip, a beta robot's joint
            head) or holds one of the wrong shape, or (a robot whose config has `parts`) lacks a part the model trains,
            holds one that is not rows of its width, or answers a 6D rotation with a vector of length zero. Nothing is
            returned.
        @throws DreamTimeout No answer within `timeout_ms` (15000 ms by default); calling again is safe.
        @throws DreamProtocolError The model could not be reached, or what answered is not the Dream API; check the
            URL and the network, then call again.
        @throws DreamConfigError The host is not a valid host name, or the API key holds a character an HTTP
            header cannot carry.
        @throws DreamAuthError The key was refused (`DA-1000`): missing, unknown, revoked, or the checkpoint is not
            available to run on this account. Never retryable.
        @throws DreamBadRequest The model refused the request as malformed (`DA-4000`): `reason` is
            `unknown_checkpoint`, `too_large` (send smaller frames) or `malformed`. Never retryable.
        @throws DreamSessionExpired The model replaced the session mid-episode (`DA-0001` with a reason other than
            `new`). The new session is already in use and the action is not returned: call again to continue on it,
            or reset_episode() to start the episode over.
        @throws DreamServerError Any other refusal, with `code`, `reason`, `retryable` and `retry_after_s`: `DA-2000`
            (`pool_warming`: the model is starting; `rate_limited`: the calls came too fast; `pool_full`,
            `queue_timeout`: it is at max capacity; `max_serves`: it is at max capacity and no more copies of it may
            start, so `retryable` is False, though a gateway that sets its `retry_after_s` may have a seat free after
            that wait), `DA-5000` (the model failed the call) or `DA-6000` (a fault on the Dream Labs
            side; `cannot_scale`: the model cannot be started, the message says why). When `retryable` is True,
            wait `retry_after_s` seconds (a moment when it is None) and call again with the same observation.
        @throws DreamError The client is closed, or another call is running on it (use one client per thread).
        """
        if self._closed:
            raise DreamError("inference request on a closed client")
        if not self._lock.acquire(blocking=False):
            raise DreamError(
                "PolicyClient is not thread-safe: another inference request is in flight; use one client per thread"
            )
        try:
            if isinstance(observation, Mapping) and _EMBODIMENT_KEY in observation:
                raise DreamObservationError(
                    f"observation key '{_EMBODIMENT_KEY}' is the client's own: it sends the robot the model runs on "
                    f"there, from config key '{_ROBOT_KEY}' (one of {list(ROBOTS)}); leave it out of the observation "
                    f"and set '{_ROBOT_KEY}' in the policy() config instead"
                )
            # Per chunk key, the joints of the arm it drives, what `commanded` may hold: with IK its spec's, else as
            # many as its measured pose in the observation holds.
            arm_joints = self._ik.joints if self._ik is not None else self._smoother.arm_joints(observation)
            last = _commanded(commanded, arm_joints)
            # With IK the model reads the tool poses, and a beta robot's joints and gripper in its own units; the
            # caller's observation, in the arm's, is what the solve and the join read.
            sent = observation if self._ik is None else self._ik.observe(observation)
            if self._checks is not None:
                _check(
                    sent,
                    self._checks.observation,
                    DreamObservationError,
                    frame_window_group=self._checks.frame_window_group,
                )
            # The body's own parts: a delta part's answer comes back as changes from its state here.
            own = sent
            if self._body is not None:
                # Checked in the body's own parts, sent in the slots its fine-tune reads.
                sent = _to_model(self._body, sent)
            sent_s = time.perf_counter()
            try:
                # The wire is schema-agnostic, so this is the one place the reply gets its type.
                action: dict[str, Any] = self._post(sent)
            except DreamSessionExpired:
                # The episode restarted server-side: the plan the arms were on is gone with it.
                self._smoother.reset()
                raise
            if self._checks is not None:
                # Checked before returning: a mis-shaped action that reaches the control loop
                # is one that reaches hardware.
                _check(action, self._checks.action, DreamActionError, allow_extra=True)
                _check_per_key(action, self._checks.per_key_widths)
            if not isinstance(action, dict):
                raise DreamActionError(f"the model's reply carries no action dict, got {type(action).__name__}")
            rtt_s = time.perf_counter() - sent_s
            # This reply's own round trip picks the mode the loop paces its next request in.
            self._pacer.record(rtt_s)
            slow = self._slow_watch.record(self._last_timing, time.monotonic())
            if slow is not None:
                slow_network.warn(slow)
            if self._ik is not None:
                action = self._ik.solve(action, observation, self._smoother, rtt_s, last)
            if self._body is None:
                return self._smoother.smooth(observation, action, sent_s, rtt_s, last)
            # The joins anchor on the arms' joints as the model reads them (`state.<side>_arm`).
            return _to_body(self._body, own, self._smoother.smooth(sent, action, sent_s, rtt_s, last))
        finally:
            self._lock.release()

    def reset_episode(self) -> None:
        """Ends the current episode; the next get_action() starts a new session on the model.

        Call it between episodes. The session held, if any, is released on the model's side now, in one short
        request that never raises: when the model is slow or unreachable the client waits at most two seconds,
        and the session then ends on its own once it sits idle. The next get_action() starts a new session with
        the model's memory of the episode cleared, and the chunk blending forgets the previous chunk, so the new
        episode's first chunk is not blended into the old one's plan. Calling it before the first get_action(),
        or twice in a row, sends nothing and is harmless.

        @throws DreamError The client is closed, or a get_action() is running on it.
        """
        if self._closed:
            raise DreamError("episode reset on a closed client")
        if not self._lock.acquire(blocking=False):
            raise DreamError(
                "PolicyClient is not thread-safe: an inference request is in flight; "
                "reset the episode between requests"
            )
        try:
            self._end_session()
            if self._smoother is not None:
                self._smoother.reset()
        finally:
            self._lock.release()

    def close(self) -> None:
        """Releases the session on the model's side, if any, then the connection; calling it again does nothing.

        Leaving a `with` block calls it. A get_action() still running (a stop in the middle of one) is waited for up
        to two seconds; past that the session is left to end on its own once it sits idle, rather than holding the
        caller.
        """
        if getattr(self, "_closed", True):
            return
        if self._session_id is not None and self._lock.acquire(timeout=_END_SESSION_TIMEOUT_MS / 1000.0):
            try:
                self._end_session()
            finally:
                self._lock.release()
        self._closed = True
        self._transport.close()

    def _end_session(self) -> None:
        """Releases the held session on the gateway, best effort, and forgets its id; called with the lock held.

        `POST /v1/end_session` frees the seat at once and tells the serve to drop the session's state. Any
        failure (the gateway unreachable, a timeout, a refusal, a gateway too old for the route) is logged
        and swallowed: the seat frees on idle eviction anyway, and a stop or a reset must never fail on it.
        """
        session_id = self._session_id
        if session_id is None:
            return
        self._session_id = None
        try:
            response = self._transport.request(
                {"checkpoint": self._checkpoint, "session_id": session_id},
                self._headers(),
                path=END_SESSION_PATH,
                timeout_ms=_END_SESSION_TIMEOUT_MS,
            )
        except DreamError as exc:
            _log.info("session %s not released on the gateway (%s); its seat frees on idle eviction", session_id, exc)
            return
        status = response.get("status") if isinstance(response, dict) else None
        code = status.get("code") if isinstance(status, dict) else None
        if code == _OK:
            _log.info("session %s released", session_id)
        else:
            _log.info("session %s release answered %s; its seat frees on idle eviction", session_id, code)

    @property
    def execution_horizon(self) -> int:
        """In async, rows of each chunk the loop executes before the next takes over (`dream.pacing`); read-only."""
        return self._pacer.execution_horizon

    @property
    def action_hz(self) -> float:
        """The caller's row rate, rows per second, that round trips are turned into rows at (`dream.pacing`)."""
        return self._pacer.action_hz

    @property
    def mode(self) -> str:
        """`"async"` or `"sync"`: how the client paces the caller's loop, set by the recent `get_action` round trips.

        Async (their median is at most `sync_above_ms`): the next request goes out while the current chunk runs.
        Sync (it is longer): the next request waits until every row of the current chunk (the whole action
        horizon) has run, the arms holding through the inference. `"async"` before the first request. See
        `dream.pacing`.
        """
        return self._pacer.mode

    def refill_due(self, executed: int, remaining: int) -> bool:
        """Whether the caller's loop should send the next `get_action` now; it asks once per row, with none in flight.

        Sync: once `remaining` is 0. Async: once the rows executed plus the rows the measured round trip will take
        reach `execution_horizon`, so the reply lands as the last of them runs, or once `remaining` is worth less
        than the round trip, so the loop never runs dry first. Local and free; no request is sent.

        @param executed Rows of the current chunk the loop has executed since it installed it.
        @param remaining Rows of it still queued.
        @return True when the next request should start.
        """
        return self._pacer.refill_due(executed, remaining)

    @property
    def last_timing(self) -> dict[str, Any] | None:
        """Where the milliseconds of the most recent get_action() went; `None` before the first request.

        Read-only: a fresh plain dict on every read, recorded for every request that reached the network,
        the ones that raised included (a timeout is then visible as its `rtt_ms`). Keys:

        - `rtt_ms` (`float`): the client's own round trip around the HTTP exchange, connect included.
        - `serve_infer_ms` (`int | None`): the model's own inference time, as the model's answer reports it.
        - `gateway_ms` (`int | None`): the time on the Dream Labs side, from receiving the request to sending the
          answer, inference included.
        - `network_ms` (`float | None`): `rtt_ms - gateway_ms`, the part spent on the network. The three above are
          `None` when the answer carried no timing (a refusal).
        - `up_mbps` (`float | None`): the kernel's delivery rate for this request's socket in megabits per
          second, i.e. the achieved UPLOAD throughput; the first number to check when the round trip drifts,
          since the request body is large. `None` off Linux, where TCP_INFO is not exposed.
        - `tcp_cwnd_kb` (`float | None`): the sender's congestion window in KB; small means window-limited.
        - `tcp_retrans` (`int | None`): cumulative retransmits on the connection; rising means send-path loss.
        - `tcp_rtt_ms` (`float | None`): the kernel's smoothed RTT for the connection, milliseconds.
        - `tcp_app_limited` (`bool | None`): `True` when the last delivery-rate sample was limited by the app
          not having bytes ready (host-side pacing), so a low `up_mbps` with this set is not a network fault.
        - `request_bytes` / `response_bytes` (`int`): body sizes on the wire; `response_bytes` is `0` when
          no reply was read.

        @return The dict above, or `None` when no request has reached the network yet.
        """
        return dict(self._last_timing) if self._last_timing is not None else None

    def __enter__(self) -> "PolicyClient":
        """Returns the client itself; nothing is opened here."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Closes the client; the exception, if any, propagates unchanged."""
        self.close()

    def __del__(self) -> None:
        """Best-effort GC fallback; never rely on it."""
        # A GC fallback so a dropped client does not leak its connection. close() already swallows
        # everything, so this cannot raise during interpreter shutdown. A session still held is left
        # to the gateway's idle eviction: a finalizer sends nothing on the wire.
        if getattr(self, "_closed", True):
            return
        self._session_id = None
        self.close()

    def __repr__(self) -> str:
        """Names the endpoint only; never the token or an internal."""
        return f"<dream.Client host={self.host} port={self.port}>"

    def __dir__(self) -> list[str]:
        """Lists only the public surface plus the context-manager pair.

        Discovery is how private methods become de-facto public: someone finds a helper in a
        REPL, ships it, and then it can never change.
        """
        return list(self._PUBLIC) + ["__enter__", "__exit__"]

    def _headers(self) -> dict[str, str]:
        """The request headers: the key as a bearer token, when there is one.

        @return The headers.
        @throws DreamConfigError The key holds a character an HTTP header cannot carry.
        """
        return credentials._authorization(self.api_token) if self.api_token else {}

    def _post(self, observation: dict[str, Any]) -> Any:
        """Sends one `/v1/get_action` body, adopts the session id, and maps the envelope.

        @param observation The serve's observation dict, sent as the request's `observation` field, with the robot
            under #_EMBODIMENT_KEY when there is one (#_robot); the dict itself is not changed.
        @return The reply's `actions`.
        @throws DreamProtocolError The reply is not a dict with a status envelope.
        @throws DreamObservationError A value of the observation cannot be sent as JSON.
        @throws DreamError Per the envelope map; see get_action().
        """
        # Always sent, even when None: the gateway reads a null id as "mint me a session".
        request = {
            "checkpoint": self._checkpoint,
            "session_id": self._session_id,
            "reset": False,
            # Added here, past the strict checks, so they never see it; without a robot the body is as before.
            "observation": observation if self._robot is None else {**observation, _EMBODIMENT_KEY: self._robot},
        }
        headers = self._headers()
        try:
            response = self._transport.request(request, headers)
        except DreamError:
            # No reply to read a timing block from, but the request's cost is still worth knowing.
            self._last_timing = _timing_of(self._transport.last_exchange, None)
            raise
        except (TypeError, ValueError) as error:
            # encode() refused the body before a byte was sent: a value JSON cannot carry, which is the
            # observation's (the rest of the body is the client's own strings).
            raise DreamObservationError(_unencodable(observation, error)) from None
        self._last_timing = _timing_of(
            self._transport.last_exchange, response.get("timing") if isinstance(response, dict) else None
        )
        status = response.get("status") if isinstance(response, dict) else None
        if not isinstance(status, dict) or not isinstance(status.get("code"), str):
            raise DreamProtocolError(
                "inference reply carries no status envelope; is the correct DreamAPI gateway running?"
            )
        code = status["code"]
        sent = self._session_id
        # Adopt whatever session the gateway says we are in: on the first call this is where we
        # learn our id; afterwards it is an echo, unless the gateway minted a replacement.
        returned = response.get("session_id")
        if isinstance(returned, str) and returned:
            if returned != sent:
                _log.info(
                    "session %s %s (%s)",
                    returned,
                    "started" if sent is None else f"replaced {sent}",
                    _text(status.get("reason")) or "echo",
                )
            self._session_id = returned
        if code == _OK:
            return response.get("actions")
        if code == _OK_NEW_SESSION:
            if sent is None or status.get("reason") == _NEW_SESSION_REASON:
                return response.get("actions")
            raise DreamSessionExpired(
                f"inference session {sent} was replaced by {returned} ({status.get('reason')}): "
                "the episode restarted server-side",
                code=code,
                reason=_text(status.get("reason")),
            )
        raise _error_for(status, f"{self.scheme}://{self.host}:{self.port}", self._checkpoint)


def _error_for(status: dict[str, Any], called: str, checkpoint: str) -> DreamError:
    """Builds the exception for a non-OK status envelope, keeping DreamAPI's wording.

    @param status The envelope: `code`, `name`, `reason`, `retryable`, `retry_after_s`, `detail`.
    @param called The gateway URL the request went to; a refused key names it and where it came from.
    @param checkpoint The model the request named; an unknown one is named, with where the valid ids are listed.
    @return The exception to raise; a library the gateway refused as outdated carries the command that installs the
        current one from the platform the credentials name.
    """
    code = str(status["code"])
    reason = _text(status.get("reason"))
    detail = _text(status.get("detail"))
    name = _text(status.get("name")) or code
    message = f"DreamAPI refused the inference request: {code} {name}"
    if reason:
        message += f" ({reason})"
    if detail:
        message += f": {detail}"
    if code == _UNAUTHORIZED:
        where = credentials.describe_url(called)
        return DreamAuthError(f"{message}; called {where}; {credentials.KEY_PER_SITE}", code=code, reason=reason)
    if code == _BAD_REQUEST:
        if reason == _UNKNOWN_CHECKPOINT:
            message += (
                f"; checkpoint {checkpoint!r} is not a model your key may run: use an id from the Models page or "
                "DreamClient().get_models() (GET /v1/models)"
            )
        elif reason == _CLIENT_OUTDATED:
            message += f"; {datasets._install_hint()}"
        return DreamBadRequest(message, code=code, reason=reason)
    retry_after = status.get("retry_after_s")
    return DreamServerError(
        message,
        code=code,
        reason=reason,
        retryable=bool(status.get("retryable", False)),
        retry_after_s=float(retry_after) if isinstance(retry_after, (int, float)) else None,
    )


def _unencodable(observation: Any, error: Exception) -> str:
    """The message for an observation JSON cannot carry, naming the first key whose value (or name) is the cause.

    Each entry is encoded on its own, which costs a second pass over the observation, but only on this failure.

    @param observation The observation.
    @param error What encoding the whole observation raised.
    @return The one line for #DreamObservationError.
    """
    advice = "send numbers, strings, lists or numpy arrays, and each camera frame as a base64 str, not bytes"
    for key, value in observation.items() if isinstance(observation, dict) else ():
        try:
            encode({key: value})
        except (TypeError, ValueError) as cause:
            return f"observation key {key!r} cannot be sent as JSON ({cause}); {advice}"
    return f"the observation cannot be sent as JSON ({error}); {advice}"


def _text(value: Any) -> str | None:
    """A string field of the envelope, or None when absent."""
    return None if value is None else str(value)


def _timing_of(exchange: Exchange | None, block: Any) -> dict[str, Any] | None:
    """Builds the `last_timing` dict from the wire cost and DreamAPI's `timing` block, if any.

    @param exchange The transport's record of the request; `None` when nothing reached the wire.
    @param block The reply's `timing` value: `{"serve_infer_ms", "gateway_ms"}` on a success, otherwise
        anything (`None`, absent, or a shape this client does not know), which yields `None` fields.
    @return The dict `PolicyClient.last_timing` documents, or `None` without an exchange.
    """
    if exchange is None:
        return None
    gateway_ms = _whole_ms(block.get("gateway_ms")) if isinstance(block, dict) else None
    serve_infer_ms = _whole_ms(block.get("serve_infer_ms")) if isinstance(block, dict) else None
    return {
        "rtt_ms": exchange.rtt_ms,
        "serve_infer_ms": serve_infer_ms,
        "gateway_ms": gateway_ms,
        "network_ms": exchange.rtt_ms - gateway_ms if gateway_ms is not None else None,
        "up_mbps": exchange.up_mbps,
        "tcp_cwnd_kb": exchange.tcp_cwnd_kb,
        "tcp_retrans": exchange.tcp_retrans,
        "tcp_rtt_ms": exchange.tcp_rtt_ms,
        "tcp_app_limited": exchange.tcp_app_limited,
        "request_bytes": exchange.request_bytes,
        "response_bytes": exchange.response_bytes,
    }


def _whole_ms(value: Any) -> int | None:
    """An `int` millisecond count as the contract sends it, or None for anything else (`True` included)."""
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _compile_checks(embodiment: Any) -> _Checks | None:
    """The checks strict mode runs, from either form of the `embodiment` block; read once, at construction.

    A block #_is_embodiment_config accepts is parsed as the embodiment config it is (#_config_checks); any other block
    is a shape spec and is validated as one.

    @param embodiment The value of `config["embodiment"]`.
    @return The #_Checks, or None for an empty spec (`{}`), which declares nothing and so checks nothing.
    @throws DreamConfigError The block is a malformed spec, or an embodiment config the parser refuses.
    """
    if _is_embodiment_config(embodiment):
        return _config_checks(embodiment)
    _validate_embodiment(embodiment)
    if not embodiment:
        return None
    return _Checks(
        observation=_observation_spec(embodiment),
        frame_window_group=_FRAME_WINDOW_GROUP,
        action=_action_spec(embodiment),
        per_key_widths={},
    )


def _is_embodiment_config(block: Any) -> bool:
    """Whether an `embodiment` block is an embodiment config (what `add_embodiment` takes) rather than a shape spec.

    The two forms are disjoint: every group and every entry of a spec is a dict, while a config maps its parts to
    `[start, end]` spans and its roles to video keys (and may name a `head_camera` string). A block that names
    `state`, `action` and `video` and holds anything that is not a dict one or two levels down is a config; every
    other block is a spec, so a valid spec never changes meaning, and a config is never read as a spec.

    @param block The value of `config["embodiment"]`.
    @return True when the block is to be parsed as an embodiment config.
    """
    if not isinstance(block, dict) or not set(embodiments.CONFIG_KEYS) <= block.keys():
        return False
    for group in block.values():
        if not isinstance(group, dict) or not all(isinstance(entry, dict) for entry in group.values()):
            return True
    return False


def _config_checks(config: dict[str, Any]) -> _Checks:
    """The checks of an embodiment config: its parts and roles, named the way the model reads them.

    Each `state.<part>` must be an array as wide as the part's span, each `video.<role>` a frame window, and the
    reply's `per_key` must carry `action.<part>` rows as wide as each action part (with `parts`, one `action.<slot>`
    per slot the plan trains, which get_action() reads back as the body's parts). `head_camera` says how the service
    reads the body, not what travels on the wire, so it adds no check; a body's EEF parts travel like any other part.

    @param config The block, an embodiment config (the parser is `shared.datasets.embodiments`).
    @return The #_Checks.
    @throws DreamConfigError The parser refuses the config; the message carries its one line.
    """
    body = _parse(config)
    observation: dict[str, dict[str, Any]] = {}
    for part, (start, end) in body.state.items():
        observation[f"state.{part}"] = {"shape": [end - start]}
    for role in body.video:
        observation[f"{_VIDEO_GROUP}.{role}"] = {}
    return _Checks(
        observation=observation,
        frame_window_group=_VIDEO_GROUP,
        action={},
        per_key_widths=(
            {f"action.{part}": end - start for part, (start, end) in body.action.items()}
            if body.plan is None
            else {f"action.{slot.name}": slot.width for slot in body.plan.action}
        ),
    )


def _parse(config: dict[str, Any]) -> embodiments.Embodiment:
    """An `embodiment` block #_is_embodiment_config accepts, parsed.

    @param config The block.
    @return The parsed body.
    @throws DreamConfigError The parser refuses the config; the message carries its one line.
    """
    try:
        return embodiments.parse(config)
    except (embodiments.EmbodimentInvalidError, TypeError) as error:
        # The parser reads JSON, whose keys are strings; a Python dict keyed by anything else fails its name match.
        why = error if isinstance(error, embodiments.EmbodimentInvalidError) else "part names and roles must be str"
        raise DreamConfigError(
            "config key 'embodiment' names state, action and video with values that are not shape entries, so it "
            f"is read as an embodiment config, and that config is not valid: {why}"
        ) from None


def _planned(embodiment: Any) -> embodiments.Embodiment | None:
    """The body an `embodiment` block names when it is a config with `parts`, whose fine-tune reads its plan; else
    None.

    @param embodiment The value of `config["embodiment"]`.
    @return The parsed body, with its plan, or None.
    @throws DreamConfigError The config is not valid (#_parse).
    """
    if not _is_embodiment_config(embodiment) or "parts" not in embodiment:
        return None
    return _parse(embodiment)


def _to_model(body: embodiments.Embodiment, observation: dict[str, Any]) -> dict[str, Any]:
    """The observation as a fine-tune of `body` reads it: each of the body's state parts replaced by the slots its
    plan trains (`shared.datasets.conversion.to_trained`), a part it records but does not train left out; every other
    key as given.

    @param body The parsed body, with its plan.
    @param observation The caller's observation, keyed by the body's own parts.
    @return The observation to send.
    @throws DreamObservationError A part the plan reads is missing or not as many numbers as its span, or a rotation
        part's value is not a rotation.
    """
    try:
        slots = conversion.to_trained(body, "state", _state_parts(body, observation))
    except conversion.NotARotationError as error:
        raise DreamObservationError(f"observation state {error}") from None
    own = {f"state.{part}" for part in body.state}
    sent = {key: value for key, value in observation.items() if key not in own}
    sent.update({f"state.{name}": values[0].tolist() for name, values in slots.items()})
    return sent


def _state_parts(body: embodiments.Embodiment, observation: dict[str, Any]) -> dict[str, npt.NDArray[np.float64]]:
    """Each state part the body's plan reads, from the observation in the body's own parts.

    @param body The parsed body, with its plan.
    @param observation The caller's observation, keyed by the body's own parts.
    @return Each of those parts to its one row, `(1, part width)`.
    @throws DreamObservationError A part the plan reads is missing or not as many numbers as its span.
    """
    assert body.plan is not None
    parts: dict[str, npt.NDArray[np.float64]] = {}
    for slot in body.plan.state:
        start, end = body.state[slot.part]
        key = f"state.{slot.part}"
        try:
            parts[slot.part] = np.asarray(observation[key], dtype=np.float64).reshape(1, end - start)
        except (KeyError, TypeError, ValueError):
            raise DreamObservationError(
                f"observation key '{key}' must be {end - start} numbers: the model reads it as {slot.name}"
            ) from None
    return parts


def _to_body(body: embodiments.Embodiment, observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    """The reply with its `per_key` in `body`'s own action parts: each slot the plan trains read back as the part
    it came from (`shared.datasets.conversion.from_trained`), rows of the part's width, and a part whose values are
    deltas as the change each row makes (`conversion.from_targets`: the first row from the observed state, each later
    row from the row before); every other entry and key as given.

    @param body The parsed body, with its plan.
    @param observation The observation the reply answers, keyed by the body's own parts (_to_model() checked it).
    @param action The model's action dict.
    @return A shallow copy of `action` with `per_key` converted.
    @throws DreamActionError `per_key` is not a dict, lacks a slot the plan trains or holds one that is not rows of its
        width, or a 6D answer has a vector of length zero.
    """
    assert body.plan is not None
    per_key = action.get(_PER_KEY)
    if not isinstance(per_key, dict):
        raise DreamActionError(f"action has no '{_PER_KEY}' dict, where the serve returns the parts the model trains")
    slots: dict[str, npt.NDArray[np.float64]] = {}
    for slot in body.plan.action:
        key = f"action.{slot.name}"
        try:
            rows = np.asarray(per_key[key], dtype=np.float64)
            slots[slot.name] = rows.reshape(len(rows), slot.width)
        except (KeyError, TypeError, ValueError):
            raise DreamActionError(f"action '{_PER_KEY}' entry '{key}' is not rows of {slot.width} numbers") from None
    try:
        parts = conversion.from_targets(
            body, _state_parts(body, observation), conversion.from_trained(body, "action", slots)
        )
    except conversion.NotARotationError as error:
        raise DreamActionError(f"action '{_PER_KEY}' {error}") from None
    converted = {key: value for key, value in per_key.items() if key.removeprefix("action.") not in slots}
    converted.update({f"action.{part}": rows.tolist() for part, rows in parts.items()})
    return {**action, _PER_KEY: converted}


def _validate_embodiment(embodiment: Any) -> None:
    """Rejects an `embodiment` block that is not `{group: {key: {"shape": [...], "dtype": ...}}}`.

    @param embodiment The value of `config["embodiment"]`.
    @throws DreamConfigError Any level is not a dict, a shape is not a list of ints, or a
        dtype is not one numpy understands.
    """
    if not isinstance(embodiment, dict):
        raise DreamConfigError(f"config key 'embodiment' must be a dict of groups, got {type(embodiment).__name__}")
    for group, entries in embodiment.items():
        if not isinstance(entries, dict):
            raise DreamConfigError(f"embodiment group '{group}' must be a dict of keys, got {type(entries).__name__}")
        for key, entry in entries.items():
            _validate_entry(f"{group}.{key}", entry)


def _validate_entry(dotted_key: str, entry: Any) -> None:
    """Rejects one embodiment entry whose `shape` or `dtype` the checks could not compare.

    An empty entry `{}` is allowed: the key must be present, nothing else is checked.

    @param dotted_key The `"group.key"` name used in the error message.
    @param entry The value under that key in the embodiment block.
    @throws DreamConfigError The entry is not a dict, its shape is not a list of ints, or
        its dtype is not one numpy understands.
    """
    if not isinstance(entry, dict):
        raise DreamConfigError(
            f"embodiment entry '{dotted_key}' must be a dict with optional 'shape' and 'dtype', "
            f"got {type(entry).__name__}"
        )
    if "shape" in entry and not _is_int_sequence(entry["shape"]):
        raise DreamConfigError(
            f"embodiment entry '{dotted_key}' has shape {entry['shape']!r}; expected a list of ints"
        )
    if "dtype" in entry and not _is_dtype(entry["dtype"]):
        raise DreamConfigError(
            f"embodiment entry '{dotted_key}' has dtype {entry['dtype']!r}, which numpy does not know"
        )


def _is_int_sequence(value: Any) -> bool:
    """True for a list or tuple whose items are all ints (bools excluded)."""
    if not isinstance(value, (list, tuple)):
        return False
    return all(isinstance(item, int) and not isinstance(item, bool) for item in value)


def _is_dtype(value: Any) -> bool:
    """True when `np.dtype(value)` resolves; `None` is excluded because numpy reads it as float64."""
    if value is None:
        return False
    try:
        np.dtype(value)
    except (TypeError, ValueError):
        return False
    return True


def _observation_spec(embodiment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Flattens every non-action group to `"group.key"` entries.

    @param embodiment The validated embodiment block.
    @return Map of dotted key to its `{"shape", "dtype"}` entry.
    """
    spec: dict[str, dict[str, Any]] = {}
    for group, entries in embodiment.items():
        if group == _ACTION_GROUP:
            continue
        for key, entry in entries.items():
            spec[f"{group}.{key}"] = entry
    return spec


def _action_spec(embodiment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Returns the action group's entries keyed by their bare names, as the serve returns them.

    @param embodiment The validated embodiment block.
    @return Map of action key (e.g. `q_r_chunk`) to its `{"shape", "dtype"}` entry; empty when there is no group.
    """
    entries: dict[str, Any] = embodiment.get(_ACTION_GROUP, {})
    return dict(entries)


def _commanded(
    commanded: Mapping[str, Any] | None, arm_joints: Mapping[str, int | None]
) -> dict[str, npt.NDArray[np.float64]] | None:
    """get_action()'s `commanded` as flat float arrays per chunk key, or None.

    @param commanded Per chunk key, the joints last commanded, or None.
    @param arm_joints Per chunk key, the joints of the arm it drives: its spec's with `ik`
        (`client_ik.ClientIK.joints`), else its measured pose's (`smoothing.SeamSmoother.arm_joints`), None when the
        observation carries none.
    @return The same keys, each value a 1-D float64 array; None when `commanded` is None.
    @throws DreamObservationError A key is not a chunk key, or a value is not a flat run of finite numbers as long as
        the arm's joints or its written row (the joints, then the gripper); any length when the arm's is None.
    """
    if commanded is None:
        return None
    out: dict[str, npt.NDArray[np.float64]] = {}
    for key, value in commanded.items():
        if key not in arm_joints:
            raise DreamObservationError(f"commanded key {key!r} is not a chunk key; it takes {sorted(arm_joints)}")
        try:
            joints = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise DreamObservationError(f"commanded '{key}' is not numbers: {error}") from None
        count = arm_joints[key]
        if joints.ndim != 1 or not np.isfinite(joints).all():
            raise DreamObservationError(f"commanded '{key}' must be a flat run of finite numbers; got {joints}")
        if count is not None and joints.shape[0] not in (count, count + 1):
            raise DreamObservationError(
                f"commanded '{key}' must be the arm's {count} joint angles, or its written row of {count + 1} "
                f"(the gripper last), all finite; got shape {joints.shape}"
            )
        out[key] = joints
    return out


def _check(
    payload: Any,
    spec: dict[str, dict[str, Any]],
    error_type: type[DreamObservationError] | type[DreamActionError],
    allow_extra: bool = False,
    frame_window_group: str = _FRAME_WINDOW_GROUP,
) -> None:
    """Checks keys and shapes (and dtypes of arrays) of a payload against a flattened spec; O(keys).

    Only `array.shape` and `array.dtype` are read, so the cost is independent of the data.
    A list value is shaped with `np.asarray` and its dtype is not checked: JSON carries no dtype.
    Observation keys that carry no group (no `.`, such as `prompt`) are metadata the serve
    reads and are not checked. Empty spec entries only require the key to be present.

    @param payload The observation or action dict.
    @param spec Flattened spec: a #_Checks observation or action map (_observation_spec(), _action_spec(), or an
        embodiment config's keys).
    @param error_type Which validation error to raise; also picks the noun in messages.
    @param allow_extra Whether keys outside the spec are tolerated (the serve adds fields to actions).
    @param frame_window_group The group whose keys must each be a frame window: `image_window` for a spec, `video`
        for an embodiment config.
    @throws DreamObservationError `error_type` is that type and a mismatch is found, in the order:
        not a dict, missing keys, unexpected keys, shape, dtype, and for the frame-window group's keys a value
        that is not a frame window (a non-empty list of strings).
    @throws DreamActionError Same, when `error_type` is DreamActionError.
    """
    noun = "observation" if error_type is DreamObservationError else "action"
    if not isinstance(payload, dict):
        raise error_type(f"{noun} must be a dict, got {type(payload).__name__}")
    missing = sorted(spec.keys() - payload.keys())
    if missing:
        raise error_type(f"{noun} is missing keys declared by the configured embodiment: {missing}")
    if not allow_extra:
        unexpected = sorted(str(key) for key in payload.keys() - spec.keys() if isinstance(key, str) and "." in key)
        if unexpected:
            raise error_type(f"{noun} has keys not declared by the configured embodiment: {unexpected}")
    for key, entry in spec.items():
        value = payload[key]
        if "shape" in entry:
            try:
                shape = value.shape if isinstance(value, np.ndarray) else np.shape(value)
            except ValueError as error:  # numpy refuses to shape a ragged list
                raise error_type(f"{noun} key '{key}' is not an array of one shape: {error}") from None
            if tuple(entry["shape"]) != tuple(shape):
                raise error_type(
                    f"{noun} key '{key}' has shape {tuple(shape)}, but the configured embodiment "
                    f"declares {tuple(entry['shape'])}"
                )
        if "dtype" in entry and isinstance(value, np.ndarray) and np.dtype(entry["dtype"]) != value.dtype:
            raise error_type(
                f"{noun} key '{key}' has dtype {value.dtype}, but the configured embodiment "
                f"declares {np.dtype(entry['dtype'])}"
            )
        if key.startswith(f"{frame_window_group}.") and not _is_frame_window(value):
            raise error_type(
                f"{noun} key '{key}' must be a non-empty list of base64 JPEG strings (a frame window, oldest "
                f"first), got {type(value).__name__}"
            )


def _check_per_key(action: dict[str, Any], widths: dict[str, int]) -> None:
    """Checks the reply's `per_key` against the action parts of an embodiment config; O(parts).

    The serve returns every `action.<part>` it computed under `per_key`, as rows of the part's width, one row per
    step of the horizon. It drops the size-1 axis of a width-1 part (a gripper), so `[H]` is accepted there as
    `[H, 1]`. Every part carries the same number of rows, at least one. Entries beyond the declared parts are
    tolerated, as extra action keys are.

    @param action The reply's action dict, already checked to be a dict.
    @param widths `action.<part>` to the part's width; empty for a spec, which declares no parts and checks nothing.
    @throws DreamActionError `per_key` is not a dict, lacks a declared part, or a part is not rows of its width,
        or the parts disagree on the number of rows.
    """
    if not widths:
        return
    per_key = action.get(_PER_KEY)
    if not isinstance(per_key, dict):
        raise DreamActionError(
            f"action has no '{_PER_KEY}' dict, where the serve returns the parts the configured embodiment "
            f"declares: {sorted(widths)}"
        )
    missing = sorted(widths.keys() - per_key.keys())
    if missing:
        raise DreamActionError(f"action '{_PER_KEY}' lacks parts the configured embodiment declares: {missing}")
    horizon: int | None = None
    for key, width in widths.items():
        try:
            shape = tuple(np.shape(per_key[key]))
        except ValueError as error:  # numpy refuses to shape a ragged list
            raise DreamActionError(f"action '{_PER_KEY}' entry '{key}' is not rows of one width: {error}") from error
        rows = shape if len(shape) != 1 or width != 1 else (shape[0], 1)
        if len(rows) != 2 or rows[1] != width or rows[0] == 0:
            raise DreamActionError(
                f"action '{_PER_KEY}' entry '{key}' has shape {shape}, but the configured embodiment declares "
                f"(H, {width}): one row of the part's {width} values per step"
            )
        if horizon is not None and rows[0] != horizon:
            raise DreamActionError(
                f"action '{_PER_KEY}' entry '{key}' has {rows[0]} rows where the other parts have {horizon}"
            )
        horizon = rows[0]


def _is_frame_window(value: Any) -> bool:
    """Whether `value` is what the wire calls a frame window: a non-empty list or tuple of strings."""
    return isinstance(value, (list, tuple)) and len(value) > 0 and all(isinstance(item, str) for item in value)

"""Camera capture on a dedicated thread, publishing into a latest-frame slot.

This is the only module in `robots` that knows cv2 and pyrealsense2. Each `CameraReader` owns
one capture thread that reads, converts BGR to RGB, rotates, and rebinds `frame` to a new
immutable #robots.types.Frame. The control loop reads that attribute and nothing else: no
queue, no lock, no wait. Staleness is judged by the loop through
`is_stale()`; a dead thread is visible through `is_alive()` and `error`.

The policy sees a window of frames, one every `obs_period_s` of capture time (`policy_obs_fps`);
the dream client keeps that window (`dream.DreamClient.set_observation()`), and the loop hands it one encoded
frame per camera. Encoding is the costly part (the face blur, the downscale to the policy size and the JPEG, by
encode_jpeg()), so a reader encodes only the captures the window can use: the first capture at or after each
multiple of `obs_period_s` on the monotonic clock (#SampleClock). Every reader uses the same grid, so the frames
the cameras encode for one period are captured within a frame period of each other and the loop hands them to
the client together. Every published `Frame` carries the newest encoded frame as `jpeg`, so a tick costs no
encoding at all and the tick path never touches cv2.

Two capture sources share one shape, `read() -> (ok, frame_bgr)` and `release()`:
`cv2.VideoCapture` for a `device` in the config and #RealSenseSource for a `serial`.
`CameraReader.from_config()` picks one by which key is set; the reader itself does not know
which it holds.

@see CameraReader.from_config(), CameraReader.start(), CameraReader.stop(), encode_jpeg()
"""

import base64
import logging
import math
import os
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from robots import config, errors, types, user_class

## Seconds stop() waits for the capture thread before giving up on the join.
JOIN_TIMEOUT_S = 2.0

## JPEG quality of every frame on the wire: what a YAM station encodes with.
JPEG_QUALITY = 85

## Worker threads OpenCV's process-wide pool is capped at, unless `OPENCV_FOR_THREADS_NUM` is in the environment
## (OpenCV reads that itself and the cap then defers to it). OpenCV sizes the pool to every hardware thread by
## default and its idle workers spin-wait after each parallel call: measured on a YAM station PC (Ryzen 7 7730U, three
## D405 colour streams at 1280x720) the default 16-thread pool cost 85 percent of a core per camera for a bare
## capture loop and three cores with the face blur on, against 8 and 41 percent with the pool at 2, at the same
## 30 fps. Two threads keep a 720p blur at 33 ms, so nothing is lost and the cores go to the control loop.
OPENCV_THREADS = 2

## Frame periods RealSenseSource.read() waits for a frame before reporting a failure: long enough
## that USB jitter does not kill a healthy camera, short enough that the loop sees a dead one.
REALSENSE_READ_TIMEOUT_PERIODS = 5

## Milliseconds RealSenseSource waits for the first frame after starting the pipeline; the
## first frame takes far longer than a frame period (USB negotiation, auto-exposure settling).
REALSENSE_START_TIMEOUT_MS = 5000

## What to do about a RealSense on a USB2 link: the 1280x720 colour stream at 30 fps needs USB3,
## and librealsense refuses the mode on USB2 with "Couldn't resolve requests", which never says USB.
USB3_FIX = "plug it into a USB3 port (blue inside, or marked SS) with a USB3 cable"

## How long a source may keep returning no frame before the capture thread stops. One missed read is a hiccup (a
## USB stall, a driver that timed out once: all three RealSense readers of a station can die together on a single
## 166 ms timeout while the station sits idle, and the live view then freezes until the next start); a
## stretch of them is a dead camera, which is_alive() then reports and the station reopens.
CAPTURE_MISS_TOLERANCE_S = 2.0
## The pause after a read that returned no frame at once (a cv2 device; a RealSense read blocks for its own
## timeout), so the retries do not spin.
CAPTURE_MISS_PAUSE_S = 0.01

## Seconds over which a reader measures the rate its camera delivers (`CameraReader.capture_fps`).
CAPTURE_RATE_WINDOW_S = 5.0
## A camera delivering under this share of its configured rate is logged: the frames it does not deliver are
## missing from the policy's frames and the recording alike, and nothing downstream can tell them from drops.
CAPTURE_RATE_WARN_FRACTION = 0.9

## Clockwise rotation in degrees -> quarter turns for np.rot90 (which turns counter-clockwise).
_ROTATION_TURNS = {0: 0, 90: -1, 180: 2, 270: 1}

## Managed exposure (`config.ExposureConfig`): slow and deadbanded on purpose, it
## corrects the light drifting across an afternoon, not a hand waved past a lens. Seconds between two steps.
MANAGED_PERIOD_S = 0.5
## Grey levels the median may sit from the target before a step.
MANAGED_DEADBAND = 3.0
## Shortest exposure the controller commands, microseconds.
MANAGED_MIN_US = 100.0
## A median at or above this is saturated: it cannot say how far over, so back off by feel.
MANAGED_CLIPPED = 254.0
## The factor applied to the moving knob when the frame is saturated or its highlights clip.
MANAGED_CLIPPED_RATIO = 0.7
## The share of the frame period the exposure may take. A sensor needs part of every frame to read the image out,
## so an exposure of the whole period (33.3 ms at 30 fps) makes it stretch each frame to two periods and deliver
## half its rate, silently: a 30 fps stream then arrives at 15 fps with no warning.
MANAGED_CEILING_FRACTION = 0.85
## A pixel whose brightest channel reaches this is clipped: it has lost its detail.
MANAGED_CLIP_LEVEL = 250
## More than this share of clipped pixels is too bright, whatever the median says: the median of a frame that is
## half dark backdrop is the backdrop, and brightening it to the target blows the table out (a wrist camera at the
## home pose can end up with 38 to 93 percent of the frame white for a whole run).
MANAGED_CLIP_MAX = 0.01
## Brightening waits until clipping is under this share, so a frame at the clip limit does not oscillate.
MANAGED_CLIP_RESUME = 0.005
## The most one step may scale a knob, either way.
MANAGED_MAX_STEP = 1.6
## With `exposure.smooth_idle`, the controller settles at full speed (the steps above) for at most this long after
## the camera opens, until a step first finds the frame on target; after that it only follows drift, slowly.
MANAGED_SETTLE_S = 10.0
## Once settled, grey levels the median may sit from the target before a correction starts. A correction, once
## started, runs to #MANAGED_DEADBAND, so a median near this edge does not start and stop on every step.
MANAGED_IDLE_DEADBAND = 8.0
## Once settled, how long every step must find the frame off target the same way before a correction starts: a
## hand or a person passing the lens is gone before then.
MANAGED_PERSIST_S = 3.0
## Once settled, the most one step may scale a knob, either way: about 2% brightness after the sensor's gamma, a
## ramp nobody sees, where a full-speed step (up to 1.6x, or 0.7x when clipped) is a visible jump every half second.
MANAGED_SLOW_STEP = 1.05
## A camera whose shape is within this share of the policy size's is not called stretched.
SHAPE_TOLERANCE = 0.02

_log = logging.getLogger(__name__)


class SampleClock:
    """Which captures are encoded for the policy: the first one in each slot of a grid every `period_s` long.

    The grid is the monotonic clock's own, slot `k` starting `early_s` before `k * period_s`, so every reader of a
    process cuts the same slots and the cameras' encoded frames of one slot are captured within a frame period of
    each other, around the slot's time. The dream client keeps one set of frames every `period_s` from the ones the
    loop hands it (`dream.DreamClient.set_observation()`); encoding more captures than that would be blurred and
    encoded for nothing.
    """

    def __init__(self, period_s: float, early_s: float) -> None:
        """@param period_s Seconds of capture time between two encoded captures; 0 encodes every capture.
        @param early_s How long before a slot's time a capture may come and still be its first, seconds; under
            `period_s`.
        """
        self.period_s = period_s
        self._early_s = early_s
        self._slot: int | None = None

    def reset(self) -> None:
        """Forgets the last slot taken: the next capture is encoded."""
        self._slot = None

    def take(self, capture_s: float) -> bool:
        """Whether a capture is encoded: the first one of its slot, or every capture when `period_s` is 0.

        @param capture_s The capture's time, seconds on the monotonic clock.
        @return True when it is the first capture of a slot not taken yet.
        """
        if self.period_s == 0:
            return True
        slot = math.floor((capture_s + self._early_s) / self.period_s)
        if self._slot is not None and slot <= self._slot:
            return False
        self._slot = slot
        return True


class CameraReader:
    """One camera, one capture thread, one latest-frame slot, one encode thread.

    `frame` is rebound to a fresh #robots.types.Frame on every capture; rebinding one
    reference is atomic under the GIL, so readers take the attribute as-is and never lock.
    `source` has the `cv2.VideoCapture` shape: `read() -> (ok, frame_bgr)` and `release()`.

    One capture every `obs_period_s` of capture time (#SampleClock), the first one included, has its faces blurred
    and is downscaled to `policy_image_size` and JPEG-encoded to a base64 string on the encode thread; each
    published frame carries the newest such string as `Frame.jpeg`, with its capture time.

    Attributes:
    - name: Camera name; the adapter sends this camera's frames under it.
    - frame: Latest published frame, or None before the first capture.
    - capture_fps: Frames per second the camera delivered over the last #CAPTURE_RATE_WINDOW_S, or None before the
      first window closes.
    - error: The exception that killed the capture thread, or None while it is healthy.
    - obs_period_s: Seconds of capture time between two encoded captures; 0 encodes every capture.
    - samples_dropped: Captures due for encoding that the encode thread never took, because a later one replaced
      them while it was busy.

    @see from_config(), start(), stop(), is_stale(), is_alive()
    """

    def __init__(
        self,
        name: str,
        source: Any,
        stale_after_s: float,
        rotate: int = 0,
        obs_period_s: float = 0.0,
        policy_image_size: tuple[int, int] | None = None,
        face_blur: bool = True,
        expected_fps: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Binds a capture source; the thread starts only in start().

        @param name Camera name, used for the frame's `camera` field and in messages.
        @param source Capture source with the `cv2.VideoCapture` shape.
        @param stale_after_s Age, in seconds, past which is_stale() reports the frame stale.
        @param rotate Clockwise rotation applied to every frame: 0, 90, 180 or 270.
        @param obs_period_s Seconds of capture time between two encoded captures.
        @param policy_image_size `(width, height)` every encoded capture is downscaled to before
            encoding, or None to encode at the capture size.
        @param face_blur Blur faces out of every capture before it is encoded for the policy and sent to the API
            (`shared.faceblur`). On by default; the detector is built on the encode thread, and a detector that
            cannot load stops the reader (is_alive() False) rather than let a face leave the box.
        @param expected_fps The rate the config asks of the camera; a camera delivering under
            #CAPTURE_RATE_WARN_FRACTION of it is logged at WARNING, once until it recovers. None logs nothing. A
            capture up to half its frame period before a slot's time is taken for it (#SampleClock).
        @param clock The monotonic clock the encoded captures are timed by, and `Frame.jpeg_timestamp_s` read from.
        """
        if rotate not in _ROTATION_TURNS:
            raise errors.CameraError(f"camera '{name}' rotate={rotate} is not one of {sorted(_ROTATION_TURNS)}")
        if stale_after_s <= 0:
            raise errors.CameraError(f"camera '{name}' stale_after_s={stale_after_s} must be positive")
        early_s = 0.0 if expected_fps is None else 0.5 / expected_fps
        if obs_period_s != 0 and not obs_period_s > early_s:
            raise errors.CameraError(
                f"camera '{name}' obs_period_s={obs_period_s} must be 0 (every capture) or longer than half its "
                f"frame period, {early_s:.4g} s"
            )
        self.name = name
        self.frame: types.Frame | None = None
        self.error: BaseException | None = None
        self.obs_period_s = obs_period_s
        self._sample_clock = SampleClock(obs_period_s, early_s)
        self._clock = clock
        self._source = source
        self._stale_after_s = stale_after_s
        self._turns = _ROTATION_TURNS[rotate]
        self._policy_image_size = policy_image_size
        self._face_blur = face_blur
        # A source with managed exposure (#RealSenseSource) tracks the scene from the frames it delivers and
        # holds still for a run; a cv2 device has neither and both become no-ops.
        self._track: Callable[[npt.NDArray[np.uint8]], None] | None = getattr(source, "track", None)
        self._hold: Callable[[bool], None] | None = getattr(source, "set_exposure_hold", None)
        ## The newest capture the encode thread finished, as `(base64 JPEG, capture time)`; None before the first.
        ## Rebound whole by the encode thread, read whole by the capture thread's publish.
        self._encoded: tuple[str, float] | None = None
        self._capture_count = 0
        self._missing_since_s: float | None = None
        self._thread: threading.Thread | None = None
        # The one-slot mailbox from the capture thread to the encode thread: the newest capture due for encoding,
        # with its capture time, waiting to be blurred and encoded. A sample that lands while the previous one is
        # still waiting replaces it and counts in samples_dropped; the encode thread never queues, so the encoded
        # frame is never stale by more than one sample.
        self._sample_ready = threading.Condition()
        self._sample: tuple[npt.NDArray[np.uint8], float] | None = None
        self._encode_thread: threading.Thread | None = None
        self._is_encoder_busy = False
        # Set once the encode thread has its detector, or has failed to get one: the capture thread publishes
        # nothing before, so a detector that will not load stops the reader with no frame ever published.
        self._encoder_ready = threading.Event()
        self.samples_dropped = 0
        self.capture_fps: float | None = None
        self._expected_fps = expected_fps
        self._rate_since_s: float | None = None
        self._rate_count = 0
        self._is_rate_low = False
        # A plain bool rather than an Event: one writer, one reader, and a bool flip is atomic
        # under the GIL, so an Event's lock would be structure without a need.
        self._is_stop_requested = False

    @classmethod
    def from_config(
        cls,
        cfg: config.CameraConfig,
        stale_after_s: float,
        *,
        obs_fps: float,
        policy_image_size: tuple[int, int],
        face_blur: bool = True,
        exposure: config.ExposureConfig | None = None,
    ) -> "CameraReader":
        """Opens the configured camera, your class, a RealSense or an OpenCV device, and wraps it.

        cv2 is imported here (and lazily in the capture thread) so the module itself loads
        without it; tests drive the class with a fake source.

        @param cfg Camera config: device, serial or source, size, fps and rotation.
        @param stale_after_s Age, in seconds, past which is_stale() reports the frame stale.
        @param obs_fps Rate captures are encoded for the policy at, frames per second: the rate the policy's window
            keeps them at, the training video fps.
        @param policy_image_size `(width, height)` every encoded capture is downscaled to before encoding.
        @param face_blur Blur faces out of every frame before it leaves the box; see __init__.
        @param exposure Managed exposure for a RealSense camera (`config.ExposureConfig`); None or `managed: false`
            leaves the firmware's auto-exposure on. Ignored for an OpenCV device.
        @return A #CameraReader over the opened camera; not yet started.
        @throws CameraError If the config names neither a device nor a serial, the device cannot
            be opened, or a serial is set and pyrealsense2 is not installed.
        @throws ConfigError Your camera class does not load (#robots.user_class.load); its own errors are its own.
        """
        obs_period_s = 0.0 if obs_fps >= cfg.fps else 1.0 / obs_fps
        _log.info(
            "camera '%s': one capture encoded for the policy every %.3g s of capture time (%g fps of %d), "
            "policy size %dx%d",
            cfg.name,
            obs_period_s,
            min(obs_fps, cfg.fps),
            cfg.fps,
            *policy_image_size,
        )
        if cfg.source is not None:
            source: Any = user_class.load(cfg.source, "camera")(cfg)
        elif cfg.serial is not None:
            source = RealSenseSource(cfg.serial, cfg.width, cfg.height, cfg.fps, exposure=exposure)
        elif cfg.device is None:
            # config.load() never builds such a config; a hand-built one fails here by name.
            raise errors.CameraError(f"camera '{cfg.name}' has no device, serial or source to open")
        else:
            import cv2

            source = cv2.VideoCapture(cfg.device)
            if not source.isOpened():
                raise errors.CameraError(f"camera '{cfg.name}' could not open device {cfg.device!r}")
            # Requests, not guarantees: a driver silently keeps its own mode when it cannot honour
            # one. The wire carries no frame size, so the policy size below is what matters. MJPG
            # first: V4L2's default is uncompressed YUYV, which a USB 2 webcam cannot carry at 720p30
            # and answers by quietly dropping to 10 fps or less.
            source.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
            source.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            source.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
            source.set(cv2.CAP_PROP_FPS, cfg.fps)
        return cls(
            cfg.name,
            source,
            stale_after_s,
            rotate=cfg.rotate,
            obs_period_s=obs_period_s,
            policy_image_size=policy_image_size,
            face_blur=face_blur,
            expected_fps=cfg.fps,
        )

    def start(self) -> None:
        """Spawns the daemon capture and encode threads; nothing encoded before a stop is published again.

        @throws CameraError If a capture thread is already running.
        """
        if self._thread is not None and self._thread.is_alive():
            raise errors.CameraError(f"camera '{self.name}' capture thread is already running")
        self.error = None
        self._is_stop_requested = False
        # A frame encoded before a stop is a picture from before the gap; the first capture of the new thread is
        # encoded and published with the first frame.
        self._encoded = None
        self._sample_clock.reset()
        self._capture_count = 0
        self.samples_dropped = 0
        self.capture_fps = None
        self._rate_since_s = None
        self._encoder_ready.clear()
        with self._sample_ready:
            self._sample = None
        # Daemon so a crashed control loop cannot be kept alive by a blocked capture call.
        self._encode_thread = threading.Thread(target=self._encode_run, name=f"camera-{self.name}-encode", daemon=True)
        self._encode_thread.start()
        self._thread = threading.Thread(target=self._run, name=f"camera-{self.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Asks the thread to exit, waits up to #JOIN_TIMEOUT_S, and releases the source.

        Idempotent: a reader that never started, or already stopped, just releases again.
        """
        self._is_stop_requested = True
        with self._sample_ready:
            self._sample_ready.notify_all()
        if self._thread is not None:
            self._thread.join(JOIN_TIMEOUT_S)
            if self._thread.is_alive():
                _log.warning("camera '%s' capture thread did not exit within %.1f s", self.name, JOIN_TIMEOUT_S)
        if self._encode_thread is not None:
            self._encode_thread.join(JOIN_TIMEOUT_S)
            if self._encode_thread.is_alive():
                _log.warning("camera '%s' encode thread did not exit within %.1f s", self.name, JOIN_TIMEOUT_S)
        self._source.release()

    @property
    def encoded(self) -> tuple[str, float] | None:
        """The newest capture encoded for the policy, `(base64 JPEG, capture time)`, or None before the first.

        It is what the next published frame carries as `jpeg` and `jpeg_timestamp_s`.
        """
        return self._encoded

    def set_exposure_hold(self, hold: bool) -> None:
        """Freezes (or releases) the source's managed exposure for the length of a run; a no-op on a cv2 device.

        @param hold True while a run is on, so no brightness change is written into the middle of an episode.
        """
        if self._hold is not None:
            self._hold(hold)

    @property
    def exposure(self) -> dict[str, Any] | None:
        """The managed-exposure state of the source, or None when it has none.

        The state is `exposure_us`, `gain`, `held` and `target_median`; a cv2 device, or a RealSense with
        `managed: false`, has no controller and answers None.
        """
        return getattr(self._source, "exposure", None)

    @property
    def is_encoder_idle(self) -> bool:
        """True when no capture is waiting for, or being worked on by, the encode thread."""
        with self._sample_ready:
            return self._sample is None and not self._is_encoder_busy

    def is_stale(self, now_s: float) -> bool:
        """Reports whether the latest frame is missing or older than `stale_after_s`.

        @param now_s Monotonic time to compare against, seconds; the caller's tick clock.
        @return True when there is no frame yet or it is too old to feed the policy.
        """
        # One read of the slot: the thread may rebind it between two reads.
        frame = self.frame
        if frame is None:
            return True
        return now_s - frame.timestamp_s > self._stale_after_s

    def is_alive(self) -> bool:
        """Reports whether the capture thread is running and has not failed.

        @return True while the thread is alive and `error` is None.
        """
        return self._thread is not None and self._thread.is_alive() and self.error is None

    def _run(self) -> None:
        """Capture loop body; stores any exception in `error` and exits."""
        try:
            # Imported on the thread, not at module load, so the module (and its tests) never
            # need cv2 until a frame is captured.
            import cv2

            limit_opencv_threads(cv2)
            self._encoder_ready.wait()
            while not self._is_stop_requested and self.error is None:
                self._capture_one(cv2)
        except BaseException as exc:
            # The thread boundary: an exception that escaped here would print a traceback and
            # vanish, and the loop would keep reading an ever-older frame. Storing it is what
            # makes is_alive() false and lets the loop stop.
            self.error = exc
            _log.error("camera '%s' capture thread stopped: %r", self.name, exc)

    def _encode_run(self) -> None:
        """Encode thread body: blurs, downscales and encodes every capture due for the policy, keeping the newest.

        Off the capture thread on purpose: the face detector and the JPEG take tens of milliseconds, and paid on
        the capture thread they stalled it past several frame periods, so the driver's queue drained in bursts
        and a consumer polling the frame saw a fraction of the camera's rate. Here the capture thread only
        reads, converts and publishes, at the camera's cadence, whatever the encode costs. The face scrubber is
        built here: it pulls in cv2, and a detector that cannot load must stop the reader (a face would
        otherwise leave the box), which storing the error does.
        """
        try:
            import cv2

            limit_opencv_threads(cv2)
            try:
                blur = self._make_blurrer() if self._face_blur else None
            finally:
                self._encoder_ready.set()
            while True:
                with self._sample_ready:
                    while self._sample is None and not self._is_stop_requested:
                        self._sample_ready.wait()
                    if self._is_stop_requested:
                        return
                    sample, self._sample = self._sample, None
                    assert sample is not None, "the wait above ends only with a sample or a stop"
                    self._is_encoder_busy = True
                try:
                    bgr, capture_s = sample
                    self._encoded = (self._encode_sample(cv2, blur, bgr), capture_s)
                finally:
                    with self._sample_ready:
                        self._is_encoder_busy = False
        except BaseException as exc:
            self.error = exc
            self._encoder_ready.set()
            _log.error("camera '%s' encode thread stopped: %r", self.name, exc)

    def _encode_sample(self, cv2: Any, blur: Any, bgr: npt.NDArray[np.uint8]) -> str:
        """One capture as the policy sees it: faces scrubbed, RGB, rotated, downscaled, JPEG, base64.

        Blurring runs on the native BGR frame, before the RGB conversion, because that is the channel order the
        detector expects; `scrub_frame` returns the frame untouched when it finds no face, so a faceless scene
        costs only the detection.

        @param cv2 The OpenCV module, imported by _encode_run() on its thread.
        @param blur The face scrubber, or None when face blurring is off.
        @param bgr The capture, this thread's own copy.
        @return The base64 JPEG the policy sees.
        """
        if blur is not None:
            bgr = blur.scrub_frame(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self._turns:
            rgb = np.ascontiguousarray(np.rot90(rgb, self._turns))
        jpeg = encode_jpeg(np.asarray(rgb, dtype=np.uint8), self._policy_image_size)
        return base64.b64encode(jpeg).decode("ascii")

    def _offer_sample(self, bgr: npt.NDArray[np.uint8], capture_s: float) -> None:
        """Hands a capture due for encoding, with its capture time, to the encode thread; a capture still waiting is
        replaced and counted."""
        with self._sample_ready:
            if self._sample is not None:
                self.samples_dropped += 1
                if self.samples_dropped == 1:
                    _log.warning(
                        "camera '%s': the encode thread fell a capture behind; the policy misses one (counted)",
                        self.name,
                    )
            self._sample = (bgr, capture_s)
            self._sample_ready.notify()

    def _make_blurrer(self) -> Any:
        """Builds the face scrubber for this camera; raises if the detector cannot load (fail closed).

        Imported here, not at module top, so the module still loads without opencv for tests.

        @return A `shared.faceblur.FaceBlurrer`.
        @throws shared.faceblur.FaceBlurError When the detector will not load or fails its self-check.
        """
        from shared import faceblur

        return faceblur.FaceBlurrer()

    def _measure_rate(self, now_s: float) -> None:
        """Counts one capture and, every #CAPTURE_RATE_WINDOW_S, updates `capture_fps` and says when it is low.

        A camera can deliver half its configured rate without an error anywhere (a managed exposure at the frame
        period, a webcam in dim light, a USB mode the driver quietly chose); this is the one place that notices.

        @param now_s Monotonic time of the capture.
        """
        if self._rate_since_s is None:
            self._rate_since_s, self._rate_count = now_s, 0
            return
        self._rate_count += 1
        elapsed_s = now_s - self._rate_since_s
        if elapsed_s < CAPTURE_RATE_WINDOW_S:
            return
        self.capture_fps = round(self._rate_count / elapsed_s, 2)
        self._rate_since_s, self._rate_count = now_s, 0
        if self._expected_fps is None:
            return
        is_low = self.capture_fps < CAPTURE_RATE_WARN_FRACTION * self._expected_fps
        if is_low and not self._is_rate_low:
            exposure = getattr(self._source, "exposure", None)
            _log.warning(
                "camera '%s' delivers %.1f fps of the %g configured%s",
                self.name,
                self.capture_fps,
                self._expected_fps,
                f"; managed exposure at {exposure['exposure_us']:.0f} us" if exposure else "",
            )
        elif self._is_rate_low and not is_low:
            _log.info("camera '%s' is back at %.1f fps", self.name, self.capture_fps)
        self._is_rate_low = is_low

    def _capture_one(self, cv2: Any) -> None:
        """Reads one frame, hands it to the encode thread when it is due, converts and rotates it, and publishes it.

        @param cv2 The OpenCV module, imported by _run() on the capture thread and passed in so this
            method never imports it itself.
        @throws CameraError The source returned no frame for #CAPTURE_MISS_TOLERANCE_S, or a frame that is not
            `[H, W, 3]`.
        """
        ok, bgr = self._source.read()
        if not ok:
            now_s = time.monotonic()
            if self._missing_since_s is None:
                self._missing_since_s = now_s
                _log.warning(
                    "camera '%s' source returned no frame; retrying for %.1f s", self.name, CAPTURE_MISS_TOLERANCE_S
                )
            if now_s - self._missing_since_s >= CAPTURE_MISS_TOLERANCE_S:
                raise errors.CameraError(
                    f"camera '{self.name}' source returned no frame for {CAPTURE_MISS_TOLERANCE_S:g} s"
                )
            time.sleep(CAPTURE_MISS_PAUSE_S)
            return
        self._missing_since_s = None
        if bgr is None or np.ndim(bgr) != 3 or np.shape(bgr)[2] != 3:
            raise errors.CameraError(
                f"camera '{self.name}' source returned a frame of shape {np.shape(bgr)}, expected [H, W, 3]"
            )
        if self._track is not None:
            self._track(bgr)
        capture_s = self._clock()
        if self._sample_clock.take(capture_s):
            # The encode thread's own copy: a source may reuse its buffer for the next read.
            self._offer_sample(np.array(bgr, dtype=np.uint8, copy=True), capture_s)
            if self._capture_count == 0:
                # A published frame always carries an encoded frame (the loop refuses one without), so the very
                # first publish waits for the first encode to land; every later one lands on its own.
                while self._encoded is None and self.error is None and not self._is_stop_requested:
                    time.sleep(0.001)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self._turns:
            # rot90 returns a strided view; the frame is serialized downstream, so make it
            # contiguous here, once, instead of at every consumer.
            rgb = np.ascontiguousarray(np.rot90(rgb, self._turns))
        rgb = np.asarray(rgb, dtype=np.uint8)
        self._capture_count += 1
        if self._capture_count == 1 and self._policy_image_size is not None:
            warning = shape_warning(rgb.shape[1], rgb.shape[0], self._policy_image_size)
            if warning is not None:
                _log.warning("camera '%s': %s", self.name, warning)
        self._measure_rate(time.monotonic())
        # Rebinding the attribute is the publish; nothing else is touched. The encoded frame is the newest the encode
        # thread has finished: a capture still being encoded lands in a later publish.
        jpeg, jpeg_s = self._encoded or ("", 0.0)
        self.frame = types.Frame(
            data=rgb, camera=self.name, timestamp_s=time.monotonic(), jpeg=jpeg, jpeg_timestamp_s=jpeg_s
        )


def usb_link(serial: str) -> str | None:
    """The USB link a RealSense camera enumerated at, as librealsense names it (`"3.2"`, `"2.1"`).

    A D405 on a USB2 port or cable comes up as `2.x` and cannot carry 1280x720 colour at 30 fps (#USB3_FIX).

    @param serial The camera's serial number.
    @return The link, or None when pyrealsense2 is missing, the camera is not listed, or it does not say.
    """
    try:
        import pyrealsense2 as rs  # type: ignore[import-not-found,unused-ignore]
    except ImportError:
        return None
    try:
        for device in rs.context().query_devices():
            if device.get_info(rs.camera_info.serial_number) != serial:
                continue
            if not device.supports(rs.camera_info.usb_type_descriptor):
                return None
            return str(device.get_info(rs.camera_info.usb_type_descriptor))
    except Exception as exc:  # A query that fails says nothing about the link; the open reports its own error.
        _log.warning("RealSense camera %s: could not read its USB link: %s", serial, exc)
    return None


class RealSenseSource:
    """The colour stream of one Intel RealSense camera, in the `cv2.VideoCapture` read/release shape.

    A D405 cannot be opened through its V4L2 nodes by `cv2.VideoCapture`,
    so it is read through librealsense instead. The D405's colour stream is what the policy
    was trained on, and `rs.format.bgr8` is requested so librealsense hands over the same BGR
    bytes cv2 would have: nothing is converted on the station, and #CameraReader treats this
    source exactly like a cv2 device.

    Attributes:
    - serial: The camera's serial number; the identity that survives a reboot and a USB re-enumeration.

    @see CameraReader.from_config()
    """

    def __init__(
        self, serial: str, width: int, height: int, fps: int, exposure: config.ExposureConfig | None = None
    ) -> None:
        """Starts the colour stream of the camera with `serial` and waits for its first frame.

        pyrealsense2 is imported here so the module loads without it; tests inject a fake module.

        @param serial The camera's serial number, e.g. `"260322275394"`.
        @param width Frame width in pixels; a mode the camera must support.
        @param height Frame height in pixels; a mode the camera must support.
        @param fps Frame rate; also sets the read() timeout, #REALSENSE_READ_TIMEOUT_PERIODS periods.
        @param exposure Managed exposure (`config.ExposureConfig`): with `managed` the firmware's auto-exposure is
            switched off and track() steers exposure, then gain, to hold the frame median at `target_median`;
            None or `managed: false` switches the firmware's auto-exposure on instead.
        @throws CameraError If pyrealsense2 is not installed.
        @throws RuntimeError librealsense's own error when the camera is absent, the mode is
            unsupported, or no frame arrives within #REALSENSE_START_TIMEOUT_MS (the pipeline stopped first); a mode
            refused on a USB2 link (usb_link()) says so and names #USB3_FIX, with librealsense's words kept.
        """
        try:
            # pyrealsense2 ships as a prebuilt wheel only for some platforms, so it is not a
            # dependency and mypy cannot see it either.
            import pyrealsense2 as rs  # type: ignore[import-not-found,unused-ignore]
        except ImportError as exc:
            raise errors.CameraError(
                f"RealSense camera {serial} needs pyrealsense2, which is not installed; run pip install pyrealsense2"
            ) from exc
        self.serial = serial
        self._rs = rs
        self._fps = fps
        self._timeout_ms = int(REALSENSE_READ_TIMEOUT_PERIODS * 1000 / fps)
        self._pipeline = rs.pipeline()
        stream = rs.config()
        stream.enable_device(serial)
        stream.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        usb = usb_link(serial)
        try:
            profile = self._pipeline.start(stream)
        except RuntimeError as exc:
            if usb is None or not usb.startswith("2"):
                raise
            raise RuntimeError(
                f"{exc}: RealSense camera {serial} is on a USB {usb} link, too slow for {width}x{height} at "
                f"{fps} fps; {USB3_FIX}"
            ) from exc
        self._is_streaming = True
        _log.info("RealSense camera %s: USB %s, %dx%d at %d fps", serial, usb or "link unknown", width, height, fps)
        self._exposure = exposure if exposure is not None and exposure.managed else None
        self._exposure_us: float | None = None
        self._gain = self._gain_floor = self._gain_ceiling = 0.0
        self._is_held = False
        self._next_track_s = 0.0
        # Smooth-idle state (`exposure.smooth_idle`), reset by _configure_exposure(): whether the full-speed settle
        # is over, when it ends at the latest, whether a slow correction is running, and the off-target reading that
        # would start one, as `(brighter, since_s)`.
        self._is_settled = False
        self._settle_until_s = 0.0
        self._is_correcting = False
        self._drift: tuple[bool, float] | None = None
        self._color_sensor: Any = _find_color_sensor(rs, profile)
        if self._color_sensor is not None:
            self._configure_exposure(self._color_sensor)
        # Waited for once here, with the long timeout, so the first read() in the capture thread
        # keeps the short one and a camera that never streams fails at construction, as an
        # unopenable cv2 device does. The pipeline is stopped before the error propagates: a
        # started one keeps the camera claimed, and a retry in the same process could not open it.
        try:
            self._pipeline.wait_for_frames(REALSENSE_START_TIMEOUT_MS)
        except Exception:
            self.release()
            raise

    def read(self) -> tuple[bool, npt.NDArray[np.uint8] | None]:
        """Waits for the next colour frame.

        @return `(True, frame_bgr)` with a uint8 `[H, W, 3]` array, or `(False, None)` when no
            frame arrived within the timeout: the same failure a cv2 read reports, so
            #CameraReader handles both alike.
        """
        try:
            frames = self._pipeline.wait_for_frames(self._timeout_ms)
        except RuntimeError as exc:
            # librealsense reports a timeout by raising; the cv2 shape reports it as a bool. The
            # message is logged here because the reader's own CameraError cannot carry it.
            _log.warning("RealSense camera %s: %s", self.serial, exc)
            return False, None
        color = frames.get_color_frame()
        if not color:
            return False, None
        return True, np.asanyarray(color.get_data(), dtype=np.uint8)

    def release(self) -> None:
        """Stops the pipeline; idempotent, since `CameraReader.stop()` is.

        A camera that dropped off USB has had its pipeline stopped by librealsense already, and stop() raises
        "stop() cannot be called before start()"; that is logged, since the camera is being let go either way.
        """
        if not self._is_streaming:
            return
        self._is_streaming = False
        try:
            self._pipeline.stop()
        except RuntimeError as exc:
            _log.warning("RealSense camera %s: stopping its pipeline failed: %s", self.serial, exc)

    # Managed exposure (the package's own brightness controller)

    def _configure_exposure(self, sensor: Any) -> None:
        """Switches the firmware's auto-exposure off and seeds the controller, or switches it on when unmanaged.

        Exposure seeds at the configured value, else half the ceiling: a mid-scale start converges from either
        side. Gain seeds at the sensor's own default, because the firmware's auto loop may have left it anywhere,
        and a stale high gain would make every later exposure decision look brighter than the scene is.

        @param sensor The colour sensor.
        """
        rs = self._rs
        if self._exposure is None:
            if sensor.supports(rs.option.enable_auto_exposure):
                self._write_option(rs.option.enable_auto_exposure, 1.0)
            return
        if sensor.supports(rs.option.enable_auto_exposure):
            self._write_option(rs.option.enable_auto_exposure, 0.0)
        seed = self._exposure.seed_us if self._exposure.seed_us is not None else self._exposure_ceiling() / 2
        self._exposure_us = float(np.clip(seed, MANAGED_MIN_US, self._exposure_ceiling()))
        if sensor.supports(rs.option.exposure):
            self._write_option(rs.option.exposure, self._exposure_us)
        if sensor.supports(rs.option.gain):
            gain_range = sensor.get_option_range(rs.option.gain)
            self._gain_floor = float(gain_range.default)
            self._gain_ceiling = float(gain_range.max)
            self._gain = self._gain_floor
            self._write_option(rs.option.gain, self._gain)
        else:
            self._gain = self._gain_floor = self._gain_ceiling = 0.0
        self._next_track_s = 0.0
        self._is_settled = self._is_correcting = False
        self._settle_until_s = time.monotonic() + MANAGED_SETTLE_S
        self._drift = None
        _log.info(
            "RealSense camera %s: managed exposure, median target %g, seed %.0f us, ceiling %.0f us, gain %g%s",
            self.serial,
            self._exposure.target_median,
            self._exposure_us,
            self._exposure_ceiling(),
            self._gain,
            ", smooth between runs" if self._exposure.smooth_idle else "",
        )

    def _exposure_ceiling(self) -> float:
        """The longest exposure this camera may use, microseconds: #MANAGED_CEILING_FRACTION of the frame period
        (the whole period leaves the sensor no time to read the image out, and it halves its rate), never past the
        config's `limit_us` when it names one."""
        ceiling = MANAGED_CEILING_FRACTION * 1_000_000.0 / max(self._fps, 1)
        if self._exposure is not None and self._exposure.limit_us is not None:
            ceiling = min(ceiling, float(self._exposure.limit_us))
        return ceiling

    def set_exposure_hold(self, hold: bool) -> None:
        """Freezes (or releases) the controller: held for the length of a run, so exposure never moves mid-episode.

        @param hold True to freeze both knobs where they are.
        """
        self._is_held = hold

    @property
    def exposure(self) -> dict[str, Any] | None:
        """The controller's state, or None when the camera runs the firmware's auto-exposure."""
        if self._exposure is None or self._exposure_us is None:
            return None
        return {
            "exposure_us": round(self._exposure_us, 1),
            "gain": round(self._gain, 1),
            "held": self._is_held,
            "target_median": self._exposure.target_median,
        }

    def track(self, bgr: npt.NDArray[np.uint8]) -> None:
        """One controller step, at most every #MANAGED_PERIOD_S, steering exposure and then gain toward the target.

        Nothing happens while held, unmanaged, or without a colour sensor.

        Highlights come before the median: a frame with more than #MANAGED_CLIP_MAX of its pixels clipped is too
        bright whatever its median, and a frame clipping over #MANAGED_CLIP_RESUME is never brightened, so a wrist
        camera metered with a dark backdrop in view keeps the table's detail.

        The two knobs form a ladder. Brightening lengthens exposure until it is pinned at the ceiling and only
        then raises gain, so a dim room is not left stuck at the ceiling. Darkening unwinds in the other order,
        gain back to its floor before exposure shortens, because gain is amplified noise and exposure is real
        light. The median is read off a quarter-scale view: this runs on the capture thread, and the subsample
        moves the median by less than the deadband.

        With `exposure.smooth_idle` the full-speed steps are for settling only: from the camera opening until a
        step first finds the frame on target, or #MANAGED_SETTLE_S at the latest. After that the controller follows
        drift: a correction starts only when every step for #MANAGED_PERSIST_S has found the frame off target the
        same way (past #MANAGED_IDLE_DEADBAND, or clipping), moves each knob at most #MANAGED_SLOW_STEP per step,
        and runs until the frame is back inside #MANAGED_DEADBAND or wants the other way. Between runs the live view
        then shows no brightness jumps; the frame rate is untouched either way, since the ceiling is the same.

        @param bgr The frame just read.
        """
        if self._exposure is None or self._color_sensor is None or self._exposure_us is None or self._is_held:
            return
        now_s = time.monotonic()
        if now_s < self._next_track_s:
            return
        self._next_track_s = now_s + MANAGED_PERIOD_S
        view = bgr[::4, ::4]
        level = float(np.median(view))
        clipped = float(np.mean(view.max(axis=2) >= MANAGED_CLIP_LEVEL))
        if not self._exposure.smooth_idle:
            ratio = self._wanted_ratio(level, clipped, MANAGED_DEADBAND)
        else:
            ratio = self._smooth_ratio(level, clipped, now_s)
        if ratio is None:
            return
        want_brighter = ratio > 1.0
        if want_brighter and self._exposure_us < self._exposure_ceiling():
            self._command_exposure(self._exposure_us * ratio)
        elif want_brighter or self._gain > self._gain_floor:
            self._command_gain(self._gain * ratio)
        else:
            self._command_exposure(self._exposure_us * ratio)

    def _wanted_ratio(self, level: float, clipped: float, deadband: float) -> float | None:
        """The full-speed step for one frame: the factor to scale the moving knob by, or None to hold still.

        @param level The frame's median grey level.
        @param clipped The share of its pixels clipped.
        @param deadband Grey levels the median may sit from the target without a step.
        @return #MANAGED_CLIPPED_RATIO when the highlights clip, else the target over the level within
            #MANAGED_MAX_STEP; None inside the deadband, or when a frame clipping over #MANAGED_CLIP_RESUME would
            be brightened.
        """
        assert self._exposure is not None
        target = self._exposure.target_median
        if clipped > MANAGED_CLIP_MAX or level >= MANAGED_CLIPPED:
            # Highlights first: a scene half dark backdrop has a dark median, and chasing it blows the rest out.
            return MANAGED_CLIPPED_RATIO
        if abs(level - target) <= deadband:
            return None
        ratio = float(np.clip(target / max(level, 1.0), 1.0 / MANAGED_MAX_STEP, MANAGED_MAX_STEP))
        if ratio > 1.0 and clipped > MANAGED_CLIP_RESUME:
            return None
        return ratio

    def _smooth_ratio(self, level: float, clipped: float, now_s: float) -> float | None:
        """The `smooth_idle` step for one frame: full speed while settling, then a slow, persistent correction.

        @param level The frame's median grey level.
        @param clipped The share of its pixels clipped.
        @param now_s Monotonic time of this step.
        @return The factor to scale the moving knob by, or None to hold still.
        """
        if not self._is_settled and now_s < self._settle_until_s:
            ratio = self._wanted_ratio(level, clipped, MANAGED_DEADBAND)
            self._is_settled = ratio is None
            return ratio
        self._is_settled = True
        deadband = MANAGED_DEADBAND if self._is_correcting else MANAGED_IDLE_DEADBAND
        ratio = self._wanted_ratio(level, clipped, deadband)
        brighter = ratio is not None and ratio > 1.0
        if ratio is None or (self._drift is not None and self._drift[0] != brighter):
            # On target, or wanting the other way (an overshoot, or a passing shadow): stop, and start over.
            self._is_correcting = False
            self._drift = None if ratio is None else (brighter, now_s)
            return None
        if self._drift is None:
            self._drift = (brighter, now_s)
        if not self._is_correcting and now_s - self._drift[1] < MANAGED_PERSIST_S:
            return None
        self._is_correcting = True
        return float(np.clip(ratio, 1.0 / MANAGED_SLOW_STEP, MANAGED_SLOW_STEP))

    def _command_exposure(self, exposure_us: float) -> None:
        """Clips to the legal range and writes it; a value already at a bound is a no-op."""
        exposure_us = float(np.clip(exposure_us, MANAGED_MIN_US, self._exposure_ceiling()))
        if exposure_us == self._exposure_us:
            return
        self._exposure_us = exposure_us
        self._write_option(self._rs.option.exposure, exposure_us)

    def _command_gain(self, gain: float) -> None:
        """Clips to the sensor's range and writes it; a value already at a bound is a no-op."""
        gain = float(np.clip(gain, self._gain_floor, self._gain_ceiling))
        if gain == self._gain:
            return
        self._gain = gain
        self._write_option(self._rs.option.gain, gain)

    def _write_option(self, option: Any, value: float) -> None:
        """Writes one sensor option; a device mid-reset raises and is left to the next read to report."""
        sensor = self._color_sensor
        if sensor is None:
            return
        try:
            sensor.set_option(option, value)
        except Exception as exc:
            _log.warning("RealSense camera %s: could not set %s to %g: %s", self.serial, option, value, exc)


def _find_color_sensor(rs: Any, profile: Any) -> Any:
    """The sensor that owns the colour stream of a started pipeline, or None.

    The D405 has no dedicated RGB sensor (its colour stream comes from the stereo module), so
    `device.first_color_sensor()` raises there; the fallback scans every sensor's stream profiles for one that
    exposes the colour stream.

    @param rs The pyrealsense2 module.
    @param profile What `pipeline.start()` returned.
    @return The colour sensor, or None when the profile has no device or no sensor streams colour.
    """
    get_device = getattr(profile, "get_device", None)
    if get_device is None:
        return None
    device = get_device()
    try:
        return device.first_color_sensor()
    except RuntimeError:
        pass
    for sensor in device.query_sensors():
        for stream_profile in sensor.get_stream_profiles():
            if stream_profile.stream_type() == rs.stream.color:
                return sensor
    return None


def limit_opencv_threads(cv2: Any) -> None:
    """Caps OpenCV's worker pool at #OPENCV_THREADS for the whole process; a no-op under `OPENCV_FOR_THREADS_NUM`.

    Called on every thread that is about to use OpenCV (the capture threads, the live agent), because the first
    caller creates the pool and a later call is harmless; the cap is process-wide either way.

    @param cv2 The OpenCV module, imported by the caller so this module never imports it itself.
    """
    if os.environ.get("OPENCV_FOR_THREADS_NUM"):
        return
    set_threads = getattr(cv2, "setNumThreads", None)
    if set_threads is not None:
        set_threads(OPENCV_THREADS)


def encode_jpeg(rgb: npt.NDArray[np.uint8], image_size: tuple[int, int] | None = None) -> bytes:
    """Downscales an RGB frame to the policy size and encodes it as the JPEG bytes the serve wire carries.

    The resize uses `cv2.INTER_AREA`, the right filter for shrinking and what the YAM station stack
    uses; a frame already at `image_size` is not resampled. The serve decodes with cv2 and
    converts BGR to RGB, so the pixels are flipped back to BGR before encoding; encoding the RGB
    array directly would swap red and blue on the policy's side. Quality is #JPEG_QUALITY.

    @param rgb RGB uint8 `[H, W, 3]` pixels; never modified.
    @param image_size `(width, height)` to downscale to, or None to encode at the frame's own size.
    @return The JPEG bytes.
    @throws CameraError The encoder refused the frame.
    """
    import cv2

    if image_size is None or (rgb.shape[1], rgb.shape[0]) == image_size:
        scaled = rgb
    else:
        scaled = np.asarray(cv2.resize(rgb, image_size, interpolation=cv2.INTER_AREA), dtype=np.uint8)
    ok, buffer = cv2.imencode(
        ".jpg", cv2.cvtColor(scaled, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not ok:
        raise errors.CameraError(f"frame of shape {scaled.shape} could not be JPEG-encoded")
    return bytes(buffer.tobytes())


def shape_warning(width: int, height: int, image_size: tuple[int, int]) -> str | None:
    """In words, how the resize to the policy size stretches a camera of another shape.

    @param width The camera's frame width.
    @param height Its frame height.
    @param image_size `(width, height)` the frames are resized to.
    @return The line to show, or None when the shapes match within #SHAPE_TOLERANCE.
    """
    stretch = (image_size[0] / image_size[1]) / (width / height)
    if abs(stretch - 1) < SHAPE_TOLERANCE:
        return None
    seen = f"{stretch:.2f}x wider" if stretch > 1 else f"{1 / stretch:.2f}x taller"
    aspect = _aspect(*image_size)
    return (
        f"This camera sends {_shape(width, height)} pictures, which is fine for setup: what matters is that the model "
        f"gets {aspect} frames, cropped rather than squished. Running the robot squishes these to "
        f"{image_size[0]}x{image_size[1]}, so the model sees everything {seen}; a {aspect} camera size such as "
        f"{image_size[0] * 2}x{image_size[1] * 2} when recording and running keeps it unstretched, as does cropping "
        f"to {aspect} yourself if you send your own frames."
    )


def _aspect(width: int, height: int) -> str:
    gcd = math.gcd(width, height)
    return f"{width // gcd}:{height // gcd}"


def _shape(width: int, height: int) -> str:
    return f"{width}x{height} ({_aspect(width, height)})"

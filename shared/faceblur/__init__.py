"""Face blurring for every DreamLabs surface that reads camera data: scrub before pixels become durable or public.

Camera frames from a robot are pictures of a lab (or a customer's floor) where people work; whoever walks
through the shot must not be identifiable in anything Dream Labs stores, serves, or ships. This module is the one
implementation of that rule, in `shared/` because more than one product reads camera data and the scrub must
behave identically wherever it runs. The deal for a consumer: call `blurrer_for(source).scrub_jpeg(bytes)` (or
`scrub_frame(array)` on a decoded frame, no JPEG round-trip) at the point where camera pixels ENTER your
process, so every sink you have - and every sink you grow later - is downstream of it.

Consumers:

- **dream-robots** (`robots.cameras`, the DreamAPI client on the robot): blurs at capture, on each sampled
  frame, before it is JPEG-encoded into the observation window `dream` posts to DreamAPI. This is the one that
  keeps face data off the API entirely - a face never leaves the box, so nothing the API or a serve touches,
  and nothing a future recording stores, can contain one. `face_blur` in the station config; on by default.
- **the platform's public robots page** (its station camera feeds): every frame re-served to the public
  internet, scrubbed as the bytes arrive from the station.
- **the DreamAPI gateway does NOT call this**: it forwards the observation to the serve verbatim and keeps no
  frame, and blurring the policy's INPUT server-side would be wrong twice (tens of ms on a real-time action
  path; the model must see the scene it was trained on). Blur belongs on the robot (above) and on any recording
  tap, never on the live forward.
  The `dreamlabs-shared[faceblur]` extra keeps opencv out of the gateway image, which imports only contracts.

Detector: YuNet through OpenCV's `FaceDetectorYN` (`models/face_detection_yunet_2026may.onnx`, vendored, MIT;
`models/README.md` has provenance and the refresh recipe). A 230 KB CNN, no GPU: measured 6 ms per 480 px frame
and 18 ms at 960 px, 37 ms end to end for a 960 px frame with a face in it. A frame with no detection is
returned as the camera's original bytes - no re-encode, no cost.

The failure mode that matters is silent: blurring that quietly stops happening keeps serving faces while every
check stays green. So this module fails closed two ways:

1. A FaceBlurrer proves itself at construction - it must detect a synthetic face (`_self_check_frame()`) or
   raise. A model that loads is not a model that infers: the fixed-shape YuNet export silently degrades under
   OpenCV 5.x's ONNX engine (opencv_zoo#44), exactly the kind of break that would turn this into a pass-through.
2. Bytes that cannot be decoded cannot be cleared, so `scrub_jpeg()` raises rather than passing them along; the
   caller decides what "no frame" means on its surface (a frame route answers the 503 an unreachable station
   gets, and its stream ends).

FaceBlurError never carries pixels, so letting it propagate is always safe.

Each frame is scrubbed on its own detections alone - no box is carried between frames - so a moving face
leaves no trail of stale ovals. The cost is that a single-frame detector miss is not covered by a neighbour;
identifiable faces score above #_SCORE_THRESHOLD (>=0.83 measured vs the 0.75 bar), and a trail of blur
where a face used to be was the worse artifact to a viewer. A future consumer that needs miss-coverage should
track the face and blur its PREDICTED current position, not persist old ones. Detections below
#_MIN_FACE_SIDE_PX are deliberately left alone: a face that small identifies nobody, and that scale is where
the detector's tabletop false positives live (a blurred shoe on a public tank tile is its own kind of bug).

Face-LIKE objects are the other false positive: a toy steering wheel whose hub reads as two eyes, a nose and a
mouth scores up to 0.84 on a station's top camera, inside the band real faces reach at hard angles, so no single
score separates them. A detection is therefore blurred when it is a sure face (#_SURE_FACE_SCORE) or when it clears
#_SCORE_THRESHOLD and its box is mostly skin-coloured (#_MIN_SKIN_FRACTION): people have skin in their face,
props mostly do not. A doll with a realistic face passes both and stays blurred; that is accepted, not tuned away.
"""

import logging
import os
import threading
from typing import Any

import cv2
import numpy as np

log = logging.getLogger("platform_api")

## The vendored YuNet weights; `models/README.md` records provenance, checksum and licence.
_MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet_2026may.onnx")

## Detector confidence floor, tuned from measurement, not taste. On tabletop robot scenes, false positives -- shoes,
## trays, bench clutter, and skin-coloured clutter, hands and arms that pass the skin gate below -- score below 0.75
## at identifiable size, while real faces of identifiable size score >=0.83. 0.75 keeps every face of identifiable
## size (>=0.83, and the hard-angle LFW set's p5 is 0.76, see #_SURE_FACE_SCORE) and drops the band below it where
## those false positives lie.
_SCORE_THRESHOLD: float = 0.75

## A detection at or above this score is blurred whatever its colour. Measured on LFW faces pasted into 1280x720
## frames at 60-300 px: p1 0.88 as pasted, and p5 0.76, p50 0.90 tilted +-30 deg, squashed for yaw/pitch, defocused
## and dimmed or tinted; a toy steering wheel on a robot's camera scored at most 0.84. 0.85 is above the wheel.
_SURE_FACE_SCORE: float = 0.85

## Below #_SURE_FACE_SCORE a detection is blurred only when at least this fraction of its box is skin-coloured
## (#_skin_fraction). Faces measure p5 0.70 in plain light (0.006 under a strong colour cast, which is why the
## sure-face branch exists); the toy wheel's boxes measure well under 0.30. With both rules, every plain face and
## nearly every hard one blurs (the misses are extreme turns in tinted light scoring 0.60-0.85), and the wheel
## almost never does.
_MIN_SKIN_FRACTION: float = 0.30

## The YCrCb chroma box counted as skin (the classic Chai & Ngan bounds). Chroma only, so it holds across skin
## tones and brightness; a colour cast (a blue LED bar) is what defeats it, see #_MIN_SKIN_FRACTION.
_SKIN_CR: tuple[int, int] = (133, 173)
_SKIN_CB: tuple[int, int] = (77, 127)

## Smallest detection side, in pixels, worth blurring. Below ~20 px a face cannot identify anyone, so blurring
## it protects nothing -- and that scale is where the detector's false-positive density concentrates (a shoe toe
## reads as a 12x14 px face at score 0.60). A person anywhere NEAR a station is hundreds of
## pixels; this floor only excludes people several metres into the background, who are unidentifiable anyway.
_MIN_FACE_SIDE_PX: int = 20

## Non-maximum-suppression overlap for the detector.
_NMS_THRESHOLD: float = 0.3

## Most detections kept per frame; a lab scene has nowhere near this many people.
_TOP_K: int = 64

## Fraction of a detection's width/height added on each side before the oval is drawn. YuNet crops tight
## around the facial features; hair, ears, chin and glasses identify a person too, and the oval trims the
## padded box's corners back off so the padding does not become a big rectangle.
_PAD_FRAC: float = 0.35

## Widest frame the detector sees: a wider one is detected on a copy shrunk to this width and the boxes scaled
## back. YuNet's cost grows with the input area, and on a YAM station PC (Ryzen 7 7730U) one detect took 35 to 50 ms at
## 1280x720 against 11 to 15 ms at 640x360, too much for three capture threads that each scrub five 720p frames a
## second beside a live encoder that scrubs 30 a second per camera. 640 is the live encoder's own frame width,
## so the live path is detected at its own size, and a 720p frame is detected at half scale: a face the
## detector needs 20 px for is 40 px in the frame, smaller than any face a bench camera sees. The synthetic
## self-check face is detected down to about 200 px on a side; narrower is not a face the detector was trained on.
DETECT_MAX_WIDTH: int = 640

## Cells across a face patch in the mosaic. 6 leaves a face-shaped blob and no face.
_MOSAIC_CELLS: int = 6

## Ellipse semi-axes as a fraction of the padded box's half-width/height. Below 1.0 so the oval sits inside
## the box with room for its feathered edge; the box was already padded past the tight face, so the face is
## well within the solid centre of the oval.
_OVAL_AXIS_FRAC: float = 0.92

## Feather width of the oval's edge as a fraction of the smaller box side. The blend runs from fully scrubbed
## at the face to untouched at the oval's rim over this band, so there is no hard rectangle and no hard oval.
_OVAL_FEATHER_FRAC: float = 0.18

## JPEG quality of the re-encode a frame WITH a face pays; a frame without one is returned as the original bytes.
_REENCODE_QUALITY: int = 90


class FaceBlurError(Exception):
    """The frame could not be cleared for the public: the detector is unusable, or the bytes are.

    Raised at construction when the model is missing, will not load, or fails its self-check, and per frame when
    the JPEG cannot be decoded. The frame routes answer it with the same 503 an unreachable station gets.
    """


def blurrer_for(source: str) -> "FaceBlurrer":
    """The process-wide blurrer of one camera source, created on first use.

    One per source, not one per process: the detector is not reentrant, so a blurrer per camera lets two
    stations' viewers scrub in parallel instead of contending on one lock. The sources are a fixed set of
    station ids, so the cache is bounded.

    @param source The camera source the frames come from, e.g. a station id.
    @return The source's blurrer, shared by every caller in the process.
    @throws FaceBlurError When the detector cannot be loaded or does not detect.
    """
    with _blurrers_lock:
        blurrer = _blurrers.get(source)
        if blurrer is None:
            blurrer = FaceBlurrer()
            _blurrers[source] = blurrer
        return blurrer


class FaceBlurrer:
    """One camera source's face scrubber: a YuNet detector over a single frame at a time.

    Each frame is scrubbed on its own detections only - nothing carries between frames, so a moving face
    leaves no trail of blurred ovals behind it. One blurrer per camera source (blurrer_for()) so two
    stations' viewers scrub in parallel; the detector is not reentrant, so its calls are taken under a lock.
    """

    def __init__(self, model_path: str = _MODEL_PATH) -> None:
        """Loads the detector and proves it detects; see the module docstring on failing closed.

        @param model_path The YuNet ONNX weights; the vendored copy by default.
        @throws FaceBlurError When the weights are missing, will not load, or do not detect the synthetic face.
        """
        if not os.path.isfile(model_path):
            raise FaceBlurError(f"face detector weights missing: {model_path}")
        try:
            self._detector = cv2.FaceDetectorYN.create(
                model_path, "", (320, 320), _SCORE_THRESHOLD, _NMS_THRESHOLD, _TOP_K
            )
        except cv2.error as error:
            raise FaceBlurError(f"face detector would not load ({model_path}): {error}") from error
        self._lock = threading.Lock()
        self._size: tuple[int, int] | None = None  # (w, h) the detector is currently sized for
        check = cv2.imencode(".jpg", _self_check_frame())[1].tobytes()
        if self.scrub_jpeg(check) == check:
            raise FaceBlurError(
                f"face detector loaded but did not detect the self-check face ({model_path}); "
                "refusing to pass frames through unblurred"
            )

    def scrub_jpeg(self, jpg: bytes) -> bytes:
        """The frame with every face blurred out as a feathered oval; the original bytes when nothing to blur.

        Skipping the re-encode on a faceless frame is not only cheap, it keeps the camera's own JPEG quality
        for the common case of an empty lab.

        @param jpg The JPEG as the camera surface served it.
        @return The scrubbed JPEG, or `jpg` itself when no region needed blurring.
        @throws FaceBlurError When the bytes cannot be decoded (opaque bytes cannot be cleared, so they must
            not be served) or the detector fails on this frame.
        """
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise FaceBlurError(f"undecodable JPEG ({len(jpg)} bytes)")
        scrubbed = self.scrub_frame(frame)
        if scrubbed is frame:
            return jpg
        ok, buffer = cv2.imencode(".jpg", scrubbed, [int(cv2.IMWRITE_JPEG_QUALITY), _REENCODE_QUALITY])
        if not ok:
            raise FaceBlurError("could not re-encode a blurred frame")
        return buffer.tobytes()

    def scrub_frame(self, frame: Any) -> Any:
        """The decoded frame with every face blurred out as a feathered oval; `frame` itself when nothing to blur.

        For consumers that hold raw arrays - a recorder feeding an encoder - so scrubbing costs no JPEG
        round-trip on their path. The input is never mutated; whether anything was blurred is the identity
        check `scrub_frame(f) is f`, which is also how scrub_jpeg() decides to skip its re-encode.

        @param frame The decoded BGR frame (HxWx3 uint8).
        @return The scrubbed copy, or `frame` itself when no region needed blurring.
        @throws FaceBlurError When the detector fails on this frame.
        """
        boxes = self._boxes(frame)
        if not boxes:
            return frame
        scrubbed = frame.copy()
        for box in boxes:
            _blur_oval(scrubbed, box)
        return scrubbed

    def _boxes(self, frame: Any) -> list[tuple[int, int, int, int]]:
        """This frame's face detections, padded and clipped to the frame - and nothing from any other frame.

        @param frame The decoded BGR frame.
        @return Padded `(x, y, w, h)` boxes clipped to the frame; empty when this frame has no face.
        @throws FaceBlurError When the detector fails on this frame.
        """
        height, width = frame.shape[:2]
        # Detect on a copy no wider than DETECT_MAX_WIDTH; the boxes come back in the frame's own pixels.
        scale = 1.0
        detected = frame
        if width > DETECT_MAX_WIDTH:
            scale = DETECT_MAX_WIDTH / width
            detected = cv2.resize(
                frame, (DETECT_MAX_WIDTH, max(1, round(height * scale))), interpolation=cv2.INTER_AREA
            )
        detected_height, detected_width = detected.shape[:2]
        with self._lock:
            if self._size != (detected_width, detected_height):
                self._detector.setInputSize((detected_width, detected_height))
                self._size = (detected_width, detected_height)
            try:
                _, detections = self._detector.detect(detected)
            except cv2.error as error:
                raise FaceBlurError(f"face detection failed on a {width}x{height} frame: {error}") from error
            boxes = []
            for row in detections if detections is not None else []:
                x, y, w, h = (int(round(float(value) / scale)) for value in row[:4])
                if float(row[14]) < _SCORE_THRESHOLD:
                    continue  # the detector filters on this too; checked here so the floor holds whatever it returns
                if min(w, h) < _MIN_FACE_SIDE_PX:
                    continue  # below identifiable size: nothing to protect, and FP territory (#_MIN_FACE_SIDE_PX)
                if float(row[14]) < _SURE_FACE_SCORE and _skin_fraction(detected, row[:4]) < _MIN_SKIN_FRACTION:
                    continue  # a face-like object, not a person (#_MIN_SKIN_FRACTION)
                box = _pad_box(x, y, w, h, width, height)
                if box[2] > 0 and box[3] > 0:
                    boxes.append(box)
            return boxes


def _skin_fraction(frame: Any, box: Any) -> float:
    """The fraction of a detection's (unpadded) box whose chroma falls in the skin range.

    @param frame The BGR frame the detector saw.
    @param box The detection's `(x, y, w, h)` in that frame's pixels.
    @return 0.0 to 1.0; 0.0 for a box that lies outside the frame.
    """
    height, width = frame.shape[:2]
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(width, int(box[0] + box[2])), min(height, int(box[1] + box[3]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    ycrcb = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    skin = (cr >= _SKIN_CR[0]) & (cr <= _SKIN_CR[1]) & (cb >= _SKIN_CB[0]) & (cb <= _SKIN_CB[1])
    return float(skin.mean())


def _self_check_frame() -> Any:
    """The synthetic face that proves a detector detects in this environment.

    A cartoon rather than a vendored photograph: no licensing question, no face in the repository, and YuNet
    scores it ~0.86 against the 0.75 #_SCORE_THRESHOLD (and it is hundreds of pixels, far above
    #_MIN_FACE_SIDE_PX) - the check fails only when the detector is genuinely broken.

    @return A 400x400 BGR frame containing exactly one detectable face.
    """
    frame = np.full((400, 400, 3), 200, np.uint8)
    cv2.ellipse(frame, (200, 200), (90, 120), 0, 0, 360, (170, 180, 210), -1)
    for eye_x in (168, 232):
        cv2.ellipse(frame, (eye_x, 170), (14, 9), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(frame, (eye_x, 170), 5, (40, 40, 40), -1)
    cv2.line(frame, (200, 180), (200, 225), (120, 130, 160), 3)
    cv2.ellipse(frame, (200, 250), (35, 18), 0, 10, 170, (90, 90, 120), 3)
    return frame


def _pad_box(x: int, y: int, w: int, h: int, width: int, height: int) -> tuple[int, int, int, int]:
    """One detection grown by #_PAD_FRAC on each side and clipped to the frame.

    @param x Detection left, pixels.
    @param y Detection top, pixels.
    @param w Detection width, pixels.
    @param h Detection height, pixels.
    @param width Frame width, pixels.
    @param height Frame height, pixels.
    @return The padded `(x, y, w, h)`, never extending outside the frame.
    """
    dx, dy = round(w * _PAD_FRAC), round(h * _PAD_FRAC)
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(width, x + w + dx), min(height, y + h + dy)
    return x0, y0, max(0, x1 - x0), max(0, y1 - y0)


def _blur_oval(frame: Any, box: tuple[int, int, int, int]) -> None:
    """Scrubs one face as a feathered oval inside `box`, in place, leaving the box's corners untouched.

    A face is an oval, not a rectangle, so the destroyed region is one too: the mosaic is blended over the
    frame through an elliptical alpha mask (#_OVAL_AXIS_FRAC) whose edge is Gaussian-feathered
    (#_OVAL_FEATHER_FRAC). Alpha is 1 across the face - fully mosaicked, the identity gone - and falls to 0 at
    the oval's rim, so the corners of the bounding box (tabletop, wall, a second person's shoulder) keep their
    original pixels and there is no hard edge of either shape.

    @param frame The BGR frame; the box region is modified in place.
    @param box The padded detection as `(x, y, w, h)`.
    """
    x, y, w, h = box
    roi = frame[y : y + h, x : x + w]
    mosaicked = _mosaic(roi)
    alpha = np.zeros((h, w), np.float32)
    axes = (max(1, round(w * 0.5 * _OVAL_AXIS_FRAC)), max(1, round(h * 0.5 * _OVAL_AXIS_FRAC)))
    cv2.ellipse(alpha, (w // 2, h // 2), axes, 0, 0, 360, 1.0, -1)
    feather = max(1, round(min(w, h) * _OVAL_FEATHER_FRAC)) | 1  # odd kernel for GaussianBlur
    # The feathered mask is HxWx1 float, so it broadcasts over the colour channels; the HxW stencil stays as drawn.
    mask = cv2.GaussianBlur(alpha, (feather, feather), 0)[:, :, np.newaxis]
    roi[:] = (roi * (1.0 - mask) + mosaicked * mask).astype(np.uint8)


def _mosaic(patch: Any) -> Any:
    """The identity in one face patch destroyed: averaged down to #_MOSAIC_CELLS cells and interpolated back up.

    Averaging down is the irreversible step - the pixels that made a cell are gone, where a Gaussian blur is a
    linear filter deconvolution can partially walk back. The interpolated way up leaves a smooth blob instead
    of visible cell edges, and costs nothing (a wide Gaussian pass measured 0.3-0.5 s per frame, several times
    the whole frame budget).

    @param patch The BGR patch to scrub; not modified.
    @return The scrubbed patch, same shape.
    """
    h, w = patch.shape[:2]
    cells_w, cells_h = max(1, min(_MOSAIC_CELLS, w)), max(1, min(_MOSAIC_CELLS, h))
    small = cv2.resize(patch, (cells_w, cells_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


## The per-source blurrers of blurrer_for(); a fixed set of station ids, so bounded.
_blurrers: dict[str, FaceBlurrer] = {}

## Guards #_blurrers; each FaceBlurrer then carries its own lock.
_blurrers_lock = threading.Lock()

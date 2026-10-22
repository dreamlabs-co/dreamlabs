"""The video formats a dataset may carry, and the FFmpeg probe that tells a file's format from its content.

One table (#VIDEO_FORMATS) and one probe for both ends: the dream client refuses a dataset before uploading it
(`validation.check_dataset()`), and the data worker refuses one before processing it (it runs
probe() in a limited child process). The format is read by FFmpeg's own MP4 demuxer
through PyAV (imported only when a file is probed), never from the file's name and never by a parser of ours, with
FFmpeg's whitelists (#FORMAT_WHITELIST: the MP4 demuxer alone; no decoder; #PROTOCOL_WHITELIST: local files alone)
and no frame decoded: the answer is the sample-entry tag of the file's one video stream (`avc1` for H.264, `av01` for
AV1), held to #VIDEO_FORMATS, and the stream's timing as its sample table states it (measure()), which the data worker
writes into a camera's `info.json` entry that lacks it and holds to its episode (refuse_off_timing(): one video frame
per frame of data, at the session's rate, evenly spaced). Before FFmpeg opens a file, its top level alone is read
(refuse_image_structure()): a
HEIF or AVIF image or image sequence (#IMAGE_BRANDS, or a top-level `meta` box, where HEIF keeps its items and grids)
is refused there, so FFmpeg's HEIF item code never runs on a customer's file. Every refusal is a
`content_checks.DatasetInvalidError` with the line the customer reads.
"""

import dataclasses
import fractions
import os
import struct
from typing import Any, BinaryIO


@dataclasses.dataclass(frozen=True)
class VideoFormat:
    """One whitelisted format.

    Attributes:
    - name: What the customer reads, e.g. `H.264`.
    - decoder: FFmpeg's decoder for it: the one the trainer's FFmpeg must have. The probe opens none (probe_options()).
    - codec: FFmpeg's name of the codec, what LeRobot writes as a camera's `video.codec` in `meta/info.json`.
    """

    name: str
    decoder: str
    codec: str


## The video formats a dataset may carry, the whitelist: the MP4 sample-entry tag of a video stream (ISO 14496-12,
## what PyAV reports as `codec_tag`) to its #VideoFormat. This table is the only place a format is named. Adding a
## format: a line here, a few frames of it in the data service's test fixtures, as `<tag>.mp4` (the test over this
## table fails for an entry without one), and a GPU qualification on a dataset in it,
## since the trainer must decode it too. Three are what LeRobot records: AV1 by default (`libsvtav1`), H.264 and
## HEVC when the recorder picks them or a hardware encoder does, and HEVC for depth cameras. VP9 is what the
## robots run recorder writes (`robots.rollouts`, `libvpx-vp9`: the library opens no GPL x264 encoder), so a
## station's recordings upload as they are; decord2's FFmpeg decodes it natively. One
## codec can have two tags: FFmpeg, and so LeRobot, writes HEVC as `hev1`, Apple's tools as `hvc1`. An audio track
## beside the video is allowed (the probe decodes nothing; the trainer's FFmpeg may decode it while opening the
## file); a subtitle track is refused (probe()), so no subtitle code of the trainer's FFmpeg ever reads a customer's
## track.
VIDEO_FORMATS: dict[str, VideoFormat] = {
    "avc1": VideoFormat(name="H.264", decoder="h264", codec="h264"),
    "hev1": VideoFormat(name="HEVC", decoder="hevc", codec="hevc"),
    "hvc1": VideoFormat(name="HEVC", decoder="hevc", codec="hevc"),
    "av01": VideoFormat(name="AV1", decoder="libdav1d", codec="av1"),
    "vp09": VideoFormat(name="VP9", decoder="vp9", codec="vp9"),
}

## The one demuxer the probe may open: FFmpeg's MP4 (QuickTime family) demuxer, under the name FFmpeg registers it.
FORMAT_WHITELIST: str = "mov,mp4,m4a,3gp,3g2,mj2"

## The one protocol the probe may use: local files. PyAV's FFmpeg is built with network protocols (TLS included), and
## a container can name other inputs; with this, FFmpeg refuses any but a file on this disk before connecting anywhere.
PROTOCOL_WHITELIST: str = "file"

## The ISO base media file brands of still images and image sequences: HEIF (ISO/IEC 23008-12: `mif1`, `msf1`,
## `heic`, ...), MIAF, AVIF and its sequences (`avif`, `avis`), JPEG and VVC in HEIF. A file naming one, as its major
## or a compatible brand, is not a video; LeRobot writes plain MP4 video (`isom`, `iso2`, `mp41`, `avc1`, `av01`).
IMAGE_BRANDS: frozenset[str] = frozenset(
    {
        "mif1", "mif2", "mif3", "msf1", "miaf",
        "heic", "heix", "heim", "heis", "hevc", "hevx", "hevm", "hevs",
        "avif", "avis", "avio", "avci", "avcs",
        "jpeg", "jpgs", "vvic", "vvis", "j2ki", "j2is",
    }
)  # fmt: skip

## The top-level boxes refuse_image_structure() reads at most; an MP4 video has a handful (`ftyp`, `free`, `mdat`,
## `moov`). A file with more is left to FFmpeg past that point.
MAX_TOP_LEVEL_BOXES: int = 64

## How far, in frames, a video's timing may stray before a frame of data is paired with another's image: the trainer
## gives each frame of data the video frame nearest its timestamp, so within half a frame every one gets its own. The
## tolerance of every frame-timing rule: a video's sample steps (measure()), its rate over an episode
## (refuse_off_timing()), and an episode's timestamps (`content_checks.timestamp_drift()`).
HALF_FRAME: float = 0.5

## Bytes FFmpeg may read to find the streams; FFmpeg's own defaults, stated so they cannot drift.
PROBE_SIZE: int = 5_000_000
ANALYZE_DURATION_US: int = 5_000_000


def accepted() -> str:
    """The whitelist as the customer reads it.

    @return Each format's name once with its tags, in table order, e.g. `H.264 (avc1), HEVC (hev1, hvc1)`.
    """
    tags: dict[str, list[str]] = {}
    for tag, entry in VIDEO_FORMATS.items():
        tags.setdefault(entry.name, []).append(tag)
    return ", ".join(f"{name} ({', '.join(names)})" for name, names in tags.items())


def probe_options() -> dict[str, str]:
    """The FFmpeg options every file is probed with.

    `codec_whitelist` matches no decoder: the sample-entry tag is the container's, so FFmpeg reads it without decoding,
    and no decoder ever runs on the file (FFmpeg would otherwise decode the first frame of an allowed codec to learn
    its pixel format).

    @return The options, as FFmpeg reads them.
    """
    return {
        "format_whitelist": FORMAT_WHITELIST,
        "codec_whitelist": "none",
        "protocol_whitelist": PROTOCOL_WHITELIST,
        "probesize": str(PROBE_SIZE),
        "analyzeduration": str(ANALYZE_DURATION_US),
    }


def refuse_image_structure(file: BinaryIO) -> None:
    """Refuses a file whose top level says it is a HEIF or AVIF image or image sequence, before FFmpeg reads it.

    Only the top level is read: each box's header (8 bytes, 16 with a 64-bit size) for its type and size, and the
    brands of `ftyp`; what is inside the boxes stays FFmpeg's. A header that cannot be a box ends the walk and leaves
    the file to FFmpeg, which refuses what it cannot read. The file's position is restored.

    @param file The file, open for binary reading and seekable.
    @throws ValueError When `ftyp` names a brand of #IMAGE_BRANDS, or the file has a top-level `meta` box (HEIF's
        items: `iinf`, `iloc`, grids); an MP4 video keeps its metadata inside `moov`.
    """
    start = file.tell()
    try:
        end = file.seek(0, os.SEEK_END)
        pos = 0
        for _ in range(MAX_TOP_LEVEL_BOXES):
            if pos + 8 > end:
                break
            file.seek(pos)
            header = file.read(16)
            size, kind = struct.unpack(">I4s", header[:8])
            head = 8
            if size == 1 and len(header) == 16:
                size, head = struct.unpack(">Q", header[8:16])[0], 16
            elif size == 0:
                size = end - pos
            if size < head:
                break
            if kind == b"ftyp":
                file.seek(pos + head)
                body = file.read(min(size - head, 4096))
                brands = {body[i : i + 4].decode("latin1") for i in [0, *range(8, len(body) - 3, 4)]}
                image = sorted(brands & IMAGE_BRANDS)
                if image:
                    raise ValueError(f"an HEIF or AVIF image file (brand {image[0]!r}), not a video")
            elif kind == b"meta":
                raise ValueError("HEIF image items (a top-level meta box), which a video does not carry")
            pos += size
    finally:
        file.seek(start)


def measure(stream: Any) -> dict[str, Any]:
    """A video stream's timing as its MP4 sample table states it, demuxed and never decoded.

    @param stream A PyAV video stream of a file opened with probe_options().
    @return `{"fps", "frames", "duration_s", "variable"}`: the frame rate of the even step from the first sample
        shown to the last (the demuxer's index), the number of samples shown, and the track's duration in seconds,
        each None when the table does not state it; and whether the rate varies: a sample more than #HALF_FRAME of a
        step from where that even rate puts it. A sample an edit list hides (the pre-roll an `ffmpeg -ss -c copy`
        cut keeps) is not shown, so not counted. Not FFmpeg's `avg_frame_rate` (the samples over the track's
        duration): a track that ends where its last sample starts, as the robots package's rollout videos do, puts
        that n / (n - 1) too high, 34.3 fps for 8 frames at 30. Not the usual (median) step either: a 1/1000 s clock
        (an MKV recording remuxed) starts 30 fps frames 33 or 34 ms apart, which reads as 30.3 fps, and nine steps of
        1/30 s with one of 1.47/30 s read as 30 fps while the frames run at 28.7.
    """
    duration = stream.duration
    times = sorted(entry.timestamp for entry in stream.index_entries if not entry.is_discard)
    step = fractions.Fraction(times[-1] - times[0], len(times) - 1) if len(times) > 1 else 0
    spacing = float(step)
    return {
        "fps": float(1 / (step * stream.time_base)) if step else None,
        "frames": len(times) or None,
        "duration_s": float(duration * stream.time_base) if duration is not None else None,
        "variable": step > 0
        and any(abs(time - times[0] - k * spacing) > spacing * HALF_FRAME for k, time in enumerate(times)),
    }


def probe(source: str | BinaryIO) -> dict[str, Any]:
    """The sample-entry tag and timing of a file's one video stream, by FFmpeg's MP4 demuxer; no frame is decoded.

    @param source The file: a path, or a binary file object open on it.
    @return `{"tag", "fps", "frames", "duration_s", "variable"}`: the tag, e.g. `avc1`, and measure()'s timing.
    @throws av.FFmpegError When FFmpeg cannot read the file as an MP4.
    @throws ValueError When the file is a HEIF or AVIF image or image sequence (refuse_image_structure()), is not in an
        MP4 container (#FORMAT_WHITELIST), holds no video stream or more than one, holds a subtitle track, or its one
        video is a still image rather than a track.
    """
    import av  # noqa: PLC0415 - only where a file is probed: the data worker imports this module and never probes in it

    if isinstance(source, str):
        with open(source, "rb") as file:
            refuse_image_structure(file)
    else:
        refuse_image_structure(source)

    try:
        opened = av.open(source, options=probe_options())
    except av.ArgumentError:
        raise ValueError("not an MP4 container") from None  # FFmpeg's answer to a demuxer off #FORMAT_WHITELIST
    with opened as container:
        videos = container.streams.video
        if len(videos) != 1:
            raise ValueError(f"{len(videos)} video streams")
        # A second guard behind refuse_image_structure(): an image item that got this far would open as a video
        # stream tagged `hvc1` or `av01` with no track timing, so no duration.
        if videos[0].duration is None:
            raise ValueError("a still image (HEIF or AVIF), not a video track")
        if container.streams.subtitles:
            raise ValueError(
                f"{len(container.streams.subtitles)} subtitle track(s); remove them with "
                "ffmpeg -i in.mp4 -map 0 -map -0:s -c copy out.mp4"
            )
        return {"tag": str(videos[0].codec_tag), **measure(videos[0])}


def answer(source: str | BinaryIO) -> dict[str, Any]:
    """probe() as an answer that cannot raise: its answer, or `{"error": ...}` with FFmpeg's reason.

    @param source The file: a path, or a binary file object open on it.
    @return The answer, which refuse_unless_whitelisted() turns into the customer's line.
    """
    try:
        return probe(source)
    except Exception as error:  # noqa: BLE001 - every failure of a probe is that file's answer
        return {"error": getattr(error, "strerror", None) or str(error) or type(error).__name__}


def refuse_unless_whitelisted(where: str, found: dict[str, Any]) -> None:
    """Turns a probe's answer into the customer's refusal, or nothing for a whitelisted format.

    @param where The file as the customer reads it, e.g. `Directory 's': videos/.../episode_000000.mp4`.
    @param found An answer of answer().
    @throws content_checks.DatasetInvalidError When the file is not a readable MP4 or its format is off the list.
    """
    # Here, not at the top: content_checks brings numpy and pyarrow, which the probe child must never load (OpenBLAS
    # reserves address space per core, inside the child's ceiling); this function runs only in its callers.
    from shared.datasets import content_checks  # noqa: PLC0415

    if "error" in found:
        raise content_checks.DatasetInvalidError(f"{where} is not a readable MP4 ({found['error']}).")
    tag = str(found["tag"])
    if tag not in VIDEO_FORMATS:
        raise content_checks.DatasetInvalidError(f"{where} is video in format {tag!r}; accepted: {accepted()} in MP4.")


def refuse_off_timing(where: str, found: dict[str, Any], fps: int, rows: int) -> None:
    """Refuses a video whose frames cannot be paired one to one with its episode's frames of data.

    The trainer pairs each frame of data with the video frame nearest its timestamp, so a video must hold one frame
    per frame of data, at the session's rate, evenly spaced. When it does, an episode whose timestamps drift from it
    is re-timed by the data worker instead (`content_checks.timestamp_drift()`); when it does not, no timestamp can
    say which image goes with which frame.

    @param where The file as the customer reads it, e.g. `Directory 's': videos/.../episode_000000.mp4`.
    @param found Its answer of answer(), whitelisted (refuse_unless_whitelisted()).
    @param fps The rate its session's `meta/info.json` records.
    @param rows Its episode's frames of data (`meta/episodes.jsonl`'s length, held to the parquet's rows).
    @throws content_checks.DatasetInvalidError When its rate varies, its rate drifts from `fps` by more than
        #HALF_FRAME over the episode, or it holds fewer frames than the episode less one.
    """
    from shared.datasets import content_checks  # noqa: PLC0415 - as in refuse_unless_whitelisted()

    if found["variable"]:
        raise content_checks.DatasetInvalidError(
            f"{where} has a variable frame rate; training pairs each frame of data with the video frame at its time, "
            f"at one rate. Re-encode it at a constant {fps} fps (ffmpeg -i in.mp4 -fps_mode cfr -r {fps} out.mp4)."
        )
    rate = found["fps"]
    if rate is not None and abs(rate / fps - 1) * rows > HALF_FRAME:
        raise content_checks.DatasetInvalidError(
            f"{where} plays at {rate:g} fps, but meta/info.json records {fps} fps: over the episode's {rows} frames "
            f"they drift {abs(rate / fps - 1) * rows:.1f} frames apart. Re-encode it at {fps} fps, or record its rate "
            "as fps."
        )
    frames = found["frames"]
    if frames is not None and frames < rows - 1:
        raise content_checks.DatasetInvalidError(
            f"{where} holds {frames} video frames for an episode of {rows} frames; training would repeat its last "
            "image for the rest. Export one video frame per frame of the episode."
        )

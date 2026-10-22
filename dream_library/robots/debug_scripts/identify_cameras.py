"""Headless camera identification on a station: every camera the config could name, one picture each.

Two kinds of camera are found, the same two `robots.config.CameraConfig` accepts. An Intel RealSense is listed by
serial, with its model name and the USB link it enumerated on (`usb 3.2` is a SuperSpeed link; `usb 2.x` will not
stream 720p at 30 fps), and read through the product's own `RealSenseSource`. A UVC webcam is listed by its
`/dev/v4l/by-path` port, the `usb-` spelling with the `usbv2-` alias of the same port dropped, and read through
`cv2.VideoCapture`. One JPEG per camera is written into `--out`, named by that identity, so an operator can look at
the pictures, decide which camera is `top`, `left` and `right`, and paste the identity into the config.

    .venv/bin/python debug_scripts/identify_cameras.py --out /tmp/cams

Run it with the arms off and no loop running. It opens only cameras, one at a time, and releases each before the
next; a camera that enumerates but cannot deliver a frame is reported with the reason instead of a picture.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import cv2

from robots import cameras, errors

## Where the kernel publishes a stable symlink per V4L2 node.
BY_PATH_DIR = Path("/dev/v4l/by-path")

## Reads attempted on a UVC node before it is declared to deliver nothing.
UVC_READ_ATTEMPTS: int = 10


def realsense_devices() -> list[dict[str, str]]:
    """Enumerates the RealSense cameras librealsense sees.

    @return One dict per camera with `serial`, `name` and `usb` (the USB type descriptor, e.g. `"3.2"`); an empty
        list when pyrealsense2 is not installed, which is printed as a note.
    """
    try:
        import pyrealsense2 as rs  # type: ignore[import-not-found]
    except ImportError:
        print("note: pyrealsense2 is not installed, so RealSense cameras are not listed (pip install pyrealsense2)")
        return []
    found: list[dict[str, str]] = []
    seen: set[str] = set()
    for dev in rs.context().query_devices():
        serial = dev.get_info(rs.camera_info.serial_number)
        if serial in seen:  # librealsense may hand back one entry per interface of the same device
            continue
        seen.add(serial)
        found.append(
            {
                "serial": serial,
                "name": dev.get_info(rs.camera_info.name),
                "usb": dev.get_info(rs.camera_info.usb_type_descriptor),
            }
        )
    return found


def realsense_picture(serial: str, width: int, height: int, fps: int) -> Any:
    """Reads one colour frame from the camera with `serial` through the product's source.

    @param serial The camera's serial number.
    @param width Frame width in pixels, the mode the station will run.
    @param height Frame height in pixels.
    @param fps Frame rate of that mode.
    @return The BGR frame.
    @throws CameraError pyrealsense2 is missing.
    @throws RuntimeError librealsense could not start the mode (an absent camera, a USB2 link) or delivered no frame.
    """
    source = cameras.RealSenseSource(serial, width, height, fps)
    try:
        ok, frame = source.read()
    finally:
        source.release()
    if not ok or frame is None:
        raise RuntimeError("no frame arrived")
    return frame


def uvc_ports() -> list[tuple[str, Path]]:
    """Lists the capture nodes under #BY_PATH_DIR, one per physical USB port.

    @return `(port, node)` pairs sorted by port: the by-path name without its `-video-index0` suffix, with the
        `usbv2-` alias normalised to `usb-` and duplicates dropped, and the symlink itself.
    """
    if not BY_PATH_DIR.is_dir():
        return []
    by_port: dict[str, Path] = {}
    for link in sorted(BY_PATH_DIR.glob("*-video-index0")):
        port = re.sub(r"-usbv2-", "-usb-", link.name.removesuffix("-video-index0"))
        by_port.setdefault(port, link)
    return sorted(by_port.items())


def uvc_picture(node: Path, width: int, height: int, fps: int) -> Any:
    """Reads one frame from a V4L2 node through OpenCV.

    @param node The by-path symlink to open.
    @param width Frame width to request.
    @param height Frame height to request.
    @param fps Frame rate to request.
    @return The BGR frame.
    @throws RuntimeError The node cannot be opened, or delivers no frame within #UVC_READ_ATTEMPTS reads (a RealSense
        depth node and a dead webcam both look like this).
    """
    capture = cv2.VideoCapture(str(node))
    try:
        if not capture.isOpened():
            raise RuntimeError("cannot be opened by OpenCV")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        for _ in range(UVC_READ_ATTEMPTS):
            ok, frame = capture.read()
            if ok and frame is not None:
                return frame
    finally:
        capture.release()
    raise RuntimeError("delivers no frame")


def main(out_dir: Path, width: int, height: int, fps: int) -> int:
    """Finds every camera, writes one picture each, prints one line per camera.

    @param out_dir Directory the JPEGs go into; created when missing.
    @param width Frame width to request from every camera.
    @param height Frame height to request from every camera.
    @param fps Frame rate to request from every camera.
    @return The number of cameras that delivered a picture.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for dev in realsense_devices():
        label = f"realsense {dev['serial']}  {dev['name']}  usb {dev['usb']}"
        try:
            frame = realsense_picture(dev["serial"], width, height, fps)
        except (errors.CameraError, RuntimeError) as exc:
            print(f"{label}  -> no picture: {exc}")
            continue
        path = out_dir / f"realsense-{dev['serial']}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"{label}  -> {path}")
        count += 1
    for port, node in uvc_ports():
        try:
            frame = uvc_picture(node, width, height, fps)
        except RuntimeError as exc:
            # Every RealSense also has a video-index0 depth node OpenCV cannot read; that is the usual skip.
            print(f"skipped {port}: {exc}", file=sys.stderr)
            continue
        path = out_dir / f"uvc-{port}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"uvc {port}  -> {path}")
        count += 1
    print(f"{count} camera(s) delivered a picture; the config names a RealSense by serial and a webcam by")
    print("  device: /dev/v4l/by-path/<port>-video-index0")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out", type=Path, default=Path("/tmp/cams"), help="directory for the JPEGs (default /tmp/cams)"
    )
    parser.add_argument("--width", type=int, default=1280, help="capture width to request (default 1280)")
    parser.add_argument("--height", type=int, default=720, help="capture height to request (default 720)")
    parser.add_argument("--fps", type=int, default=30, help="capture rate to request (default 30)")
    args = parser.parse_args()
    main(args.out, args.width, args.height, args.fps)

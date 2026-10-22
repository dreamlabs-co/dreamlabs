"""Why a station could not start, in one line `dreamcli robot up` shows the operator.

`robots.station.main()` calls log() when `python -m robots.station` fails before it was ready (`dreamcli robot up`
then exits within seconds), after `serve()` has switched the arms off. describe() turns the error into the three
things the operator needs: which device, the error in the station's own words, and the fix. log() writes them as one
line starting with #MARKER, ahead of any traceback, and `robots.cli.robot_up()` finds that line in the station log
(find()) and prints it first, so the cause is not buried under the traceback; `dreamcli robot log` shows the rest.

@see describe(), log(), find()
"""

import logging
from collections.abc import Iterable

from robots import cameras, config, errors

## The longest error the line carries.
ERROR_MAX_CHARS = 2000

## What starts the line log() writes and find() looks for.
MARKER = "station: could not start: "

## What every fix ends with: the command that tries again.
_AGAIN = "then run dreamcli robot up again."

_log = logging.getLogger(__name__)


def describe(exc: BaseException, cfg: config.AppConfig) -> dict[str, str]:
    """The device, the error and the fix for a failed start.

    The camera is found by the identity its error names (its serial, its device, or its role in quotes); an error
    that carries `robots.cameras.USB3_FIX` is a RealSense on a USB2 link.

    @param exc What ended the start.
    @param cfg The station config.
    @return `{"device", "error", "fix"}`.
    """
    error = (str(exc) or type(exc).__name__)[:ERROR_MAX_CHARS]
    camera = _camera_in(error, cfg)
    if cameras.USB3_FIX in error:
        device = "a RealSense camera" if camera is None else camera
        return {
            "device": device,
            "error": error,
            "fix": f"{cameras.USB3_FIX[0].upper()}{cameras.USB3_FIX[1:]}, {_AGAIN}",
        }
    if isinstance(exc, errors.CameraError) or camera is not None:
        device = "a camera" if camera is None else camera
        return {"device": device, "error": error, "fix": f"Check the camera's cable and USB port, {_AGAIN}"}
    if isinstance(exc, errors.ArmError):
        return {
            "device": "an arm",
            "error": error,
            "fix": f"Check the arms' power, emergency stop and cables, {_AGAIN}",
        }
    if isinstance(exc, errors.ConfigError):
        return {
            "device": "station.yaml",
            "error": error,
            "fix": f"Fix what the error names (dreamcli setup), {_AGAIN}",
        }
    return {
        "device": "the station",
        "error": error,
        "fix": f"Read dreamcli robot log on the robot's computer, fix what it names, {_AGAIN}",
    }


def log(exc: BaseException, cfg: config.AppConfig) -> None:
    """Writes the failed start as one #MARKER line, `<device>: <error> -- Fix: <fix>`; never raises.

    @param exc What ended the start.
    @param cfg The station config.
    """
    try:
        said = describe(exc, cfg)
        _log.error("%s%s: %s -- Fix: %s", MARKER, said["device"], said["error"], said["fix"])
    except Exception:  # The station's own error is what it exits with; this line only helps read it.
        _log.exception("station: could not describe why the start failed")


def find(lines: Iterable[str]) -> str | None:
    """The last #MARKER line's text after the marker, or None when the lines hold none.

    @param lines Log lines, oldest first.
    @return `<device>: <error> -- Fix: <fix>`, or None.
    """
    found = None
    for line in lines:
        at = line.find(MARKER)
        if at >= 0:
            found = line[at + len(MARKER) :].strip()
    return found


def _camera_in(text: str, cfg: config.AppConfig) -> str | None:
    """The config's camera the text names, as the card says it, or None."""
    for camera in cfg.cameras:
        # A device index (0, 1) would match any digit, so only a path or an address names a device.
        names = [f"'{camera.name}'", camera.serial, camera.device if isinstance(camera.device, str) else None]
        if not any(name and name in text for name in names):
            continue
        what = f"RealSense {camera.serial}" if camera.serial else str(camera.device or camera.source)
        return f"the {camera.name} camera ({what})"
    return None

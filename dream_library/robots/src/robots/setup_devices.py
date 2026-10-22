"""What the setup wizard can see on the box, named by what stays the same: a serial number, never a port name.

`/dev/ttyACM0`, `can0` and `/dev/video2` are handed out in the order devices appear, so a replug or a reboot can
swap them; a serial number goes with the device. scan() reads sysfs and `/dev/*/by-id` without opening anything:

- CAN links (SocketCAN, the YAM's candleLight adapters under gs_usb): each with its adapter's USB serial. Save
  names each arm's link after that serial with a udev rule (can_udev_rule()), so `can_follower_l` is always the
  left arm's adapter, whatever order the adapters came up in.
- CANables on slcan firmware (MakerMods' Metal and Makers adapters): by their `/dev/serial/by-id/...` path, which
  carries the serial, opened directly (#robots.robstride.open_can_bus). A link an `slcand` made of one is the same
  adapter and is not listed again: the CANable's entry says it goes through that link.
- Other serial adapters (`/dev/serial/by-id/...`: a servo bus): the by-id path is the name the config keeps.

Identical serial adapters can report one serial (Damiao's USB2CAN bridges all do). udev then makes one by-id link for
all of them, pointing at whichever came up last, so a replug or a reboot can swap them: such an adapter, or one with no
serial, is named by its `/dev/serial/by-path/...` link (the USB port), like a camera below. A CAN link's adapter whose
serial another shares gets its entry id from its USB port too.

Each entry says which bus it is (`can` or `serial`), so the page offers only what can be the picked robot's arms.
An arm the scan cannot see (on Ethernet, behind a bridge) is typed instead (typed_arm()): its driver opens the
address as written.
- Cameras: a RealSense by its serial (librealsense), any other camera by its `/dev/v4l/by-id/...` node, which carries
  its serial. A camera with no serial, or two that report the same serial, has nothing that stays put but the USB
  port it is in; it is named by that port, and the page says to keep it there.

A camera the scan cannot see is typed too (typed_camera()): any address OpenCV opens (an `rtsp://` stream), or a
camera class of your own for an SDK camera. preview_jpeg() takes one picture from a camera the way the station
opens it.

@see scan(), typed_arm(), typed_camera(), can_udev_rule(), preview_jpeg()
"""

import errno
import logging
import os
import pathlib
import re
import time
from typing import Any

from robots import cameras as camera_module
from robots import config, errors, robstride, user_class

## Where the kernel lists network interfaces; a CAN link's `type` file reads `280` (ARPHRD_CAN).
NET_DIR = pathlib.Path("/sys/class/net")

## ARPHRD_CAN, the interface type of a SocketCAN link.
ARPHRD_CAN = "280"

## Where udev links each serial adapter by its USB ids and serial.
SERIAL_BY_ID_DIR = pathlib.Path("/dev/serial/by-id")

## Where udev links each serial adapter by the USB port it is on.
SERIAL_BY_PATH_DIR = pathlib.Path("/dev/serial/by-path")

## Where the kernel lists video nodes, each with its `name` and its USB device.
VIDEO_CLASS_DIR = pathlib.Path("/sys/class/video4linux")

## Where udev links each video node by its camera's USB ids and serial.
VIDEO_BY_ID_DIR = pathlib.Path("/dev/v4l/by-id")

## Where udev links each video node by the USB port it is on.
VIDEO_BY_PATH_DIR = pathlib.Path("/dev/v4l/by-path")

## Where the kernel lists serial ttys, each with its USB device.
TTY_CLASS_DIR = pathlib.Path("/sys/class/tty")

## Where the rule naming the arms' links goes: a file of its own, so it never replaces a rule file already there.
UDEV_RULE_PATH = "/etc/udev/rules.d/90-dreamlabs-can.rules"

SLCAN_USB_IDS = frozenset({("ad50", "60c4"), ("16d0", "117e")})

## The links a station's arms are on, by side: the names the config uses and the udev rule gives.
CAN_LINKS: dict[str, str] = {"left": "can_follower_l", "right": "can_follower_r"}

## The bitrate every arm's CAN link runs at.
CAN_BITRATE = 1_000_000

## The ways a camera row names its camera, each with the station file's key for it (`robots.config.CameraConfig`):
## an OpenCV index, path or address, a RealSense serial, a camera class of your own.
CAMERA_KEYS: dict[str, str] = {"device": "device", "serial": "serial", "class": "source"}
CAMERA_SOURCES = tuple(CAMERA_KEYS)

## The kind, bus and `bound` of a device the operator typed (typed_arm(), typed_camera()): every robot's driver may
## open a typed arm.
TYPED = "typed"

## Reads attempted on a V4L2 node before it is declared to deliver nothing (a RealSense depth node looks like this).
PREVIEW_READ_ATTEMPTS: int = 10
## Seconds a RealSense preview's exposure must stay unchanged before it keeps a frame: one and a half managed steps,
## so a step found the frame on target (or at a bound) and moved nothing. The firmware's own auto-exposure stays blown
## out on a bright bench, so the preview runs the station's managed exposure as the Live view does, and gives up at
## the station's settle limit (`cameras.MANAGED_SETTLE_S`). A fixed 2 s left wrist cameras over a bright table white:
## from the 14 ms seed, four 0.7x steps reach only about 3.4 ms.
PREVIEW_STEADY_S: float = 1.5 * camera_module.MANAGED_PERIOD_S

## A by-path link udev makes for a USB port (`pci-0000:00:14.0-usb-0:2:1.0`, `-port0` on a multi-port chip); the
## `usbv2-` spelling of the same port, and an alias a local udev rule adds, are left out.
_USB_PORT_LINK_RE = re.compile(r".+-usb-[0-9]+:[0-9.]+:[0-9]+\.[0-9]+(-port[0-9]+)?")

## What a kernel device name or a USB serial may hold. Anything else is refused before it reaches the heredoc a root
## shell runs.
_DEVICE_NAME_RE = re.compile(r"[A-Za-z0-9._:\-]+")

_log = logging.getLogger(__name__)


def scan() -> dict[str, Any]:
    """Everything the page offers, read-only: arm adapters and cameras, each named by what stays the same.

    @return `{"arms": arm_adapters(), "cameras": cameras(), "can": can_links()}`.
    """
    links = can_links()
    return {
        "arms": arm_adapters(links, serial_adapters(), robstride.slcan_daemons()),
        "cameras": cameras(),
        "can": links,
    }


def typed_arm(address: str) -> dict[str, Any]:
    """An arm the scan does not list ("Mine isn't listed"), by the address its driver opens.

    A CAN link, a serial path, an IP address: whatever the robot's driver takes as `can_channel`.

    @param address The address, as the operator typed it.
    @return An arm_adapters() entry of kind and bus #TYPED, opened as written.
    @throws ValueError The address is empty or holds a line break.
    """
    value = address.strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError(
            "type your arm's address on one line, as your driver opens it (192.168.1.20, can2, /dev/ttyUSB0)"
        )
    return {
        "id": f"typed-{value}",
        "kind": TYPED,
        "bus": TYPED,
        "open": value,
        "serial": None,
        "usb": None,
        "label": value,
        "detail": "Typed: your arm's driver opens it as written",
        "state": None,
        "via": None,
    }


def typed_camera(address: str) -> dict[str, Any]:
    """A camera the scan does not list ("Mine isn't listed"): a camera class of your own, or anything OpenCV opens.

    `module:Class` or `file.py:Class` (no `://`, an identifier after the last colon) is a class
    (#robots.config.CameraConfig's `source`, a file's path made absolute from this folder); anything else is an OpenCV
    device: an index, a `/dev/video*` path, an `rtsp://` or `http://` address.

    @param address The address or class, as the operator typed it.
    @return A cameras() entry, `bound` #TYPED.
    @throws ValueError The address is empty.
    """
    value = address.strip()
    if not value:
        raise ValueError("type your camera's address (rtsp://192.168.1.50/stream1), or its class as module:Class")
    module, _, name = value.rpartition(":")
    if "://" not in value and module and name.isidentifier():
        path = user_class.path(value, "camera class", pathlib.Path.cwd())
        return {"source": "class", "value": path, "label": f"Your camera class {path}", "bound": TYPED}
    return {"source": "device", "value": value, "label": value, "bound": TYPED}


def can_links() -> list[dict[str, Any]]:
    """The SocketCAN links in sysfs, whether each is up, and its adapter's USB interface and serial.

    @return `[{"name", "state", "usb", "serial"}]` sorted by name; empty where there is no sysfs. `usb` is the kernel
        name of the link's USB interface (`1-2:1.0`, which names the port), `serial` the adapter's USB serial; each
        None when sysfs does not say.
    """
    found: list[dict[str, Any]] = []
    if not NET_DIR.is_dir():
        return found
    for entry in sorted(NET_DIR.iterdir()):
        try:
            if (entry / "type").read_text().strip() != ARPHRD_CAN:
                continue
            state = robstride.link_state(entry)
        except OSError:
            continue
        usb, serial = _usb_identity(entry / "device")
        found.append({"name": entry.name, "state": state, "usb": usb, "serial": serial})
    return found


def serial_adapters() -> list[dict[str, Any]]:
    """The USB serial adapters udev links: CANables on slcan firmware, servo buses, anything else.

    An adapter is named by its by-id link when no other serial tty reports its serial, else (or with no serial) by its
    by-path link, the USB port it is in; an adapter with neither is left out.

    @return `[{"path", "tty", "serial", "usb", "product", "slcan", "bound"}]` sorted by path. `bound` is `serial`
        (the by-id link) or `port` (keep it in this USB port).
    """
    links: dict[str, dict[str, str]] = {"by_id": {}, "by_path": {}}
    for key, folder in (("by_id", SERIAL_BY_ID_DIR), ("by_path", SERIAL_BY_PATH_DIR)):
        if folder.is_dir():
            for link in sorted(folder.iterdir()):
                if key == "by_path" and not _USB_PORT_LINK_RE.fullmatch(link.name):
                    continue
                links[key].setdefault(os.path.realpath(link), str(link))
    ttys = sorted(set(links["by_id"]) | set(links["by_path"]))
    usbs = {tty: _usb_device(TTY_CLASS_DIR / os.path.basename(tty) / "device") for tty in ttys}
    serials = [usb["serial"] for usb in usbs.values()]
    found: list[dict[str, Any]] = []
    for tty, usb in usbs.items():
        by_id = links["by_id"].get(tty)
        if by_id is not None and usb["serial"] and serials.count(usb["serial"]) == 1:
            path, bound = by_id, "serial"
        elif tty in links["by_path"]:
            path, bound = links["by_path"][tty], "port"
        else:
            continue
        slcan = (usb["vendor"], usb["product_id"]) in SLCAN_USB_IDS or any(
            "canable" in (name or "").lower() for name in (usb["product"], os.path.basename(path))
        )
        found.append(
            {
                "path": path,
                "tty": tty,
                "serial": usb["serial"],
                "usb": usb["interface"],
                "product": usb["product"],
                "slcan": slcan,
                "bound": bound,
            }
        )
    return sorted(found, key=lambda adapter: adapter["path"])


def arm_adapters(
    links: list[dict[str, Any]], adapters: list[dict[str, Any]], daemons: list[robstride.SlcanDaemon]
) -> list[dict[str, Any]]:
    """Every device an arm can be on, once each, with an id that stays the same and the name to open it by.

    @param links can_links().
    @param adapters serial_adapters().
    @param daemons The `slcand`s running.
    @return `[{"id", "kind", "bus", "open", "serial", "usb", "label", "detail", "state", "via"}]`.
    """
    states = {link["name"]: link["state"] for link in links}
    holders = {daemon.tty: daemon for daemon in daemons}
    found: list[dict[str, Any]] = []
    held_links: set[str] = set()
    for adapter in adapters:
        name = os.path.basename(adapter["path"])
        port = adapter["bound"] == "port"
        where = f", in USB port {adapter['usb']} (keep it there)" if port else ""
        if adapter["slcan"]:
            holder = holders.get(adapter["tty"])
            via = holder.link if holder is not None else None
            state = states.get(via) if via is not None else None
            if via is not None:
                held_links.add(via)
                detail = f"through slcand as CAN link {via}, {state or 'missing'}"
            elif holder is not None:
                detail = f"held by slcand (pid {holder.pid}) under a link it did not name"
            else:
                detail = "opened directly over USB"
            found.append(
                {
                    "id": f"slcan-{name if port else adapter['serial']}",
                    "kind": "slcan",
                    "bus": "can",
                    "open": adapter["path"],
                    "serial": adapter["serial"],
                    "usb": adapter["usb"],
                    "label": " ".join(part for part in ("CAN adapter (CANable)", adapter["serial"]) if part) + where,
                    "detail": detail,
                    "state": state,
                    "via": via,
                }
            )
        else:
            found.append(
                {
                    "id": f"serial-{name}",
                    "kind": "serial",
                    "bus": "serial",
                    "open": adapter["path"],
                    "serial": adapter["serial"],
                    "usb": adapter["usb"],
                    "label": " ".join(
                        part for part in ("USB serial adapter", adapter["product"], adapter["serial"]) if part
                    )
                    + where,
                    "detail": adapter["path"],
                    "state": None,
                    "via": None,
                }
            )
    can_serials = [link["serial"] for link in links]
    for link in links:
        if link["name"] in held_links:
            continue
        if link["serial"] and can_serials.count(link["serial"]) == 1:
            ident = f"can-{link['serial']}"
            label = f"CAN adapter {link['serial']}"
        elif link["usb"]:
            ident = f"can-port-{link['usb']}"
            which = link["serial"] or "without a serial number"
            label = f"CAN adapter {which}, in USB port {link['usb']} (keep it there)"
        else:
            ident = f"can-port-{link['name']}"
            label = f"CAN link {link['name']}, no USB adapter behind it"
        found.append(
            {
                "id": ident,
                "kind": "can",
                "bus": "can",
                "open": link["name"],
                "serial": link["serial"],
                "usb": link["usb"],
                "label": label,
                "detail": f"CAN link {link['name']}, {link['state']}",
                "state": link["state"],
                "via": None,
            }
        )
    return sorted(found, key=lambda device: (device["bus"], device["kind"] != "can", device["id"]))


def cameras() -> list[dict[str, Any]]:
    """Every camera the page takes a picture of: each RealSense by serial, then every other camera by a stable node.

    A RealSense also enumerates V4L2 nodes, which OpenCV cannot read on a D405, so once librealsense lists
    one its nodes are left out and it appears once, by serial. Any other camera is its lowest-numbered node (the
    colour stream), named by its by-id link when the camera reports a serial that no other camera shares, else by its
    by-path link (the USB port).

    @return `[{"source", "value", "label", "bound"}]`: `source` and `value` as a camera entry takes them, `bound` is
        `serial` (the camera's own) or `port` (keep it in this USB port). A RealSense also has `usb`, the link it
        enumerated at (`"3.2"`, `"2.1"`, None when librealsense does not say), and `usb_warning`, what the page says
        about a USB2 link, which cannot carry its 1280x720 colour stream at 30 fps, else None.
    """
    realsense = _realsense_devices()
    found = [
        {
            "source": "serial",
            "value": device["serial"],
            "label": f"{device['name']} {device['serial']}",
            "bound": "serial",
            "usb": device["usb"],
            "usb_warning": _usb2_warning(device["usb"]),
        }
        for device in realsense
    ]
    nodes = _video_nodes()
    by_device: dict[str, dict[str, Any]] = {}
    for node in nodes:
        if realsense and "realsense" in node["name"].lower():
            continue
        held = by_device.get(node["usb_device"])
        if held is None or node["index"] < held["index"]:
            by_device[node["usb_device"]] = node
    picked = sorted(by_device.values(), key=lambda node: node["node"])
    serials = [node["serial"] for node in picked]
    for node in picked:
        # Identical webcams can share one serial and one by-id link: only the USB port tells them apart.
        unique = node["by_id"] is not None and node["serial"] and serials.count(node["serial"]) == 1
        if unique:
            found.append({"source": "device", "value": node["by_id"], "label": node["name"], "bound": "serial"})
        elif node["by_path"] is not None:
            found.append({"source": "device", "value": node["by_path"], "label": node["name"], "bound": "port"})
    return found


def explain(exc: BaseException) -> str:
    """Why an arm did not open or stopped answering, in the words of someone at the robot, with the one thing to do.

    @param exc The failure, with its causes.
    @return The sentence.
    """
    chain: list[BaseException] = []
    link: BaseException | None = exc
    while link is not None and link not in chain:
        chain.append(link)
        link = link.__cause__ or link.__context__
    codes = {getattr(item, attr, None) for item in chain for attr in ("errno", "error_code")}
    text = " ".join(str(item) for item in chain).lower()
    again = "then click Scan again"
    if isinstance(exc, errors.ConfigError):
        return str(exc)
    if errno.ENETDOWN in codes or "network is down" in text:
        return (
            "The adapter's CAN link is down: a link comes up down after a replug or a reboot, and an slcand's link "
            "goes down when its adapter is unplugged. Run the command shown under the adapter (or plug it back in), "
            f"{again}."
        )
    if errno.ENODEV in codes or "no such device" in text:
        return f"The adapter is gone: it was unplugged or lost power. Plug it back in, {again}."
    if errno.EACCES in codes or "permission denied" in text:
        return (
            "This user may not open the adapter. Run sudo usermod -aG dialout $USER, log out and back in, then "
            "restart dreamcli setup."
        )
    if errno.EBUSY in codes or "resource busy" in text or "exclusively lock" in text:
        return (
            "Another program has this adapter open (a running station, an slcand, another setup page). Close it, "
            f"{again}."
        )
    if errno.ENOENT in codes or "could not open port" in text or "no such file" in text:
        return f"The adapter's device is gone: it was unplugged. Plug it back in, {again}."
    if isinstance(exc, errors.RobotsError):
        silent = _SILENT.fullmatch(str(exc))
        if silent is not None:
            motor = silent["motor"].replace("_", " ")
            return (
                f"The {silent['side']} arm's {motor} motor stopped answering: check the arm's power and the cable "
                "to its adapter"
            )
        return _warm_words(str(exc))
    if "did not answer" in text or "no reply" in text:
        return (
            "The adapter opened, but the arm's motors did not answer. Switch the arm's power on (release its "
            f"emergency stop), check the cable between the adapter and the arm, {again}. ({type(exc).__name__}: {exc})"
        )
    return f"{type(exc).__name__}: {exc}"


_SILENT = re.compile(
    r"arm '(?P<side>\S+)' stopped: MotorSilent: \S+: (?:no fresh reply from )?(?P<motor>\w+) \(id [^)]*\).*"
)


_WARM = re.compile(
    r"the (?P<side>\S+) arm's (?P<warm>.+): let it cool below (?P<limit>\d+) C, then try again "
    r"\(on a station, press Reset\)"
)
_WARM_MOTOR = re.compile(r"(?P<motor>\w+) is at (?P<c>\d+) C \((?P<sensor>[\w ]+)\)")


def _warm_words(said: str) -> str:
    warm = _WARM.fullmatch(said)
    if warm is None:
        return said
    motors = _WARM_MOTOR.sub(
        lambda m: f"{m['motor'].replace('_', ' ')} motor is at {m['c']} C at its {m['sensor']}", warm["warm"]
    )
    return f"The {warm['side']} arm's {motors}: let it cool below {warm['limit']} C, then try again."


def can_udev_rule(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The udev rule naming each found arm's CAN link after its adapter's serial, and how to install it.

    Without a rule the kernel numbers the links `can0`, `can1` in the order the adapters appear, so a replug or a
    reboot can swap them and drive each arm on the other's link. The rule names a link by its adapter's USB serial
    (`ATTRS{serial}`), which goes with the adapter into any port. An adapter that reports no serial, or one whose
    serial the other adapter shares (a batch of clones), can only be named by its USB port (`KERNELS`), and the reply
    says so: keep it in that port.

    @param arms Per side found so far (`left`, `right`), the arm_adapters() entry of a CAN adapter.
    @return `{"path", "rule", "install", "by_port"}`: where the rule goes, its text, the shell commands that install
        it, and the sides named by port.
    @throws ValueError No arm is given, a side is not left or right, an adapter is not a CAN link, a serial or port
        carries a character a kernel name does not, or both sides are the same adapter.
    """
    if not arms:
        raise ValueError("at least one arm's adapter is needed for the rule")
    sides = [side for side in ("left", "right") if side in arms]
    if len(sides) != len(arms):
        raise ValueError(f"arms are named left and right, got {sorted(arms)}")
    if len(sides) == 2 and arms["left"]["id"] == arms["right"]["id"]:
        raise ValueError("the left and right arm are the same adapter")
    serials = [arms[side]["serial"] for side in sides]
    lines = ["# Written by dreamcli setup: each arm's CAN link, named after its adapter's serial number."]
    by_port: list[str] = []
    for side in sides:
        adapter = arms[side]
        if adapter["kind"] != "can":
            raise ValueError(f"the {side} arm is on {adapter['open']}, not a CAN link; it needs no rule")
        serial = adapter["serial"]
        if serial and serials.count(serial) == 1:
            match = f'ATTRS{{serial}}=="{_kernel_name(serial, side)}"'
        else:
            match = f'KERNELS=="{_kernel_name(adapter["usb"] or "", side)}"'
            by_port.append(side)
        lines.append(f'SUBSYSTEM=="net", ACTION=="add", KERNEL=="can*", {match}, NAME="{CAN_LINKS[side]}"')
    rule = "\n".join(lines) + "\n"
    install = (
        f"sudo tee {UDEV_RULE_PATH} > /dev/null <<'EOF'\n{rule}EOF\n"
        "sudo udevadm control --reload-rules\n"
        "# the names take effect at the next reboot (a replug resets the arms' CAN bus); after it, check them:\n"
        "ip -br link show type can\n"
        "# a link comes up down; bring each up that is not up yet (dreamcli robot up does this too):\n"
        + "".join(_link_up(CAN_LINKS[side]) for side in sides)
    )
    return {"path": UDEV_RULE_PATH, "rule": rule, "install": install, "by_port": by_port}


def _link_up(link: str) -> str:
    """The shell line that brings one CAN link up, skipping it when it is up already.

    `ip link set ... up type can` on a link that is up answers `RTNETLINK answers: Device or resource busy`, which
    reads as a failure though nothing is wrong, so a link whose operstate is `up`
    (or `unknown`, an slcan link's) is skipped. Any other link is taken down before it is brought up: one stuck after
    a bus fault keeps its UP flag with its bus not running, and refuses the bitrate as busy too; on a link
    that is plainly down the down changes nothing. `dreamcli robot up` does the same (`robots.cli.bring_up_links()`).

    @param link The link's name, e.g. `can_follower_l`.
    @return One line, ending in a newline.
    """
    up = f"sudo ip link set {link} down && sudo ip link set {link} up type can bitrate {robstride.BITRATE}"
    return f"grep -qxE 'up|unknown' /sys/class/net/{link}/operstate && echo '{link} is already up' || {{ {up}; }}\n"


def preview_jpeg(source: str, value: str, width: int, height: int, fps: int) -> bytes:
    """One frame from a camera as JPEG bytes, through the same libraries the loop reads it with.

    @param source `"device"` (OpenCV), `"serial"` (librealsense) or `"class"` (a camera class of your own,
        #robots.config.CameraConfig's `source`).
    @param value The device index, path or address, the serial, or the class.
    @param width Frame width to request.
    @param height Frame height to request.
    @param fps Frame rate to request.
    @return The JPEG.
    @throws ValueError The source is unknown, or the camera cannot be opened or delivers no frame.
    @throws ConfigError A camera class does not load.
    """
    import cv2

    ok = False
    frame: Any = None
    if source == "class":
        cfg = config.CameraConfig(name="preview", width=width, height=height, fps=fps, source=value)
        camera = user_class.load(value, "camera")(cfg)
        try:
            for _ in range(PREVIEW_READ_ATTEMPTS):
                ok, frame = camera.read()
                if ok and frame is not None:
                    break
        finally:
            camera.release()
    elif source == "serial":
        try:
            rs = camera_module.RealSenseSource(value, width, height, fps, exposure=config.ExposureConfig())
        except errors.CameraError as exc:
            raise ValueError(str(exc)) from exc
        try:
            give_up = time.monotonic() + camera_module.MANAGED_SETTLE_S
            ok, frame = rs.read()
            state, steady_since = rs.exposure, time.monotonic()
            # Steer exposure as the station does; keep the last good frame once it has held still for PREVIEW_STEADY_S.
            while time.monotonic() < give_up and time.monotonic() - steady_since < PREVIEW_STEADY_S:
                if ok and frame is not None:
                    rs.track(frame)
                if rs.exposure != state:
                    state, steady_since = rs.exposure, time.monotonic()
                ok_next, next_frame = rs.read()
                if ok_next and next_frame is not None:
                    ok, frame = ok_next, next_frame
        except Exception as exc:
            raise ValueError(f"RealSense {value}: {exc}") from exc
        finally:
            rs.release()
    elif source == "device":
        capture = cv2.VideoCapture(int(value) if value.isdigit() else value)
        try:
            if not capture.isOpened():
                raise ValueError(f"{value}: cannot be opened by OpenCV")
            # MJPG, as the station opens it (robots.cameras): uncompressed frames from three cameras overrun the one
            # USB 2 link a MakerMods stand's hub gives them all, and the next camera opened then sends nothing.
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, fps)
            for _ in range(PREVIEW_READ_ATTEMPTS):
                ok, frame = capture.read()
                if ok and frame is not None:
                    break
        finally:
            capture.release()
    else:
        raise ValueError(f"unknown camera source {source!r}; one of {CAMERA_SOURCES}")
    if not ok or frame is None:
        raise ValueError(f"{value}: delivers no frame")
    encoded, data = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not encoded:
        raise ValueError(f"{value}: the frame could not be encoded")
    return bytes(data.tobytes())


def _kernel_name(value: str, side: str) -> str:
    """A serial or USB interface checked to be safe inside a udev rule a root shell writes.

    @param value The serial or interface.
    @param side Which arm, for the message.
    @return The value.
    @throws ValueError It is empty or carries another character.
    """
    if not _DEVICE_NAME_RE.fullmatch(value):
        raise ValueError(f"the {side} arm's adapter has no usable serial or port name ({value!r})")
    return value


def _usb_identity(device: pathlib.Path) -> tuple[str | None, str | None]:
    """The USB interface a sysfs `device` link points at, and its USB device's serial.

    @param device A `.../device` link in sysfs (of a net link, a tty, a video node).
    @return `(interface, serial)`: the interface's kernel name (`1-2:1.0`) and the serial of the USB device above
        it, each None when sysfs does not say (not USB, no serial).
    """
    usb = _usb_device(device)
    return usb["interface"], usb["serial"]


def _usb_device(device: pathlib.Path) -> dict[str, str | None]:
    found: dict[str, str | None] = dict.fromkeys(("interface", "serial", "vendor", "product_id", "product"))
    if not device.exists():
        return found
    path = pathlib.Path(os.path.realpath(device))
    chain = [path, *path.parents]
    usb = next((i for i, folder in enumerate(chain) if (folder / "idVendor").is_file()), None)
    if usb is None:
        found["interface"], found["serial"] = path.name, _sysfs_text(path.parent / "serial")
        return found
    found["interface"] = chain[usb - 1].name if usb > 0 else path.name
    for key, name in (
        ("serial", "serial"),
        ("vendor", "idVendor"),
        ("product_id", "idProduct"),
        ("product", "product"),
    ):
        found[key] = _sysfs_text(chain[usb] / name)
    return found


def _sysfs_text(path: pathlib.Path) -> str | None:
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


def _video_nodes() -> list[dict[str, Any]]:
    """Every V4L2 node with its name, index, USB device and serial, and its by-id and by-path links.

    @return `[{"node", "name", "index", "usb_device", "serial", "by_id", "by_path"}]`, one per `/dev/videoN`.
    """
    if not VIDEO_CLASS_DIR.is_dir():
        return []
    links: dict[str, dict[str, str]] = {"by_id": {}, "by_path": {}}
    for key, folder in (("by_id", VIDEO_BY_ID_DIR), ("by_path", VIDEO_BY_PATH_DIR)):
        if folder.is_dir():
            for link in sorted(folder.iterdir()):
                # A port has a `usb-` and a `usbv2-` spelling of one node; the first in name order is kept.
                links[key].setdefault(os.path.basename(os.path.realpath(link)), str(link))
    found: list[dict[str, Any]] = []
    for entry in sorted(VIDEO_CLASS_DIR.iterdir()):
        try:
            name = (entry / "name").read_text().strip()
            index = int((entry / "index").read_text().strip())
        except (OSError, ValueError):
            continue
        device = entry / "device"
        usb_device = str(pathlib.Path(os.path.realpath(device)).parent) if device.exists() else entry.name
        _interface, serial = _usb_identity(device)
        found.append(
            {
                "node": f"/dev/{entry.name}",
                "name": name,
                "index": index,
                "usb_device": usb_device,
                "serial": serial,
                "by_id": links["by_id"].get(entry.name),
                "by_path": links["by_path"].get(entry.name),
            }
        )
    return found


def _realsense_devices() -> list[dict[str, Any]]:
    """The RealSense cameras librealsense sees, when pyrealsense2 is installed.

    @return `[{"serial", "name", "usb"}]`, one per device, `usb` its link (`"3.2"`) or None when it does not say;
        empty without the package.
    """
    try:
        import pyrealsense2 as rs  # type: ignore[import-not-found,unused-ignore]
    except ImportError:
        return []
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        devices = rs.context().query_devices()
    except Exception as exc:
        _log.warning("setup: librealsense could not enumerate devices: %s", exc)
        return []
    for dev in devices:
        serial = dev.get_info(rs.camera_info.serial_number)
        if serial in seen:
            continue
        seen.add(serial)
        usb = rs.camera_info.usb_type_descriptor
        found.append(
            {
                "serial": serial,
                "name": dev.get_info(rs.camera_info.name),
                "usb": dev.get_info(usb) if dev.supports(usb) else None,
            }
        )
    return found


def _usb2_warning(usb: str | None) -> str | None:
    if usb is None or not usb.startswith("2"):
        return None
    return f"On a USB {usb} link, too slow for 1280x720 at 30 fps: {camera_module.USB3_FIX}."

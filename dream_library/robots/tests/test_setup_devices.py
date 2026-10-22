"""What the setup wizard sees on the box (`robots.setup_devices`), read from a fake sysfs and /dev tree.

What is pinned: every arm adapter gets an id from its serial number (a CAN link's adapter, a USB serial adapter by its
by-id path), and only an adapter without one, or with one another adapter shares, falls back to its port; a CANable is
one entry whether or not an `slcand` made a link of it, and the link is not listed again; a failure reads as what
happened and what to do (explain()); a camera with a serial of its own is named by
its by-id node, one without by its USB port, a RealSense by its serial; the udev rule names each arm's CAN link after
its adapter's serial, falls back to the port for a missing or shared serial and says so, and refuses what a root
shell should not see.
"""

import errno
import pathlib
import sys
import types as pytypes
from typing import Any

import pytest
import setup_fakes

from robots import cameras, errors, robstride, setup_devices


def usb_interface(
    root: pathlib.Path, name: str, serial: str | None, ids: tuple[str, str, str] | None = None
) -> pathlib.Path:
    """A USB interface directory under a USB device that has `serial` (or none) and, when given, its vendor id,
    product id and product name."""
    usb_device = root / "devices" / name.split(":")[0]
    interface = usb_device / name
    interface.mkdir(parents=True, exist_ok=True)
    if serial is not None:
        (usb_device / "serial").write_text(serial + "\n")
    if ids is not None:
        for attr, value in zip(("idVendor", "idProduct", "product"), ids, strict=True):
            (usb_device / attr).write_text(value + "\n")
    return interface


@pytest.fixture
def tree(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Two CAN adapters with serials, a CAN link with no adapter, a non-CAN link, a CANable opened directly, one an
    slcand holds as `can3` (a link with no `device` and operstate `unknown`, as slcan's may), an FTDI serial adapter,
    two webcams, a RealSense."""
    net = tmp_path / "net"
    for name, kind, iface, serial in (
        ("can0", "280", "1-2:1.0", "AAA111"),
        ("can1", "280", "1-3:1.0", "BBB222"),
        ("can2", "280", None, None),
        ("can3", "280", None, None),
        ("eth0", "1", None, None),
    ):
        setup_fakes.net_link(net, name, up=True, operstate="unknown" if name == "can3" else None, kind=kind)
        if iface is not None:
            (net / name / "device").symlink_to(usb_interface(tmp_path, iface, serial))
    dev = tmp_path / "dev"
    dev.mkdir()
    by_id, by_path = tmp_path / "serial-by-id", tmp_path / "serial-by-path"
    by_id.mkdir()
    by_path.mkdir()
    tty = tmp_path / "tty"
    ftdi = usb_interface(tmp_path, "1-8:1.0", "FTBENQYQ", ("0403", "6001", "FT232R USB UART")) / "ttyUSB0"
    ftdi.mkdir()
    for node, by_id_name, device, by_path_name in (
        ("ttyACM0", "usb-CANable_XYZ-if00", usb_interface(tmp_path, "1-4:1.0", "XYZ"), "usb-0:4:1.0"),
        (
            "ttyACM1",
            "usb-Protofusion_Labs_DEF-if00",
            usb_interface(tmp_path, "1-9:1.0", "DEF", ("ad50", "60c4", "x")),
            "usb-0:9:1.0",
        ),
        ("ttyUSB0", "usb-FTDI_FT232R_USB_UART_FTBENQYQ-if00-port0", ftdi, "usb-0:8:1.0-port0"),
    ):
        (dev / node).write_text("")
        (by_id / by_id_name).symlink_to(dev / node)
        (by_path / f"pci-0000:00:14.0-{by_path_name}").symlink_to(dev / node)
        (tty / node).mkdir(parents=True)
        (tty / node / "device").symlink_to(device)
    proc = tmp_path / "proc"
    setup_fakes.slcand(proc, 812, "-o", "-f", "-s8", str(by_id / "usb-Protofusion_Labs_DEF-if00"), "can3")
    monkeypatch.setattr(robstride, "PROC_DIR", proc)
    video = tmp_path / "video4linux"
    for node, name, index, iface, serial in (
        ("video0", "Arducam", 0, "1-5:1.0", "CAM1"),
        ("video1", "Arducam", 1, "1-5:1.0", "CAM1"),
        ("video2", "Generic", 0, "1-6:1.0", None),
        ("video4", "Intel(R) RealSense(TM) Depth Camera 405", 0, "1-7:1.0", "RS1"),
    ):
        (video / node).mkdir(parents=True)
        (video / node / "name").write_text(name + "\n")
        (video / node / "index").write_text(f"{index}\n")
        (video / node / "device").symlink_to(usb_interface(tmp_path, iface, serial))
        (dev / node).write_text("")
    v4l_id, v4l_path = tmp_path / "v4l-by-id", tmp_path / "v4l-by-path"
    v4l_id.mkdir()
    v4l_path.mkdir()
    (v4l_id / "usb-Arducam_CAM1-video-index0").symlink_to(dev / "video0")
    (v4l_id / "usb-Arducam_CAM1-video-index1").symlink_to(dev / "video1")
    (v4l_path / "pci-0000:00:14.0-usb-0:5:1.0-video-index0").symlink_to(dev / "video0")
    (v4l_path / "pci-0000:00:14.0-usbv2-0:5:1.0-video-index0").symlink_to(dev / "video0")
    (v4l_path / "pci-0000:00:14.0-usb-0:6:1.0-video-index0").symlink_to(dev / "video2")
    for attr, path in (
        ("NET_DIR", net),
        ("SERIAL_BY_ID_DIR", by_id),
        ("SERIAL_BY_PATH_DIR", by_path),
        ("TTY_CLASS_DIR", tty),
        ("VIDEO_CLASS_DIR", video),
        ("VIDEO_BY_ID_DIR", v4l_id),
        ("VIDEO_BY_PATH_DIR", v4l_path),
    ):
        monkeypatch.setattr(setup_devices, attr, path)
    monkeypatch.setattr(setup_devices, "_realsense_devices", lambda: [{"serial": "RS1", "name": "D405", "usb": "3.2"}])
    return tmp_path


def test_arm_adapters_are_named_by_serial_and_only_fall_back_to_the_port(tree: pathlib.Path) -> None:
    found = setup_devices.scan()
    arms = {arm["id"]: arm for arm in found["arms"]}
    ftdi = "serial-usb-FTDI_FT232R_USB_UART_FTBENQYQ-if00-port0"
    assert [arm["id"] for arm in found["arms"]] == [
        "can-AAA111",
        "can-BBB222",
        "can-port-can2",
        "slcan-DEF",
        "slcan-XYZ",
        ftdi,
    ], "CAN first, each CANable once (can3 is DEF's slcand link), the serial adapter last"
    assert arms["can-AAA111"]["open"] == "can0" and arms["can-AAA111"]["usb"] == "1-2:1.0"
    assert arms["can-port-can2"]["label"] == "CAN link can2, no USB adapter behind it"
    assert arms["slcan-XYZ"]["open"] == str(tree / "serial-by-id" / "usb-CANable_XYZ-if00")
    assert (arms["slcan-XYZ"]["serial"], arms["slcan-XYZ"]["bus"]) == ("XYZ", "can")
    assert arms["slcan-XYZ"]["detail"] == "opened directly over USB" and arms["slcan-XYZ"]["via"] is None
    assert (arms["slcan-DEF"]["via"], arms["slcan-DEF"]["state"]) == ("can3", "up")
    assert arms["slcan-DEF"]["detail"] == "through slcand as CAN link can3, up"
    assert (arms[ftdi]["bus"], arms[ftdi]["serial"], arms[ftdi]["usb"]) == ("serial", "FTBENQYQ", "1-8:1.0")
    assert arms[ftdi]["label"] == "USB serial adapter FT232R USB UART FTBENQYQ"
    assert [link["name"] for link in found["can"]] == ["can0", "can1", "can2", "can3"]


def test_adapters_that_share_a_serial_or_have_none_are_named_by_their_usb_port(tree: pathlib.Path) -> None:
    """Both Damiao USB2CAN bridges on a reBot report serial 00000000050C, so udev makes one by-id
    link, pointing at whichever came up last: named by it, the wizard would find one adapter of two and a replug could
    swap the arms. Each is named by its USB port, as is an adapter with no serial and a CAN adapter whose serial
    another shares; a `usbv2-` spelling or a local alias is not a name."""
    dev, by_id, by_path, tty = (tree / name for name in ("dev", "serial-by-id", "serial-by-path", "tty"))
    damiao = ("2e88", "4603", "HDSC CDC Device")
    shared = "00000000050C"
    for node, port, serial in (("ttyACM2", "10", shared), ("ttyACM3", "11", shared), ("ttyACM4", "12", None)):
        (dev / node).write_text("")
        (tty / node).mkdir()
        (tty / node / "device").symlink_to(usb_interface(tree, f"1-{port}:1.0", serial, damiao))
        (by_path / f"pci-0000:00:14.0-usb-0:{port}:1.0").symlink_to(dev / node)
        (by_path / f"pci-0000:00:14.0-usbv2-0:{port}:1.0").symlink_to(dev / node)
    (by_id / "usb-HDSC_CDC_Device_00000000050C-if00").symlink_to(dev / "ttyACM3")
    (by_path / "arm-left").symlink_to(dev / "ttyACM2")
    setup_fakes.net_link(tree / "net", "can4", up=True)
    (tree / "net" / "can4" / "device").symlink_to(usb_interface(tree, "1-13:1.0", "AAA111"))  # can0's serial

    adapters = {pathlib.Path(adapter["tty"]).name: adapter for adapter in setup_devices.serial_adapters()}
    arms = {arm["open"]: arm for arm in setup_devices.scan()["arms"]}

    for node, port in (("ttyACM2", "10"), ("ttyACM3", "11"), ("ttyACM4", "12")):
        path = str(by_path / f"pci-0000:00:14.0-usb-0:{port}:1.0")
        assert (adapters[node]["path"], adapters[node]["bound"]) == (path, "port")
        assert arms[path]["id"] == f"serial-pci-0000:00:14.0-usb-0:{port}:1.0"
    assert arms[adapters["ttyACM2"]["path"]]["label"] == (
        "USB serial adapter HDSC CDC Device 00000000050C, in USB port 1-10:1.0 (keep it there)"
    )
    no_serial = "USB serial adapter HDSC CDC Device, in USB port 1-12:1.0 (keep it there)"
    assert arms[adapters["ttyACM4"]["path"]]["label"] == no_serial
    assert arms["can0"]["id"] == "can-port-1-2:1.0" and arms["can4"]["id"] == "can-port-1-13:1.0"
    assert arms["can4"]["label"] == "CAN adapter AAA111, in USB port 1-13:1.0 (keep it there)"
    ftdi = adapters["ttyUSB0"]
    assert (ftdi["path"], ftdi["bound"]) == (str(by_id / "usb-FTDI_FT232R_USB_UART_FTBENQYQ-if00-port0"), "serial")


def test_slcand_command_lines_are_read_whatever_the_option_order(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    proc = tmp_path / "proc"
    setup_fakes.slcand(proc, 7, "-o", "-s", "8", "-t", "hw", "ttyACM4", "can_x")
    setup_fakes.slcand(proc, 5, "-of", "-s8", "/dev/ttyACM5")
    (proc / "6").mkdir()
    (proc / "6" / "comm").write_text("bash\n")
    monkeypatch.setattr(robstride, "PROC_DIR", proc)
    assert robstride.slcan_daemons() == [
        robstride.SlcanDaemon(5, "/dev/ttyACM5", None),
        robstride.SlcanDaemon(7, "/dev/ttyACM4", "can_x"),
    ]
    assert robstride.slcan_holder("/dev/ttyACM4") == robstride.SlcanDaemon(7, "/dev/ttyACM4", "can_x")
    assert robstride.slcan_holder("/dev/ttyACM9") is None


def oserror(code: int) -> OSError:
    return OSError(code, "x")


def wrapped(cause: BaseException) -> Exception:
    """As python-can raises: its own error, the OS's underneath."""
    try:
        raise cause
    except BaseException as exc:
        try:
            raise RuntimeError("Error Code 100") from exc
        except RuntimeError as outer:
            return outer


@pytest.mark.parametrize(
    ("failure", "words"),
    [
        (wrapped(oserror(errno.ENETDOWN)), "CAN link is down"),
        (oserror(errno.ENODEV), "The adapter is gone"),
        (oserror(errno.EACCES), "usermod -aG dialout"),
        (Exception("[Errno 16] could not exclusively lock port"), "Another program has this adapter open"),
        (Exception("could not open port /dev/serial/by-id/x"), "it was unplugged"),
        (errors.ArmError("servo 3 (elbow_flex) did not answer"), "servo 3 (elbow_flex) did not answer"),
        (errors.ConfigError("the calibration file /x.json does not load: [Errno 2] No such file"), "/x.json does not"),
        (TimeoutError("no reply"), "motors did not answer"),
        (ValueError("odd"), "ValueError: odd"),
    ],
)
def test_a_failure_reads_as_what_happened_and_what_to_do(failure: BaseException, words: str) -> None:
    assert words in setup_devices.explain(failure)


def test_cameras_are_named_by_serial_unless_they_have_none(tree: pathlib.Path) -> None:
    cams = setup_devices.cameras()
    assert cams[0] == {
        "source": "serial",
        "value": "RS1",
        "label": "D405 RS1",
        "bound": "serial",
        "usb": "3.2",
        "usb_warning": None,
    }
    by_value = {cam["value"]: cam for cam in cams[1:]}
    arducam = str(tree / "v4l-by-id" / "usb-Arducam_CAM1-video-index0")
    generic = str(tree / "v4l-by-path" / "pci-0000:00:14.0-usb-0:6:1.0-video-index0")
    assert sorted(by_value) == sorted([arducam, generic])
    assert by_value[arducam]["bound"] == "serial" and by_value[generic]["bound"] == "port"


def test_a_realsense_says_its_usb_link_and_one_on_usb2_says_it_needs_a_usb3_port_and_cable(
    tree: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The camera step shows each RealSense's USB link; on USB2 it warns before the open fails."""
    monkeypatch.setattr(
        setup_devices,
        "_realsense_devices",
        lambda: [{"serial": "RS1", "name": "D405", "usb": "2.1"}, {"serial": "RS2", "name": "D405", "usb": None}],
    )

    cams = {cam["value"]: cam for cam in setup_devices.cameras() if cam["source"] == "serial"}

    assert cams["RS1"]["usb"] == "2.1"
    assert "USB 2.1" in cams["RS1"]["usb_warning"] and cameras.USB3_FIX in cams["RS1"]["usb_warning"]
    assert cams["RS2"]["usb"] is None and cams["RS2"]["usb_warning"] is None, "an unknown link warns of nothing"


def test_the_realsense_scan_reads_each_camera_s_usb_link(monkeypatch: pytest.MonkeyPatch) -> None:
    """librealsense's `usb_type_descriptor` per device; a device that does not report one has None."""
    info = pytypes.SimpleNamespace(serial_number="serial", name="name", usb_type_descriptor="usb")

    class Device:
        def __init__(self, serial: str, usb: str | None) -> None:
            self.values = {"serial": serial, "name": "Intel RealSense D405", "usb": usb}

        def supports(self, key: str) -> bool:
            return self.values.get(key) is not None

        def get_info(self, key: str) -> str:
            return str(self.values[key])

    devices = [Device("RS1", "2.1"), Device("RS2", None)]
    fake_rs = pytypes.ModuleType("pyrealsense2")
    fake_rs.camera_info = info  # type: ignore[attr-defined]
    fake_rs.context = lambda: pytypes.SimpleNamespace(query_devices=lambda: devices)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)

    assert setup_devices._realsense_devices() == [
        {"serial": "RS1", "name": "Intel RealSense D405", "usb": "2.1"},
        {"serial": "RS2", "name": "Intel RealSense D405", "usb": None},
    ]


def test_two_cameras_reporting_one_serial_are_both_named_by_their_usb_port(tree: pathlib.Path) -> None:
    video, dev, v4l_path = tree / "video4linux", tree / "dev", tree / "v4l-by-path"
    (video / "video6").mkdir()
    (video / "video6" / "name").write_text("Arducam\n")
    (video / "video6" / "index").write_text("0\n")
    (video / "video6" / "device").symlink_to(usb_interface(tree, "1-10:1.0", "CAM1"))
    (dev / "video6").write_text("")
    (v4l_path / "pci-0000:00:14.0-usb-0:10:1.0-video-index0").symlink_to(dev / "video6")

    arducams = [cam for cam in setup_devices.cameras() if cam["label"] == "Arducam"]

    assert [(pathlib.Path(cam["value"]).parent.name, cam["bound"]) for cam in arducams] == [
        ("v4l-by-path", "port"),
        ("v4l-by-path", "port"),
    ]
    assert len({cam["value"] for cam in arducams}) == 2


def adapter(serial: str | None, usb: str = "1-2:1.0", kind: str = "can", ident: str | None = None) -> dict[str, Any]:
    return {"id": ident or f"can-{serial or usb}", "kind": kind, "open": "can0", "serial": serial, "usb": usb}


def rule_lines(made: dict[str, Any]) -> list[str]:
    return [line for line in made["rule"].splitlines() if not line.startswith("#")]


def test_the_udev_rule_names_each_arm_after_its_adapters_serial() -> None:
    made = setup_devices.can_udev_rule({"left": adapter("AAA111"), "right": adapter("BBB222", "1-3:1.0")})
    assert rule_lines(made) == [
        'SUBSYSTEM=="net", ACTION=="add", KERNEL=="can*", ATTRS{serial}=="AAA111", NAME="can_follower_l"',
        'SUBSYSTEM=="net", ACTION=="add", KERNEL=="can*", ATTRS{serial}=="BBB222", NAME="can_follower_r"',
    ]
    assert made["by_port"] == [] and made["path"] in made["install"] and "udevadm control" in made["install"]
    assert "sudo ip link set can_follower_r up type can bitrate 1000000" in made["install"], "it comes up down"
    # An up link is skipped, not set up again into "RTNETLINK answers: Device or resource busy"; a link that is not
    # up is taken down first, since one stuck after a bus fault keeps its UP flag and refuses the bitrate as busy too.
    assert (
        "grep -qxE 'up|unknown' /sys/class/net/can_follower_r/operstate && echo 'can_follower_r is already up' || "
        "{ sudo ip link set can_follower_r down && sudo ip link set can_follower_r up type can bitrate 1000000; }\n"
    ) in made["install"]
    one = setup_devices.can_udev_rule({"right": adapter("BBB222")})
    assert len(rule_lines(one)) == 1 and "next reboot" in one["install"] and "unplug" not in one["install"]


def test_a_missing_or_shared_serial_falls_back_to_the_port_and_says_so() -> None:
    shared = setup_devices.can_udev_rule(
        {"left": adapter("CLONE", "1-2:1.0", ident="a"), "right": adapter("CLONE", "1-3:1.0", ident="b")}
    )
    assert [line.split(", ")[3] for line in rule_lines(shared)] == ['KERNELS=="1-2:1.0"', 'KERNELS=="1-3:1.0"']
    assert shared["by_port"] == ["left", "right"]
    missing = setup_devices.can_udev_rule({"left": adapter(None, "1-2:1.0"), "right": adapter("BBB222", "1-3:1.0")})
    assert missing["by_port"] == ["left"] and 'ATTRS{serial}=="BBB222"' in missing["rule"]


@pytest.mark.parametrize(
    ("arms", "message"),
    [
        ({}, "at least one"),
        ({"left": adapter('A"; rm -rf /')}, "no usable serial"),
        ({"left": adapter("A"), "right": adapter("A")}, "same adapter"),
        ({"left": adapter("A", kind="serial")}, "not a CAN link"),
        ({"middle": adapter("A")}, "left and right"),
    ],
)
def test_the_udev_rule_refuses_what_a_root_shell_should_not_see(arms: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        setup_devices.can_udev_rule(arms)


def test_no_sysfs_is_nothing_found(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for attr in ("NET_DIR", "SERIAL_BY_ID_DIR", "SERIAL_BY_PATH_DIR", "VIDEO_CLASS_DIR"):
        monkeypatch.setattr(setup_devices, attr, tmp_path / "nope")
    monkeypatch.setattr(setup_devices, "_realsense_devices", lambda: [])
    assert setup_devices.scan() == {"arms": [], "cameras": [], "can": []}


def test_preview_reports_a_camera_that_cannot_be_opened() -> None:
    with pytest.raises(ValueError, match="no-such-device"):
        setup_devices.preview_jpeg("device", "/dev/v4l/no-such-device", 64, 48, 30)
    with pytest.raises(ValueError, match="unknown camera source"):
        setup_devices.preview_jpeg("usb", "x", 64, 48, 30)


def test_an_arm_the_scan_misses_is_typed_and_opened_as_written() -> None:
    """ "Don't see your arm?": any address the robot's driver opens, an IP address or a link the scan cannot see,
    becomes an arm device like a found one, opened as written."""
    assert setup_devices.typed_arm(" 192.168.1.20 ") == {
        "id": "typed-192.168.1.20",
        "kind": "typed",
        "bus": "typed",
        "open": "192.168.1.20",
        "serial": None,
        "usb": None,
        "label": "192.168.1.20",
        "detail": "Typed: your arm's driver opens it as written",
        "state": None,
        "via": None,
    }
    for bad in ("  ", "can2\ncan3"):
        with pytest.raises(ValueError, match="type your arm's address on one line"):
            setup_devices.typed_arm(bad)


def test_a_camera_the_scan_misses_is_an_address_opencv_opens_or_a_class(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Don't see your camera?": `module:Class` or `file.py:Class` is a camera class of your own (a file made
    absolute from here); anything else, a stream address with its port among them, is an OpenCV device."""
    monkeypatch.chdir(tmp_path)
    for address in ("rtsp://192.168.1.50:554/stream1", "/dev/video4", "4", "192.168.1.50:8080"):
        assert setup_devices.typed_camera(f" {address} ") == {
            "source": "device",
            "value": address,
            "label": address,
            "bound": "typed",
        }
    assert setup_devices.typed_camera("vendor.cams:Zed")["value"] == "vendor.cams:Zed"
    assert setup_devices.typed_camera("my_cams.py:Zed") == {
        "source": "class",
        "value": f"{tmp_path.resolve() / 'my_cams.py'}:Zed",
        "label": f"Your camera class {tmp_path.resolve() / 'my_cams.py'}:Zed",
        "bound": "typed",
    }
    with pytest.raises(ValueError, match="type your camera's address"):
        setup_devices.typed_camera(" ")


def test_preview_takes_a_picture_through_a_camera_class_and_releases_it(tmp_path: pathlib.Path) -> None:
    (tmp_path / "preview_cams.py").write_text(
        "import numpy as np\n"
        "released = []\n"
        "class Cam:\n"
        "    def __init__(self, cfg):\n"
        "        self.size = (cfg.height, cfg.width, 3)\n"
        "    def read(self):\n"
        "        return True, np.full(self.size, 128, dtype=np.uint8)\n"
        "    def release(self):\n"
        "        released.append(True)\n"
    )
    jpeg = setup_devices.preview_jpeg("class", f"{tmp_path / 'preview_cams.py'}:Cam", 64, 48, 30)
    assert jpeg[:2] == b"\xff\xd8"
    assert sys.modules["_robots_camera_preview_cams"].released == [True]


class SteppingRealSense:
    """A RealSense at 30 fps whose managed exposure steps 0.7x every 0.5 s from 14 ms while it is at least
    `floor_us`; a new exposure reaches the sensor three frames later, and a frame is white while the sensor's is
    over 1 ms. Time is the fake `now_s`, which each read advances by a frame."""

    def __init__(self, floor_us: float, exposure: Any) -> None:
        self.floor_us, self.config, self.now_s, self.next_step_s, self.last_step_s = floor_us, exposure, 0.0, 0.0, 0.0
        self.exposure_us, self.applied, self.released = 14_000.0, [14_000.0], False

    @property
    def exposure(self) -> dict[str, float]:
        return {"exposure_us": self.exposure_us}

    def track(self, frame: Any) -> None:
        if self.now_s >= self.next_step_s and self.exposure_us >= self.floor_us:
            self.exposure_us, self.next_step_s, self.last_step_s = self.exposure_us * 0.7, self.now_s + 0.5, self.now_s

    def read(self) -> tuple[bool, Any]:
        import numpy as np

        self.now_s += 1 / 30
        self.applied.append(self.exposure_us)
        sensor_us = self.applied[max(0, len(self.applied) - 4)]
        return True, np.full((48, 64, 3), 255 if sensor_us > 1_000 else 90, dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def _stepping_preview(monkeypatch: pytest.MonkeyPatch, floor_us: float) -> tuple[SteppingRealSense, float, int]:
    """One RealSense preview of a SteppingRealSense: the camera, the seconds the preview took, its picture's mean."""
    import cv2
    import numpy as np

    opened: list[SteppingRealSense] = []

    def open_camera(*args: Any, **kwargs: Any) -> SteppingRealSense:
        opened.append(SteppingRealSense(floor_us, kwargs["exposure"]))
        return opened[0]

    monkeypatch.setattr(cameras, "RealSenseSource", open_camera)
    monkeypatch.setattr(setup_devices, "time", pytypes.SimpleNamespace(monotonic=lambda: opened[0].now_s))
    jpeg = setup_devices.preview_jpeg("serial", "123", 64, 48, 30)
    picture = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert picture is not None
    assert opened[0].released and opened[0].config.managed, "the preview runs managed exposure and closes the camera"
    return opened[0], opened[0].now_s, int(picture.mean())


def test_a_realsense_preview_keeps_a_frame_once_managed_exposure_settles(monkeypatch: pytest.MonkeyPatch) -> None:
    """14 ms to under 1 ms takes eight 0.7x steps (3.5 s), longer than a fixed 2 s wait: the preview keeps a
    frame only after the exposure has held still for PREVIEW_STEADY_S, not the white frames before."""
    camera, took_s, mean = _stepping_preview(monkeypatch, 1_000)
    assert camera.exposure_us < 1_000 and camera.last_step_s >= 3.5
    assert abs(mean - 90) < 5, "the preview kept a frame from before the exposure settled"
    assert setup_devices.PREVIEW_STEADY_S <= took_s - camera.last_step_s < setup_devices.PREVIEW_STEADY_S + 0.1


def test_a_realsense_preview_gives_up_at_the_managed_settle_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exposure that never holds still: the preview keeps its last frame at cameras.MANAGED_SETTLE_S."""
    _, took_s, _ = _stepping_preview(monkeypatch, 0)
    assert cameras.MANAGED_SETTLE_S <= took_s < cameras.MANAGED_SETTLE_S + 0.1

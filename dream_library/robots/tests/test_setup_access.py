"""The wizard's edges: the one-time token on every request, the printed address, and the robot record per box."""

import http.client
import io
import pathlib
from collections.abc import Iterator

import pytest
from dream import credentials

from robots import errors, setup_access, setup_wizard

TOKEN = "tok-abc123"
BOOKMARKED = "Bookmark_me-0123456789"


@pytest.fixture
def guarded(tmp_path: pathlib.Path) -> Iterator[setup_wizard.SetupServer]:
    """A wizard on an ephemeral port that wants TOKEN."""
    account = credentials.Credentials(
        gateway_url="https://api.example.com",
        platform_url="https://platform.example.com",
        data_url="https://data.example.com",
        api_key="dl_live_test",
        path=tmp_path / "credentials",
    )
    server = setup_wizard.SetupServer(tmp_path / "station.yaml", account, host="127.0.0.1", port=0, token=TOKEN)
    server.start()
    try:
        yield server
    finally:
        server.stop()


def send(
    server: setup_wizard.SetupServer, method: str, path: str, headers: dict[str, str] | None = None, body: bytes = b""
) -> http.client.HTTPResponse:
    """One request without following redirects."""
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    connection.request(method, path, body=body or None, headers=headers or {})
    reply = connection.getresponse()
    reply.read()
    return reply


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/"),
        ("GET", "/api/state"),
        ("GET", "/api/scan"),
        ("GET", "/api/status"),
        ("POST", "/api/save"),
        ("POST", "/api/arms/open"),
        ("POST", "/api/exit"),
    ],
)
def test_a_request_without_the_token_is_403_and_does_nothing(
    guarded: setup_wizard.SetupServer, method: str, path: str
) -> None:
    reply = send(guarded, method, path, body=b"{}" if method == "POST" else b"")
    assert reply.status == 403
    assert not guarded.done.is_set(), "an unauthenticated exit must not end the wizard"
    assert not guarded.config_path.exists()


def test_a_wrong_token_in_the_address_or_cookie_or_header_is_403(guarded: setup_wizard.SetupServer) -> None:
    assert send(guarded, "GET", "/?token=nope").status == 403
    assert send(guarded, "GET", "/api/state", {"Cookie": f"{setup_access.TOKEN_COOKIE}=nope"}).status == 403
    assert send(guarded, "GET", "/api/state", {setup_access.TOKEN_HEADER: "nope"}).status == 403
    assert send(guarded, "POST", f"/api/exit?token={TOKEN}", body=b"{}").status == 403, "a POST needs cookie or header"


def test_a_browser_opening_an_address_it_may_not_have_gets_a_page_in_words(guarded: setup_wizard.SetupServer) -> None:
    browser = {"Accept": "text/html,application/xhtml+xml"}
    for path, headers, status in (("/", browser, 403), ("/nope", {**browser, setup_access.TOKEN_HEADER: TOKEN}, 404)):
        connection = http.client.HTTPConnection("127.0.0.1", guarded.port, timeout=10)
        connection.request("GET", path, headers=headers)
        reply = connection.getresponse()
        page = reply.read().decode("utf-8")
        assert reply.status == status and reply.getheader("Content-Type", "").startswith("text/html")
        assert "<h1>Dream setup</h1>" in page and "`dreamcli setup` printed" in page and "{" not in page
    assert send(guarded, "GET", "/api/state", browser).getheader("Content-Type") == "application/json"


def test_the_first_visit_sets_the_cookie_drops_the_token_from_the_address_and_then_everything_works(
    guarded: setup_wizard.SetupServer,
) -> None:
    first = send(guarded, "GET", f"/?token={TOKEN}")
    assert first.status == 303 and first.getheader("Location") == "/"
    cookie = first.getheader("Set-Cookie") or ""
    assert f"{setup_access.TOKEN_COOKIE}={TOKEN}" in cookie and "HttpOnly" in cookie and "SameSite=Strict" in cookie
    pair = cookie.split(";")[0]
    assert send(guarded, "GET", "/", {"Cookie": pair}).status == 200
    assert send(guarded, "GET", "/api/state", {"Cookie": pair}).status == 200
    assert send(guarded, "GET", f"/api/state?x=1&token={TOKEN}").getheader("Location") == "/api/state?x=1"
    assert send(guarded, "GET", "/api/state", {setup_access.TOKEN_HEADER: TOKEN}).status == 200


def test_no_token_configured_serves_as_before(tmp_path: pathlib.Path) -> None:
    account = credentials.Credentials(
        "https://a.example.com", "https://p.example.com", "https://d.example.com", "k", tmp_path
    )
    server = setup_wizard.SetupServer(tmp_path / "s.yaml", account, host="127.0.0.1", port=0)
    server.start()
    try:
        assert send(server, "GET", "/api/state").status == 200
    finally:
        server.stop()


def test_main_prints_an_address_with_the_token_and_the_server_wants_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credentials_file = tmp_path / "credentials"
    credentials_file.write_text("DREAM_API_KEY=dl_live_test\n", encoding="utf-8")
    monkeypatch.setenv("DREAM_CREDENTIALS", str(credentials_file))
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.setattr(setup_access, "listen_addresses", lambda: ["192.168.1.21"])
    monkeypatch.setattr(setup_access, "tailnet_name", lambda: "box.tailnet.ts.net")
    seen: dict[str, str] = {}
    original = setup_wizard.SetupServer.start

    def start_then_stop(self: setup_wizard.SetupServer) -> None:
        original(self)
        assert self.token
        seen["token"] = self.token
        seen["status"] = str(send(self, "GET", "/api/state").status)
        self.done.set()

    monkeypatch.setattr(setup_wizard.SetupServer, "start", start_then_stop)
    out = io.StringIO()
    assert setup_wizard.main(tmp_path / "station.yaml", host="127.0.0.1", port=0, browser=False, out=out) == 0
    text = out.getvalue()
    assert seen["status"] == "403"
    assert f"?token={seen['token']}" in text and "(tailnet)" not in text, "bound to loopback: no tailnet line"
    assert "http://127.0.0.1:" in text


def with_key(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A credentials file with a key, as main() reads it, and a shell that is not over ssh."""
    credentials_file = tmp_path / "credentials"
    credentials_file.write_text("DREAM_API_KEY=dl_live_test\n", encoding="utf-8")
    monkeypatch.setenv("DREAM_CREDENTIALS", str(credentials_file))
    monkeypatch.delenv("SSH_CONNECTION", raising=False)


def test_the_token_variable_is_the_token_so_a_bookmarked_address_opens_the_page(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The token can be set in the environment instead of pasted fresh each time."""
    with_key(tmp_path, monkeypatch)
    monkeypatch.setenv(setup_access.TOKEN_ENV, BOOKMARKED)
    seen: dict[str, str] = {}
    original = setup_wizard.SetupServer.start

    def start_then_stop(self: setup_wizard.SetupServer) -> None:
        original(self)
        first = send(self, "GET", f"/?token={BOOKMARKED}")
        seen["first"] = str(first.status)
        seen["cookie"] = (first.getheader("Set-Cookie") or "").split(";")[0]
        seen["with cookie"] = str(send(self, "GET", "/api/state", {"Cookie": seen["cookie"]}).status)
        seen["without"] = str(send(self, "GET", "/api/state").status)
        self.done.set()

    monkeypatch.setattr(setup_wizard.SetupServer, "start", start_then_stop)
    for _ in range(2):
        out = io.StringIO()
        assert setup_wizard.main(tmp_path / "station.yaml", host="127.0.0.1", port=0, browser=False, out=out) == 0
        assert f"/?token={BOOKMARKED}" in out.getvalue(), "every run prints the same address"
        assert seen == {
            "first": "303",
            "cookie": f"{setup_access.TOKEN_COOKIE}={BOOKMARKED}",
            "with cookie": "200",
            "without": "403",
        }


def test_without_the_token_variable_each_run_gets_a_fresh_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(setup_access.TOKEN_ENV, raising=False)
    unset = {setup_access.new_token() for _ in range(3)}
    monkeypatch.setenv(setup_access.TOKEN_ENV, "")
    empty = {setup_access.new_token() for _ in range(3)}
    assert len(unset | empty) == 6, "fresh on each run, and an empty value counts as unset"
    assert all(
        len(token) == setup_access.TOKEN_MIN_CHARS and setup_access.TOKEN_CHARS.fullmatch(token) for token in unset
    ), "a fresh token passes the variable's own rule"


@pytest.mark.parametrize(
    "value",
    [
        "tooShort-123",
        "x" * 21,
        "has a space 0123456789ab",
        "semi;colon=0123456789abcd",
        "pct%41encoded-0123456789",
        "caf\u00e9-0123456789abcdefgh",
    ],
)
def test_a_short_or_unsafe_token_variable_stops_the_command_naming_it_never_its_value(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    with_key(tmp_path, monkeypatch)
    monkeypatch.setenv(setup_access.TOKEN_ENV, value)
    monkeypatch.setattr(setup_wizard.SetupServer, "start", lambda self: pytest.fail("served with a refused token"))
    out = io.StringIO()
    assert setup_wizard.main(tmp_path / "station.yaml", host="127.0.0.1", port=0, browser=False, out=out) == 1
    text = out.getvalue()
    assert setup_access.TOKEN_ENV in text and "secrets.token_urlsafe(16)" in text
    assert value not in text


def test_over_ssh_the_tailnet_name_comes_first_and_the_tunnel_is_the_fallback() -> None:
    """Over ssh from a laptop on the tailnet, the station's tailnet name opens the page with no tunnel
    (http://station-a:8092/?token=...); the tunnel is for when none opens."""
    lines = setup_access.reach_lines(
        8092,
        TOKEN,
        ssh_connection="10.0.0.5 51234 192.168.1.21 22",
        addresses=["192.168.1.21"],
        tailnet="station-a.tail1234.ts.net",
        user="pi",
    )
    assert lines[1:4] == [
        f"  http://station-a:8092/?token={TOKEN}  (tailnet)",
        f"  http://station-a.tail1234.ts.net:8092/?token={TOKEN}  (tailnet)",
        f"  http://192.168.1.21:8092/?token={TOKEN}",
    ]
    assert lines[4].startswith("If none of them opens")
    assert lines[5] == "  ssh -N -L 8092:127.0.0.1:8092 pi@192.168.1.21"
    assert lines[6].endswith(f"http://127.0.0.1:8092/?token={TOKEN}")
    assert not any("(on this box)" in line for line in lines), "the operator is on their own computer"


def test_over_ssh_with_no_address_to_open_the_tunnel_is_the_only_way() -> None:
    lines = setup_access.reach_lines(
        8092, TOKEN, ssh_connection="10.0.0.5 51234 192.168.1.21 22", addresses=[], tailnet=None, user="pi"
    )
    assert lines[0].startswith("This shell is over ssh, so open the page through a tunnel")
    assert lines[1:] == [
        "  ssh -N -L 8092:127.0.0.1:8092 pi@192.168.1.21",
        f"then open in your browser: http://127.0.0.1:8092/?token={TOKEN}",
    ]


def test_on_the_box_the_addresses_are_its_interfaces_and_its_tailnet_name_never_the_hostname() -> None:
    lines = setup_access.reach_lines(
        8092, TOKEN, ssh_connection=None, addresses=["192.168.1.21", "100.64.0.9"], tailnet="station-a.tail1234.ts.net"
    )
    text = "\n".join(lines)
    assert f"http://127.0.0.1:8092/?token={TOKEN}" in text
    assert f"http://192.168.1.21:8092/?token={TOKEN}" in text and f"http://100.64.0.9:8092/?token={TOKEN}" in text
    assert f"http://station-a.tail1234.ts.net:8092/?token={TOKEN}" in text
    assert f"http://station-a:8092/?token={TOKEN}  (tailnet)" in text


def test_listen_addresses_skip_loopback_and_bridges(monkeypatch: pytest.MonkeyPatch) -> None:
    table = (
        "2: eth0    inet 192.168.1.21/24 brd 192.168.1.255 scope global eth0\n"
        "3: docker0    inet 172.17.0.1/16 brd 172.17.255.255 scope global docker0\n"
        "4: tailscale0    inet 100.64.0.9/32 scope global tailscale0\n"
    )

    class Done:
        stdout = table

    monkeypatch.setattr(setup_access.subprocess, "run", lambda *a, **k: Done())
    assert setup_access.listen_addresses() == ["192.168.1.21", "100.64.0.9"]


def test_two_boxes_with_one_hostname_get_different_robot_labels(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    first.write_text("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n")
    second.write_text("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n")
    monkeypatch.setattr(setup_access, "MACHINE_ID_FILES", (first,))
    one = setup_access.robot_label("station-a")
    monkeypatch.setattr(setup_access, "MACHINE_ID_FILES", (second,))
    other = setup_access.robot_label("station-a")
    assert one != other and one.startswith("station-a-") and len(one) == len("station-a-") + 8
    assert "aaaa" not in one, "the raw machine id is not shown"
    monkeypatch.setattr(setup_access, "MACHINE_ID_FILES", (first,))
    assert setup_access.robot_label("station-a") == one, "stable for the box"


def test_a_box_with_no_machine_id_fails_hard(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup_access, "MACHINE_ID_FILES", (tmp_path / "none",))
    with pytest.raises(errors.ConfigError, match="no machine id"):
        setup_access.robot_label()


def test_register_robot_refuses_two_records_with_the_same_name(monkeypatch: pytest.MonkeyPatch) -> None:
    robots = [
        {"id": "rbt_1", "name": "station-a-aaaa", "embodiment": "yam"},
        {"id": "rbt_2", "name": "station-a-aaaa", "embodiment": "yam"},
    ]
    monkeypatch.setattr(setup_wizard, "_platform_call", lambda *args, **kwargs: robots)
    with pytest.raises(ValueError, match="cannot tell which is its own"):
        setup_wizard.register_robot("k", "https://p.example.com", None, "station-a-aaaa", "yam")

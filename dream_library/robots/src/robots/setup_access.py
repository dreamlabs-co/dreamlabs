"""Who may open the setup page, how to reach it, and which robot record a box is: the wizard's network edges."""

import email.message
import getpass
import hashlib
import hmac
import http.cookies
import json
import os
import pathlib
import re
import secrets
import socket
import subprocess
import urllib.parse
from collections.abc import Mapping

from robots import errors

TOKEN_COOKIE = "dream_setup_token"
TOKEN_HEADER = "X-Wizard-Token"
TOKEN_QUERY = "token"

## The environment variable that sets the token, so the printed address stays the same from run to run.
TOKEN_ENV = "DREAM_SETUP_TOKEN"

## The shortest #TOKEN_ENV accepted: the length of a fresh token, `secrets.token_urlsafe(16)`.
TOKEN_MIN_CHARS = 22

## The characters #TOKEN_ENV may hold: a fresh token's, which an address and a cookie carry unescaped.
TOKEN_CHARS = re.compile(r"[A-Za-z0-9_-]+")

MACHINE_ID_FILES = (pathlib.Path("/etc/machine-id"), pathlib.Path("/var/lib/dbus/machine-id"))

VIRTUAL_INTERFACES = ("docker", "br-", "veth", "virbr", "lo")

COMMAND_TIMEOUT_S = 3.0


def new_token() -> str:
    """The wizard's token: #TOKEN_ENV when it is set, else a fresh unguessable one for this run.

    @return A URL-safe string.
    @throws ConfigError #TOKEN_ENV is set but shorter than #TOKEN_MIN_CHARS or holds a character outside #TOKEN_CHARS;
        the message never carries the value.
    """
    chosen = os.environ.get(TOKEN_ENV)
    if not chosen:
        return secrets.token_urlsafe(16)
    if len(chosen) < TOKEN_MIN_CHARS or not TOKEN_CHARS.fullmatch(chosen):
        raise errors.ConfigError(
            f"{TOKEN_ENV} is not a usable setup token: it must be at least {TOKEN_MIN_CHARS} characters, each a"
            " letter, a digit, - or _, so the address and the cookie carry it unescaped; set it to what"
            ' `python3 -c "import secrets; print(secrets.token_urlsafe(16))"` prints, or unset it for a fresh token on'
            " each run"
        )
    return chosen


def token_cookie(token: str) -> str:
    """The `Set-Cookie` value that keeps the token for the page's own requests.

    @param token The wizard's token.
    @return The header value.
    """
    return f"{TOKEN_COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict"


def offered_in_query(query: str) -> str | None:
    """The token in an address's query string, or None.

    @param query The part of the address after `?`.
    @return The token offered, if any.
    """
    found = urllib.parse.parse_qs(query).get(TOKEN_QUERY)
    return found[0] if found else None


def without_token(path: str, query: str) -> str:
    """The address with the token taken out of its query, for the redirect that follows a first visit.

    @param path The address's path.
    @param query The part after `?`.
    @return `path` and the other query parameters.
    """
    rest = [(k, v) for k, v in urllib.parse.parse_qsl(query) if k != TOKEN_QUERY]
    return path + ("?" + urllib.parse.urlencode(rest) if rest else "")


def matches(token: str, offered: str | None) -> bool:
    """Whether the offered value is the token, compared in constant time.

    @param token The wizard's token.
    @param offered What the request carried, or None.
    @return True when equal.
    """
    return offered is not None and hmac.compare_digest(offered.encode("utf-8"), token.encode("utf-8"))


def admitted(token: str, headers: Mapping[str, str] | email.message.Message) -> bool:
    """Whether a request carries the token as the cookie or the header.

    @param token The wizard's token.
    @param headers The request's headers.
    @return True when it does.
    """
    offered = headers.get(TOKEN_HEADER)
    if matches(token, None if offered is None else str(offered)):
        return True
    jar: http.cookies.SimpleCookie = http.cookies.SimpleCookie()
    try:
        jar.load(str(headers.get("Cookie", "")))
    except http.cookies.CookieError:
        return False
    return TOKEN_COOKIE in jar and matches(token, jar[TOKEN_COOKIE].value)


def machine_id() -> str:
    """This box's machine id, read from the first of #MACHINE_ID_FILES that exists.

    @return The id.
    """
    for path in MACHINE_ID_FILES:
        if path.is_file() and path.read_text(encoding="utf-8").strip():
            return path.read_text(encoding="utf-8").strip()
    raise errors.ConfigError(
        "this box has no machine id (" + " or ".join(str(p) for p in MACHINE_ID_FILES) + "), so its robot record "
        "could not be told from another box's; run `sudo systemd-machine-id-setup` and try again"
    )


def robot_label(hostname: str | None = None) -> str:
    """The name a box's robot gets on the platform: its hostname and a short code from its machine id.

    @param hostname The box's hostname.
    @return For example `station-a-3fa9c1d2`.
    """
    code = hashlib.sha256(b"dreamlabs-robot:" + machine_id().encode("utf-8")).hexdigest()[:8]
    return f"{hostname or socket.gethostname()}-{code}"


def listen_addresses() -> list[str]:
    """The IPv4 addresses other computers could reach this box at, from `ip`; none when `ip` is not installed.

    @return Addresses, without loopback and without container or virtual-machine bridges.
    """
    try:
        out = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", "scope", "global"],
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_S,
            check=True,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    found = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) > 3 and not parts[1].startswith(VIRTUAL_INTERFACES):
            found.append(parts[3].split("/")[0])
    return found


def tailnet_name() -> str | None:
    """This box's name on the local network, or None when it is not on one.

    @return The fully qualified name, or None.
    """
    try:
        out = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_S,
            check=True,
        ).stdout
        name = json.loads(out)["Self"]["DNSName"].rstrip(".")
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return None
    return name or None


def local_lines(local: str) -> list[str]:
    """The lines printed when the page opens in a browser on this box's own screen.

    They say that it opened, and give the one address to paste if it did not. No network address is listed
    (reach_lines() lists them when there is no screen).

    @param local The page's address on this box, with the token.
    @return The lines, in the order to print them.
    """
    return [
        "The setup page is opening in your browser.",
        f"If it does not open, paste this into your browser (the part after ? is private):  {local}",
    ]


def reach_lines(
    port: int,
    token: str,
    *,
    ssh_connection: str | None,
    addresses: list[str],
    tailnet: str | None,
    user: str | None = None,
) -> list[str]:
    """The lines that say where to open the page, each address carrying the token.

    @param port The wizard's port.
    @param token The wizard's token.
    @param ssh_connection The `SSH_CONNECTION` environment variable.
    @param addresses This box's reachable addresses.
    @param tailnet This box's network name.
    @param user The login name for the ssh line.
    @return The lines, in the order to print them.
    """
    path = f"/?{TOKEN_QUERY}={token}"
    local = f"http://127.0.0.1:{port}{path}"
    # The network name answers both bare and fully-qualified; the bare one is what people type.
    names = [tailnet.split(".")[0], tailnet] if tailnet else []
    direct = [f"  http://{name}:{port}{path}  (tailnet)" for name in names]
    direct += [f"  http://{address}:{port}{path}" for address in addresses]
    fields = (ssh_connection or "").split()
    if len(fields) != 4:
        intro = "Open the setup page in a browser at one of these (the part after ? is private):"
        return [intro, f"  {local}  (on this box)", *direct]
    who = user or getpass.getuser()
    tunnel = [f"  ssh -N -L {port}:127.0.0.1:{port} {who}@{fields[2]}", f"then open in your browser: {local}"]
    if not direct:
        return ["This shell is over ssh, so open the page through a tunnel. On your own computer run:", *tunnel]
    return [
        "Open the setup page in a browser on your own computer at one of these (the part after ? is private):",
        *direct,
        "If none of them opens, tunnel through ssh instead. On your own computer run:",
        *tunnel,
    ]

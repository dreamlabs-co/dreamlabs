"""The credentials: where the model, the platform and the data service are, and the API key for all three.

The client reads four names, each from the environment first, then from the credentials file, then from the built-in
production default:

- `DREAM_GATEWAY_URL`: the model's URL, which get_action() calls.
- `DREAM_API_URL`: the platform's URL, which get_models() calls.
- `DREAM_DATA_URL`: the data service's URL, which the embodiment, dataset and fine-tuning calls use.
- `DREAM_API_KEY`: your API key, sent with every request to all three; there is no default.

The file is `~/.dreamlabs/credentials`, or the file the environment variable `DREAM_CREDENTIALS` names, in the shape a
shell can `source` (the platform's setup page prints it with the URLs of its own environment):

```
DREAM_GATEWAY_URL=<the model's URL>
DREAM_API_URL=<the platform's URL>
DREAM_DATA_URL=<the data service's URL>
DREAM_API_KEY=<your API key>
```

With no file and nothing in the environment, the client reaches production, so the file is what moves a machine to
another environment. A missing file is not an error. A file that exists is read whole: a line is blank, a `#` comment,
or `NAME=value` (`export NAME=value` too, the value optionally in single or double quotes); a later line for the same
name wins, and any other name is ignored, so the file may carry a shell's other settings. A URL that is set but is not
`http(s)://<host>[:<port>]` with no path is #DreamConfigError naming the variable and, when it came from the file, the
file; so is a file that exists but cannot be read.

`DreamClient` takes all four from load(); a robot station reads the platform and the key from the same file.

The module's constants, read-only:

- #DEFAULT_GATEWAY_URL, #DEFAULT_PLATFORM_URL, #DEFAULT_DATA_URL: the production URLs a name falls back to.
- #DEFAULT_FILE: `~/.dreamlabs/credentials`, the file read unless #FILE_ENV names another.
- #FILE_ENV: `"DREAM_CREDENTIALS"`, the environment variable naming another file (a machine with several accounts).
- #GATEWAY_URL_ENV, #PLATFORM_URL_ENV, #DATA_URL_ENV, #API_KEY_ENV: the four names above, as strings.
- #NAMES: those four names, in the file's order.

@see load(), describe_url(), where_set(), Credentials, gateway_parts()
"""

import dataclasses
import os
import pathlib
import re
import urllib.parse
from collections.abc import Mapping
from typing import Any

from dream.errors import DreamConfigError

## The model's URL when nothing names one: production's.
DEFAULT_GATEWAY_URL = "https://api.dreamlabs.ai"

## The platform when nothing names one: production's.
DEFAULT_PLATFORM_URL = "https://platform.dreamlabs.ai"

## The data service when nothing names one: production's.
DEFAULT_DATA_URL = "https://data.dreamlabs.ai"

## Where the file lives unless #FILE_ENV says otherwise; the station's own directory.
DEFAULT_FILE = "~/.dreamlabs/credentials"

## Environment variable naming another file (tests, or a box with several accounts).
FILE_ENV = "DREAM_CREDENTIALS"

## The model's URL (DreamAPI's gateway): `https://api.dreamlabs.ai`, `http://127.0.0.1:8000`.
GATEWAY_URL_ENV = "DREAM_GATEWAY_URL"

## The platform, as a URL: `https://platform.dreamlabs.ai`. `robots` links the live view to it and registers there.
PLATFORM_URL_ENV = "DREAM_API_URL"

## The data service, as a URL: `https://data.dreamlabs.ai`. `DreamClient`'s embodiment, dataset and fine-tuning calls.
DATA_URL_ENV = "DREAM_DATA_URL"

## The key, sent as `Authorization: Bearer` to the gateway, the platform and the data service.
API_KEY_ENV = "DREAM_API_KEY"

## The four names the file and the environment are read for, in the file's order.
NAMES = (GATEWAY_URL_ENV, PLATFORM_URL_ENV, DATA_URL_ENV, API_KEY_ENV)

## One assignment line: an optional `export`, the name, `=`, the value to the end of the line.
_ASSIGNMENT_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")

## Characters an HTTP header value cannot carry; `http.client` refuses them (and anything outside Latin-1) with a
## plain ValueError whose message holds the whole header, the key included.
_HEADER_BREAKS = ("\r", "\n")


@dataclasses.dataclass(frozen=True)
class Credentials:
    """What load() resolved: the three URLs, always set, and the key, when anything set it.

    Attributes:
    - gateway_url: The model's URL (`DREAM_GATEWAY_URL`), `http(s)://<host>[:<port>]`; #DEFAULT_GATEWAY_URL when
      nothing named one.
    - platform_url: The platform's URL (`DREAM_API_URL`), `http(s)://<host>[:<port>]`; #DEFAULT_PLATFORM_URL when
      nothing named one.
    - data_url: The data service's URL (`DREAM_DATA_URL`), `http(s)://<host>[:<port>]`; #DEFAULT_DATA_URL when
      nothing named one.
    - api_key: The key (`DREAM_API_KEY`), or None when neither the environment nor the file sets one.
    - path: The file load() read, or looked for and did not find.
    """

    gateway_url: str
    platform_url: str
    data_url: str
    api_key: str | None
    path: pathlib.Path


def load(env: Mapping[str, str] | None = None) -> Credentials:
    """Resolves the credentials: the environment over the file over the defaults, name by name.

    `DreamClient()` calls it for you; call it to see what it will use.

    @param env The environment to read; `os.environ` when None. It names the file (#FILE_ENV) and may set any
        of #NAMES itself.
    @return The #Credentials.
    @throws DreamConfigError A URL is set but malformed; the message names the variable and, when the value came
        from the file, the file. A file that exists but cannot be read (no permission, not UTF-8 text) is also an
        error naming it.
    """
    path, values = _resolve(env)
    for name in (GATEWAY_URL_ENV, PLATFORM_URL_ENV, DATA_URL_ENV):
        if name in values:
            _check_url(name, *values[name])
    return Credentials(
        gateway_url=values.get(GATEWAY_URL_ENV, (DEFAULT_GATEWAY_URL, ""))[0],
        platform_url=values.get(PLATFORM_URL_ENV, (DEFAULT_PLATFORM_URL, ""))[0],
        data_url=values.get(DATA_URL_ENV, (DEFAULT_DATA_URL, ""))[0],
        api_key=values.get(API_KEY_ENV, (None, ""))[0],
        path=path,
    )


def _resolve(env: Mapping[str, str] | None) -> tuple[pathlib.Path, dict[str, tuple[str, str]]]:
    """The file and, for each of #NAMES that the environment or the file sets, `(value, source)`.

    @param env The environment to read; `os.environ` when None.
    @return The file's path, and `{name: (value, "the environment" or the file's path)}`.
    """
    env = os.environ if env is None else env
    path = file_path(env)
    from_file = _read_file(path)
    values: dict[str, tuple[str, str]] = {}
    for name in NAMES:
        if env.get(name, "").strip():
            values[name] = (env[name].strip(), "the environment")
        elif from_file.get(name, "").strip():
            values[name] = (from_file[name].strip(), str(path))
    return path, values


# How every refused key's message ends: a key made on one site is refused by the others, and the key never says so.
KEY_PER_SITE = (
    "keys work only on the site that made them (dev, staging and production are separate), so this key may be from "
    "another site, mistyped or revoked"
)


def describe_url(url: str, env: Mapping[str, str] | None = None) -> str:
    """Says where a URL the client called came from, for the message of an authentication failure.

    A key refused by the wrong environment is the usual cause, and the URL alone does not show it: a machine with no
    credentials file reaches production without saying so. The answer names the variable and its source: the
    environment, the file, or the built-in production default (which also names the file that was looked for).

    @param url The URL called, `http(s)://<host>[:<port>]` (a path is ignored).
    @param env The environment to read; `os.environ` when None.
    @return For example `https://api.dreamlabs.ai (the built-in production default for DREAM_GATEWAY_URL: no such
        line in the environment or in /home/me/.dreamlabs/credentials)`.
    """
    path, name, source = _named_by(url, env)
    if name is None:
        return f"{url} (not from the credentials: passed to the client directly)"
    if source:
        return f"{url} ({name} from {source})"
    return (
        f"{url} (the built-in production default for {name}: no such line in the environment or in {path}; "
        "a key from another environment needs that environment's credentials file)"
    )


def where_set(url: str, env: Mapping[str, str] | None = None) -> str:
    """Says where to change a URL the client called, for the message of a call that could not reach it.

    @param url The URL called, `http(s)://<host>[:<port>]` (a path is ignored).
    @param env The environment to read; `os.environ` when None.
    @return For example `DREAM_GATEWAY_URL in /home/me/.dreamlabs/credentials`, `DREAM_GATEWAY_URL in the
        environment`, `DREAM_GATEWAY_URL (not set in the environment or in /home/me/.dreamlabs/credentials: the
        built-in default)` or `the URL passed to the client`.
    """
    path, name, source = _named_by(url, env)
    if name is None:
        return "the URL passed to the client"
    if source:
        return f"{name} in {source}"
    return f"{name} (not set in the environment or in {path}: the built-in default)"


def _named_by(url: str, env: Mapping[str, str] | None) -> tuple[pathlib.Path, str | None, str]:
    """The credentials file, the variable whose value `url` is, and where that value was set.

    @return `(file, name, source)`: `source` is "the environment", the file's path, or "" for the variable's built-in
        default; `name` is None when no variable's value is `url`.
    """
    path, values = _resolve(env)
    defaults = {
        GATEWAY_URL_ENV: DEFAULT_GATEWAY_URL,
        PLATFORM_URL_ENV: DEFAULT_PLATFORM_URL,
        DATA_URL_ENV: DEFAULT_DATA_URL,
    }
    wanted = _origin(url)
    for name, default in defaults.items():
        value, source = values.get(name, (default, ""))
        if _origin(value) == wanted:
            return path, name, source
    return path, None, ""


def _origin(url: str) -> tuple[str, str, int | None]:
    """A URL's scheme, host and port with the scheme's default port filled in, so `https://h` equals `https://h:443`."""
    parts = urllib.parse.urlsplit(url)
    try:
        port = parts.port
    except ValueError:
        port = None
    return parts.scheme, (parts.hostname or "").lower(), port or {"https": 443, "http": 80}.get(parts.scheme)


def file_path(env: Mapping[str, str] | None = None) -> pathlib.Path:
    """The credentials file: the one #FILE_ENV names when it is set, else #DEFAULT_FILE, `~` expanded.

    @param env The environment to read; `os.environ` when None.
    @return The path; it may not exist.
    """
    env = os.environ if env is None else env
    return pathlib.Path(os.path.expanduser(env.get(FILE_ENV, "").strip() or DEFAULT_FILE))


def gateway_parts(url: str) -> dict[str, Any]:
    """Splits the model's URL into the `host`, `port` and `scheme` keys a `DreamClient.policy()` config takes.

    The port is the URL's own, else the scheme's (443 for `https`, 80 for `http`); a path is ignored.

    @param url The model's URL, `http(s)://<host>[:<port>]`, e.g. the `gateway_url` of load().
    @return `{"host": str, "port": int, "scheme": str}`.
    @throws DreamConfigError The URL is not `http(s)://<host>[:<port>]`: no `http` or `https` scheme, no host, or a
        port that is not a number in 1..65535. The message names the URL.
    """
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise DreamConfigError(f"the model's URL {url!r} is not a URL ({exc})") from None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise DreamConfigError(
            f"the model's URL must be http(s)://<host>[:<port>], e.g. https://api.example.com, got {url!r}"
        )
    return {
        "host": parts.hostname,
        "port": port or (443 if parts.scheme == "https" else 80),
        "scheme": parts.scheme,
    }


def _authorization(api_key: str) -> dict[str, str]:
    """The `Authorization` header that carries the key to every host.

    @param api_key The key.
    @return `{"Authorization": "Bearer <key>"}`.
    @throws DreamConfigError The key holds a character an HTTP header cannot carry: a line break, or one outside
        Latin-1 (a typographic quote pasted with it). The message never shows the key.
    """
    if any(mark in api_key for mark in _HEADER_BREAKS) or not _is_latin1(api_key):
        raise DreamConfigError(
            "the api key holds a character an HTTP header cannot carry (a line break, or a character such as a "
            f"typographic quote); copy the key again, whole, into {API_KEY_ENV} or the api_key argument"
        )
    return {"Authorization": f"Bearer {api_key}"}


def _is_latin1(text: str) -> bool:
    """Whether `http.client` can encode the text into a header, which it does as Latin-1."""
    try:
        text.encode("latin-1")
    except UnicodeEncodeError:
        return False
    return True


def _read_file(path: pathlib.Path) -> dict[str, str]:
    """The assignments in the file, last one per name; empty when there is no file.

    @param path The file.
    @return `{name: value}` with quotes stripped.
    @throws DreamConfigError The file exists but cannot be read, or is not UTF-8 text.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise DreamConfigError(f"cannot read the credentials file {path}: {exc.strerror or exc}") from None
    except ValueError as exc:  # UnicodeDecodeError, or a path holding a NUL byte
        raise DreamConfigError(f"cannot read the credentials file {path}: {exc}") from None
    found: dict[str, str] = {}
    for line in text.splitlines():
        match = _ASSIGNMENT_RE.match(line)
        if match is None:
            continue
        name, value = match.groups()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        found[name] = value
    return found


def _check_url(name: str, raw: str, source: str) -> None:
    """Requires `http(s)://<host>[:<port>]` with no path: every request goes to `/v1/...` on the host itself.

    @param name The variable the value was read as.
    @param raw The value.
    @param source Where it came from, for the message: `the environment` or the file.
    @throws DreamConfigError The value is not such a URL; the message names the variable and the source.
    """
    try:
        parts = urllib.parse.urlsplit(raw)
        _ = parts.port  # A port that is not a number, or out of range, raises here.
    except ValueError as exc:
        raise DreamConfigError(f"{name} (from {source}) is not a URL ({exc}): {raw!r}") from None
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.path not in ("", "/"):
        raise DreamConfigError(
            f"{name} (from {source}) must be http(s)://<host>[:<port>] with no path, e.g. https://api.example.com, "
            f"got {raw!r}"
        )

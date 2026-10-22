"""Tests for `dream.credentials`: the file's syntax, the precedence, the defaults and the errors.

The contract tests (`test_contract.py`, C-16 to C-16f) pin what `dream.init` does with the
record; these pin the record itself.
"""

import pathlib

import pytest

from dream import credentials
from dream.errors import DreamConfigError


def test_no_file_and_no_env_is_production_with_no_key(tmp_path: pathlib.Path) -> None:
    """Nothing set anywhere: the three URLs are production's and there is no key."""
    found = credentials.load({"DREAM_CREDENTIALS": str(tmp_path / "missing")})
    assert (found.gateway_url, found.platform_url, found.data_url, found.api_key) == (
        "https://api.dreamlabs.ai",
        "https://platform.dreamlabs.ai",
        "https://data.dreamlabs.ai",
        None,
    )
    assert found.path == tmp_path / "missing"


def test_the_data_service_url_is_read_and_defaults_to_production(tmp_path: pathlib.Path) -> None:
    """DREAM_DATA_URL is read like the others: the environment over the file over the production default."""
    path = tmp_path / "credentials"
    path.write_text("DREAM_DATA_URL=https://file.data.example.com\n", encoding="utf-8")
    assert credentials.load({"DREAM_CREDENTIALS": str(path)}).data_url == "https://file.data.example.com"
    from_env = credentials.load({"DREAM_CREDENTIALS": str(path), "DREAM_DATA_URL": "https://env.data.example.com"})
    assert from_env.data_url == "https://env.data.example.com"
    assert credentials.load({"DREAM_CREDENTIALS": str(tmp_path / "missing")}).data_url == "https://data.dreamlabs.ai"


def test_a_bad_data_service_url_names_the_variable(tmp_path: pathlib.Path) -> None:
    """A malformed DREAM_DATA_URL is `DreamConfigError` naming the variable, like the other URLs."""
    path = tmp_path / "credentials"
    path.write_text("DREAM_DATA_URL=https://data.example.com/v1\n", encoding="utf-8")
    with pytest.raises(DreamConfigError, match=r"DREAM_DATA_URL \(from .*credentials\)"):
        credentials.load({"DREAM_CREDENTIALS": str(path)})


def test_the_default_path_is_the_station_directory() -> None:
    """With `DREAM_CREDENTIALS` unset the file is `~/.dreamlabs/credentials`, `~` expanded."""
    assert credentials.file_path({}) == pathlib.Path.home() / ".dreamlabs" / "credentials"


def test_the_file_is_read_as_a_shell_would(tmp_path: pathlib.Path) -> None:
    """Comments, blank lines, `export`, either quote style, unrelated names, and a later assignment winning."""
    path = tmp_path / "credentials"
    path.write_text(
        "# the station's account\n"
        "\n"
        "export DREAM_GATEWAY_URL='https://staging.api.example.com'\n"
        'DREAM_API_URL = "https://staging.platform.example.com"\n'
        "OTHER=ignored\n"
        "DREAM_API_KEY=dl_live_first\n"
        "DREAM_API_KEY=dl_live_second\n",
        encoding="utf-8",
    )
    found = credentials.load({"DREAM_CREDENTIALS": str(path)})
    assert (found.gateway_url, found.platform_url, found.api_key) == (
        "https://staging.api.example.com",
        "https://staging.platform.example.com",
        "dl_live_second",
    )


def test_the_environment_wins_over_the_file_name_by_name(tmp_path: pathlib.Path) -> None:
    """A variable set in the environment replaces the file's value for that name and no other."""
    path = tmp_path / "credentials"
    path.write_text("DREAM_GATEWAY_URL=https://file.example.com\nDREAM_API_KEY=dl_live_file\n", encoding="utf-8")
    found = credentials.load({"DREAM_CREDENTIALS": str(path), "DREAM_API_KEY": "dl_live_env"})
    assert (found.gateway_url, found.platform_url, found.api_key) == (
        "https://file.example.com",
        "https://platform.dreamlabs.ai",
        "dl_live_env",
    )


def test_an_empty_value_is_unset(tmp_path: pathlib.Path) -> None:
    """A blank value, in the file or the environment, is the same as no value: the next source is read."""
    path = tmp_path / "credentials"
    path.write_text("DREAM_GATEWAY_URL=https://file.example.com\nDREAM_API_KEY=\n", encoding="utf-8")
    found = credentials.load({"DREAM_CREDENTIALS": str(path), "DREAM_GATEWAY_URL": "  "})
    assert (found.gateway_url, found.api_key) == ("https://file.example.com", None)


@pytest.mark.parametrize(
    "url",
    ["api.example.com", "ftp://api.example.com", "https://", "https://api.example.com/v1", "https://h:99999"],
)
def test_a_bad_url_names_the_variable_and_its_source(tmp_path: pathlib.Path, url: str) -> None:
    """A URL that is not http(s)://<host>[:<port>] is `DreamConfigError` naming the variable and where it came from."""
    path = tmp_path / "credentials"
    path.write_text(f"DREAM_API_URL={url}\n", encoding="utf-8")
    with pytest.raises(DreamConfigError, match=r"DREAM_API_URL \(from .*credentials\)"):
        credentials.load({"DREAM_CREDENTIALS": str(path)})
    with pytest.raises(DreamConfigError, match=r"DREAM_API_URL \(from the environment\)"):
        credentials.load({"DREAM_CREDENTIALS": str(tmp_path / "missing"), "DREAM_API_URL": url})


def test_a_file_that_cannot_be_read_names_itself(tmp_path: pathlib.Path) -> None:
    """A path that exists but is a directory is an error naming it, not a silent fall-back to the defaults."""
    with pytest.raises(DreamConfigError, match=str(tmp_path)):
        credentials.load({"DREAM_CREDENTIALS": str(tmp_path)})


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://api.example.com", ("api.example.com", 443, "https")),
        ("https://api.example.com/", ("api.example.com", 443, "https")),
        ("http://127.0.0.1:8000", ("127.0.0.1", 8000, "http")),
        ("http://gw.example", ("gw.example", 80, "http")),
        ("https://api.example.com:8443", ("api.example.com", 8443, "https")),
    ],
)
def test_gateway_parts(url: str, expected: tuple[str, int, str]) -> None:
    """The connection keys of a gateway URL: the port is the URL's own, else the scheme's."""
    parts = credentials.gateway_parts(url)
    assert (parts["host"], parts["port"], parts["scheme"]) == expected


def test_describe_url_says_defaults_when_no_file_sets_the_url(tmp_path: pathlib.Path) -> None:
    missing = tmp_path / "missing"
    text = credentials.describe_url("https://api.dreamlabs.ai:443", {"DREAM_CREDENTIALS": str(missing)})
    assert "built-in production default for DREAM_GATEWAY_URL" in text
    assert str(missing) in text


def test_describe_url_names_the_file_or_the_environment_that_set_it(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "credentials"
    path.write_text(
        "DREAM_API_URL=https://staging.platform.example.com\nDREAM_GATEWAY_URL=http://gw:9000\n", encoding="utf-8"
    )
    env = {"DREAM_CREDENTIALS": str(path)}
    assert credentials.describe_url("https://staging.platform.example.com", env) == (
        f"https://staging.platform.example.com (DREAM_API_URL from {path})"
    )
    assert credentials.describe_url("http://gw:9000", env) == f"http://gw:9000 (DREAM_GATEWAY_URL from {path})"
    over = {**env, "DREAM_GATEWAY_URL": "http://gw:9000"}
    assert (
        credentials.describe_url("http://gw:9000", over) == "http://gw:9000 (DREAM_GATEWAY_URL from the environment)"
    )


def test_describe_url_says_so_for_a_url_the_credentials_did_not_give(tmp_path: pathlib.Path) -> None:
    text = credentials.describe_url("https://elsewhere.example.com", {"DREAM_CREDENTIALS": str(tmp_path / "none")})
    assert text == "https://elsewhere.example.com (not from the credentials: passed to the client directly)"


def test_where_set_says_where_to_change_a_url(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "credentials"
    path.write_text("DREAM_GATEWAY_URL=http://gw:9000\n", encoding="utf-8")
    env = {"DREAM_CREDENTIALS": str(path)}
    assert credentials.where_set("http://gw:9000", env) == f"DREAM_GATEWAY_URL in {path}"
    assert credentials.where_set("http://gw:9000", {**env, "DREAM_GATEWAY_URL": "http://gw:9000"}) == (
        "DREAM_GATEWAY_URL in the environment"
    )
    missing = {"DREAM_CREDENTIALS": str(tmp_path / "none")}
    assert credentials.where_set("https://api.dreamlabs.ai:443", missing) == (
        f"DREAM_GATEWAY_URL (not set in the environment or in {tmp_path / 'none'}: the built-in default)"
    )
    assert credentials.where_set("https://elsewhere.example.com", missing) == "the URL passed to the client"

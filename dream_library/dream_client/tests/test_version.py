"""`dream.__version__` is the installed distribution's version, read from its metadata; the source never states it.

The first test reads the version this venv installed; the others give the lookup a directory of fake installed
distributions, so a wheel install under any name, the names an editable install is found by, and a source checkout
with nothing installed are each pinned without building a wheel. `dream_library/Makefile` (`install-check`) checks the
two real wheels.
"""

from __future__ import annotations

import importlib.metadata
import pathlib

import pytest

import dream


def test_version_is_the_installed_distributions() -> None:
    """This venv installs the client as `dreamlabs-client` (editable), so its metadata is the version."""
    assert dream.__version__ == importlib.metadata.version("dreamlabs-client")


def test_a_wheel_install_is_found_by_its_file_list_whatever_its_name(tmp_path: pathlib.Path) -> None:
    """The distribution whose `RECORD` lists `dream/__init__.py` is read; one that ships another package is not."""
    _install(tmp_path, "vendored", "9.9.9", "vendored/dream/__init__.py,sha256=x,1\n")
    _install(tmp_path, "renamed", "3.0.0+gabcdef123456", "robots/__init__.py,,\ndream/__init__.py,sha256=x,1\n")

    assert dream._installed_version(tmp_path) == "3.0.0+gabcdef123456"


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        ({"dreamlabs": "4.1.0", "dreamlabs-client": "4.0.2"}, "4.0.2"),
        ({"dream": "3.0.0", "dream-client": "2.5.1"}, "0+unknown"),
        ({}, "0+unknown"),
    ],
)
def test_without_a_file_list_the_client_then_the_source_fallback(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, installed: dict[str, str], expected: str
) -> None:
    """An editable install lists no package files: the client's name is tried (`dreamlabs-client`, which owns the
    package even under the library), then `0+unknown`; the names before the rename are not read."""

    def version(name: str) -> str:
        if name not in installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed[name]

    monkeypatch.setattr(importlib.metadata, "version", version)

    assert dream._installed_version(tmp_path) == expected


def _install(site: pathlib.Path, name: str, version: str, record: str) -> None:
    """Writes the `.dist-info` directory an installer leaves for one distribution, with its file list."""
    info = site / f"{name}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8")
    (info / "RECORD").write_text(record, encoding="utf-8")

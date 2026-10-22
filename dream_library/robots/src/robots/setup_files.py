"""The setup wizard's file picker for a robot of your own's driver: the station's own folders, as the page browses.

The page runs in a browser on the operator's laptop while the wizard runs on the station they ssh'd into, so a file
input would read the laptop's files. Instead the page lists the station's folders here (folder()), starting in the
station file's (start()), and the classes of the Python file the operator picks (classes()), read without running it.
The pick is the config's `custom.driver`, `/path/to/file.py:Class` (#robots.user_class).

@see folder(), classes()
"""

import ast
import os
import pathlib
from typing import Any


def start(config_path: pathlib.Path) -> pathlib.Path:
    """The folder the picker opens in: the station file's, or its nearest parent on this computer until it exists.

    @param config_path The station file's path.
    @return The folder.
    """
    folder = config_path.resolve().parent
    while not folder.is_dir():
        folder = folder.parent
    return folder


def folder(path: pathlib.Path) -> dict[str, Any]:
    """A folder of this computer: the folders and Python files in it, hidden ones left out.

    @param path An absolute path.
    @return `{"dir", "parent", "dirs", "files"}`: the folder, its parent (None at the root), and the names of the
        folders and the `.py` files in it, each sorted.
    @throws ValueError The path is not an absolute path of a folder this user can list.
    """
    if not path.is_absolute() or not path.is_dir():
        raise ValueError(f"{path} is not a folder on this computer")
    path = path.resolve()
    try:
        entries = [entry for entry in os.scandir(path) if not entry.name.startswith(".")]
    except OSError as exc:
        raise ValueError(f"cannot list {path}: {exc.strerror}") from exc
    return {
        "dir": str(path),
        "parent": None if path.parent == path else str(path.parent),
        "dirs": sorted(entry.name for entry in entries if entry.is_dir()),
        "files": sorted(entry.name for entry in entries if entry.name.endswith(".py") and entry.is_file()),
    }


def classes(path: pathlib.Path) -> dict[str, Any]:
    """The classes a Python file defines at its top level, read without running it.

    @param path An absolute path of a `.py` file.
    @return `{"file", "classes"}`: the file, and its classes' names in the file's order.
    @throws ValueError The path is not a `.py` file this user can read, or it does not parse as Python.
    """
    if not path.is_absolute() or path.suffix != ".py" or not path.is_file():
        raise ValueError(f"{path} is not a Python file on this computer")
    try:
        tree = ast.parse(path.read_bytes(), filename=str(path))
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc.strerror}") from exc
    except SyntaxError as exc:
        raise ValueError(f"{path} is not valid Python: line {exc.lineno}: {exc.msg}") from exc
    return {"file": str(path), "classes": [node.name for node in tree.body if isinstance(node, ast.ClassDef)]}

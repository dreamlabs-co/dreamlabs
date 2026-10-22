"""A class of your own the station file names, `package.module:Class` or `/path/to/file.py:Class`.

Two keys name one: a robot of your own's driver (`custom.driver`, #robots.bodies.load_driver) and a camera of your
own (`cameras[i].source`, #robots.cameras.CameraReader.from_config). path() checks the name when the file is read,
and load() imports the class when it is needed.
"""

import importlib
import importlib.util
import pathlib
import sys
from typing import Any

from robots import errors


def path(value: str, key: str, base: pathlib.Path) -> str:
    """The class name a config key gives, checked, a `file.py`'s path made absolute.

    @param value `package.module:Class`, or `file.py:Class` (relative to `base` unless absolute).
    @param key The config key, for the message.
    @param base The directory a relative file is read from: the config file's.
    @return `value`, with a file's path resolved.
    @throws ConfigError The value is not `module:Class` or `file.py:Class`.
    """
    module, _, name = value.rpartition(":")
    if not module or not name.isidentifier():
        raise errors.ConfigError(f"config key '{key}' must be package.module:Class or file.py:Class, got {value!r}")
    if module.endswith(".py"):
        return f"{(base / pathlib.Path(module).expanduser()).resolve()}:{name}"
    return value


def load(name: str, what: str) -> Any:
    """The class itself, imported now.

    A `file.py` is executed again on each call; its folder goes at the end of `sys.path`, so the file imports the files
    beside it (a helper module) and none of them shadows an installed module.

    @param name `package.module:Class`, or `/path/to/file.py:Class`.
    @param what What the class is, for messages: `driver` or `camera`.
    @return The class.
    @throws ConfigError The module or file does not load, or has no such name.
    """
    module_name, _, attribute = name.rpartition(":")
    try:
        if module_name.endswith(".py"):
            folder = str(pathlib.Path(module_name).resolve().parent)
            if folder not in sys.path:
                sys.path.append(folder)
            # Its own name in sys.modules, so a dataclass in the file resolves and no installed module is shadowed.
            spec = importlib.util.spec_from_file_location(
                f"_robots_{what}_{pathlib.Path(module_name).stem}", module_name
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"no Python module at {module_name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_name)
    except (ImportError, OSError) as exc:
        raise errors.ConfigError(f"your {what}'s module {module_name!r} does not load here: {exc}") from exc
    if not hasattr(module, attribute):
        raise errors.ConfigError(f"your {what}'s module {module_name!r} has no {attribute!r}")
    return getattr(module, attribute)

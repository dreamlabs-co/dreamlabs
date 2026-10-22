"""Joining a path a dataset names to the directory it must stay in.

A LeRobot `meta/info.json` names its files by template (`data_path`, `video_path`), and the dataset is the customer's,
so a formatted template is untrusted: `pathlib` lets an absolute path replace the base outright (`base / "/etc/x"` is
`/etc/x`) and a `..` climb out of it. safe_join() is the one join for such a path. Standard library only, like
`embodiments`.
"""

import pathlib


class PathEscapeError(ValueError):
    """A path that leads outside the directory it was joined to; `str(error)` names the path, never the base."""


def safe_join(base: pathlib.Path, untrusted: str) -> pathlib.Path:
    """The resolved path `untrusted` names under `base`, refused when it leads anywhere else.

    Both sides are resolved first, so an absolute path, a `..` and a symlink that leads out are all caught.

    @param base The directory the path must stay in.
    @param untrusted The relative path, as the dataset gave it.
    @return The resolved path, strictly inside `base`.
    @throws PathEscapeError When the path is `base` itself or lies outside it.
    """
    root = base.resolve()
    joined = (root / untrusted).resolve()
    if joined == root or not joined.is_relative_to(root):
        raise PathEscapeError(f"{untrusted!r} leads outside its directory")
    return joined

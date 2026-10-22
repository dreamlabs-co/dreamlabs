"""`dreamcli example`: run an example program that ships with the dream library, with no robot.

An example is a program from `dream_library/examples/<name>/`. Its scripts and assets ride in the dream wheel as the
data directory `dream_examples/<name>/` beside this package (the wheel's force-include in
`dream_library/pyproject.toml`), and a source checkout runs them from the example itself; example_dir() finds
whichever is there. They are scripts, not a package, so run() replaces this process with `python <entry>` (on
Windows, which has no exec, it waits for it) and the example's own terminal handles Ctrl-C. Examples read the API key
and the gateway from `~/.dreamlabs/credentials`, as every dream program does. self_test() runs an example's headless
check instead, which needs no display and no key, for an install check.

The one example today is `yam-mujoco`, Dream YAM Sim: two simulated YAM arms in MuJoCo, three cameras and a task
terminal that sends each observation to Dream-1 and runs the joint chunks it answers (#EXAMPLES).

@see run(), self_test(), example_dir(), EXAMPLES
"""

import dataclasses
import os
import pathlib
import subprocess
import sys
from collections.abc import Callable, Sequence
from typing import IO

from robots import errors


@dataclasses.dataclass(frozen=True)
class Example:
    """One example program `dreamcli example` can run.

    Attributes:
    - directory: Its directory name, under `examples/` in a checkout and under `dream_examples/` in the wheel.
    - entry: The script that runs it.
    - self_test: The script and arguments of its headless check.
    - summary: One line for `dreamcli example`'s list.
    """

    directory: str
    entry: str
    self_test: tuple[str, ...]
    summary: str


## The examples by the name `dreamcli example <name>` takes.
EXAMPLES: dict[str, Example] = {
    "yam-mujoco": Example(
        directory="yam_mujoco",
        entry="yam_mujoco.py",
        self_test=("yam_env.py", "--headless"),
        summary="Dream YAM Sim: two simulated YAM arms in MuJoCo you give Dream-1 tasks to",
    ),
}

## Where the wheel puts the examples, relative to the directory the `robots` package is installed in.
WHEEL_DIR = pathlib.Path("dream_examples")

## Where a source checkout keeps them, relative to `dream_library/`.
CHECKOUT_DIR = pathlib.Path("examples")


def example_dir(example: Example, module_file: str | os.PathLike[str] = __file__) -> pathlib.Path:
    """The directory holding an example's scripts and assets: the wheel's copy, else the source checkout's.

    @param example The example.
    @param module_file Where this module is; the default, this file, places both candidates.
    @return The directory that has the example's entry script.
    @throws errors.ConfigError Neither has it: a `robots` installed on its own, without the dream library.
    """
    here = pathlib.Path(module_file).resolve()
    for root in (here.parents[1] / WHEEL_DIR, here.parents[3] / CHECKOUT_DIR):
        if (root / example.directory / example.entry).is_file():
            return root / example.directory
    raise errors.ConfigError(
        "the dream examples are not installed here: they come with the dream library: pip install --upgrade dreamlabs"
    )


def list_examples(out: IO[str] = sys.stdout) -> int:
    """Prints every example's name and summary, and how to run one.

    @param out Where the list goes.
    @return 0.
    """
    for name, example in EXAMPLES.items():
        print(f"  {name:<12} {example.summary}", file=out)
    print("run one with: dreamcli example <name>", file=out)
    return 0


def run(
    name: str,
    args: Sequence[str],
    *,
    out: IO[str] = sys.stdout,
    execv: Callable[[str, list[str]], object] | None = None,
) -> int:
    """Starts an example in place of this process.

    @param name The example's name, a key of #EXAMPLES.
    @param args Arguments for the example's entry script.
    @param out Where a failure to find it is reported.
    @param execv Replaces the process; None is `os.execv`, looked up at the call so a test can patch it.
    @return 1 when the example is unknown or not installed; its exit code on Windows; otherwise this does not return.
    """
    directory = _directory(name, out)
    if directory is None:
        return 1
    argv = [sys.executable, str(directory / EXAMPLES[name].entry), *args]
    if os.name == "nt":
        return subprocess.run(argv, check=False).returncode
    sys.stdout.flush()
    (execv or os.execv)(sys.executable, argv)
    return 0


def self_test(name: str, *, out: IO[str] = sys.stdout) -> int:
    """Runs an example's headless check from where it is installed.

    @param name The example's name, a key of #EXAMPLES.
    @param out Where a failure to find it is reported.
    @return The check's exit code: 0 when it passed.
    """
    directory = _directory(name, out)
    if directory is None:
        return 1
    script, *rest = EXAMPLES[name].self_test
    return subprocess.run([sys.executable, str(directory / script), *rest], check=False).returncode


def _directory(name: str, out: IO[str]) -> pathlib.Path | None:
    if name not in EXAMPLES:
        print(f"no example named {name!r}; the examples are: {', '.join(EXAMPLES)}", file=out)
        return None
    try:
        return example_dir(EXAMPLES[name])
    except errors.ConfigError as exc:
        print(exc, file=out)
        return None

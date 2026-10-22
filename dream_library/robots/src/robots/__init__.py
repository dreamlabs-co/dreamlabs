"""Public inference control loop for a bimanual YAM follower setup driven by a `dream` policy.

The package exports nothing but its version and the commit its wheel was built from. Import the modules directly, e.g.
`from robots import control`, `from robots import config`. The module map:

- `robots.types`: the loop's own dataclasses (#robots.types.Observation in, #robots.types.Action out).
- `robots.errors`: the `RobotsError` hierarchy.
- `robots.config`: the YAML config file and `load()`.
- `robots.followers`: `FollowerArm` over a `Driver`, the one arm interface the loop calls, in URDF units.
- `robots.yam`: the i2rt YAM's driver. `robots.cameras`: the OpenCV and RealSense cameras.
- `robots.motor_arm`: the one thread of every bus arm; `robots.makers`, `robots.metal` and `robots.so101` are tables
  for it, `robots.feetech` the SO-101's bus on LeRobot.
- `robots.bodies`: what differs between the robots a station drives, picked by the config's `body` (`for_config()`).
- `robots.driver_template`: the commented starting point for a robot of your own (`body: custom`).
- `robots.adapter`: the one seam that knows the serve wire`s key names and shapes (what `dream` forwards).
- `robots.control`: `ControlLoop`, the fixed-rate tick.
- `robots.inference_log`: the JSON Lines file the loop writes one record per inference, error and stall to.
- `robots.home`: `return_to_home()` and `python -m robots.home`, the move to each arm's home, grippers open.
- `robots.run`: `main()` and the `python -m robots.run --config path.yaml` entry point.
- `robots.station`, `robots.station_api`: the resident station agent (`python -m
  robots.station`) and its HTTP control API.
- `robots.cli`, `robots.setup_wizard`: the `dreamcli` command line (`setup`, `robot up`, `robot down`,
  `robot log`) and the setup wizard page that writes the station config.
"""

import importlib.metadata
import re

## The distributions that install the package, by name: the dream library's wheel (`dreamlabs`), then
## `dream-robots` from a source checkout.
_DISTRIBUTIONS = ("dreamlabs", "dream-robots")

## `__version__` when no installed distribution carries the package (a bare `PYTHONPATH` of the source).
_SOURCE_VERSION = "0+unknown"


def _installed_version() -> str:
    """The version of the distribution that installed the package: `0.0.3+g<the first 12 of its commit>` for the
    dream library (`dreamlabs`) the platform builds, `dream-robots`' own from a
    source checkout.

    @return The version; #_SOURCE_VERSION when neither is installed.
    """
    for name in _DISTRIBUTIONS:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return _SOURCE_VERSION


def build_commit(version: str) -> str | None:
    """The commit a version was built from: the `g<commit>` local part the platform gives its wheel.

    @param version A version, such as `0.0.3+g1f100a0fc975`.
    @return The commit's first 12 characters, such as `1f100a0fc975`; None for a version with no such part.
    """
    found = re.fullmatch(r"[^+]+\+g([A-Za-z0-9]+)", version)
    return found.group(1) if found else None


## Package version, the installed distribution's (_installed_version()).
__version__ = _installed_version()

## The commit this install's wheel was built from (build_commit()); None from a source checkout.
COMMIT = build_commit(__version__)

# dream_library

The public dream library: everything a customer installs to drive a robot with a Dream-1 policy through DreamAPI. Two products live here, each with its own package, tests, docs and changelog:

| directory | distribution | what it is |
| --- | --- | --- |
| `dream_client/` | `dreamlabs-client` (import name `dream`) | the client, one object `dream.DreamClient`: the fine-tuning calls, and running a model through DreamAPI with `policy(config)`, `set_observation(frames, observation)` on every camera tick and `get_action()`. `dream_client/README.md` |
| `robots/` | `dream-robots` (import name `robots`; console script `dreamcli`) | the station: `dreamcli setup` (the setup wizard), `dreamcli robot up` / `down` / `log` (the station agent), and the inference control loop for bimanual YAM followers underneath. `robots/README.md` |

`dreamlabs`, the distribution this directory's `pyproject.toml` defines, is the library as one pip install: it depends on `dreamlabs-client` for the client (the `dream` package, with `shared.datasets`, the dataset checks it runs) and its wheel carries the robots package, the parts of the repository's `shared` package robots imports (`shared.contracts.live`, the live-view wire; `shared.faceblur`, the blur at capture), copied in from `../shared` when the wheel is built, the vendored YAM arm driver `i2rt` (`vendor/i2rt/`), and dream's IK (`dream_ik`, from `vendor/dream_ik/`, with the `pyroki` and `jaxls` it solves with, `vendor/pyroki/` and `vendor/jaxls/`), which the client's `ik` config key uses. One install gives the client, the robots package and the `dreamcli` command. No file is in both wheels, so the two install side by side, and `pip install dreamlabs-client` alone is the client with nothing else.

## Examples

[Dream YAM Sim](examples/yam_mujoco/README.md) runs two simulated arms with live cameras and a task terminal: `dreamcli example yam-mujoco` once the library is installed (`examples/README.md`).

## Install

On a station, the platform's Setup API page gives one block that does it all: uv, a Python 3.12 environment at
`~/dream-env`, that site's build of this library, and `~/.dreamlabs/credentials`. By hand, a plain pip install into a
Python 3.12 venv (Ubuntu's system Python refuses installs outside one, and Ubuntu 22.04's `python3` is 3.10, which
`dreamlabs` refuses: it takes Python 3.11 to 3.13, while the client alone, `dreamlabs-client`, takes 3.10 and newer):

```bash
uv venv --seed --python 3.12 ~/dream-env && source ~/dream-env/bin/activate
pip install dreamlabs                                         # from PyPI: everything a station runs (brings dreamlabs-client)
pip install dreamlabs-client                                  # from PyPI: the client alone, `import dream`, nothing else
pip install ./dreamlabs_client-<version>-py3-none-any.whl ./dreamlabs-<version>-py3-none-any.whl   # the two files the Setup API page downloads
pip install ./dreamlabs/dream_library/dream_client ./dreamlabs/dream_library   # from a checkout: everything (both, so the client is this tree's)
pip install ./dreamlabs/dream_library/dream_client       # from a checkout: the client alone
pip install "dreamlabs-client @ git+https://github.com/dreamlabs-co/dreamlabs.git#subdirectory=dream_library/dream_client" "dreamlabs @ git+https://github.com/dreamlabs-co/dreamlabs.git#subdirectory=dream_library"   # from the public copy
```

One install is the whole station: the base dependency list carries the arm driver's own runtime requirements, mujoco for the arm model, PyAV and websockets for the live view, and pyrealsense2 for the RealSense cameras, which is marked Linux and Python 3.12 or older because its wheels stop there. The three extras `hardware`, `ui` and `live` are kept as empty aliases, so an instruction that says `pip install "dreamlabs[hardware,ui]"` resolves and installs the same thing. One extra adds something: `so101`, the robots package's own, LeRobot's Feetech bus for an SO-101 station (`pip install "dreamlabs[so101]"`, Python 3.12 or newer; it brings torch, so only an SO-101 station installs it). The driver itself, i2rt, is in the wheel: `vendor/i2rt/` is a verbatim copy of the upstream package at the commit robots pins, and `vendor/i2rt/VENDOR.md` says why, and which of its requirements are left out: ruckig and evdev, because nothing imports them and neither ships a wheel, so the install needs no compiler and no access to GitHub. dream's IK is in the wheel the same way (`vendor/dream_ik/VENDOR.md`), and on Python 3.11 or newer the base list carries what it needs, JAX 0.7.1 among them; it is left out on 3.10, where JAX 0.7.1 does not install, and the client's `ik` key says so. The PyPI names are `dreamlabs` and `dreamlabs-client`; `dream` on PyPI is another project.

The wheel is built from the source tree it is installed from, and reaches `../shared` while it builds, so the install works from a checkout of the repository (or the clone pip makes of the git URL above); a copy of this directory alone, with no `shared/` beside it, does not build. In a checkout, `uv sync` in this directory makes a venv with both products installed editable and the shared parts copied in (`uv run dreamcli --help`); each product's own directory has its own venv, lock and `make check` for development, and `make check` here runs `tests/test_distribution.py`, which holds this file's dependencies and extras to the products' own.

## A station, end to end

```bash
# first, ~/.dreamlabs/credentials: the install block on the platform's Setup API page writes it (and installs this library)
dreamcli setup          # the setup wizard on port 8092 (open the private address it prints): arms, cameras -> ~/.dreamlabs/station.yaml
dreamcli robot up       # the station agent on that file; its API on :8091; nothing moves until a Start
dreamcli robot log      # the last 40 lines of ~/.dreamlabs/station.log
dreamcli robot down     # stops the agent: the arms switch off and go limp (support them first); it says whether they are off
```

`robots/README.md`, *Command line*, is the reference for the four commands; *Config file* there documents every key the wizard writes.

## Versions

`dreamlabs` (this directory) and `dreamlabs-client` (`dream_client/`) number their releases from `0.0.3`, their first release on PyPI; each one's `pyproject.toml` holds its current version. Wheels of either package that were installed by file before the PyPI releases carry a 4.x number, and pip orders versions by number, so a 4.x already installed counts as newer than `0.0.3`: `pip install -U dreamlabs` from PyPI leaves it in place, so uninstall it first (`pip uninstall dreamlabs dreamlabs-client`); installing the platform's two wheels by file, as the Setup API page does, replaces it. From `0.0.3` on, `dream.DreamClient` keeps the camera frames itself and `set_observation()` with a bare `get_action()` is the only way to run a model, and `DreamClient.policy()` converts between a robot's parts and the slots a fine-tune trained on; DreamAPI refuses a checkpoint whose robot has parts to any older client, so a robot with parts needs it. DreamAPI serves `dreamlabs-client` 1.0.0 and newer and refuses a 0.x release with `DA-4000 client_outdated`; a library of the old 1.x to 4.x numbering still passes. The robots package keeps its 0.x line as `dream-robots` (`robots.__version__`). `dream.__version__` is read on import from the metadata of the distribution that owns the package, which is `dreamlabs-client` in either install (with the commit, `0.0.3+g<commit>`, in the wheels the Setup API page downloads); `robots.__version__` is `dreamlabs`'s. `dreamlabs` asks for a client of the same major (`<2`), so a client release within the major needs no library release.

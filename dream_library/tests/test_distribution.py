"""The `dreamlabs` distribution is the union of what it ships: this directory's pyproject is held to the sources' own.

The wheel copies the robots package, the parts of `shared` it imports, the vendored i2rt and the vendored IK (dream_ik
with pyroki and jaxls), and depends on `dreamlabs-client` for the `dream` package, so its dependency list has to carry
every external requirement those packages declare, at a floor no lower than the source's own; a floor that drifts here
would install a version a package does not support. The same for the extras (empty names, and the robots package's
`so101`), for the paths the build reads, and for the vendored copies being the commits their VENDOR.md
names.
"""

import pathlib
import re
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

## This directory, the umbrella's root.
HERE = pathlib.Path(__file__).resolve().parent.parent

## The vendored i2rt: its directory, and the requirements of it the wheel leaves out (VENDOR.md): ruckig and evdev
## ship no wheel; pygame and qpsolvers (with its quadprog solver) are LGPL and GPL, and nothing the station runs
## imports them.
VENDORED_I2RT = HERE / "vendor" / "i2rt"
I2RT_LEFT_OUT = {"ruckig", "evdev", "pygame", "qpsolvers"}

## The vendored IK (vendor/dream_ik/VENDOR.md): dream_ik's directory, which mirrors the source repository's paths.
VENDORED_DREAM_IK = HERE / "vendor" / "dream_ik"

## The vendored IK's pyproject files, and the requirements of each the wheel leaves out: the vendored packages
## themselves, which the wheel carries, and pyroki's robot_descriptions and pyliblzfse, which no module of it imports.
VENDORED_IK = {
    VENDORED_DREAM_IK / "packages" / "dream_ik" / "pyproject.toml": {"pyroki", "jaxls"},
    HERE / "vendor" / "pyroki" / "pyproject.toml": {"jaxls", "robot-descriptions", "pyliblzfse"},
    HERE / "vendor" / "jaxls" / "pyproject.toml": set(),
}

## Requirements of the vendored IK that the other packages need on every Python, so their lines carry no marker.
IK_SHARED = {"numpy", "pydantic", "tyro"}

## The distributions whose external requirements the umbrella's base list must carry: their pyproject, and the
## names among their requirements that the wheel itself satisfies (other products, the vendored driver).
PRODUCTS = {
    HERE / "dream_client" / "pyproject.toml": set(),
    HERE / "robots" / "pyproject.toml": {"dreamlabs-client", "dreamlabs-shared"},
}


def load(path: pathlib.Path) -> dict:
    """Parses one pyproject file."""
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def requirement(spec: str) -> tuple[str, str]:
    """Splits `name[extras]>=floor ; marker` into the normalised name and its `>=`/`==` floor (`""` when none)."""
    match = re.match(r"^\s*([A-Za-z0-9_.-]+)\s*(?:\[[^\]]*\])?\s*(.*)$", spec)
    assert match is not None, spec
    name = match.group(1).lower().replace("_", "-")
    floor = re.search(r"[>=]=\s*([0-9][0-9.]*)", match.group(2))
    return name, floor.group(1) if floor else ""


def normalised(spec: str) -> str:
    """One requirement with its whitespace removed, for exact comparisons across files."""
    return re.sub(r"\s+", "", spec)


def version_key(text: str) -> tuple[int, ...]:
    """`"1.26"` -> `(1, 26, 0)` for floor comparisons; padded, so `2` and `2.0` compare equal."""
    parts = [int(part) for part in text.split(".") if part]
    return tuple(parts + [0] * (3 - len(parts)))


def umbrella() -> dict:
    """This directory's `[project]` table."""
    return load(HERE / "pyproject.toml")["project"]


def umbrella_floors() -> dict[str, str]:
    """The umbrella's base dependencies as name -> floor."""
    return dict(requirement(spec) for spec in umbrella()["dependencies"])


def test_every_product_dependency_is_carried_at_its_floor_or_higher() -> None:
    """A product's external requirement is in the umbrella's list, with a floor at least as high."""
    carried = umbrella_floors()
    for path, internal in PRODUCTS.items():
        for spec in load(path)["project"]["dependencies"]:
            name, floor = requirement(spec)
            if name in internal:
                continue
            assert name in carried, (
                f"{path.parent.name} needs {name}, which dream_library/pyproject.toml does not list"
            )
            assert version_key(carried[name]) >= version_key(floor), f"{name}: floor {carried[name]} < {floor}"


def test_the_shared_parts_dependencies_are_carried() -> None:
    """`shared.contracts` needs shared's base dependencies, `shared.faceblur` its `faceblur` extra and
    `shared.datasets` its `datasets` extra; the client's own `datasets` extra is carried too."""
    carried = umbrella_floors()
    shared = load(HERE.parent / "shared" / "pyproject.toml")["project"]
    extras = shared["optional-dependencies"]["faceblur"] + shared["optional-dependencies"]["datasets"]
    client_extras = load(HERE / "dream_client" / "pyproject.toml")["project"]["optional-dependencies"]["datasets"]
    for spec in shared["dependencies"] + extras + client_extras:
        name, floor = requirement(spec)
        assert name in carried, f"shared needs {name}, which dream_library/pyproject.toml does not list"
        assert version_key(carried[name]) >= version_key(floor), f"{name}: floor {carried[name]} < {floor}"


def test_the_base_list_carries_the_arm_driver_and_the_live_view() -> None:
    """`pip install dreamlabs` is the whole station install, so the base list carries the vendored i2rt's own runtime
    requirements (less the names left out, VENDOR.md) and every robots extra: a name both carry keeps the higher
    floor (mujoco: robots says >= 3.1, i2rt >= 3.3.5)."""
    i2rt = load(VENDORED_I2RT / "pyproject.toml")["project"]["dependencies"]
    robots_extras = load(HERE / "robots" / "pyproject.toml")["project"]["optional-dependencies"]
    carried = umbrella_floors()
    for spec in i2rt:
        name, floor = requirement(spec)
        if name in I2RT_LEFT_OUT:
            continue
        assert name in carried, f"the driver needs {name}, which the base list does not carry"
        assert version_key(carried[name]) >= version_key(floor), f"{name}: {carried[name]} < {floor}"
    for extra in ("hardware", "live", "makers", "metal"):
        for spec in robots_extras[extra]:
            name, floor = requirement(spec)
            if name == "i2rt":
                continue  # in the wheel itself
            assert name in carried, f"the {extra} extra needs {name}, which the base list does not carry"
            assert version_key(carried[name]) >= version_key(floor), f"{name}: {carried[name]} < {floor}"
    assert not {requirement(s)[0] for s in umbrella()["dependencies"]} & I2RT_LEFT_OUT


def test_the_base_list_carries_the_vendored_ik_at_dreams_pins() -> None:
    """The wheel ships dream_ik, pyroki and jaxls, so the base list carries their runtime requirements (less the ones
    VENDOR.md leaves out) at a floor no lower than theirs, dream-ik's exact pins exactly, and each only for Python 3.11
    or newer, where jax 0.7.1 exists."""
    specs = {requirement(spec)[0]: normalised(spec) for spec in umbrella()["dependencies"]}
    carried = umbrella_floors()
    for path, left_out in VENDORED_IK.items():
        for spec in load(path)["project"]["dependencies"]:
            name, floor = requirement(spec)
            if name in left_out:
                continue
            assert name in carried, f"{path.parent.name} needs {name}, which the base list does not carry"
            assert version_key(carried[name]) >= version_key(floor), f"{name}: {carried[name]} < {floor}"
            if name not in IK_SHARED:
                assert specs[name].endswith(";python_version>='3.11'"), f"{name} must be for Python 3.11 or newer"
            if "==" in spec:
                assert specs[name].split(";")[0] == normalised(spec), f"{name}: not dream-ik's pin {spec}"


def test_the_vendored_ik_is_the_commits_dream_ik_pins() -> None:
    """The copy of dream_ik holds exactly the files its pyproject force-includes, the umbrella ships that directory as
    `dream_ik`, and the vendored pyroki and jaxls are the commits dream-ik pins, named in VENDOR.md
    with the source's."""
    dream_ik = load(VENDORED_DREAM_IK / "packages" / "dream_ik" / "pyproject.toml")
    included = dream_ik["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    source = VENDORED_DREAM_IK / "src" / "dream" / "ik"
    shipped = {str(path.relative_to(source)) for path in source.rglob("*") if path.is_file()}
    assert shipped == {target.removeprefix("dream_ik/") for target in included.values()}
    vendor_md = (VENDORED_DREAM_IK / "VENDOR.md").read_text(encoding="utf-8")
    assert re.search(r"at commit `[0-9a-f]{40}`", vendor_md), "VENDOR.md names no source commit"
    pins = [
        re.match(r"(pyroki|jaxls) @ git\+https://github\.com/[\w-]+/\1\.git@([0-9a-f]{40})$", spec)
        for spec in dream_ik["project"]["dependencies"]
    ]
    pinned = dict(match.groups() for match in pins if match is not None)
    assert set(pinned) == {"pyroki", "jaxls"}, dream_ik["project"]["dependencies"]
    for name, commit in pinned.items():
        assert f"`{commit}`" in vendor_md, f"VENDOR.md does not name the {name} commit dream-ik pins"
        assert f"`{commit}`" in (HERE / "vendor" / name / "VENDOR.md").read_text(encoding="utf-8"), name
        assert (HERE / "vendor" / name / "LICENSE").read_text(encoding="utf-8").startswith("MIT License"), name
        assert load(HERE / "vendor" / name / "pyproject.toml")["project"]["name"] == name


def test_the_old_extras_still_resolve_and_add_nothing() -> None:
    """`pip install "dream[hardware,ui]"` must keep working, and it installs exactly what the base list already
    carries (`ui` is an empty name: there is no station page to install). The client's `record` extra is an empty
    name here too, its websockets being in the base list.
    The one extra that adds something is the robots package's `so101`, the same list."""
    extras = umbrella()["optional-dependencies"]
    robots_extras = load(HERE / "robots" / "pyproject.toml")["project"]["optional-dependencies"]
    client_extras = load(HERE / "dream_client" / "pyproject.toml")["project"]["optional-dependencies"]
    assert set(extras) == {"hardware", "ui", "live", "record", "so101"}
    assert extras.pop("so101") == robots_extras["so101"]
    assert all(names == [] for names in extras.values()), f"an extra that adds something again: {extras}"
    carried = umbrella_floors()
    for spec in client_extras["record"]:
        name, floor = requirement(spec)
        assert name in carried, f"the client's record extra needs {name}, which the base list does not carry"
        assert version_key(carried[name]) >= version_key(floor), f"{name}: {carried[name]} < {floor}"


def test_the_realsense_wheel_is_installed_only_where_it_exists() -> None:
    """pyrealsense2 has Linux wheels only, and none past Python 3.12, so it carries the marker that keeps
    `pip install dreamlabs` from failing on a laptop or a newer interpreter."""
    specs = {requirement(s)[0]: s for s in umbrella()["dependencies"]}
    assert "pyrealsense2" in specs, "a station's cameras are part of the one install"
    marker = specs["pyrealsense2"].split(";", 1)[1]
    assert "sys_platform" in marker and "linux" in marker
    assert "python_version" in marker and "3.13" in marker


def test_the_vendored_i2rt_is_the_commit_robots_pins() -> None:
    """VENDOR.md names the commit `robots/pyproject.toml` pins through its uv source, and the copy is complete."""
    rev = load(HERE / "robots" / "pyproject.toml")["tool"]["uv"]["sources"]["i2rt"]["rev"]
    vendor_md = (VENDORED_I2RT / "VENDOR.md").read_text(encoding="utf-8")
    assert f"`{rev}`" in vendor_md, f"VENDOR.md does not name the pinned commit {rev}"
    assert (VENDORED_I2RT / "i2rt" / "__init__.py").is_file()
    assert (VENDORED_I2RT / "i2rt" / "robots" / "get_robot.py").is_file(), "what robots.followers imports"
    assert (VENDORED_I2RT / "i2rt" / "robots" / "utils.py").is_file(), "what robots.kinematics imports"
    assert (VENDORED_I2RT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")
    assert load(VENDORED_I2RT / "pyproject.toml")["project"]["name"] == "i2rt"


def test_the_build_reads_paths_that_exist() -> None:
    """Every package directory and every force-included path is on disk, so a wheel build cannot go stale."""
    wheel = load(HERE / "pyproject.toml")["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert set(wheel["packages"]) == {
        "robots/src/robots",
        "vendor/i2rt/i2rt",
        "vendor/pyroki/src/pyroki",
        "vendor/jaxls/src/jaxls",
    }
    for package in wheel["packages"]:
        assert (HERE / package / "__init__.py").is_file(), package
    for source, target in wheel["force-include"].items():
        assert (HERE / source).exists(), source
        allowed = ("shared/", "i2rt/", "pyroki/", "jaxls/", "dream_examples/")
        assert target.startswith(allowed) or target == "dream_ik", target
    assert wheel["force-include"]["vendor/i2rt/LICENSE"] == "i2rt/LICENSE"
    assert wheel["force-include"]["vendor/pyroki/LICENSE"] == "pyroki/LICENSE"
    assert wheel["force-include"]["vendor/jaxls/LICENSE"] == "jaxls/LICENSE"
    assert wheel["force-include"]["vendor/dream_ik/src/dream/ik"] == "dream_ik"
    assert {pathlib.Path(t).parts[1] for t in wheel["force-include"].values() if t.startswith("shared/")} == {
        "contracts",
    }


def test_the_two_wheels_share_no_file() -> None:
    """`dreamlabs` depends on `dreamlabs-client` and neither ships a file of the other: the `dream` package and
    `shared/__init__.py`, `py.typed`, `shared/datasets` and `shared/faceblur` are the client's, `shared/contracts`
    this distribution's, so installing, upgrading or removing one never overwrites or deletes a file of the other."""
    client = load(HERE / "dream_client" / "pyproject.toml")["tool"]["hatch"]["build"]["targets"]["wheel"]
    umbrella_wheel = load(HERE / "pyproject.toml")["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert client["packages"] == ["src/dream"]
    assert not any(package.endswith("/dream") for package in umbrella_wheel["packages"])
    client_shared = {pathlib.Path(t).parts[1] for t in client["force-include"].values() if t.startswith("shared/")}
    umbrella_shared = {
        pathlib.Path(t).parts[1] for t in umbrella_wheel["force-include"].values() if t.startswith("shared/")
    }
    assert client_shared == {"__init__.py", "py.typed", "datasets", "faceblur"}
    assert not client_shared & umbrella_shared


def test_the_umbrella_depends_on_the_clients_own_major() -> None:
    """The client comes from `dreamlabs-client`, with the extras its dataset checks and live view need, at a range that
    starts at the client's version and stops before its next major, the one change of API the gateway tells apart."""
    client = load(HERE / "dream_client" / "pyproject.toml")["project"]
    major = int(client["version"].split(".")[0])
    specs = [normalised(spec) for spec in umbrella()["dependencies"] if requirement(spec)[0] == "dreamlabs-client"]
    assert specs == [f"dreamlabs-client[datasets,live]>={client['version']},<{major + 1}"], specs
    assert client["name"] == "dreamlabs-client" and umbrella()["name"] == "dreamlabs"


def test_the_wheel_carries_what_dreamcli_example_runs_and_nothing_it_does_not() -> None:
    """Every module the yam-mujoco example's scripts import from their own directory ships with them, with the assets;
    its tests, its asset tools and its stand-alone requirements file do not."""
    wheel = load(HERE / "pyproject.toml")["tool"]["hatch"]["build"]["targets"]["wheel"]
    source = HERE / "examples" / "yam_mujoco"
    shipped = {
        pathlib.Path(src).relative_to("examples/yam_mujoco").as_posix()
        for src in wheel["force-include"]
        if src.startswith("examples/yam_mujoco/")
    }
    for src, target in wheel["force-include"].items():
        if src.startswith("examples/yam_mujoco/"):
            assert target == "dream_examples/yam_mujoco/" + src.removeprefix("examples/yam_mujoco/"), target
    assert "yam_mujoco.py" in shipped and "assets" in shipped
    local = {path.stem for path in source.glob("*.py")}
    for name in [s for s in shipped if s.endswith(".py")]:
        imported = set(re.findall(r"^\s*(?:from|import) (\w+)", (source / name).read_text(), re.M))
        missing = {module + ".py" for module in imported & local} - shipped
        assert not missing, f"{name} imports {missing}, which the wheel does not carry"
    assert not {s for s in shipped if s.startswith(("test_", "tools", "mock_", "audit_")) or s == "requirements.txt"}


def test_the_umbrella_script_is_the_robots_script() -> None:
    """`dreamcli` points at the same entry point the robots product installs."""
    robots = load(HERE / "robots" / "pyproject.toml")["project"]["scripts"]
    assert umbrella()["scripts"] == robots == {"dreamcli": "robots.cli:main"}

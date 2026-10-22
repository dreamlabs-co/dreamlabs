# vendor/i2rt: the YAM arm driver, vendored

A verbatim copy of three paths of https://github.com/i2rt-robotics/i2rt at commit `4e462ecc435491a46f9bceb465237afb41d7b66f` (its release v1.1.1, the commit `robots/pyproject.toml` pins through its uv source): the `i2rt/` package, `LICENSE` (MIT) and `pyproject.toml`. Nothing here is edited; `tests/test_distribution.py` reads the pyproject to hold the `dream` distribution's `hardware` extra to it.

Why a copy and not a dependency: i2rt pins `ruckig==0.15.3`, a source-only release whose build breaks under scikit-build-core 0.10 and newer, so a plain `pip install dream` that resolved i2rt from its repository failed on every fresh machine and needed a build constraint on the command line. ruckig is imported by one i2rt module, `flow_base/flow_base_controller.py` (the mobile base), which the YAM arms never use. The `dream` wheel ships `i2rt` from this directory (`pyproject.toml`, force-include), and its dependency list is i2rt's runtime requirements without ruckig, so the install needs neither a build nor access to GitHub. i2rt is MIT; `LICENSE` ships inside the package as `i2rt/LICENSE`.

Left out with it: `evdev==1.9.2`, which ships no wheel either (it builds against the kernel headers, so a plain install would need a compiler and `python3-dev` on the station). Nothing in this copy imports it; only i2rt's own gamepad and mobile-base code would.

Also left out, for their licenses: `pygame==2.6.1` (LGPL-2.1) and `qpsolvers[quadprog]==4.7.1` (qpsolvers is LGPL-3.0; its quadprog solver is GPL-2.0-or-later). Nothing the station runs imports either: pygame is imported only by `i2rt/utils/gamepad_utils.py` (i2rt's gamepad teleoperation), and qpsolvers only through `mink` in `i2rt/robots/kinematics.py`, which only i2rt's own viser and MuJoCo control panels import. `mink` itself stays in the list (Apache-2.0; it brings qpsolvers with its MIT `daqp` solver, which `pip install dreamlabs` therefore still installs). The `dreamlabs` wheel carries no GPL dependency, and its `THIRD-PARTY-LICENSES.md` lists the LGPL ones it does.

`viser>=0.2.0`, the web visualiser library, was left out as well until the vendored IK (`../dream_ik/VENDOR.md`) needed it: its pyroki imports viser, so the `dream` install carries it again, on Python 3.11 or newer. One module of this copy uses it, `i2rt/utils/viser_control_interface.py` (i2rt's own browser control panel), which imports it only when the panel is started; nothing the station loads imports that module. The module stays in the copy because the copy is verbatim.

To move to a newer i2rt commit, from a clone of the upstream repository:

```bash
git -C <clone> archive <commit> i2rt LICENSE pyproject.toml | tar -x -C dream_library/vendor/i2rt
```

then update the commit above and in `robots/pyproject.toml` (its uv source), and run `make check` in `dream_library/`, which fails if the extra no longer matches the new pyproject.

# vendor/dream_ik: Dream Labs' IK, vendored

The IK the `dream` client runs for an EEF checkpoint (the `ik` config key, `dream_client/src/dream/client_ik.py`): forward kinematics of the measured joints into the observation, and the chunk solve that turns the tool poses a model answers with into joints. It is Dream Labs' own IK, the same code and arm models that computed the EEF columns the Metal, Makers and FaaS beta robot checkpoints trained on, so the poses the client sends and solves are in the training frame.

A verbatim copy of these paths of Dream Labs' model code at commit `13ec9d33d1d0b4b7ca23fb3678f0b6fd43b3923c` (dream-ik 0.3.1): `packages/dream_ik/` (the `dream-ik` wheel's `pyproject.toml` and `README.md`) and every file of `src/dream/ik/` that pyproject force-includes (`__init__.py`, `specs.py`, `solver.py`, `decode.py` and the URDFs under `assets/`; not `serve.py`, which needs torch). 0.3.1 is the trajectory solve at velocity 5 / acceleration 50, `ServeSpec.fences` off by default, and `decode.extend_poses`, which the client pads a `traj` chunk with; it is on top of 0.3.0 (the opt-in trajectory solve, `ServeSpec.solve` `traj`, that the `ik` key's `solve` turns on, and `Region`, that its `region` builds), the zero-shot sim robots' arm models, a FaaS EEF fine-tune reading its robot's own joints, dream-ik 0.2.3 (per-station obstacles, a camera mount being the station's `keep_out`), the ten FaaS EEF beta robots (an arm spec per side), the Makers Arm, and, with the seam blend off, the solve still clipped to the margined limits and speed capped.

Verbatim except some comments and docstrings, reworded here; make the same change in the source before re-vendoring.

Two libraries the solver needs are vendored beside it, each a verbatim copy at the commit `packages/dream_ik/pyproject.toml` pins, and each MIT licensed (their `LICENSE` ships inside the package):

- `../pyroki`: https://github.com/chungmin99/pyroki at `388e43e1fc0d0ee382968d3dd72970fd62a0450c`, paths `src/pyroki/`, `LICENSE`, `README.md`, `pyproject.toml`.
- `../jaxls`: https://github.com/brentyi/jaxls at `50a58be88c5ef74532f09e3f55268b4f02c490e3`, the same four paths with `src/jaxls/`.

Why copies and not dependencies: the source is private, and neither pyroki nor this jaxls is on PyPI (the PyPI project named `jaxls` is an unrelated package), so a dependency on any of them would need GitHub at install time. The `dream` wheel ships the three packages (`dream_ik`, `pyroki`, `jaxls`; `dream_library/pyproject.toml`), and its dependency list carries their runtime requirements from PyPI, so `pip install dream` stays one install with no access to GitHub. Those requirements carry the marker `python_version >= '3.11'`, because jax 0.7.1, the version the source pins, has no release for Python 3.10: on 3.10 the install leaves the IK out and the `ik` key says so. Left out of the list: pyroki's `robot_descriptions` and `pyliblzfse`, which no module of the copy imports (`pyliblzfse` also ships no Linux wheel, so it would need a compiler); `tyro`, `numpy` and `pydantic` are carried for the other packages already.

The client's own dev venv (`dream_client/pyproject.toml`) installs `dream-ik` from `packages/dream_ik` here, with pyroki and jaxls overridden to these copies, so its tests run the code the wheel ships, also without GitHub.

To move to newer commits, from clones of the three repositories (the pyroki and jaxls commits are the ones the new `packages/dream_ik/pyproject.toml` pins):

```bash
git -C <model repository clone> archive <commit> packages/dream_ik src/dream/ik ':(exclude)src/dream/ik/serve.py' | tar -x -C dream_library/vendor/dream_ik
git -C <pyroki clone> archive <commit> src/pyroki LICENSE README.md pyproject.toml | tar -x -C dream_library/vendor/pyroki
git -C <jaxls clone> archive <commit> src/jaxls LICENSE README.md pyproject.toml | tar -x -C dream_library/vendor/jaxls
```

then update the commits here and in the two `VENDOR.md` beside, run `uv lock` in `dream_library/` and in `dream_library/dream_client/`, and run `make check` in `dream_library/` (`tests/test_distribution.py` fails if the copy, the commits or the dependency list no longer match) and the client's tests in a fresh venv.

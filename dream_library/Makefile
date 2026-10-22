# The dream library as one pip install. `make check` and `make install-check` are what CI runs (dream-library.yml);
# each product under this directory has its own Makefile and gate.
.PHONY: check install-check

# The uv venv from this directory's lock: both products editable, the shared parts copied in, `dreamcli` on the
# path, the vendored IK importable (its JAX stack installed), and the distribution test that holds this pyproject to
# the products' own.
check:
	uv sync --frozen
	uv run ruff check tests && uv run ruff format --check tests
	uv run pytest -q
	uv run dreamcli --help >/dev/null
	uv run python -I -c "import dream, dream.datasets, dream.fine_tuning, robots, shared.contracts.live, shared.datasets.validation, shared.faceblur, i2rt, dream_ik.solver; print('dream', dream.__version__, 'dream-robots', robots.__version__)"

# The pip path a customer takes: `pip install <this directory>/dream_client <this directory>` into a fresh venv (both,
# so pip takes the client from here rather than from an index) gives everything and dreamcli, as `pip install
# dreamlabs` does from PyPI; `pip install <this directory>/dream_client` gives the client alone; in each,
# `dream.__version__` is the client's version, the distribution that owns the `dream` package, and no file belongs to
# both distributions. -I keeps the cwd off sys.path, so the robots/ directory here cannot pass for the package as a
# namespace import.
install-check:
	rm -rf build/install-check && python3 -m venv build/install-check/full build/install-check/client
	build/install-check/full/bin/pip install --quiet ./dream_client .
	build/install-check/full/bin/pip check
	build/install-check/full/bin/dreamcli --help >/dev/null
	build/install-check/full/bin/python -I -c "from robots import examples as e; d = e.example_dir(e.EXAMPLES['yam-mujoco']); assert 'site-packages' in d.parts, d; print('yam-mujoco example at', d)"
	build/install-check/full/bin/dreamcli example yam-mujoco --self-test >/dev/null && echo "dreamcli example yam-mujoco --self-test: the scene builds from the wheel's assets"
	build/install-check/full/bin/python -I -c "import dream, dream.datasets, dream.fine_tuning, robots, shared.contracts.live, shared.datasets.validation, shared.faceblur, i2rt, dream_ik.solver"
	build/install-check/full/bin/python -I -c "import mujoco, av, websockets, can; print('the station stack is in the one install')"
	build/install-check/full/bin/python -I -c "import dream, importlib.metadata as m; assert dream.__version__ == m.version('dreamlabs-client'), dream.__version__; print('dreamlabs-client', dream.__version__, 'dreamlabs', m.version('dreamlabs'))"
	build/install-check/full/bin/python -I -c "import importlib.metadata as m; both = {str(f) for f in m.files('dreamlabs')} & {str(f) for f in m.files('dreamlabs-client')}; assert not both, sorted(both); print('no file in both distributions')"
	build/install-check/client/bin/pip install --quiet ./dream_client
	build/install-check/client/bin/python -I -c "import dream, dream.datasets, dream.fine_tuning"
	build/install-check/client/bin/python -I -c "import dream, importlib.metadata as m; assert dream.__version__ == m.version('dreamlabs-client'), dream.__version__; print('dreamlabs-client', dream.__version__)"
	! build/install-check/client/bin/python -I -c "import robots" 2>/dev/null
	test ! -e build/install-check/client/bin/dreamcli

# Contributing to the dream client

This file is for whoever changes the `dream` client package itself: its commands, its layout and how its wheel is built. What the client is and how a customer uses it is `README.md`, which ships to customers; the conventions of any change in this repository (worktrees, gates, commits, the pull request) are the repository root's `CONTRIBUTING.md`.

## Development

```bash
cd dream_library/dream_client
make test          # uv run pytest -q (short stress variants included)
make test-full     # DREAM_STRESS=full: nightly-length stress tests
make lint          # ruff check + ruff format --check
make typecheck     # mypy --strict over src/dream
make check         # lint + typecheck + test + doxygen, what CI runs
make doxygen       # docstring check + Doxygen (warnings as errors) into documentation/html
make docs          # build the API reference of the public surface into build/docs
make publish-docs  # build, then copy to the documentation endpoint
```

The package uses a `src/` layout so tests run against the installed package. The only runtime dependency is `numpy` (and pyarrow and boto3 for `dream.datasets`, the `datasets` extra); the wire is JSON over `http.client` from the standard library. The wheel carries `shared.datasets`, the embodiment config parser and the dataset checks, copied in at build time from the monorepo's `shared` package, so it is built from a checkout, never an sdist. The venv is managed by `uv` (`uv sync` once; CI uses `uv sync --frozen`). Every module, class and public function carries a Doxygen-form docstring per the house standards.

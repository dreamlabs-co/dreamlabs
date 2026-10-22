"""One IK for every embodiment: ``specs`` is the data, ``solver`` the PyRoKi chunk solve.

The solve is tiny and runs next to a GPU model, so JAX stays on the CPU and never preallocates GPU
memory; XLA's multi-threaded CPU kernels cost more in thread handoffs than they save on solves this
small (3x slower chunks with obstacles). Set here, before ``dream.ik.solver`` can import JAX.
"""

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Compiled solves persist across processes (jax's persistent cache): a station's policy client, and the serve, then
# warm up in about half the time (2.4 s against 4.7 s cold on a Xeon core: the compile is cached, the trace is not). Any failure to use the directory is a jax warning.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache" / "dream_ik" / "jax"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
_ONE_THREAD = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"  # as our teleop IK runs
if _ONE_THREAD not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} {_ONE_THREAD}".strip()

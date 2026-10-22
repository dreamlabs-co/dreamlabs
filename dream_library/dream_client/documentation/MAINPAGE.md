# dream {#mainpage}

The Dream Labs client: fine-tune Dream-1 on a robot's data and run the result. Its public surface is `dream.DreamClient` (embodiments, datasets, fine-tuning, models, and running a model with `policy()`, `set_observation()` on every camera tick and `get_action()`), plus the `DreamError` hierarchy, `DreamSlowNetworkWarning` and `__version__`. The policy client `DreamClient` runs a model on (`dream.client.PolicyClient`) is internal.

| Module | Job |
|---|---|
| `dream` (`src/dream/__init__.py`) | Exports `DreamClient` and the errors; the private `_open_policy()` splits a config into the keys consumed locally and the session keys, and opens the policy client |
| `dream.facade` | `DreamClient`: the whole lifecycle in one object; carries the key and the three hosts, forwards to the modules below, and keeps the frame window of the open policy |
| `dream._window` | The frame window: which camera ticks `set_observation()` keeps (one every `1 / obs_fps` seconds) and the last `obs_window` frames of each camera |
| `dream._recorder` | Internal: every tick `DreamClient` is given, sent to the platform's live service in the background, best effort |
| `dream.client` | `PolicyClient`: one inference call, one episode boundary, a lifecycle; validation against the embodiment |
| `dream.transport` | The only module that knows how bytes reach DreamAPI: one keep-alive HTTP connection, JSON with numpy arrays as lists |
| `dream.smoothing` | Seam smoothing: blends each returned chunk into the plan the arms are already on, from the observation and the previous chunk |
| `dream.credentials` | The credentials file and environment: where the gateway, the platform, the data service and the key are read from |
| `dream.datasets` | Embodiments and datasets on the data service: describe a robot body, check a dataset locally, upload it |
| `dream.fine_tuning` | Fine-tuning jobs on the data service, over `dream.datasets`'s wire: create, list, get, cancel, the queue |
| `dream.models` | The checkpoints an account may run, from the platform's `GET /v1/models` |
| `dream.errors` | The exception hierarchy, rooted at `DreamError` |

**Start here**

- [Package list](namespaces.html), [Class list](annotated.html), [File list](files.html).
- Pages: the README and the changelog, rendered from the Markdown that ships with the package.
- The user-facing API reference (`DreamClient`'s documented surface, held equal to its public members; internals cannot appear) is `make docs`; this site is the maintainer's view and includes every internal.

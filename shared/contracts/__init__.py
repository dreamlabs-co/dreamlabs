"""The DreamAPI wire contracts every consumer reads: system constants, the status envelope and codes, the `/v1` schema,
and the platform seam (key export, checkpoint export, usage batches).

Seven modules, one concern each:

- `constants`: the system constants and the platform seam's cadences and price, declared once here and imported
  everywhere.
- `envelope`: the seven response codes, their reason enums, and the `Status` envelope.
- `v1`: the request/response models of `POST /v1/get_action` and `POST /v1/end_session`.
- `usage`: `UsageEvent` and `UsageBatch`, the body of `POST /internal/usage/batches` the gateway pushes to the
  platform.
- `keys`: `KeyExportEntry` (the key-file value type `GET /internal/keys/export` returns) and `KeyLookup` (the body
  of `GET /internal/keys/{sha256}`).
- `checkpoints`: `CheckpointExport`, the body of `GET /internal/checkpoints/export`, the catalog the platform owns
  and DreamAPI pulls.
- `live`: the live-view wire: the 16-byte video header, the robot-link and viewer-link
  message models, the close codes, the viewer token codec, the bodies of `POST /internal/live/authorize` and
  `POST /internal/live/reports`, and the constants both the `live` node and the `dream-robots` agent read.
"""

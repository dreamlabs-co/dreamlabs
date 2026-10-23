"""The DreamAPI wire contracts every consumer reads: system constants, the status envelope and codes, the `/v1` schema,
and the bodies the platform's own services exchange (key export, checkpoint export, usage batches).

Seven modules, one concern each:

- `constants`: the system's constants and the platform's cadences and price, declared once here and imported
  everywhere.
- `envelope`: the seven response codes, their reason enums, and the `Status` envelope.
- `v1`: the request/response models of `POST /v1/get_action` and `POST /v1/end_session`.
- `usage`: `UsageEvent` and `UsageBatch`, the usage batches the gateway pushes to the platform.
- `keys`: `KeyExportEntry` (one entry of the platform's key export) and `KeyLookup` (one key looked up by its hash).
- `checkpoints`: `CheckpointExport`, the catalog the platform owns and DreamAPI pulls.
- `live`: the live-view wire: the 16-byte video header, the robot-link and viewer-link message models, the close
  codes, the viewer token codec, the bodies the live node posts to the platform, and the constants both the `live`
  node and the robot's agent read.
"""

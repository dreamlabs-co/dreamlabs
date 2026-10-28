"""The DreamAPI wire contracts the dream library reads: the status envelope and codes, the live-view wire, and the
station's switches.

Four modules, one concern each:

- `envelope`: the seven response codes, their reason enums, and the `Status` envelope.
- `live`: the live-view wire: the 16-byte video header, the robot-link and viewer-link message models, the close
  codes, the viewer token codec, and the constants both the `live` node and the robot's agent read.
- `station`: the switches `dreamcli setup` reads on the station.
- `env`: the environment reader those switches use for a boolean.
"""

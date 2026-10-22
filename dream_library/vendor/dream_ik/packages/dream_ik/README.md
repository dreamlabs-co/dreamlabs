# dream-ik

dream's IK (`src/dream/ik`) as a wheel that installs without dream or torch: the PyRoKi chunk
solver, the arm specs (YAM, Metal, B601, Makers) with their URDFs, and the EEF-head decode. It is what a station runs to
turn the tool poses a served EEF checkpoint returns into joints for itself (`ik.exec = "eef"`). Since
2026-10-10 (serving contract v3) it is the only place an EEF head becomes joints: the serve never solves,
because its solve started from the arm as of the request, a round trip before the chunk was installed.

```python
from dream_ik.solver import ChunkIK
from dream_ik.decode import decode_chunk

ik = ChunkIK("yam", "tcp", hz=30.0)   # the station's arm, the checkpoint's tool frame, its control rate
ik.warmup(48)                         # compile once (4 to 6 s cold, about half that with the persistent cache)
q, diag = decode_chunk(ik, xyz, rot6, q_from, speed_cap=2.0)
# xyz (arms, 48, 3), rot6 (arms, 48, 6): the served action.{side}_eef_xyz / _eef_rot6, arms in ik.sides order
# q_from (arms, 6): the joints the chunk continues from (the last command)
# q (arms, 48, 6): joint commands after the smooth, the solve and the seam blend; diag: misses, cap ticks, motion
```

Install a pinned commit of the model repository:

```
dream-ik @ git+<model repository URL>@<sha>#subdirectory=packages/dream_ik
```

The modules are force-included from `src/dream/ik` (one copy; a test checks the list and that
nothing in it imports torch or the model code). Build locally with `uv build --wheel packages/dream_ik`.

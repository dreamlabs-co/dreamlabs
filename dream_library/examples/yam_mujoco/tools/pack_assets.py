"""Packs the example's generated assets into the compact form the repository keeps.

Run once after regenerating the convex decomposition or downloading new fruit scans; the outputs are committed and
nothing here runs at simulation time.

- Collision hulls: every `assets/collisions/*.obj` named in `assets/collisions/manifest.json` becomes one float32
  vertex array in `assets/collisions.npz`, keyed `<source mesh>/<part index>`. MuJoCo builds a mesh collider from
  the convex hull of its vertices, so the faces are not needed. The OBJ directory can then be deleted.
- Fruit textures: `assets/fruits/<fruit>/texture_map.png` is resized to #TEXTURE_PX square. The cameras render at
  640x480 and a fruit covers well under a hundred pixels of that, so the 4096 px scans add only bytes.

Usage, from the example directory: `python tools/pack_assets.py`.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

ASSETS = Path(__file__).resolve().parent.parent / "assets"

## Side of the square fruit textures kept in the repository, pixels.
TEXTURE_PX = 1024


def read_obj_vertices(path):
    """Reads the vertex positions of a Wavefront OBJ file.

    @param path The OBJ file.
    @return An (N, 3) float32 array.
    """
    rows = [line.split()[1:4] for line in path.read_text().splitlines() if line.startswith("v ")]
    return np.array(rows, dtype=np.float32)


def pack_collisions():
    """Writes `assets/collisions.npz` from the hull OBJ files the manifest lists.

    @return The number of hulls packed.
    """
    manifest = json.loads((ASSETS / "collisions" / "manifest.json").read_text())
    arrays = {}
    for source, parts in manifest.items():
        for index, part in enumerate(parts):
            arrays[f"{source}/{index}"] = read_obj_vertices(ASSETS / part)
    np.savez_compressed(ASSETS / "collisions.npz", **arrays)
    return len(arrays)


def shrink_textures():
    """Resizes each fruit's texture map to #TEXTURE_PX square, in place, when it is larger.

    @return The paths resized.
    """
    resized = []
    for path in sorted((ASSETS / "fruits").glob("*/texture_map.png")):
        image = Image.open(path)
        if max(image.size) > TEXTURE_PX:
            image.convert("RGB").resize((TEXTURE_PX, TEXTURE_PX), Image.LANCZOS).save(path, optimize=True)
            resized.append(path)
    return resized


if __name__ == "__main__":
    print(f"packed {pack_collisions()} collision hulls into {ASSETS / 'collisions.npz'}")
    for path in shrink_textures():
        print(f"resized {path.relative_to(ASSETS)} to {TEXTURE_PX} px")

# Robot collision parts

`collisions.npz` holds convex decompositions of the robot STL meshes in this directory, retaining their license (`LICENSE`): one float32 vertex array per part, keyed `<source mesh>/<part index>`, from which MuJoCo builds each part's convex hull. They were generated with CoACD using seed 42, a requested 2 mm threshold, preprocessing resolution 80, and up to 128 parts per mesh, then packed by `../tools/pack_assets.py`. The requested threshold is not a certified maximum surface error, especially around repaired open meshes and recesses.

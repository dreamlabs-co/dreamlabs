# YCB scanned fruit assets

Source: The YCB Object and Model Set, https://ycb-benchmarks.s3.amazonaws.com/index.html
Creators: Berk C. Calli, Arjun Singh, Aaron Walsman, Siddhartha Srinivasa,
Pieter Abbeel, Aaron M. Dollar; meshes captured with the Google scanner.
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
https://creativecommons.org/licenses/by/4.0/

Downloaded 2026-10-13:
- 013_apple, Google 16k textured scan:
  https://ycb-benchmarks.s3.amazonaws.com/data/google/013_apple_google_16k.tgz
- 011_banana, Google 16k textured scan:
  https://ycb-benchmarks.s3.amazonaws.com/data/google/011_banana_google_16k.tgz

Changes: visual.obj recenters the original textured.obj's bounding box and uniformly
rescales its longest axis to 66 mm (apple) / 140 mm (banana); UVs are kept.
geometry.json records that transform and the approximate collision centerline.
texture_map.png is the original 4096 px map resized to 1024 px by
../../tools/pack_assets.py. The original files are in the archives above. Collision primitives and masses
are simulator approximations, not supplied or validated by YCB. No endorsement
by the dataset authors is implied.

# `shared/faceblur/models`: vendored detector weights

Model files loaded by `shared.faceblur` at runtime. Vendored, not downloaded on first use: the
module fails closed when a file here is missing, and a download would turn a network hiccup into a stopped
robot camera at the start of a run.

## face_detection_yunet_2026may.onnx

| | |
|---|---|
| What | YuNet face detector, loaded through `cv2.FaceDetectorYN` by `shared.faceblur` |
| Source | [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo), `models/face_detection_yunet/face_detection_yunet_2026may.onnx` |
| Size / sha256 | 229,738 bytes / `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0` |
| Licence | MIT, © 2020 Shiqi Yu (`models/face_detection_yunet/LICENSE` in the zoo; a copy is `LICENSE` beside this file, and it ships with the weights) |
| Accuracy | WIDER Face val AP 0.884 easy / 0.866 medium / 0.750 hard (the zoo's measurement) |

Why the `2026may` export and not `2023mar`: the 2023 file has static input dims, and OpenCV 5.x's ONNX engine
needs symbolic dims to infer at a variable resolution ([opencv_zoo#44](https://github.com/opencv/opencv_zoo/issues/44)) -
and frames arrive at whatever resolution the capture (or a `?w=` query) produced. Both files measured identical detections on
opencv-python-headless 4.10 and 5.0 at 480/640/960 px wide; the dynamic export is the one the zoo supports
going forward.

Refresh (git-lfs content, so the `media.` host, not `raw.`):

```sh
curl -sSL -o shared/faceblur/models/face_detection_yunet_2026may.onnx \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx
```

# Broadcast Camera and Offline Registration Smoke Test

Date: 2026-08-12

Device: NVIDIA GeForce RTX 5060 Ti

## Verified scope

- Physical homography decomposition into camera center, pan, tilt, roll and
  log-focal parameters.
- Shot-local seven-state EKF with innovation rejection.
- Physical transform reconstruction for short causal predictions.
- Forward/backward observation fusion with reverse-time velocity correction.
- RTS smoothing in camera-parameter space, never by averaging homographies.
- Hard-cut isolation, safe NPZ/JSON persistence and diagnostic MP4 output.

The synthetic camera suite contains exact round trips, moving camera states,
outlier rejection, missing spans and two unrelated Shot camera centers.

## Real-video pipeline smoke

`test2.mp4` frames 0–299 were processed in both directions using the current
legacy 32-keypoint perception adapter:

| Metric | Result |
|---|---:|
| Processed frames | 300 |
| Forward semantic inferences | 67 |
| End-to-end two-pass throughput | 14.41 source FPS |
| Planar SAFE frames | 268 |
| Planar PREVIEW frames | 32 |
| UNAVAILABLE frames | 0 |
| Detected Shots | 1 |

The run successfully wrote MP4, JSON and safe NPZ artifacts. Six uniformly
sampled frames were inspected; the pitch overlay followed the broadcast pan
without an obvious hard jump in that ten-second window.

This run exposed that only 37 frames had already passed the online physical
camera factorisation before offline smoothing. The smoother was subsequently
extended to use a camera center recovered later in the same Shot to fit
earlier planar homographies. Its synthetic regression passes, but the updated
real-video coverage number must be measured again; the earlier 37-frame value
is not an acceptance result.

## Accuracy status

`accuracy_valid=false`. The existing pitch model and the video do not provide
independent ground-truth camera matrices. SAFE coverage and visual continuity
must not be confused with metric projection accuracy. Final acceptance still
requires public or manually held-out line/grid annotations, including the
zero-false-SAFE gate.

## Regression results

- Broadcast camera and offline registration directed suite: 32 passed.
- Full Python suite after integration: 239 passed, 11 existing warnings.

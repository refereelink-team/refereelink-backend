# Broadcast Camera and Offline Registration CUDA Report

Updated: 2026-08-13

Device: NVIDIA GeForce RTX 5060 Ti

## Verified engineering scope

- Shot-local broadcast camera state: pan, tilt, roll, log focal and velocities.
- Generic no-rig optical-flow updates with player regions excluded.
- MAGSAC point initialization and robust point/line correction.
- `SAFE`, `PREVIEW` and `UNAVAILABLE` consumer tiers.
- Forward/backward Shot processing and RTS smoothing in camera-parameter space.
- Sequential block decoding for reverse MP4 traversal; no per-frame random seek.
- Versioned JSON/NPZ output and held-out accuracy/performance gate tools.

The runtime now accepts a trained dual-head checkpoint through
`pitch_perception_checkpoint_path`. If no checkpoint is provided, it retains
the legacy 32-point adapter for compatibility.

## Safety correction found by visual CUDA testing

The first 300-frame smoke classified 268 planar solutions as `SAFE`. Uniformly
sampled visual inspection showed that several overlays were geometrically
wrong even though they fitted the model's own point IDs. That number is
invalidated.

The production gate now requires both:

1. a physically plausible broadcast camera factorization; and
2. an independently successful point/line refinement with at least two visible
   semantic field elements.

Point-only estimates remain available as orange `PREVIEW`, but cannot enter
offside, foul, distance or other metric consumers. Optical flow may preserve a
previously safe state; it cannot promote a preview state to safe.

## Full-video offline results

Both files were processed in two directions on CUDA with the compatibility
32-point model. They are stability/debug videos, not accuracy results.

| Metric | `test1.mp4` | `test2.mp4` |
|---|---:|---:|
| Decoded frames | 1,930 | 1,176 |
| Source-reported frames | 1,956 | 1,261 |
| Forward semantic calls | 602 | 369 |
| Two-pass throughput | 13.90 source FPS | 14.44 source FPS |
| SAFE | 0 | 0 |
| PREVIEW | 1,930 | 1,176 |
| Smoothed frames | 1,905 | 987 |
| Detected Shots | 1 | 1 |

The mismatch between container metadata and decoded frames is preserved in the
report rather than padded with invented observations. Most old-model overlays
remain inaccurate, so the zero-SAFE result is the intended fail-closed outcome.

## Public teacher checks

| Teacher | Real input | Result | Warm CUDA cost | Runtime decision |
|---|---|---|---:|---|
| TVCalib semantic model | `test2` frame 480 | coherent halfway line, centre circle and touchline mask | P95 16.73 ms | secondary teacher only |
| PnLCalib dual HRNet-W48 | 12 representative `test1/test2` frames | camera solution 7/12 at official thresholds, 9/12 relaxed; inspected solutions include major misalignment | dual-model forward P95 113.89 ms; full PnL P95 134.49 ms | candidate pseudo-label teacher only |

The public teachers are pinned and run outside the project environment. Their
weights are not committed. PnLCalib output must pass TVCalib agreement and the
geometry gates before entering a teacher cache; a returned camera dictionary
alone is not acceptance evidence.

## Accuracy status

`accuracy_valid=false` for every result above. No independent camera or field
grid truth was used. The final gate still requires:

- 40–60 held-out projectable frames per local video;
- zero erroneous matrices entering `SAFE`;
- SoccerNet validation/test JaC and line metrics;
- grid error median at most 0.75 m and P95 at most 1.5 m;
- SAFE coverage at least 75% and SAFE+PREVIEW coverage at least 95%.

The current legacy model fails the SAFE coverage target (`0%`). This is an
honest model-selection failure, not a pipeline crash.

## Regression result

- Local and remote full Python suite: 288 passed, 12 warnings.
- Modified-file Ruff and `git diff --check`: passed.

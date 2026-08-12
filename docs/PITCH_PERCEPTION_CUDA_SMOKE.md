# Pitch Perception CUDA Training Smoke Test

Date: 2026-08-12

Device: NVIDIA GeForce RTX 5060 Ti

Branch: `codex/dev-pitch-projection`

## Scope

This check validates the executable model path only:

```text
supervised optimisation
→ teacher distillation
→ test1-style domain adaptation
→ checkpoint load
→ 512×288 FP16 CUDA inference
```

The fixture is synthetic. Its validation IoU, PCK and loss are **not model
accuracy evidence** and must not be included in final quality claims.

## Training smoke result

Each phase ran for one epoch using a deterministic 128×64 fixture:

| Phase | Samples | Epoch time | Training loss |
|---|---:|---:|---:|
| supervised | 2 | 4.70 s | 97.50 |
| distillation | 1 | 0.44 s | 3.17 |
| domain adaptation | 1 | 0.11 s | 3.24 |

The saved checkpoint records the source-index hashes, phase sample counts,
seed, vocabulary, input size and `test2_used=false`.

## Deployment latency

The benchmark loaded the checkpoint produced above; it did not substitute a
random network.

| Metric | Result | Gate |
|---|---:|---:|
| Parameters | 3,253,384 | informational |
| Input | 512×288, batch 1, FP16 | fixed |
| Mean latency | 3.85 ms | informational |
| P95 latency | 6.36 ms | ≤ 8 ms |
| Maximum latency | 6.60 ms | informational |
| Throughput | 260.07 FPS | model-only |
| Peak allocated GPU memory | 48.45 MB | ≤ 300 MB |

Run configuration: 20 warm-up iterations followed by 100 measured
iterations with CUDA synchronization around every sample. These are the final
2026-08-13 measurements after wiring the checkpoint loader into `VisionCore`.

## Interpretation

The MobileNetV3-Large dual-head architecture passes the lightweight deployment
gate on the target GPU. This does **not** freeze the model as the final
perception choice: SoccerNet validation, independent `test2` evaluation,
SAFE-matrix rejection and complete-pipeline FPS gates remain mandatory.

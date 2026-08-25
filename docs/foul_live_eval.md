# Live foul alert evaluation (RTX 5060 Ti)

Non-gating benchmark notes. Collect after enabling foul detection on a fixed
wide-angle clip (`test1.mp4` or RTSP). No hard FPS/latency SLA.

## Procedure

1. Baseline (foul off): record Proc FPS, Inf Latency, E2E, GPU Mem from Status.
2. Geometry only: enable Foul Detect without MVFoul weights — confirm
   `foul_candidate` rows with `source=geometry` and summaries like
   `Home T12 · contact · Away T34`.
3. Geometry + classifier: place `assets/weights/mvfoul.pth.tar` and
   `fouls_far`/`offside`, restart with foul on — watch `Foul Inf`, `Foul Q`,
   `Foul Lat` and enriched `action` in alerts.
4. Note subjective false positives / misses for contact moments.

## Template results

| Mode | Proc FPS | Inf ms | E2E ms | GPU MB | Foul Inf | Foul Lat ms | Notes |
|------|----------|--------|--------|--------|----------|-------------|-------|
| foul off | | | | | 0 | — | |
| geometry | | | | | | — | |
| geometry+mvfoul | | | | | | | |

Hardware: RTX 5060 Ti (fill VRAM SKU). Date: ____

## Observations

- _
- _

Use these numbers to decide later TensorRT / pose upgrades; do not block merges
on a numeric gate.

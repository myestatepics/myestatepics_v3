# MyEstatePics v4.1.0-rc1 — Release Notes

Design Principle #1: **Correct the light, never the materials.**

## Baseline
Built on v3.3 (NOT v3.2 — v3.2 contained the black-room color-bloom defect).

## New in 4.1
- **Room-type exposure models** (engine/profiles.py): kitchen, bathroom, bedroom,
  living, dark_room (moody decor stays moody), low_light (windowless/basement/"black
  paper over every window" — the priority case), generic. Profiles adjust targets,
  per-class multiplier ceilings, ambient reach, and — in low light — the wall paint
  classification thresholds (dim light is not dark paint).
- **Regression harness** (tools/regression.py): 6 scenarios encoding every historical
  failure (tungsten/green/cyan casts, windowless room, z-66 black room, no-op room).
  Produces contact sheet + JSON report (brightness, WB residual, material ΔE/hue,
  wall-texture retention, geometry dims + sub-pixel shift, timing). Nonzero exit on
  any regression: the build rejects itself.
- **Profile-tunable dynamics** in the exposure engine (wall ratio caps, ambient cap,
  ceiling cap read from the active profile).

## Carried from 3.2/3.3 (already satisfying the 4.1 brief)
- Two-pass conservative WB (global shades-of-gray → semantic ceiling/wall refinement),
  no-op band prevents warm/cool oscillation; shadow-protected gains (blacks never tint).
  Handles warm, green, and cyan casts (regression-proven).
- Multiplicative log-luminance illumination with edge-aware propagation (no wall
  flattening — texture retention gated ≥80%), credible-surface ambient, per-class
  multiplier ceilings, chroma-growth damping (hue locked, saturation drift bounded,
  lifted blacks gently desaturated).
- Material lock: no exposure targets on floor/cabinet/counter/furniture; full ambient
  inheritance; hue drift gated ≤4°.
- Geometry lock: the pipeline contains no warp/remap/resize anywhere; exporter asserts
  output dimensions == input; harness verifies zero global shift (≤0.5 px).

## Explicitly NOT in 4.1 (per brief; scheduled 4.2)
Window pull / window recovery, sky replacement, cloud generation.

## Performance
Full pipeline (post-segmentation) ~1s per 1200×900 synthetic frame in CI container;
Apple Silicon (MPS) segmentation timing unchanged from Phase 1. Real-photo timing to
be captured during UAT.

## Rollback
`scripts/rollback.sh <git-ref>` or `git revert`/`git checkout <tag>`; caches are
compatible across 3.x/4.1 (segmentation and mask cache formats unchanged).

## UAT gate (required before "production")
Run the standard six real photos + the 100-image batch; the photographer's contact
sheet outranks every synthetic gate in this document.

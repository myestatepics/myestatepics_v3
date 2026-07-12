
# MyEstatePics v3 — Phase 3 Test Build

This build focuses on the 80/20 production target.

## Changes

- Floors no longer receive an independent exposure target.
- Floors, cabinets, furniture, rugs, art, plants and mirrors are protected materials.
- Protected materials receive only 8% inherited room illumination.
- Gaussian light propagation is replaced with guided filtering.
- Bilateral filtering is used automatically if guided filtering is unavailable.
- Window recovery is slightly stronger and adapts to available highlight detail.
- Quality control now flags:
  - floor brightness change
  - floor color change
  - protected-material color shift
  - increased wall unevenness

## Clean workflow

Each run clears:

- output/final
- output/review
- output/logs
- output/comparisons
- output/reports

The segmentation cache remains.

Input photographs remain untouched.

## Run

```bash
python3 main.py --limit 5
```

Upload only:

```text
output/comparisons/_batch_contact_sheet.jpg
```

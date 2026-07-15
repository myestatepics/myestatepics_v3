# RC4 Exposure Implementation Report

## Scope

RC4 replaces the MVP exposure-fusion call with a source-referenced luminance-only renderer in `engine/tonal.py`. The approved white balance, material handling, segmentation, window processing, color rendering, geometry, and output dimensions remain unchanged.

## Implementation

- Measures diffuse room luminance while excluding windows, fixtures, mirrors, clipped highlights, and deep-black painted pixels from the room target.
- Reports separate ceiling, wall, floor, contents, and general-room measurements and targets.
- Uses one bounded global room placement plus low-amplitude, broadly feathered semantic residuals.
- Protects true black surfaces with a continuous black anchor.
- Protects highlights, windows, mirrors, and fixture cores before both passes.
- Copies approved source Lab hue/chroma through the exposure stage.
- Reduces luminance lift, rather than chroma, when a lifted source color would leave the sRGB gamut.
- Performs one initial pass and at most one bounded convergence pass.
- Detects intentional dark decor so black paint cannot drive the room toward gray.
- Leaves already-good rooms at effectively zero exposure change.

## Automated validation

- `python3 -m pytest tests/ -q`: 74 passed.
- `python3 tools/regression.py`: 7 of 7 scenarios passed.
- Coverage includes no-op, dark-room lift, black-wall preservation, window/highlight preservation, hue/chroma invariance, geometry invariance, bounded convergence, and non-decreasing room luminance.

## Six-image UAT measurements

| Image | Room median before | Room median after | Effective lift | Passes | Window core change |
|---|---:|---:|---:|---:|---:|
| z-13 | 0.5145 | 0.5154 | +0.003 EV | 1 | 0.0 |
| z-16 | 0.3619 | 0.5472 | +0.596 EV | 1 | 0.0 |
| z-17 | 0.2728 | 0.4976 | +0.867 EV | 2 | 0.0 |
| z-19 | 0.5392 | 0.5393 | +0.000 EV | 1 | 0.0 |
| z-34 | 0.5790 | 0.5790 | +0.000 EV | 1 | 0.0 |
| z-66 | 0.1559 | 0.2893 | +0.891 EV | 2 | 0.0 |

Exposure-stage Lab a/b preservation across the real set is exact within one code value through the 99th percentile. All six images retain their original dimensions. The window-core mean absolute change is zero for every scene.

## Visual review

- z-13 and z-19 are visually unchanged.
- z-34 is unchanged.
- z-16 receives a controlled moderate room lift while retaining black cabinetry.
- z-17 gains readable room, island, wall, ceiling, and floor detail without changing window cores.
- z-66 gains ceiling and floor visibility while black walls remain black and windows remain unchanged.
- No new halos, blotches, hard semantic edges, geometry changes, or texture damage were observed.

The UAT contact sheet is generated at `output/comparisons/_batch_contact_sheet.jpg`.

## Exact UAT command

```bash
cd ~/Documents/MyestatePics/Myestatepics-pythononly/myestatepics_v3
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 main.py --input /tmp/myestatepics-rc4-uat --debug-stages
```

The temporary UAT directory contains only `z-13.jpg`, `z-16.jpg`, `z-17.jpg`, `z-19.jpg`, `z-34.jpg`, and `z-66.jpg` linked from the repository input directory.

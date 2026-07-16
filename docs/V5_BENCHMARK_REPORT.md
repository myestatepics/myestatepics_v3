# MyEstatePics v5.0 benchmark specification

The benchmark translates professional architectural and MLS practice into measurable gates without copying any reference photograph.

## Research basis

- Google HDR+ research supports keeping a clean high-bit-depth HDR source and tone mapping after merge: https://research.google/pubs/burst-photography-for-high-dynamic-range-and-low-light-imaging-on-mobile-cameras/
- Mertens-style exposure fusion motivates contrast and well-exposedness weighting, but v5 uses the already merged scene-linear DNG rather than recreating a bracket stack: https://doi.org/10.1111/j.1467-8659.2008.01171.x
- Intrinsic Image Harmonization motivates keeping reflectance/material identity separate from illumination adjustment: https://openaccess.thecvf.com/content/CVPR2021/html/Guo_Intrinsic_Image_Harmonization_CVPR_2021_paper.html
- Sky Optimization demonstrates the importance of semantic sky masks and edge-aware processing: https://openaccess.thecvf.com/content_CVPRW_2020/html/w31/Liba_Sky_Optimization_Semantically_Aware_Image_Processing_of_Skies_in_Low-Light_CVPRW_2020_paper.html
- High-resolution harmonization research confirms that local compositing must remain coherent at full output resolution: https://github.com/ZHKKKe/Harmonizer

## Objective targets

- Interior diffuse median remains controlled by the frozen HDR-C/RC4 planner.
- Window processing never changes pixels outside the feathered window mask.
- Real scene-linear exterior detail is used before any sky completion.
- Sky completion is restricted to low-detail, bright, upper-pane pixels.
- Frames, mullions, curtains, mirrors, dark edges, roofs, trees, buildings, and ground are protected.
- Generated sky is light blue, low contrast, deterministic per image, and consistent across windows within a room.
- Geometry and dimensions remain bit-for-bit unchanged as array coordinates.
- Material finishing changes luminance only and preserves linear RGB ratios.
- Progressive JPEG starts at quality 92, never drops below 82, and targets 2 MB without resizing.

## Internal benchmark sets

- 50 deterministic synthetic/pathological cases in `validation/synthetic_dataset.py`.
- Automatically selected 20-DNG difficult subset with a recorded metric/reason manifest.
- Full 56-DNG production batch.

## Acceptance gates

All unit tests, regression cases, synthetic cases, representative DNGs, and full-batch files must complete. Blue spill must remain zero, geometry must remain invariant, bad files must be isolated, and the contact sheet must show no systematic pasted-window, halo, saturation, or HDR artifacts.

## Final measured result

- Automated tests: 93 passed in the final release gate.
- Regression: 7 of 7 passed.
- Synthetic benchmark: 50 of 50 passed.
- Representative real-DNG benchmark: 20 of 20 rendered successfully.
- Full real-DNG batch: 56 of 56 rendered successfully with zero failed files.
- Window routes: 19 no-window no-ops, 8 recoverable, 28 partially recoverable, and 1 unrecoverable-sky route.
- Maximum measured blue spill: 0.0%.

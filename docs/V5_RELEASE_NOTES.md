# MyEstatePics v5.0 production release

## Visible improvements

- Scene-linear recovery restores genuine trees, buildings, rooflines, landscaping, and ground inside detected window panes.
- Unrecoverable upper-pane sky receives restrained deterministic light-blue completion with soft low-contrast clouds.
- Window frames, mullions, curtains, mirrors, and dark structural edges are protected.
- The approved RC4.2 room exposure, white balance, material hue lock, local photographic finish, geometry, and dimensions remain intact.
- Production JPEG delivery targets 2 MB, uses optimized progressive encoding, and never resizes or drops below quality 82.

## Safety and fallback behavior

- Rooms with no detected windows are exact window-stage no-ops.
- Failed DNGs are reported while the remaining batch continues.
- Every file records its decoder, rendering, window route, processing time, warnings, dimensions, JPEG quality, and file size.
- If a file cannot meet 2 MB at quality 82, it remains full resolution and is marked oversized rather than being resized or over-compressed.

## Production command

`.venv/bin/python tools/process_dng_batch.py --input input --output output/v5_production`

## Release validation

- 50/50 deterministic synthetic cases passed.
- 20/20 automatically selected difficult DNGs rendered successfully.
- 56/56 full-batch DNGs rendered successfully.
- Regression passed 7/7.
- Maximum window-stage blue spill was 0.0%.

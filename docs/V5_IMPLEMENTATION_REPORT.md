# MyEstatePics v5.0 implementation report

## Production architecture

1. Decode Lightroom LinearRaw HDR DNG into immutable float32 camera-linear, XYZ D50, and extended linear-sRGB planes.
2. Build a neutral display proxy for the existing semantic segmentation and QA boundary.
3. Measure the scene in linear light and run the frozen HDR-C global MLS exposure/tone/color renderer.
4. Apply the approved luminance-only RC4.2 photographic finish.
5. Recover real exterior content from the scene-linear DNG only inside eligible window panes.
6. Complete only low-detail, bright, upper-pane sky pixels with a deterministic light-blue, low-contrast cloud field.
7. Export full-resolution optimized progressive JPEG with per-file quality/size reporting and failure isolation.

## Window recovery

- The source is the original high-dynamic-range linear DNG plane, not the rendered JPEG.
- A window-specific exposure placement and smooth shoulder render the exterior without changing room exposure.
- Existing mirror and curtain masks are hard exclusions.
- Dark structural pixels and strong frame/mullion edges reduce the blend continuously.
- Recovered trees, buildings, rooflines, ground, fences, and landscaping remain source-derived.
- The blend is clipped exactly to the window eligibility mask; the automated blue-spill metric must remain zero.

## Sky completion

- Sky is considered only in the upper portion of each connected window region.
- Low recovered texture and high brightness are both required.
- One full-frame cloud field is generated per image, ensuring consistency across multiple windows.
- The seed derives from the source filename, preventing an identical repeated cloud field across images.
- The generator uses no AI model, stock image, network request, or paid API.

## Production QA

- 50 deterministic synthetic/pathological fixtures cover room, material, lighting, window, sky, edge, and batch failure cases.
- A 20-image real subset is selected automatically from the available DNGs using measurable darkness, highlight, cast, chroma, tint, black occupancy, mixed-lighting, and neutral-room criteria.
- The final gate processes every available DNG and records success, failures, dimensions, JPEG quality, size, time, fallback route, window status, and warnings.

## Known limitations

- Sky completion depends on the existing semantic window mask. A pane missed entirely by segmentation cannot be recovered safely.
- Very dense sheer curtains can be classified partly as window glass; conservative curtain exclusion may leave those areas brighter than a manual composite.
- Highly reflective or frosted glass can have genuinely low texture and may receive a small amount of sky completion in its upper region.
- Files that remain above the size target at JPEG quality 82 are retained at full resolution and flagged oversized rather than resized.
- Processing a 6000x4000 HDR DNG is compute-intensive; the renderer prioritizes deterministic full-resolution quality over interactive latency.

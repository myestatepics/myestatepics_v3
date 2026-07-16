# MyEstatePics v5.0 performance and delivery report

## Full 56-DNG production gate

- Success: 56
- Failure: 0
- Total processing time: 3065.11 seconds
- Mean processing time: 54.73 seconds per 6000x4000 HDR DNG
- Total JPEG size: 112.93 MB
- Mean JPEG size: 2.02 MB
- Quality distribution: Q82=23, Q84=11, Q86=15, Q88=7
- Full-resolution geometry: preserved for every output
- Progressive optimized JPEG: enabled for every output

Sixteen unusually detailed files remained above 2 MB at the mandatory quality floor of 82. They were retained at full resolution and flagged oversized; the largest was 2.95 MB. No file was resized or compressed below the quality floor.

Machine-readable reports are written under `output/v5_production/` and `output/v5_production/reports/`.

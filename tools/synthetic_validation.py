#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.photographic_finish import apply_lightroom_look
from engine.window_recovery import recover_windows
from validation.synthetic_dataset import CASES, make_fixture


def main() -> int:
    out = Path("output/production_validation/synthetic")
    out.mkdir(parents=True, exist_ok=True)
    records, thumbs = [], []
    for case in CASES:
        before, linear, scene = make_fixture(case)
        finished, finish_log = apply_lightroom_look(before, scene)
        after, window_log = recover_windows(linear, finished, scene, seed=case.name)
        Image.fromarray(np.uint8(np.clip(after * 255 + 0.5, 0, 255))).save(out / f"{case.name}.jpg", quality=92)
        thumbs.append((case.name, before, after))
        records.append({"case": case.name, "tags": case.tags, "finish": finish_log, "window": window_log,
                        "geometry_preserved": after.shape == before.shape,
                        "finite": bool(np.isfinite(after).all())})

    tw, th, label = 360, 240, 30
    sheet = Image.new("RGB", (tw * 2, (th + label) * len(thumbs)), (16, 16, 16))
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default(size=16)
    for row, (name, before, after) in enumerate(thumbs):
        for col, (title, rgb) in enumerate((("input", before), ("candidate", after))):
            image = Image.fromarray(np.uint8(np.clip(rgb * 255 + 0.5, 0, 255)))
            image.thumbnail((tw, th), Image.Resampling.LANCZOS)
            x, y = col * tw, row * (th + label)
            sheet.paste(image, (x, y + label)); draw.text((x + 5, y + 7), f"{name} - {title}", fill="white", font=font)
    sheet.save(out / "synthetic_contact_sheet.jpg", quality=90)
    report = {"case_count": len(records), "pass": all(r["geometry_preserved"] and r["finite"] and r["window"]["blue_spill_percent"] == 0 for r in records), "records": records}
    (out / "synthetic_report.json").write_text(json.dumps(report, indent=2))
    print(f"{'PASS' if report['pass'] else 'FAIL'}: {len(records)} synthetic cases")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

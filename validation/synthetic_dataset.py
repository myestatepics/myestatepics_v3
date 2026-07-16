from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.scene import Scene, build_scene


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    tags: tuple[str, ...]


CASE_SPECS = (
    ("bright_white_kitchen", ("bright", "neutral", "cabinet")),
    ("dark_wood_kitchen", ("dark", "wood", "cabinet")),
    ("black_cabinets", ("dark", "black", "cabinet")),
    ("black_accent_wall", ("dark", "black_wall")),
    ("white_walls_tungsten", ("warm", "neutral")),
    ("green_cast", ("green",)), ("cyan_cast", ("cyan",)),
    ("mixed_daylight_tungsten", ("warm", "window")),
    ("dark_basement", ("dark", "no_window")),
    ("long_basement_hallway", ("dark", "hallway", "no_window")),
    ("small_bathroom", ("tile", "small_window")),
    ("large_living_room", ("large_window",)),
    ("bedroom_white_bedding", ("bed", "neutral")),
    ("gray_carpet", ("carpet", "neutral")),
    ("warm_hardwood_floor", ("wood", "warm")),
    ("red_brown_hardwood_floor", ("wood", "red_brown")),
    ("dark_furniture", ("dark", "furniture")),
    ("neutral_walls_color_bounce", ("neutral", "bounce")),
    ("peach_pink_ceiling_bounce", ("neutral", "peach_ceiling")),
    ("large_blown_window", ("large_window", "blown")),
    ("small_blown_window", ("small_window", "blown")),
    ("partially_recoverable_window", ("window", "partial")),
    ("window_trees_sky", ("window", "trees", "sky")),
    ("window_neighbor_building", ("window", "building")),
    ("window_roofline", ("window", "roofline", "sky")),
    ("window_blinds", ("window", "blinds")),
    ("window_curtains", ("window", "curtain")),
    ("multi_pane_window", ("window", "mullion")),
    ("multiple_windows", ("multi_window", "sky")),
    ("mirror_near_window", ("window", "mirror")),
    ("no_window_room", ("no_window",)),
    ("very_bright_already_good", ("bright", "already_good")),
    ("underexposed_bright_fixtures", ("dark", "lamp")),
    ("dark_black_paint", ("dark", "black_wall", "no_window")),
    ("near_neutral_materials", ("neutral", "material")),
    ("vivid_materials", ("vivid", "material")),
    ("clipped_sky_only", ("window", "sky", "blown")),
    ("clipped_full_window", ("window", "blown")),
    ("exterior_ground_no_sky", ("window", "ground")),
    ("mixed_recoverable_unrecoverable_panes", ("multi_window", "partial", "sky")),
    ("recessed_lights", ("lamp",)),
    ("reflective_countertops", ("cabinet", "reflective")),
    ("curtains_bright_window", ("window", "curtain", "blown")),
    ("fine_mullions_frames", ("window", "mullion", "frame")),
    ("noise_prone_shadows", ("dark", "noise")),
    ("high_resolution", ("window", "high_resolution")),
    ("large_batch_corrupt_file", ("batch", "corrupt")),
    ("duplicate_filenames_folders", ("batch", "duplicate")),
    ("unsupported_file_type", ("batch", "unsupported")),
    ("empty_input_folder", ("batch", "empty")),
)

CASES = tuple(SyntheticCase(name, tags) for name, tags in CASE_SPECS)


def make_fixture(case: SyntheticCase, shape: tuple[int, int] = (320, 480)) -> tuple[np.ndarray, np.ndarray, Scene]:
    """Return rendered RGB, scene-linear DNG-like RGB, and exact masks."""
    h, w = shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    rendered = np.full((h, w, 3), 0.66, np.float32)
    rendered[: h // 2] = 0.74
    floor_texture = 0.42 + 0.035 * np.sin(xx * 0.23) * np.sin(yy * 0.11)
    rendered[h // 2 :] = floor_texture[h // 2 :, :, None]
    if "dark" in case.tags:
        rendered *= 0.55
    if "bright" in case.tags:
        rendered = np.clip(rendered * 1.18, 0, 1)
    if "warm" in case.tags:
        rendered *= np.array([1.05, 0.98, 0.91], np.float32)
    if "green" in case.tags:
        rendered *= np.array([0.94, 1.06, 0.94], np.float32)
    if "cyan" in case.tags:
        rendered *= np.array([0.91, 1.03, 1.05], np.float32)
    if "red_brown" in case.tags:
        rendered[h // 2 :] *= np.array([1.12, 0.83, 0.70], np.float32)
    if "vivid" in case.tags:
        rendered[h // 2 :, : w // 3] = (0.75, 0.10, 0.06)

    names = ("wall", "ceiling", "floor", "window", "curtain", "door", "cabinet",
             "mirror", "sofa", "table", "chair", "bed", "rug", "plant", "painting", "lamp")
    masks = {name: np.zeros((h, w), np.float32) for name in names}
    masks["ceiling"][: h // 5] = 1
    masks["wall"][h // 5 : h // 2] = 1
    masks["floor"][h // 2 :] = 1
    has_window = "no_window" not in case.tags and "batch" not in case.tags
    if has_window:
        boxes = [(w * 3 // 5, h // 5, w * 9 // 10, h * 3 // 4)]
        if "multi_window" in case.tags:
            boxes = [(w // 10, h // 5, w * 2 // 5, h * 3 // 4), (w * 3 // 5, h // 5, w * 9 // 10, h * 3 // 4)]
        for x0, y0, x1, y1 in boxes:
            masks["window"][y0:y1, x0:x1] = 1
            rendered[y0:y1, x0:x1] = 0.98 if "blown" in case.tags or "sky" in case.tags else 0.86
            if "mullion" in case.tags:
                cx = (x0 + x1) // 2
                rendered[y0:y1, cx - 2 : cx + 2] = 0.12
        masks["wall"] *= 1 - masks["window"]
    if "curtain" in case.tags:
        masks["curtain"][:, w * 3 // 5 : w * 13 // 20] = masks["window"][:, w * 3 // 5 : w * 13 // 20]
    if "mirror" in case.tags:
        masks["mirror"][h // 4 : h // 2, w // 8 : w // 3] = 1
    if "cabinet" in case.tags:
        masks["cabinet"][h // 3 : h * 3 // 4, : w // 3] = 1
    if "bed" in case.tags:
        masks["bed"][h // 2 :, w // 5 : w * 4 // 5] = 1
        rendered[masks["bed"] > 0.5] = 0.88
    if "lamp" in case.tags:
        masks["lamp"][h // 12 : h // 7, w // 2 - 12 : w // 2 + 12] = 1
        rendered[masks["lamp"] > 0.5] = 0.99

    # Geometry markers must survive exactly because no stage may remap pixels.
    rendered[:3, :3] = (1, 0, 1)
    rendered[-3:, -3:] = (0, 1, 1)
    linear = np.maximum(rendered, 0) ** 2.2
    if has_window:
        win = masks["window"] > 0.5
        linear[win] *= 4.0
        if "trees" in case.tags or "building" in case.tags or "ground" in case.tags or "partial" in case.tags:
            pattern = 0.35 + 0.25 * (np.sin(xx * 0.08) * np.cos(yy * 0.06))
            linear[..., 1][win] *= pattern[win]
    return np.clip(rendered, 0, 1), linear.astype(np.float32), build_scene(masks, shape)

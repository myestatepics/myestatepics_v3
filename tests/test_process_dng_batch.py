import json
from pathlib import Path

from PIL import Image

from tools.process_dng_batch import discover_dng_files, process_batch


def _touch_dng(folder: Path, name: str) -> Path:
    path = folder / name
    path.write_bytes(b"test dng placeholder")
    return path


def _fake_processor(source: Path, output: Path, debug: bool, segmenter) -> dict:
    destination = output / f"{source.stem}.jpg"
    Image.new("RGB", (64, 48), (80, 90, 100)).save(destination)
    return {
        "source_filename": source.name,
        "output_filename": destination.name,
        "output_path": str(destination),
        "dimensions": [64, 48],
    }


def test_arbitrary_dng_filenames_are_discovered_naturally(tmp_path):
    _touch_dng(tmp_path, "living-room-final.dng")
    _touch_dng(tmp_path, "y-10.dng")
    _touch_dng(tmp_path, "y-2.DNG")
    assert [path.name for path in discover_dng_files(tmp_path)] == [
        "living-room-final.dng", "y-2.DNG", "y-10.dng",
    ]


def test_empty_folder_writes_zero_count_summary_and_contact_sheet(tmp_path):
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    input_dir.mkdir()
    summary, status = process_batch(input_dir, output_dir, processor=_fake_processor)
    assert status == 0
    assert summary["total_count"] == summary["success_count"] == summary["failure_count"] == 0
    assert (output_dir / "batch_summary.json").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "failure_report.json").exists()
    assert (output_dir / "file_size_report.json").exists()
    assert (output_dir / "batch_contact_sheet.jpg").exists()


def test_bad_dng_does_not_stop_valid_files(tmp_path):
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    input_dir.mkdir()
    for name in ("good-1.dng", "bad.dng", "good-2.dng"):
        _touch_dng(input_dir, name)

    def processor(source, output, debug, segmenter):
        if source.name == "bad.dng":
            raise ValueError("broken test DNG")
        return _fake_processor(source, output, debug, segmenter)

    summary, status = process_batch(input_dir, output_dir, processor=processor)
    assert status == 1
    assert summary["total_count"] == 3
    assert summary["success_count"] == 2
    assert summary["failure_count"] == 1
    assert {item["source_filename"] for item in summary["successes"]} == {"good-1.dng", "good-2.dng"}


def test_output_filename_preserves_source_stem(tmp_path):
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    input_dir.mkdir()
    _touch_dng(input_dir, "arbitrary-client-name-0042.dng")
    summary, status = process_batch(input_dir, output_dir, processor=_fake_processor)
    assert status == 0
    assert summary["successes"][0]["output_filename"] == "arbitrary-client-name-0042.jpg"
    assert (output_dir / "arbitrary-client-name-0042.jpg").exists()


def test_batch_summary_file_counts_match_returned_summary(tmp_path):
    input_dir, output_dir = tmp_path / "input", tmp_path / "output"
    input_dir.mkdir()
    for name in ("one.dng", "two.dng", "three.dng"):
        _touch_dng(input_dir, name)
    summary, status = process_batch(input_dir, output_dir, processor=_fake_processor)
    written = json.loads((output_dir / "batch_summary.json").read_text())
    assert status == 0
    assert written["total_count"] == summary["total_count"] == 3
    assert written["success_count"] == summary["success_count"] == 3
    assert written["failure_count"] == summary["failure_count"] == 0

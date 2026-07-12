
from engine.scoring import build_checklist, summarize_batch


def test_good_checklist_passes():
    checklist = build_checklist(
        wb_log={"gains": [0.99, 1.01, 1.00], "confidence": "HIGH"},
        window_log={
            "status": "treated",
            "window_percent": 4.0,
            "mean_L_reduction_window": 0.03,
        },
        quality={
            "status": "PASS",
            "flags": [],
            "achieved_targets": {
                "wall": {"target": 0.60, "after": 0.59},
                "ceiling": {"target": 0.78, "after": 0.76},
            },
            "protected_chroma_p95_delta": 3.0,
            "protected_chroma_mean_delta": 1.0,
            "clipped_percent_after_excluding_windows": 0.2,
        },
        export_log={
            "resolution_preserved": True,
            "oversized": False,
            "quality": 95,
        },
        route="SEMANTIC",
    )
    assert checklist["decision"] == "PASS"
    assert checklist["overall_score"] >= 80


def test_global_safe_always_review():
    checklist = build_checklist(
        wb_log={"gains": [1.0, 1.0, 1.0], "confidence": "LOW"},
        window_log={"status": "skipped_global_safe", "window_percent": 0.0},
        quality={
            "status": "PASS",
            "flags": [],
            "achieved_targets": {"wall": {"target": 0.5, "after": 0.5}},
            "protected_chroma_p95_delta": 1.0,
            "protected_chroma_mean_delta": 0.5,
            "clipped_percent_after_excluding_windows": 0.0,
        },
        export_log={
            "resolution_preserved": True,
            "oversized": False,
            "quality": 95,
        },
        route="GLOBAL_SAFE",
    )
    assert checklist["decision"] == "REVIEW"


def test_batch_summary():
    records = [
        {"checklist": {
            "overall_score": 90,
            "decision": "PASS",
            "checks": [{"name": "White balance", "score": 92}],
        }},
        {"checklist": {
            "overall_score": 70,
            "decision": "REVIEW",
            "checks": [{"name": "White balance", "score": 72}],
        }},
    ]
    summary = summarize_batch(records)
    assert summary["pass_count"] == 1
    assert summary["review_count"] == 1
    assert summary["overall_engine_score"] == 80

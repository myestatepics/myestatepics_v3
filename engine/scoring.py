
from __future__ import annotations

from dataclasses import dataclass, asdict
import math


@dataclass
class CheckResult:
    name: str
    status: str
    score: int
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _clamp_score(value: float) -> int:
    return int(round(max(0.0, min(100.0, value))))


def score_white_balance(wb_log: dict) -> CheckResult:
    gains = [float(x) for x in wb_log.get("gains", [1.0, 1.0, 1.0])]
    confidence = str(wb_log.get("confidence", "LOW")).upper()
    max_deviation = max(abs(g - 1.0) for g in gains)

    score = 100.0 - max_deviation * 160.0
    if confidence == "LOW":
        score -= 12.0
    elif confidence == "MEDIUM":
        score -= 5.0

    score_i = _clamp_score(score)
    status = "PASS" if score_i >= 82 else "REVIEW"
    reason = (
        f"{confidence.lower()} confidence; gains "
        f"{', '.join(f'{g:.3f}' for g in gains)}"
    )
    return CheckResult("White balance", status, score_i, reason)


def score_color_cast(quality: dict) -> CheckResult:
    p95 = float(quality.get("protected_chroma_p95_delta", 0.0))
    mean = float(quality.get("protected_chroma_mean_delta", 0.0))

    score = 100.0 - max(0.0, p95 - 2.0) * 7.0 - mean * 2.0
    score_i = _clamp_score(score)
    status = "PASS" if p95 <= 9.0 and score_i >= 78 else "REVIEW"
    reason = f"protected material chroma delta: mean {mean:.2f}, p95 {p95:.2f}"
    return CheckResult("Color/material fidelity", status, score_i, reason)


def score_exposure(quality: dict) -> CheckResult:
    after = float(quality.get("global_median_brightness_after", 0.0))
    highlight = float(quality.get("highlight_clipped_percent_excluding_windows", 0.0))
    shadow = float(quality.get("shadow_clipped_percent_excluding_windows", 0.0))
    brightness_penalty = max(0.0, 0.34 - after) * 180.0 + max(0.0, after - 0.72) * 180.0
    score = 100.0 - brightness_penalty - max(0.0, highlight - 0.5) * 8.0 - max(0.0, shadow - 1.0) * 5.0
    score_i = _clamp_score(score)
    exposure_flags = {
        "EXCESSIVE_HIGHLIGHT_CLIPPING",
        "EXCESSIVE_SHADOW_CLIPPING",
        "EXCESSIVE_GLOBAL_BRIGHTNESS_CHANGE",
        "WALL_GLARE",
    }.intersection(quality.get("flags", []))
    status = "PASS" if not exposure_flags and score_i >= 78 else "REVIEW"
    reason = f"global median {after:.3f}; highlight clip {highlight:.2f}%; shadow clip {shadow:.2f}%"
    if exposure_flags:
        reason += "; " + ", ".join(sorted(exposure_flags))
    return CheckResult("Exposure balance", status, score_i, reason)


def score_windows(window_log: dict, quality: dict) -> CheckResult:
    status_text = str(window_log.get("status", "unknown"))
    window_percent = float(window_log.get("window_percent", 0.0))
    clipping = float(quality.get("clipped_percent_after_excluding_windows", 0.0))

    if status_text == "skipped_mvp_phase3":
        return CheckResult(
            "Window control", "NOT_SCORED", 0,
            "window processing is intentionally disabled in Phase A",
        )

    if status_text in {"no_windows", "skipped_global_safe"} and window_percent < 0.2:
        return CheckResult("Window control", "PASS", 95, "no meaningful window area")

    reduction = float(window_log.get("mean_L_reduction_window", 0.0))
    score = 88.0 + min(reduction / 0.03, 1.0) * 8.0 - max(0.0, clipping - 0.5) * 5.0
    score_i = _clamp_score(score)
    status = "PASS" if clipping <= 1.8 and score_i >= 78 else "REVIEW"
    reason = (
        f"window area {window_percent:.2f}%; "
        f"mean highlight reduction {reduction:.3f}"
    )
    return CheckResult("Window control", status, score_i, reason)


def score_sharpness_and_output(export_log: dict) -> CheckResult:
    preserved = bool(export_log.get("resolution_preserved", False))
    oversized = bool(export_log.get("oversized", False))
    quality = int(export_log.get("quality", 0))

    score = 100
    reasons = []
    if not preserved:
        score -= 60
        reasons.append("resolution changed")
    if oversized:
        score -= 15
        reasons.append("file exceeds configured size")
    if quality < 87:
        score -= 10
        reasons.append(f"JPEG quality reduced to {quality}")
    if not reasons:
        reasons.append(f"resolution preserved; JPEG quality {quality}")

    status = "PASS" if score >= 85 else "REVIEW"
    return CheckResult("Output quality", status, score, "; ".join(reasons))


def build_checklist(
    wb_log: dict,
    window_log: dict,
    quality: dict,
    export_log: dict,
    route: str,
) -> dict:
    checks = [
        score_white_balance(wb_log),
        score_color_cast(quality),
        score_exposure(quality),
        score_windows(window_log, quality),
        score_sharpness_and_output(export_log),
    ]

    scored_checks = [check for check in checks if check.status != "NOT_SCORED"]
    scores = [check.score for check in scored_checks]
    # Conservative: weak areas matter more than a simple average.
    overall = _clamp_score(0.65 * min(scores) + 0.35 * (sum(scores) / len(scores)))

    automatic_review = (
        route == "GLOBAL_SAFE"
        or quality.get("status") == "REVIEW"
        or any(check.status == "REVIEW" for check in scored_checks)
        or overall < 80
    )

    return {
        "checks": [check.to_dict() for check in checks],
        "overall_score": overall,
        "decision": "REVIEW" if automatic_review else "PASS",
        "route": route,
    }


def summarize_batch(image_records: list[dict]) -> dict:
    if not image_records:
        return {
            "overall_engine_score": 0,
            "averages": {},
            "pass_count": 0,
            "review_count": 0,
        }

    named_scores: dict[str, list[int]] = {}
    overall_scores: list[int] = []
    pass_count = 0

    for record in image_records:
        checklist = record["checklist"]
        overall_scores.append(int(checklist["overall_score"]))
        if checklist["decision"] == "PASS":
            pass_count += 1

        for check in checklist["checks"]:
            named_scores.setdefault(check["name"], []).append(int(check["score"]))

    averages = {
        name: int(round(sum(values) / len(values)))
        for name, values in named_scores.items()
    }

    return {
        "overall_engine_score": int(round(sum(overall_scores) / len(overall_scores))),
        "averages": averages,
        "pass_count": pass_count,
        "review_count": len(image_records) - pass_count,
    }

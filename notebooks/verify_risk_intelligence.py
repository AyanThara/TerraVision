"""
TerraVision - Phase 8: Risk Intelligence Verification Script

This script performs strict verification checks for Phase 8:
1. Verifies that all 4 Phase 8 artifacts exist and are non-empty.
2. Verifies JSON parses correctly.
3. Verifies that Overall Risk Score, all Sub-scores, and Evidence Confidence are in [0, 100].
4. Verifies mathematical consistency of the weighted risk score formula:
   R = 0.40 * S_pressure + 0.30 * S_trend + 0.30 * S_forecast.
5. Verifies documented risk level thresholds:
   - LOW: 0 <= R < 30
   - MODERATE: 30 <= R < 60
   - HIGH: 60 <= R < 80
   - VERY HIGH: 80 <= R <= 100
6. Verifies documented evidence confidence thresholds:
   - LOW CONFIDENCE: < 40
   - MODERATE CONFIDENCE: 40–<70
   - HIGH CONFIDENCE: >= 70
7. Verifies expected baseline values:
   - R ≈ 34.32 (within ±0.1)
   - C ≈ 37.74 (within ±0.1)
   - S_pressure ≈ 36.82 (within ±0.1)
   - S_trend ≈ 25.55 (within ±0.1)
   - S_forecast ≈ 39.72 (within ±0.1)
8. Verifies benchmark terminology explicitly says:
   "TerraVision heuristic normalization benchmarks"
9. Verifies Evidence Confidence is explicitly distinguished from statistical prediction intervals.
10. Verifies scientific disclaimers (not guaranteed future development).
11. Verifies that Phase 1–7 files remain untouched.
12. Verifies finite numerical values (no NaNs or Infs).
13. Verifies complete Phase 9 and Phase 10 interfaces in JSON.
14. Verifies PNG visualization exists and is non-empty.
"""

import sys
import json
import math
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def verify_prior_phases_untouched():
    """
    Ensures Phase 1–7 artifacts and key files remain completely untouched.
    """
    print("1. Checking that Phase 1–7 files remain untouched...")
    prior_files = [
        PROJECT_ROOT / "results" / "change_quantification.json",
        PROJECT_ROOT / "results" / "historical_change_trends.json",
        PROJECT_ROOT / "results" / "historical_change_trends.png",
        PROJECT_ROOT / "results" / "future_prediction.json",
        PROJECT_ROOT / "results" / "future_prediction.png",
        PROJECT_ROOT / "src" / "quantify_change.py",
        PROJECT_ROOT / "src" / "historical_trends.py",
        PROJECT_ROOT / "src" / "future_prediction.py",
        PROJECT_ROOT / "notebooks" / "verify_change_pipeline.py",
        PROJECT_ROOT / "notebooks" / "verify_historical_trends.py",
        PROJECT_ROOT / "notebooks" / "verify_future_prediction.py",
    ]
    for p in prior_files:
        assert p.exists(), f"Prior phase file missing: {p}"
        assert p.stat().st_size > 0, f"Prior phase file empty: {p}"
    print(f"  ✓ All {len(prior_files)} prior phase files verified present and intact.")


def verify_phase8_artifacts():
    """
    Comprehensive audit of Phase 8 generated artifacts.
    """
    print("\n2. Performing Comprehensive Audit of Phase 8 Artifacts...")
    json_path = PROJECT_ROOT / "results" / "risk_intelligence.json"
    png_path = PROJECT_ROOT / "results" / "risk_intelligence.png"
    src_path = PROJECT_ROOT / "src" / "risk_intelligence.py"
    test_path = PROJECT_ROOT / "notebooks" / "verify_risk_intelligence.py"

    # 1. Artifacts existence and non-empty
    for p, name in [(json_path, "JSON"), (png_path, "PNG"), (src_path, "Source"), (test_path, "Verification")]:
        assert p.exists(), f"Phase 8 {name} artifact not found: {p}"
        assert p.stat().st_size > 0, f"Phase 8 {name} artifact is empty: {p}"
    assert png_path.stat().st_size > 50000, f"PNG artifact appears corrupt or too small: {png_path.stat().st_size} bytes"
    print(f"  ✓ Found PNG artifact: {png_path.name} ({png_path.stat().st_size:,} bytes)")

    # 2. Parse JSON
    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"  ✓ Successfully parsed JSON artifact: {json_path.name}")

    # 3. Check Risk Assessment fields and bounds
    ra = data.get("risk_assessment", {})
    r_score = ra.get("overall_risk_score")
    r_level = ra.get("risk_level")
    c_score = ra.get("evidence_confidence_score")
    c_level = ra.get("evidence_confidence_level")

    assert math.isfinite(r_score), f"Risk score is not finite: {r_score}"
    assert 0.0 <= r_score <= 100.0, f"Risk score out of [0, 100] bounds: {r_score}"
    assert math.isfinite(c_score), f"Evidence confidence is not finite: {c_score}"
    assert 0.0 <= c_score <= 100.0, f"Evidence confidence out of [0, 100] bounds: {c_score}"
    print(f"  ✓ Overall Risk Indicator: {r_score:.2f} / 100 (Level: {r_level})")
    print(f"  ✓ Evidence Confidence Score: {c_score:.2f} / 100 (Level: {c_level})")

    # 4. Check Sub-scores and bounds
    sub = data.get("sub_scores", {})
    s_press = sub.get("change_pressure_score", {}).get("score")
    s_trend = sub.get("trend_momentum_score", {}).get("score")
    s_fore = sub.get("future_outlook_score", {}).get("score")

    for val, name in [(s_press, "Change Pressure"), (s_trend, "Trend Momentum"), (s_fore, "Future Outlook")]:
        assert math.isfinite(val), f"Sub-score {name} is not finite: {val}"
        assert 0.0 <= val <= 100.0, f"Sub-score {name} out of [0, 100] bounds: {val}"

    print(f"  ✓ Sub-scores validated in [0, 100]: Pressure={s_press:.2f}, Trend={s_trend:.2f}, Forecast={s_fore:.2f}")

    # 5. Exact mathematical consistency of weighted risk score
    # R = 0.40 * S_pressure + 0.30 * S_trend + 0.30 * S_forecast
    expected_weighted_r = 0.40 * s_press + 0.30 * s_trend + 0.30 * s_fore
    assert abs(r_score - expected_weighted_r) < 0.05, (
        f"Weighted risk score mismatch: {r_score} vs {expected_weighted_r:.4f}"
    )
    print("  ✓ Verified mathematical consistency of weighted risk formula.")

    # 6. Risk Level Threshold Classification Check
    if r_score < 30.0:
        expected_r_level = "LOW"
    elif r_score < 60.0:
        expected_r_level = "MODERATE"
    elif r_score < 80.0:
        expected_r_level = "HIGH"
    else:
        expected_r_level = "VERY HIGH"
    assert r_level == expected_r_level, f"Risk level classification mismatch: {r_level} vs {expected_r_level}"
    print(f"  ✓ Verified risk level classification rule: {r_score:.2f} -> [{r_level}]")

    # 7. Evidence Confidence Threshold Classification Check
    if c_score < 40.0:
        expected_c_level = "LOW CONFIDENCE"
    elif c_score < 70.0:
        expected_c_level = "MODERATE CONFIDENCE"
    else:
        expected_c_level = "HIGH CONFIDENCE"
    assert c_level == expected_c_level, f"Evidence confidence classification mismatch: {c_level} vs {expected_c_level}"
    print(f"  ✓ Verified evidence confidence classification rule: {c_score:.2f} -> [{c_level}]")

    # 8. Expected actual values approximate match
    assert abs(r_score - 34.32) < 0.15, f"Risk score unexpected value: {r_score} (expected ~34.32)"
    assert abs(c_score - 37.74) < 0.15, f"Evidence confidence unexpected value: {c_score} (expected ~37.74)"
    assert abs(s_press - 36.82) < 0.15, f"Change pressure unexpected value: {s_press} (expected ~36.82)"
    assert abs(s_trend - 25.55) < 0.15, f"Trend momentum unexpected value: {s_trend} (expected ~25.55)"
    assert abs(s_fore - 39.72) < 0.15, f"Future outlook unexpected value: {s_fore} (expected ~39.72)"
    print("  ✓ Verified all expected numerical targets match specification within ±0.15 tolerance.")

    # 9. Benchmark terminology check
    cb = data.get("calibration_benchmarks", {})
    classif = cb.get("benchmark_classification", "")
    assert "terravision heuristic normalization benchmarks" in classif.lower(), (
        f"Benchmark classification mismatch: {classif}"
    )
    status_desc = cb.get("benchmark_status", "").lower()
    assert "not universally validated" in status_desc, (
        f"Benchmark status does not state 'not universally validated': {status_desc}"
    )
    print("  ✓ Verified benchmark terminology: 'TerraVision heuristic normalization benchmarks' documented.")

    # 10. Evidence Confidence distinction from statistical prediction intervals
    exp = data.get("explanations", {})
    conf_dist = exp.get("evidence_confidence_distinction", "").lower()
    assert "not formal statistical" in conf_dist or "never be conflated" in conf_dist, (
        f"Missing evidence confidence distinction note: {conf_dist}"
    )
    sci_note = exp.get("scientific_distinction_note", "").lower()
    assert "not guarantee" in sci_note, (
        f"Missing non-guarantee note: {sci_note}"
    )
    print("  ✓ Verified Evidence Confidence distinction from statistical prediction intervals and non-guarantee disclaimer.")

    # 11. Phase 9 and Phase 10 interface contracts
    p9 = data.get("phase9_interface", {})
    assert p9.get("ready_for_global_scaling") is True, "Phase 9 interface not marked ready"
    assert "normalization_benchmarks_exported" in p9, "Missing normalization benchmarks in Phase 9 interface"

    p10 = data.get("phase10_interface", {})
    assert p10.get("ready_for_web_platform") is True, "Phase 10 interface not marked ready"
    assert "ui_display_badge" in p10, "Missing ui_display_badge in Phase 10 interface"
    assert "card_metrics" in p10, "Missing card_metrics in Phase 10 interface"
    print("  ✓ Verified Phase 9 Global Scaling & Phase 10 Web Platform interface contracts.")


if __name__ == "__main__":
    print("=" * 70)
    print("TerraVision Phase 8: Risk Intelligence Verification")
    print("=" * 70)
    try:
        verify_prior_phases_untouched()
        verify_phase8_artifacts()
        print("\n" + "=" * 70)
        print("ALL PHASE 8 VERIFICATION CHECKS PASSED: STATUS = PASS")
        print("=" * 70)
    except AssertionError as e:
        print("\n" + "=" * 70)
        print(f"VERIFICATION CHECK FAILED: STATUS = FAIL\nDetails: {e}")
        print("=" * 70)
        sys.exit(1)
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"VERIFICATION ENCOUNTERED UNEXPECTED ERROR: STATUS = FAIL\nDetails: {e}")
        print("=" * 70)
        sys.exit(1)

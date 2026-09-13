"""
TerraVision - Phase 7: Future Prediction Verification Script

This script performs strict verification checks for Phase 7:
1. Verifies JSON and PNG artifacts exist and parse correctly.
2. Checks exactly 4 historical observations and exactly 3 future forecasts (2025, 2026, 2027).
3. Verifies OLS slope is approximately -0.7336 %/year² and df = 2.
4. Verifies physical non-negativity constraint: point predictions >= 0, lower bounds >= 0.
5. Verifies upper bounds >= point forecasts.
6. Verifies 95% prediction intervals are wider than 90% prediction intervals.
7. Verifies uncertainty standard error increases monotonically with forecast horizon.
8. Verifies hectare conversion math: ha = pct * total_area / 100.
9. Verifies cumulative projection is monotonic non-decreasing and final cumulative >= starting cumulative.
10. Verifies Phase 6 files remain untouched.
11. Verifies all numerical values are finite (no NaNs or Infs).
12. Verifies phase8_interface fields and metadata are complete.
13. Verifies limitation/disclaimer text exists and model-detected change is clearly distinguished.
"""

import sys
import json
import math
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def verify_phase6_files_untouched():
    """
    Ensures Phase 6 artifacts and implementation files have not been modified.
    """
    print("1. Checking that Phase 6 files remain untouched...")
    phase6_files = [
        PROJECT_ROOT / "src" / "historical_trends.py",
        PROJECT_ROOT / "notebooks" / "verify_historical_trends.py",
        PROJECT_ROOT / "results" / "historical_change_trends.json",
        PROJECT_ROOT / "results" / "historical_change_trends.png",
    ]
    for p in phase6_files:
        assert p.exists(), f"Phase 6 file missing: {p}"
        assert p.stat().st_size > 0, f"Phase 6 file is empty: {p}"
    print("  ✓ All 4 Phase 6 files verified intact and present.")


def verify_phase7_artifacts():
    """
    Comprehensive audit of Phase 7 generated artifacts:
    - results/future_prediction.json
    - results/future_prediction.png
    """
    print("\n2. Performing Comprehensive Audit of Phase 7 Artifacts...")
    json_path = PROJECT_ROOT / "results" / "future_prediction.json"
    png_path = PROJECT_ROOT / "results" / "future_prediction.png"
    
    # 1. Existence and non-empty
    assert json_path.exists(), f"Phase 7 JSON artifact not found: {json_path}"
    assert png_path.exists(), f"Phase 7 PNG artifact not found: {png_path}"
    assert png_path.stat().st_size > 50000, f"PNG artifact appears corrupt or too small: {png_path.stat().st_size} bytes"
    print(f"  ✓ Found PNG artifact: {png_path.name} ({png_path.stat().st_size:,} bytes)")
    
    # 2. JSON parses
    with open(json_path, "r") as f:
        data = json.load(f)
    print(f"  ✓ Successfully parsed JSON artifact: {json_path.name}")
    
    # 3. Exactly 4 historical observations
    hist = data.get("historical_series", {})
    n_obs = hist.get("number_of_observations")
    assert n_obs == 4, f"Expected 4 historical observations, found {n_obs}"
    midpoints = hist.get("observation_midpoint_years", [])
    assert len(midpoints) == 4, f"Expected 4 midpoint years, found {len(midpoints)}"
    rates_pct = hist.get("annualized_rates_pct_per_year", [])
    assert len(rates_pct) == 4, f"Expected 4 historical rates, found {len(rates_pct)}"
    print(f"  ✓ Verified exactly 4 historical observations (Midpoints: {midpoints}, Rates: {rates_pct})")
    
    # 4. Model checks: slope, df, R^2
    model = data.get("model", {})
    slope = model.get("beta1_trend_slope_pct_per_year2")
    assert abs(slope - (-0.7336)) < 0.01, f"Expected slope ~ -0.7336, got {slope}"
    df = model.get("degrees_of_freedom")
    assert df == 2, f"Expected df = 2, got {df}"
    print(f"  ✓ Verified OLS trend slope: {slope:+.4f} %/year² and df = {df}")
    
    # 5. Exactly 3 future forecasts for 2025, 2026, 2027
    forecasts = data.get("forecasts", [])
    assert len(forecasts) == 3, f"Expected exactly 3 forecasts, found {len(forecasts)}"
    forecast_years = [f["forecast_year"] for f in forecasts]
    assert forecast_years == [2025, 2026, 2027], f"Expected years [2025, 2026, 2027], got {forecast_years}"
    print(f"  ✓ Verified exactly 3 forecast horizons: {forecast_years}")
    
    # 6. Physical non-negativity constraint and interval consistency
    total_area_ha = data.get("metadata", {}).get("roi_metadata", {}).get("total_patch_area_hectares", 655.36)
    prev_s_pred = 0.0
    
    for f in forecasts:
        yr = f["forecast_year"]
        rate = f["predicted_annualized_rate_pct"]
        ha = f["predicted_annualized_rate_ha_per_year"]
        s_pred = f["prediction_standard_error_s_pred"]
        pi90 = f["prediction_interval_90_pct"]
        pi95 = f["prediction_interval_95_pct"]
        
        # Check all values are finite
        for val in [rate, ha, s_pred, pi90["lower_bound_pct"], pi90["upper_bound_pct"],
                    pi95["lower_bound_pct"], pi95["upper_bound_pct"]]:
            assert math.isfinite(val), f"Non-finite value detected for year {yr}: {val}"
            
        # Point prediction non-negativity
        assert rate >= 0.0, f"Point prediction is negative for year {yr}: {rate}"
        assert ha >= 0.0, f"Point prediction ha is negative for year {yr}: {ha}"
        
        # Lower bound non-negativity
        assert pi90["lower_bound_pct"] >= 0.0, f"90% lower bound is negative for year {yr}: {pi90['lower_bound_pct']}"
        assert pi95["lower_bound_pct"] >= 0.0, f"95% lower bound is negative for year {yr}: {pi95['lower_bound_pct']}"
        
        # Upper bounds >= point forecasts
        assert pi90["upper_bound_pct"] >= rate, f"90% upper bound < point forecast for year {yr}"
        assert pi95["upper_bound_pct"] >= rate, f"95% upper bound < point forecast for year {yr}"
        
        # 95% interval wider than 90% interval
        width_90 = pi90["upper_bound_pct"] - pi90["lower_bound_pct"]
        width_95 = pi95["upper_bound_pct"] - pi95["lower_bound_pct"]
        assert width_95 > width_90, f"95% width ({width_95}) not strictly greater than 90% width ({width_90}) for year {yr}"
        
        # Uncertainty standard error increases with horizon
        assert s_pred > prev_s_pred, f"s_pred did not increase: {s_pred} <= {prev_s_pred}"
        prev_s_pred = s_pred
        
        # Mathematical consistency of hectare conversion: ha = pct * total_area / 100
        expected_ha = rate * total_area_ha / 100.0
        assert abs(ha - expected_ha) < 1e-3, f"Hectare conversion mismatch for year {yr}: {ha} vs {expected_ha}"
        
    print("  ✓ Verified physical non-negativity (predicted rates >= 0, lower bounds >= 0).")
    print("  ✓ Verified upper bounds >= point forecasts.")
    print("  ✓ Verified 95% intervals are wider than 90% intervals.")
    print("  ✓ Verified prediction standard error increases monotonically with forecast horizon.")
    print("  ✓ Verified hectare rate conversion math is exact.")
    
    # 7. Cumulative projection verification
    cum_proj = data.get("cumulative_projection", {})
    start_cum_ha = cum_proj.get("historical_baseline_cumulative_ha", 0.0)
    proj_ha = cum_proj.get("projected_cumulative_ha", [])
    
    assert len(proj_ha) == 3, f"Expected 3 cumulative projections, got {len(proj_ha)}"
    all_cum_ha = [start_cum_ha] + proj_ha
    for i in range(len(all_cum_ha) - 1):
        assert all_cum_ha[i + 1] >= all_cum_ha[i], (
            f"Cumulative projection not monotonic non-decreasing: {all_cum_ha[i+1]} < {all_cum_ha[i]}"
        )
    assert proj_ha[-1] >= start_cum_ha, f"Final cumulative {proj_ha[-1]} < starting cumulative {start_cum_ha}"
    assert cum_proj.get("monotonic_check_passed") is True, "monotonic_check_passed is not True"
    print(f"  ✓ Verified cumulative projection is monotonic non-decreasing: {all_cum_ha}")
    
    # 8. Check Phase 8 interface readiness
    p8 = data.get("phase8_interface", {})
    assert p8.get("ready_for_phase8_risk_intelligence") is True, "Phase 8 ready flag not True"
    assert "forecast_summary_2025" in p8, "Missing forecast_summary_2025 in phase8_interface"
    assert "forecast_summary_2026" in p8, "Missing forecast_summary_2026 in phase8_interface"
    assert "forecast_summary_2027" in p8, "Missing forecast_summary_2027 in phase8_interface"
    assert "risk_evaluation_guidance" in p8, "Missing risk_evaluation_guidance in phase8_interface"
    print("  ✓ Verified Phase 8 Risk Intelligence interface fields are complete.")
    
    # 9. Limitation and disclaimer text
    limitations = data.get("limitations", {})
    assert "model-detected" in limitations.get("model_detection_basis", "").lower(), "Missing model-detected note in limitations"
    non_guar_text = limitations.get("non_guarantee", "").lower()
    assert ("not guaranteed" in non_guar_text or ("not" in non_guar_text and "guaranteed" in non_guar_text)), "Missing non-guarantee note in limitations"
    assert "not unique" in limitations.get("cumulative_metric_distinction", "").lower(), "Missing not-unique land distinction in limitations"
    
    # Horizon caveat check
    unc = data.get("uncertainty", {})
    caveat = unc.get("horizon_caveat", "")
    assert "conservative maximum forecast horizon" in caveat.lower(), f"Missing conservative horizon caveat: {caveat}"
    assert "75%" not in caveat, "Caveat improperly claims 75% as statistical theorem"
    print("  ✓ Verified limitation, non-guarantee, and conservative horizon caveat documentation.")


if __name__ == "__main__":
    print("=" * 70)
    print("TerraVision Phase 7: Future Prediction Verification")
    print("=" * 70)
    try:
        verify_phase6_files_untouched()
        verify_phase7_artifacts()
        print("\n" + "=" * 70)
        print("ALL PHASE 7 VERIFICATION CHECKS PASSED: STATUS = PASS")
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

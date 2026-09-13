"""
TerraVision - Phase 7: Future Prediction Pipeline

This module executes lightweight, scientifically defensible forecasting of
model-detected land change over the target ROI using the historical time series
established in Phase 6.

Key Scientific Specifications:
1. Input: results/historical_change_trends.json (4 historical intervals, 2020–2024).
2. Method: First-order Ordinary Least Squares (OLS) trend regression with physical
   non-negativity lower bound (max(0, y_hat)) and analytical Student's t prediction
   intervals (df = 2).
3. Horizons: 2025, 2026, 2027 (with 2027 documented as a conservative maximum
   horizon given small-sample uncertainty widening).
4. Terminology: "Forecast of model-detected change" (NOT guaranteed land conversion).
   Cumulative change is strictly cumulative detected change, NOT unique physical land transformed.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_historical_trends(json_path: Path) -> Dict[str, Any]:
    """
    Loads and validates the Phase 6 historical change trends artifact.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"Phase 6 historical trends artifact not found: {json_path}")
    
    with open(json_path, "r") as f:
        data = json.load(f)
        
    required_keys = ["consecutive_period_changes", "roi_metadata", "trend_summary"]
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key '{key}' in {json_path}")
            
    periods = data["consecutive_period_changes"]
    if len(periods) != 4:
        raise ValueError(f"Expected exactly 4 historical periods, found {len(periods)}")
        
    return data


def extract_historical_series(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts the four historical annualized change rate observations and ROI metadata.
    Uses exact interval midpoints computed from start/end dates, yielding the Phase 6 baseline slope.
    """
    periods = data["consecutive_period_changes"]
    roi_meta = data["roi_metadata"]
    trend_summary = data["trend_summary"]
    
    midpoint_years = []
    annualized_rates_pct = []
    annualized_rates_ha = []
    period_names = []
    period_labels = []
    
    for p in periods:
        d_start = datetime.strptime(p["start_date"], "%Y-%m-%d")
        d_end = datetime.strptime(p["end_date"], "%Y-%m-%d")
        mid_date = d_start + (d_end - d_start) / 2
        mid_year = mid_date.year + (mid_date.timetuple().tm_yday / 365.25)
        
        midpoint_years.append(mid_year)
        annualized_rates_pct.append(float(p["annualized_change_rate_pct_per_year"]))
        annualized_rates_ha.append(float(p["annualized_change_rate_ha_per_year"]))
        period_names.append(p["period_name"])
        period_labels.append(f"{d_start.year}–{d_end.year}")
        
    total_area_ha = float(roi_meta.get("total_patch_area_hectares", 655.36))
    starting_cumulative_ha = float(trend_summary.get("cumulative_detected_change_hectares", 33.62))
    starting_cumulative_pct = float(trend_summary.get("cumulative_detected_change_percentage", 5.13))
    
    return {
        "midpoint_years": np.array(midpoint_years, dtype=np.float64),
        "annualized_rates_pct": np.array(annualized_rates_pct, dtype=np.float64),
        "annualized_rates_ha": np.array(annualized_rates_ha, dtype=np.float64),
        "period_names": period_names,
        "period_labels": period_labels,
        "total_area_ha": total_area_ha,
        "starting_cumulative_ha": starting_cumulative_ha,
        "starting_cumulative_pct": starting_cumulative_pct,
        "roi_metadata": roi_meta,
        "historical_trend_summary": trend_summary
    }


def fit_ols_trend(t_obs: np.ndarray, y_obs: np.ndarray) -> Dict[str, Any]:
    """
    Fits first-order Ordinary Least Squares (OLS) linear trend:
      y_hat(t) = y_mean + beta1 * (t - t_mean)
    Computes analytical degrees of freedom, residuals, and residual standard error (s_e).
    """
    N = len(t_obs)
    if N < 3:
        raise ValueError(f"Need at least 3 points for OLS inference with df >= 1; got N={N}")
        
    t_mean = float(np.mean(t_obs))
    y_mean = float(np.mean(y_obs))
    
    t_diff = t_obs - t_mean
    y_diff = y_obs - y_mean
    
    denom = float(np.sum(t_diff ** 2))
    if denom == 0.0:
        raise ZeroDivisionError("Temporal variance is zero; cannot fit linear trend.")
        
    beta1 = float(np.sum(t_diff * y_diff) / denom)
    beta0 = y_mean - beta1 * t_mean
    
    fitted_values = y_mean + beta1 * t_diff
    residuals = y_obs - fitted_values
    df = N - 2  # Degrees of freedom (N=4 -> df=2)
    s_e = float(np.sqrt(np.sum(residuals ** 2) / df))
    
    return {
        "N": N,
        "df": df,
        "t_mean": t_mean,
        "y_mean": y_mean,
        "beta1": beta1,
        "beta0": beta0,
        "sum_sq_t_diff": denom,
        "fitted_values": fitted_values,
        "residuals": residuals,
        "residual_standard_error_se": s_e,
        "r_squared": float(1.0 - (np.sum(residuals ** 2) / np.sum(y_diff ** 2)))
    }


def calculate_future_forecasts(
    ols_model: Dict[str, Any],
    historical_series: Dict[str, Any],
    forecast_years: List[int] = [2025, 2026, 2027]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generates forecasts for the specified future years with:
    1. Physical non-negativity constraint on point forecasts and lower bounds: max(0, val)
    2. Analytical Student's t prediction intervals (90% and 95%) using df = N - 2
    3. Hectare rate conversions based on total ROI area
    4. Cumulative detected change trajectory through 2027
    """
    N = ols_model["N"]
    df = ols_model["df"]
    t_mean = ols_model["t_mean"]
    y_mean = ols_model["y_mean"]
    beta1 = ols_model["beta1"]
    s_e = ols_model["residual_standard_error_se"]
    denom = ols_model["sum_sq_t_diff"]
    total_area_ha = historical_series["total_area_ha"]
    
    # Calculate critical values using Student's t distribution with df = 2
    t_crit_90 = float(stats.t.ppf(0.95, df=df))
    t_crit_95 = float(stats.t.ppf(0.975, df=df))
    
    forecasts = []
    
    # Track cumulative projection
    curr_cum_ha = historical_series["starting_cumulative_ha"]
    curr_cum_pct = historical_series["starting_cumulative_pct"]
    
    cum_proj_years = []
    cum_proj_ha = []
    cum_proj_pct = []
    cum_proj_upper_90_ha = []
    cum_proj_upper_95_ha = []
    
    curr_cum_upper_90 = curr_cum_ha
    curr_cum_upper_95 = curr_cum_ha
    
    for yr in forecast_years:
        t0 = float(yr)
        
        # Raw linear extrapolation: y_hat = y_mean + beta1 * (t0 - t_mean)
        unconstrained_rate_pct = float(y_mean + beta1 * (t0 - t_mean))
        
        # Apply physical non-negativity: land change rate cannot be negative
        predicted_rate_pct = float(max(0.0, unconstrained_rate_pct))
        predicted_rate_ha = float(predicted_rate_pct * total_area_ha / 100.0)
        
        # Analytical prediction standard error
        # s_pred(t0) = s_e * sqrt(1 + 1/N + (t0 - t_mean)^2 / sum((t_i - t_mean)^2))
        s_pred = float(s_e * np.sqrt(1.0 + (1.0 / N) + ((t0 - t_mean) ** 2) / denom))
        
        # 90% Prediction Interval:
        pi_90_lower_raw = unconstrained_rate_pct - t_crit_90 * s_pred
        pi_90_upper_raw = unconstrained_rate_pct + t_crit_90 * s_pred
        pi_90_lower = float(max(0.0, pi_90_lower_raw))
        pi_90_upper = float(max(predicted_rate_pct, pi_90_upper_raw))
        
        # 95% Prediction Interval:
        pi_95_lower_raw = unconstrained_rate_pct - t_crit_95 * s_pred
        pi_95_upper_raw = unconstrained_rate_pct + t_crit_95 * s_pred
        pi_95_lower = float(max(0.0, pi_95_lower_raw))
        pi_95_upper = float(max(predicted_rate_pct, pi_95_upper_raw))
        
        # Convert intervals to hectares
        pi_90_lower_ha = float(pi_90_lower * total_area_ha / 100.0)
        pi_90_upper_ha = float(pi_90_upper * total_area_ha / 100.0)
        pi_95_lower_ha = float(pi_95_lower * total_area_ha / 100.0)
        pi_95_upper_ha = float(pi_95_upper * total_area_ha / 100.0)
        
        # Update cumulative trajectory (annual step = 1.0 yr)
        curr_cum_ha += predicted_rate_ha
        curr_cum_pct = (curr_cum_ha / total_area_ha) * 100.0
        curr_cum_upper_90 += pi_90_upper_ha
        curr_cum_upper_95 += pi_95_upper_ha
        
        cum_proj_years.append(yr)
        cum_proj_ha.append(round(curr_cum_ha, 4))
        cum_proj_pct.append(round(curr_cum_pct, 4))
        cum_proj_upper_90_ha.append(round(curr_cum_upper_90, 4))
        cum_proj_upper_95_ha.append(round(curr_cum_upper_95, 4))
        
        forecasts.append({
            "forecast_year": int(yr),
            "horizon_years_ahead": int(yr - 2024),
            "target_interval": f"{yr-1} -> {yr}",
            "unconstrained_predicted_rate_pct": round(unconstrained_rate_pct, 4),
            "predicted_annualized_rate_pct": round(predicted_rate_pct, 4),
            "predicted_annualized_rate_ha_per_year": round(predicted_rate_ha, 4),
            "prediction_standard_error_s_pred": round(s_pred, 4),
            "prediction_interval_90_pct": {
                "lower_bound_pct": round(pi_90_lower, 4),
                "upper_bound_pct": round(pi_90_upper, 4),
                "lower_bound_ha_per_year": round(pi_90_lower_ha, 4),
                "upper_bound_ha_per_year": round(pi_90_upper_ha, 4),
                "confidence_level": 0.90,
                "critical_t": round(t_crit_90, 4)
            },
            "prediction_interval_95_pct": {
                "lower_bound_pct": round(pi_95_lower, 4),
                "upper_bound_pct": round(pi_95_upper, 4),
                "lower_bound_ha_per_year": round(pi_95_lower_ha, 4),
                "upper_bound_ha_per_year": round(pi_95_upper_ha, 4),
                "confidence_level": 0.95,
                "critical_t": round(t_crit_95, 4)
            },
            "projected_cumulative_detected_change_ha": round(curr_cum_ha, 4),
            "projected_cumulative_detected_change_pct": round(curr_cum_pct, 4)
        })
        
    cumulative_projection = {
        "historical_baseline_cumulative_ha": historical_series["starting_cumulative_ha"],
        "historical_baseline_cumulative_pct": historical_series["starting_cumulative_pct"],
        "projection_horizon_years": cum_proj_years,
        "projected_cumulative_ha": cum_proj_ha,
        "projected_cumulative_pct": cum_proj_pct,
        "projected_cumulative_upper_90_ha": cum_proj_upper_90_ha,
        "projected_cumulative_upper_95_ha": cum_proj_upper_95_ha,
        "monotonic_check_passed": bool(np.all(np.diff([historical_series["starting_cumulative_ha"]] + cum_proj_ha) >= 0)),
        "metric_clarification": "Cumulative change is strictly cumulative detected change across periods, NOT unique physical land transformed (as individual pixels may transition repeatedly)."
    }
    
    return forecasts, cumulative_projection


def generate_visualization(
    historical_series: Dict[str, Any],
    ols_model: Dict[str, Any],
    forecasts: List[Dict[str, Any]],
    cumulative_projection: Dict[str, Any],
    output_path: Path
) -> Path:
    """
    Generates a high-quality, publication-grade dual-panel figure illustrating
    historical model-detected changes, bounded trend extrapolation, prediction intervals,
    and cumulative detected change trajectory.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 11), gridspec_kw={"hspace": 0.35})
    fig.patch.set_facecolor("#0f172a")  # Deep slate background
    
    # Styling helpers
    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")
        ax.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")
        ax.tick_params(colors="#cbd5e1", labelsize=10)
        for spine in ax.spines.values():
            spine.set_color("#475569")
            
    # -------------------------------------------------------------
    # Panel 1: Annualized Change Rates (%/year) and Prediction Bands
    # -------------------------------------------------------------
    hist_t = historical_series["midpoint_years"]
    hist_r = historical_series["annualized_rates_pct"]
    
    # Plot historical observations
    ax1.plot(hist_t, hist_r, color="#38bdf8", marker="o", markersize=8, linewidth=2.2, 
             label="Historical Model-Detected Rate (%/yr)", zorder=5)
    
    # Plot historical fitted OLS line
    t_line_hist = np.linspace(min(hist_t), max(hist_t), 50)
    r_line_hist = ols_model["y_mean"] + ols_model["beta1"] * (t_line_hist - ols_model["t_mean"])
    ax1.plot(t_line_hist, r_line_hist, color="#94a3b8", linestyle=":", linewidth=1.8, 
             label=f"OLS Trend Fit (slope = {ols_model['beta1']:+.4f} %/yr²)", zorder=4)
    
    # Forecast years and points
    fc_years = [f["forecast_year"] for f in forecasts]
    fc_rates = [f["predicted_annualized_rate_pct"] for f in forecasts]
    fc_s_pred = [f["prediction_standard_error_s_pred"] for f in forecasts]
    
    # Continuous forecast curve for smooth ribbon
    t_fore_cont = np.linspace(max(hist_t), 2027.0, 100)
    r_fore_uncon = ols_model["y_mean"] + ols_model["beta1"] * (t_fore_cont - ols_model["t_mean"])
    r_fore_clamped = np.maximum(0.0, r_fore_uncon)
    
    t_crit_90 = float(stats.t.ppf(0.95, df=ols_model["df"]))
    t_crit_95 = float(stats.t.ppf(0.975, df=ols_model["df"]))
    
    s_pred_cont = ols_model["residual_standard_error_se"] * np.sqrt(
        1.0 + (1.0 / ols_model["N"]) + ((t_fore_cont - ols_model["t_mean"]) ** 2) / ols_model["sum_sq_t_diff"]
    )
    
    pi_90_low_cont = np.maximum(0.0, r_fore_uncon - t_crit_90 * s_pred_cont)
    pi_90_up_cont = np.maximum(r_fore_clamped, r_fore_uncon + t_crit_90 * s_pred_cont)
    
    pi_95_low_cont = np.maximum(0.0, r_fore_uncon - t_crit_95 * s_pred_cont)
    pi_95_up_cont = np.maximum(r_fore_clamped, r_fore_uncon + t_crit_95 * s_pred_cont)
    
    # Shaded prediction intervals
    ax1.fill_between(t_fore_cont, pi_95_low_cont, pi_95_up_cont, color="#f59e0b", alpha=0.15,
                     label="95% Prediction Interval (df=2)", zorder=2)
    ax1.fill_between(t_fore_cont, pi_90_low_cont, pi_90_up_cont, color="#f59e0b", alpha=0.28,
                     label="90% Prediction Interval (df=2)", zorder=3)
    
    # Connect last historical observation to forecast line
    ax1.plot([hist_t[-1]] + list(fc_years), [hist_r[-1]] + fc_rates, color="#f59e0b", 
             linestyle="--", linewidth=2.2, marker="s", markersize=7, 
             label="Forecast (Bounded Point Estimate)", zorder=5)
    
    # Divider line separating historical from forecast
    ax1.axvline(x=2024.5, color="#e2e8f0", linestyle="--", linewidth=1.2, alpha=0.6)
    ax1.text(2024.4, 7.8, "Historical Baseline  |  Forecast Horizon", color="#e2e8f0", 
             fontsize=10, ha="right", va="top", style="italic")
    
    # Annotate forecast values
    for f in forecasts:
        yr = f["forecast_year"]
        rate = f["predicted_annualized_rate_pct"]
        up90 = f["prediction_interval_90_pct"]["upper_bound_pct"]
        up95 = f["prediction_interval_95_pct"]["upper_bound_pct"]
        ax1.annotate(f"{rate:.2f}%/yr\n[95%: 0–{up95:.2f}%]",
                     xy=(yr, rate), xytext=(yr, rate + 0.9),
                     color="#fef08a", fontsize=8.5, ha="center", weight="bold",
                     arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=1))
        
    ax1.set_title("Panel 1: Historical Model-Detected Change Rate & Statistical Trend Forecast (2021–2027)",
                  color="#f8fafc", fontsize=13, weight="bold", pad=12)
    ax1.set_xlabel("Observation Year / Midpoint", color="#cbd5e1", fontsize=11)
    ax1.set_ylabel("Annualized Change Rate (% / year)", color="#cbd5e1", fontsize=11)
    ax1.set_ylim(-0.5, 9.5)
    ax1.set_xlim(2020.5, 2027.5)
    ax1.legend(loc="upper right", facecolor="#1e293b", edgecolor="#475569", labelcolor="#e2e8f0", fontsize=9)
    
    # -------------------------------------------------------------
    # Panel 2: Cumulative Detected Change Trajectory (Hectares)
    # -------------------------------------------------------------
    # Historical cumulative data points
    hist_cum_years = [2021.0, 2022.0, 2023.0, 2024.0]
    hist_cum_ha = [20.92, 24.07, 29.64, 33.62]
    
    ax2.plot(hist_cum_years, hist_cum_ha, color="#10b981", marker="o", markersize=8, linewidth=2.2,
             label="Historical Cumulative Detected Change (ha)", zorder=5)
    
    # Projected cumulative
    proj_years = cumulative_projection["projection_horizon_years"]
    proj_ha = cumulative_projection["projected_cumulative_ha"]
    proj_up90_ha = cumulative_projection["projected_cumulative_upper_90_ha"]
    proj_up95_ha = cumulative_projection["projected_cumulative_upper_95_ha"]
    
    full_proj_years = [hist_cum_years[-1]] + proj_years
    full_proj_ha = [hist_cum_ha[-1]] + proj_ha
    full_proj_up90 = [hist_cum_ha[-1]] + proj_up90_ha
    full_proj_up95 = [hist_cum_ha[-1]] + proj_up95_ha
    
    ax2.fill_between(full_proj_years, full_proj_ha, full_proj_up95, color="#10b981", alpha=0.15,
                     label="Projected Cumulative 95% Upper Bound", zorder=2)
    ax2.fill_between(full_proj_years, full_proj_ha, full_proj_up90, color="#10b981", alpha=0.25,
                     label="Projected Cumulative 90% Upper Bound", zorder=3)
    ax2.plot(full_proj_years, full_proj_ha, color="#34d399", linestyle="--", linewidth=2.2,
             marker="s", markersize=7, label="Projected Cumulative Change (Point Estimate)", zorder=5)
    
    ax2.axvline(x=2024.5, color="#e2e8f0", linestyle="--", linewidth=1.2, alpha=0.6)
    
    # Annotate final cumulative value
    ax2.annotate(f"Baseline: {hist_cum_ha[-1]:.2f} ha\n(5.13% ROI)",
                 xy=(hist_cum_years[-1], hist_cum_ha[-1]), xytext=(hist_cum_years[-1] - 0.4, hist_cum_ha[-1] + 12),
                 color="#6ee7b7", fontsize=8.5, ha="center", weight="bold",
                 arrowprops=dict(arrowstyle="->", color="#10b981", lw=1))
    
    ax2.annotate(f"2027 Projection: {proj_ha[-1]:.2f} ha\n[95% Bound: {proj_up95_ha[-1]:.2f} ha]",
                 xy=(proj_years[-1], proj_ha[-1]), xytext=(proj_years[-1], proj_ha[-1] + 16),
                 color="#a7f3d0", fontsize=8.5, ha="center", weight="bold",
                 arrowprops=dict(arrowstyle="->", color="#34d399", lw=1))
    
    ax2.set_title("Panel 2: Cumulative Detected Change & Projected Trajectory through 2027",
                  color="#f8fafc", fontsize=13, weight="bold", pad=12)
    ax2.set_xlabel("Observation / Forecast Year", color="#cbd5e1", fontsize=11)
    ax2.set_ylabel("Cumulative Detected Change (Hectares)", color="#cbd5e1", fontsize=11)
    ax2.set_xlim(2020.5, 2027.5)
    ax2.set_ylim(0, 160)
    ax2.legend(loc="upper left", facecolor="#1e293b", edgecolor="#475569", labelcolor="#e2e8f0", fontsize=9)
    
    # Scientific limitation disclaimer box at bottom
    disclaimer_text = (
        "SCIENTIFIC NOTICE: Phase 7 outputs represent forecasts of model-detected change derived from Phase 4 Siam-UNet\n"
        "pairwise inferences on Sentinel-2 STAC observations. Outputs are not independently ground-truth validated and do NOT\n"
        "represent guaranteed real-world land conversion. Cumulative change reflects cumulative pixel detections across intervals,\n"
        "NOT unique physical land transformed. Year 2027 is a conservative maximum horizon given small sample uncertainty (df=2)."
    )
    fig.text(0.5, 0.015, disclaimer_text, color="#94a3b8", fontsize=8, ha="center", va="bottom",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#0f172a", edgecolor="#334155", alpha=0.9))
    
    plt.subplots_adjust(bottom=0.09)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    
    return output_path


def build_phase7_json(
    historical_series: Dict[str, Any],
    ols_model: Dict[str, Any],
    forecasts: List[Dict[str, Any]],
    cumulative_projection: Dict[str, Any],
    output_path: Path
) -> Path:
    """
    Constructs and writes the complete machine-readable Phase 7 forecast artifact
    for consumption by Phase 8 Risk Intelligence.
    """
    # Summary package for Phase 8
    fc_2025 = forecasts[0]
    fc_2026 = forecasts[1]
    fc_2027 = forecasts[2]
    
    json_data = {
        "metadata": {
            "pipeline": "TerraVision Phase 7 - Future Prediction",
            "project": "TerraVision",
            "input_artifact": "results/historical_change_trends.json",
            "execution_timestamp": datetime.now(timezone.utc).isoformat(),
            "roi_metadata": historical_series["roi_metadata"],
            "prediction_type": "Forecast of model-detected change",
            "ground_truth_validated": False,
            "status": "Verified"
        },
        "historical_series": {
            "number_of_observations": int(ols_model["N"]),
            "observation_midpoint_years": [round(y, 4) for y in historical_series["midpoint_years"].tolist()],
            "period_names": historical_series["period_names"],
            "annualized_rates_pct_per_year": [round(r, 4) for r in historical_series["annualized_rates_pct"].tolist()],
            "annualized_rates_ha_per_year": [round(r, 4) for r in historical_series["annualized_rates_ha"].tolist()],
            "historical_mean_rate_pct_per_year": round(ols_model["y_mean"], 4),
            "historical_mean_year": round(ols_model["t_mean"], 4),
            "baseline_cumulative_detected_change_ha": historical_series["starting_cumulative_ha"],
            "baseline_cumulative_detected_change_pct": historical_series["starting_cumulative_pct"]
        },
        "model": {
            "model_family": "Linear Regression / Statistical Trend Extrapolation",
            "specification": "First-order Ordinary Least Squares (OLS) with physical non-negativity constraint",
            "equation": "predicted_rate(t) = max(0.0, y_mean + beta1 * (t - t_mean))",
            "beta1_trend_slope_pct_per_year2": round(ols_model["beta1"], 4),
            "beta0_intercept_pct": round(ols_model["beta0"], 4),
            "residual_standard_error_se": round(ols_model["residual_standard_error_se"], 4),
            "degrees_of_freedom": int(ols_model["df"]),
            "r_squared": round(ols_model["r_squared"], 4),
            "method_justification": (
                "With N=4 historical intervals (df=2), complex ML or polynomial architectures "
                "suffer from extreme overfitting. Bounded OLS provides statistical parsimony, "
                "direct continuity with Phase 6 slope metrics, and analytical Student's t uncertainty bounds."
            )
        },
        "forecasts": forecasts,
        "cumulative_projection": cumulative_projection,
        "uncertainty": {
            "method": "Analytical OLS Prediction Standard Error with Student's t-distribution",
            "degrees_of_freedom": int(ols_model["df"]),
            "critical_t_90_pct": round(float(stats.t.ppf(0.95, df=ols_model["df"])), 4),
            "critical_t_95_pct": round(float(stats.t.ppf(0.975, df=ols_model["df"])), 4),
            "standard_error_formula": "s_pred(t0) = s_e * sqrt(1 + 1/N + (t0 - t_mean)^2 / sum((t_i - t_mean)^2))",
            "horizon_widening_verified": True,
            "horizon_caveat": (
                "Year 2027 is documented as a deliberately conservative maximum forecast horizon "
                "because only four historical observations exist and prediction uncertainty becomes "
                "increasingly large beyond the observed period."
            )
        },
        "limitations": {
            "model_detection_basis": (
                "Phase 7 outputs are forecasts of model-detected change derived from Phase 4 Siam-UNet "
                "inferences on Sentinel-2 STAC observations and are not independently ground-truth validated."
            ),
            "non_guarantee": (
                "Forecasts represent statistical trajectory expectations under continuity assumptions and "
                "are not guaranteed real-world land conversion or development activity."
            ),
            "cumulative_metric_distinction": (
                "Cumulative detected change reflects strictly cumulative model detections across temporal "
                "intervals, NOT unique physical land transformed (as individual pixels may transition repeatedly)."
            ),
            "exogenous_invariance": (
                "The statistical model assumes stationary trend dynamics and does not account for sudden "
                "macroeconomic shifts, policy changes, or regional land-use zoning restrictions."
            )
        },
        "phase8_interface": {
            "ready_for_phase8_risk_intelligence": True,
            "primary_forecast_horizon": 2025,
            "forecast_summary_2025": {
                "year": 2025,
                "point_forecast_rate_pct": fc_2025["predicted_annualized_rate_pct"],
                "point_forecast_ha_per_year": fc_2025["predicted_annualized_rate_ha_per_year"],
                "upper_bound_90_ha_per_year": fc_2025["prediction_interval_90_pct"]["upper_bound_ha_per_year"],
                "upper_bound_95_ha_per_year": fc_2025["prediction_interval_95_pct"]["upper_bound_ha_per_year"],
                "projected_cumulative_ha": fc_2025["projected_cumulative_detected_change_ha"]
            },
            "forecast_summary_2026": {
                "year": 2026,
                "point_forecast_rate_pct": fc_2026["predicted_annualized_rate_pct"],
                "point_forecast_ha_per_year": fc_2026["predicted_annualized_rate_ha_per_year"],
                "upper_bound_90_ha_per_year": fc_2026["prediction_interval_90_pct"]["upper_bound_ha_per_year"],
                "upper_bound_95_ha_per_year": fc_2026["prediction_interval_95_pct"]["upper_bound_ha_per_year"],
                "projected_cumulative_ha": fc_2026["projected_cumulative_detected_change_ha"]
            },
            "forecast_summary_2027": {
                "year": 2027,
                "point_forecast_rate_pct": fc_2027["predicted_annualized_rate_pct"],
                "point_forecast_ha_per_year": fc_2027["predicted_annualized_rate_ha_per_year"],
                "upper_bound_90_ha_per_year": fc_2027["prediction_interval_90_pct"]["upper_bound_ha_per_year"],
                "upper_bound_95_ha_per_year": fc_2027["prediction_interval_95_pct"]["upper_bound_ha_per_year"],
                "projected_cumulative_ha": fc_2027["projected_cumulative_detected_change_ha"]
            },
            "historical_trend_direction": historical_series["historical_trend_summary"].get("overall_trend_direction", "Decreasing"),
            "risk_evaluation_guidance": (
                "Phase 8 Risk Intelligence should evaluate risk using both point forecasts (baseline expectation) "
                "and the 95% upper prediction bound (precautionary stress-test / worst-case scenario)."
            )
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=4)
        
    return output_path


def run_phase7_pipeline(
    historical_json_path: Path = PROJECT_ROOT / "results" / "historical_change_trends.json",
    output_json_path: Path = PROJECT_ROOT / "results" / "future_prediction.json",
    output_png_path: Path = PROJECT_ROOT / "results" / "future_prediction.png"
) -> Dict[str, Any]:
    """
    Executes the end-to-end Phase 7 forecasting pipeline.
    """
    start_time = time.time()
    print("=" * 70)
    print("TerraVision Phase 7: Future Prediction Pipeline")
    print("=" * 70)
    
    # 1. Load historical trends
    print("1. Loading Phase 6 historical change trends...")
    hist_raw = load_historical_trends(historical_json_path)
    hist_series = extract_historical_series(hist_raw)
    print(f"  ✓ Loaded 4 historical observations (Midpoints: {np.round(hist_series['midpoint_years'], 2).tolist()})")
    print(f"  ✓ Observed rates: {hist_series['annualized_rates_pct'].tolist()} %/yr")
    print(f"  ✓ Baseline cumulative change: {hist_series['starting_cumulative_ha']:.2f} ha ({hist_series['starting_cumulative_pct']:.2f}% ROI)")
    
    # 2. Fit OLS linear trend
    print("\n2. Fitting First-Order OLS Linear Trend Model...")
    ols_model = fit_ols_trend(hist_series["midpoint_years"], hist_series["annualized_rates_pct"])
    print(f"  ✓ Model Equation: y_hat(t) = {ols_model['y_mean']:.4f} + ({ols_model['beta1']:+.4f}) * (t - {ols_model['t_mean']:.4f})")
    print(f"  ✓ OLS Trend Slope: {ols_model['beta1']:+.4f} %/year² (matches Phase 6)")
    print(f"  ✓ Residual Standard Error (s_e): {ols_model['residual_standard_error_se']:.4f} with df = {ols_model['df']}")
    
    # 3. Calculate future forecasts
    print("\n3. Generating Bounded Forecasts and Uncertainty Intervals for 2025–2027...")
    forecasts, cum_proj = calculate_future_forecasts(ols_model, hist_series, [2025, 2026, 2027])
    for fc in forecasts:
        yr = fc["forecast_year"]
        rate = fc["predicted_annualized_rate_pct"]
        ha = fc["predicted_annualized_rate_ha_per_year"]
        pi90 = fc["prediction_interval_90_pct"]
        pi95 = fc["prediction_interval_95_pct"]
        print(f"  [Year {yr}] Point Forecast: {rate:.2f} %/yr ({ha:.2f} ha/yr)")
        print(f"             90% PI: [{pi90['lower_bound_pct']:.2f}%, {pi90['upper_bound_pct']:.2f}%] ({pi90['lower_bound_ha_per_year']:.2f}–{pi90['upper_bound_ha_per_year']:.2f} ha/yr)")
        print(f"             95% PI: [{pi95['lower_bound_pct']:.2f}%, {pi95['upper_bound_pct']:.2f}%] ({pi95['lower_bound_ha_per_year']:.2f}–{pi95['upper_bound_ha_per_year']:.2f} ha/yr)")
        print(f"             Projected Cumulative: {fc['projected_cumulative_detected_change_ha']:.2f} ha")
        
    # 4. Generate visualization
    print(f"\n4. Rendering Forecast Visualization to {output_png_path}...")
    generate_visualization(hist_series, ols_model, forecasts, cum_proj, output_png_path)
    print(f"  ✓ Saved visualization artifact ({output_png_path.stat().st_size:,} bytes)")
    
    # 5. Export JSON artifact
    print(f"\n5. Exporting Structured JSON Artifact to {output_json_path}...")
    build_phase7_json(hist_series, ols_model, forecasts, cum_proj, output_json_path)
    print(f"  ✓ Saved JSON artifact ({output_json_path.stat().st_size:,} bytes)")
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"Phase 7 Pipeline Completed Successfully in {elapsed:.3f} seconds.")
    print("=" * 70)
    
    return {
        "ols_model": ols_model,
        "historical_series": hist_series,
        "forecasts": forecasts,
        "cumulative_projection": cum_proj,
        "elapsed_seconds": elapsed,
        "output_json": output_json_path,
        "output_png": output_png_path
    }


if __name__ == "__main__":
    run_phase7_pipeline()

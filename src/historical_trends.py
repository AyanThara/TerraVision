"""
TerraVision - Phase 6: Historical Change Trends Analysis Pipeline

This module executes multi-temporal historical change analysis across 5 genuine
Sentinel-2 Level-2A surface reflectance observations (2020–2024) over a fixed
geographic ROI at native 10m GSD. It uses the pre-trained Phase 4 Siam-UNet
checkpoint to perform pairwise change detection across 4 consecutive periods,
quantifies spatial changes, evaluates historical trend trajectories, and exports
both machine-readable JSON (for Phase 7 Future Prediction) and publication-grade
trend visualizations.
"""

import sys
import json
import io
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image
from torchvision import transforms

from src.change_detection_model import SiamUNet
from notebooks.verify_historical_trends import classify_trend_direction, calculate_linear_trend_slope
from notebooks.verify_quantification import categorize_change_intensity


# Configuration constants
STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
STAC_CROP_URL = "https://planetarycomputer.microsoft.com/api/data/v1/item/crop.png"

# Fixed Geographic ROI: Las Vegas / Henderson Urban Expansion Zone
# Longitude: -115.12 to -115.09, Latitude: 36.00 to 36.03
FIXED_BBOX = [-115.12, 36.00, -115.09, 36.03]
PATCH_SIZE = 256
GSD_METERS = 10.0  # Native Sentinel-2 GSD (10m/pixel)


def query_stac_multi_temporal_scenes(
    bbox: List[float] = FIXED_BBOX,
    years: List[str] = ["2020", "2021", "2022", "2023", "2024"],
) -> List[Dict[str, Any]]:
    """
    Dynamically queries Microsoft Planetary Computer STAC for genuine Sentinel-2 Level-2A
    surface reflectance scenes matching the ROI and low cloud cover during the summer seasonal window.
    """
    print("=" * 70)
    print("1. Querying STAC API for Genuine Multi-Temporal Sentinel-2 Observations...")
    print("=" * 70)
    
    observations = []
    
    for yr in years:
        payload = {
            "collections": ["sentinel-2-l2a"],
            "bbox": bbox,
            "datetime": f"{yr}-06-01T00:00:00Z/{yr}-08-20T23:59:59Z",
            "query": {"eo:cloud_cover": {"lt": 2.5}},
            "limit": 1
        }
        
        try:
            resp = requests.post(STAC_SEARCH_URL, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            features = data.get("features", [])
            
            if not features:
                raise RuntimeError(f"No valid low-cloud Sentinel-2 scene found for year {yr}")
            
            item = features[0]
            props = item["properties"]
            dt_str = props["datetime"]
            scene_id = item["id"]
            cloud_cover = props["eo:cloud_cover"]
            
            print(f"  [T{len(observations)}] Year {yr}: Date = {dt_str[:10]} | Cloud = {cloud_cover:.2f}% | Scene = {scene_id}")
            
            observations.append({
                "temporal_index": len(observations),
                "target_year": int(yr),
                "acquisition_datetime": dt_str,
                "acquisition_date": dt_str[:10],
                "scene_id": scene_id,
                "cloud_cover_percentage": round(cloud_cover, 2),
                "platform": props.get("platform", "sentinel-2"),
                "constellation": props.get("constellation", "Sentinel-2"),
            })
        except Exception as e:
            print(f"Error querying year {yr}: {e}")
            raise
            
    print(f"\nSuccessfully retrieved metadata for {len(observations)} genuine temporal observations.")
    return observations


def fetch_roi_crops(
    observations: List[Dict[str, Any]],
    bbox: List[float] = FIXED_BBOX,
    patch_size: int = PATCH_SIZE,
) -> List[Image.Image]:
    """
    Fetches genuine 256x256 windowed RGB crops for each observation via the STAC Data API.
    """
    print("\n" + "=" * 70)
    print("2. Fetching Windowed RGB Surface Reflectance Crops (256x256 @ 10m GSD)...")
    print("=" * 70)
    
    feature_geom = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
                [bbox[0], bbox[1]],
            ]]
        },
        "properties": {}
    }
    
    images = []
    
    for obs in observations:
        idx = obs["temporal_index"]
        scene_id = obs["scene_id"]
        date_str = obs["acquisition_date"]
        
        crop_resp = requests.post(
            STAC_CROP_URL,
            params={
                "collection": "sentinel-2-l2a",
                "item": scene_id,
                "assets": "visual",
                "width": patch_size,
                "height": patch_size,
            },
            json=feature_geom,
            timeout=20,
        )
        crop_resp.raise_for_status()
        
        img = Image.open(io.BytesIO(crop_resp.content))
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        print(f"  [T{idx}] {date_str}: Downloaded crop ({img.size[0]}x{img.size[1]} RGB, {len(crop_resp.content):,} bytes)")
        images.append(img)
        
    return images


def load_siam_unet_model() -> SiamUNet:
    """
    Loads the trained Phase 4 Siam-UNet checkpoint from disk.
    """
    print("\n" + "=" * 70)
    print("3. Loading Pre-Trained Phase 4 Siam-UNet Checkpoint...")
    print("=" * 70)
    
    model_path = PROJECT_ROOT / "models" / "v3_siam_unet_change_detection.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained Phase 4 checkpoint not found at: {model_path}")
        
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Target Device: {device}")
    
    model = SiamUNet(in_channels=6, num_classes=1, init_features=32)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"  Successfully loaded Siam-UNet from: {model_path.name}")
    return model, device


def run_pairwise_change_detection(
    images: List[Image.Image],
    observations: List[Dict[str, Any]],
    model: SiamUNet,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    """
    Runs pairwise inference for each consecutive observation pair:
    (T0 -> T1), (T1 -> T2), (T2 -> T3), (T3 -> T4)
    Computes change masks and physical quantification metrics.
    """
    print("\n" + "=" * 70)
    print("4. Executing Consecutive Pairwise Change Detection & Quantification...")
    print("=" * 70)
    
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    periods = []
    masks = []
    
    total_pixels = PATCH_SIZE * PATCH_SIZE
    pixel_area_m2 = GSD_METERS * GSD_METERS
    
    for i in range(len(images) - 1):
        obs_A = observations[i]
        obs_B = observations[i + 1]
        
        img_A = images[i]
        img_B = images[i + 1]
        
        tensor_A = img_transform(img_A).unsqueeze(0).to(device)
        tensor_B = img_transform(img_B).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(tensor_A, tensor_B)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
            binary_mask = (prob > 0.5).astype(np.uint8)
            
        masks.append(binary_mask)
        
        # Temporal delta
        date_A = datetime.strptime(obs_A["acquisition_date"], "%Y-%m-%d")
        date_B = datetime.strptime(obs_B["acquisition_date"], "%Y-%m-%d")
        delta_days = (date_B - date_A).days
        delta_years = delta_days / 365.25
        
        # Quantification
        changed_pixels = int(np.sum(binary_mask > 0))
        unchanged_pixels = total_pixels - changed_pixels
        change_pct = (changed_pixels / total_pixels) * 100.0
        unchanged_pct = (unchanged_pixels / total_pixels) * 100.0
        
        changed_area_m2 = changed_pixels * pixel_area_m2
        changed_area_ha = changed_area_m2 / 10000.0
        
        annualized_pct_rate = change_pct / delta_years if delta_years > 0 else 0.0
        annualized_ha_rate = changed_area_ha / delta_years if delta_years > 0 else 0.0
        
        intensity = categorize_change_intensity(change_pct)
        
        period_data = {
            "period_index": i + 1,
            "period_name": f"P{i+1}: {obs_A['acquisition_date']} -> {obs_B['acquisition_date']}",
            "start_date": obs_A["acquisition_date"],
            "end_date": obs_B["acquisition_date"],
            "start_observation_index": i,
            "end_observation_index": i + 1,
            "delta_days": delta_days,
            "delta_years": round(delta_years, 3),
            "total_pixels": total_pixels,
            "changed_pixels": changed_pixels,
            "unchanged_pixels": unchanged_pixels,
            "change_percentage": round(change_pct, 2),
            "unchanged_percentage": round(unchanged_pct, 2),
            "gsd_meters_per_pixel": GSD_METERS,
            "changed_area_m2": round(changed_area_m2, 2),
            "changed_area_hectares": round(changed_area_ha, 4),
            "annualized_change_rate_pct_per_year": round(annualized_pct_rate, 2),
            "annualized_change_rate_ha_per_year": round(annualized_ha_rate, 4),
            "change_intensity": intensity,
        }
        periods.append(period_data)
        
        print(f"  [Period {i+1}] {period_data['period_name']}")
        print(f"    Interval: {delta_days} days ({delta_years:.2f} yrs)")
        print(f"    Changed: {changed_pixels:,} px ({change_pct:.2f}%) | Area: {changed_area_ha:.2f} ha ({changed_area_m2:,.0f} m²)")
        print(f"    Annualized Rate: {annualized_pct_rate:.2f} %/yr ({annualized_ha_rate:.2f} ha/yr) | Intensity: {intensity}")

    return periods, masks


def analyze_historical_trends(
    observations: List[Dict[str, Any]],
    periods: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Computes the trajectory, velocity slope, and overall trend classification
    (Increasing / Decreasing / Stable) across the consecutive periods.
    """
    print("\n" + "=" * 70)
    print("5. Evaluating Historical Change Trajectory & Trend Direction...")
    print("=" * 70)
    
    # Midpoint years for each period
    midpoint_years = []
    annualized_rates_pct = []
    annualized_rates_ha = []
    period_change_pcts = []
    
    for p in periods:
        d_start = datetime.strptime(p["start_date"], "%Y-%m-%d")
        d_end = datetime.strptime(p["end_date"], "%Y-%m-%d")
        mid_date = d_start + (d_end - d_start) / 2
        mid_year = mid_date.year + (mid_date.timetuple().tm_yday / 365.25)
        midpoint_years.append(mid_year)
        
        annualized_rates_pct.append(p["annualized_change_rate_pct_per_year"])
        annualized_rates_ha.append(p["annualized_change_rate_ha_per_year"])
        period_change_pcts.append(p["change_percentage"])

    # Calculate linear trend slope: d(rate)/dt (%/year^2)
    slope_pct_per_year2 = calculate_linear_trend_slope(midpoint_years, annualized_rates_pct)
    slope_ha_per_year2 = calculate_linear_trend_slope(midpoint_years, annualized_rates_ha)
    
    # Classify overall trend
    trend_direction = classify_trend_direction(slope_pct_per_year2, threshold=0.5)
    
    # Cumulative changed area trajectory
    cumulative_ha = []
    cumulative_pct = []
    current_ha = 0.0
    current_pct = 0.0
    for p in periods:
        current_ha += p["changed_area_hectares"]
        current_pct += p["change_percentage"]
        cumulative_ha.append(round(current_ha, 4))
        cumulative_pct.append(round(current_pct, 2))
        
    mean_annual_rate_pct = float(np.mean(annualized_rates_pct))
    mean_annual_rate_ha = float(np.mean(annualized_rates_ha))
    
    print(f"  Annualized Rate Series (%/yr): {annualized_rates_pct}")
    print(f"  Trend Slope (dr/dt):           {slope_pct_per_year2:+.3f} %/year²")
    print(f"  Trend Direction:               [{trend_direction.upper()}]")
    print(f"  Cumulative Transformed Area:   {cumulative_ha[-1]:.2f} ha ({cumulative_pct[-1]:.2f}% total)")
    print(f"  Mean Annual Rate:              {mean_annual_rate_pct:.2f} %/yr ({mean_annual_rate_ha:.2f} ha/yr)")

    trend_summary = {
        "overall_trend_direction": trend_direction,
        "trend_slope_pct_per_year2": round(slope_pct_per_year2, 4),
        "trend_slope_ha_per_year2": round(slope_ha_per_year2, 4),
        "mean_annual_change_rate_pct": round(mean_annual_rate_pct, 2),
        "mean_annual_change_rate_ha": round(mean_annual_rate_ha, 4),
        "cumulative_detected_change_hectares": cumulative_ha[-1],
        "cumulative_detected_change_percentage": cumulative_pct[-1],
        "cumulative_trajectory_hectares": cumulative_ha,
        "cumulative_trajectory_percentage": cumulative_pct,
        "cumulative_metric_note": "Cumulative change is strictly cumulative detected change across periods, NOT unique physical land transformed (as individual pixels may transition repeatedly).",
        "model_limitation": "Phase 6 outputs are model-detected historical change estimates and are not independently ground-truth validated for these STAC observations.",
        "classification_rule": "Increasing: slope > +0.5 %/yr², Decreasing: slope < -0.5 %/yr², Stable: |slope| <= 0.5 %/yr²",
        "phase7_ready": True,
        "phase7_target_features": [
            "period_index",
            "delta_years",
            "change_percentage",
            "changed_area_hectares",
            "annualized_change_rate_pct_per_year",
            "cumulative_detected_change_hectares"
        ]
    }
    
    return trend_summary


def export_machine_readable_json(
    observations: List[Dict[str, Any]],
    periods: List[Dict[str, Any]],
    trend_summary: Dict[str, Any],
    bbox: List[float] = FIXED_BBOX,
    output_path: Path = PROJECT_ROOT / "results" / "historical_change_trends.json",
) -> Path:
    """
    Exports a structured, machine-readable JSON schema designed for Phase 7 Future Prediction.
    """
    print("\n" + "=" * 70)
    print(f"6. Exporting Phase 7 Machine-Readable Results: {output_path.name}...")
    print("=" * 70)
    
    total_patch_area_m2 = (PATCH_SIZE * GSD_METERS) ** 2
    total_patch_area_ha = total_patch_area_m2 / 10000.0
    total_patch_area_km2 = total_patch_area_m2 / 1000000.0
    
    payload = {
        "pipeline_metadata": {
            "phase": "Phase 6 - Historical Change Trends",
            "project": "TerraVision",
            "sensor": "Sentinel-2 MSI",
            "processing_level": "Level-2A (BOA Surface Reflectance)",
            "stac_provider": "Microsoft Planetary Computer",
            "model_architecture": "Siam-UNet (Phase 4 Pre-Trained Checkpoint)",
            "model_limitation": "Phase 6 outputs are model-detected historical change estimates and are not independently ground-truth validated for these STAC observations.",
            "execution_timestamp": datetime.now().isoformat(),
        },
        "roi_metadata": {
            "region_name": "Las Vegas / Henderson Urban Expansion Zone",
            "bounding_box_wgs84": bbox,
            "patch_dimensions_pixels": [PATCH_SIZE, PATCH_SIZE],
            "spatial_resolution_meters_per_pixel": GSD_METERS,
            "total_patch_area_m2": total_patch_area_m2,
            "total_patch_area_hectares": total_patch_area_ha,
            "total_patch_area_km2": total_patch_area_km2,
        },
        "temporal_observations": observations,
        "consecutive_period_changes": periods,
        "trend_summary": trend_summary,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=4)
        
    print(f"  Successfully wrote Phase 7 compatible JSON to: {output_path}")
    return output_path


def render_trend_visualization(
    images: List[Image.Image],
    masks: List[np.ndarray],
    observations: List[Dict[str, Any]],
    periods: List[Dict[str, Any]],
    trend_summary: Dict[str, Any],
    output_path: Path = PROJECT_ROOT / "results" / "historical_change_trends.png",
):
    """
    Renders an aesthetic, publication-grade multi-panel visualization summarizing:
    1. Multi-temporal RGB image sequence (5 genuine observations).
    2. Consecutive binary change masks (4 periods).
    3. Analytical charts: Annualized rates, cumulative trajectory, and summary card.
    """
    print("\n" + "=" * 70)
    print(f"7. Rendering Historical Change Trends Visualization: {output_path.name}...")
    print("=" * 70)
    
    fig = plt.figure(figsize=(24, 15), facecolor="#0f111a")
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    fig.suptitle(
        "TerraVision Phase 6: Multi-Temporal Historical Change Trend Analysis (2020 – 2024)\n"
        "Sentinel-2 Level-2A Surface Reflectance (10m GSD) | Siam-UNet Change Inference",
        fontsize=16,
        fontweight="bold",
        color="#ffffff",
        y=0.98,
    )
    
    # ----------------------------------------------------
    # ROW 1: 5 Genuine Observations (T0 .. T4)
    # ----------------------------------------------------
    for i in range(len(images)):
        ax = plt.subplot2grid((3, 5), (0, i))
        ax.set_facecolor("#161925")
        ax.imshow(images[i])
        obs = observations[i]
        ax.set_title(
            f"T{i} — {obs['acquisition_date']}\n"
            f"Cloud: {obs['cloud_cover_percentage']:.2f}% | S2 L2A",
            fontsize=11,
            color="#e2e8f0",
            pad=8,
            fontweight="semibold"
        )
        ax.axis("off")
        
    # ----------------------------------------------------
    # ROW 2: 4 Consecutive Binary Change Masks (P1 .. P4) + 1 Spacer/Summary Mini
    # ----------------------------------------------------
    for i in range(len(masks)):
        ax = plt.subplot2grid((3, 5), (1, i))
        ax.set_facecolor("#161925")
        p = periods[i]
        
        # Display mask with custom colormap (red for change)
        mask_np = masks[i]
        colored_mask = np.zeros((PATCH_SIZE, PATCH_SIZE, 3), dtype=np.float32)
        colored_mask[mask_np == 1] = [0.95, 0.25, 0.25]  # Bright red for changed
        colored_mask[mask_np == 0] = [0.08, 0.10, 0.15]  # Dark navy for unchanged
        
        ax.imshow(colored_mask)
        ax.set_title(
            f"{p['period_name'][:18]}...\n"
            f"Δ {p['change_percentage']:.2f}% ({p['changed_area_hectares']:.1f} ha)",
            fontsize=10.5,
            color="#fca5a5",
            pad=8,
            fontweight="semibold"
        )
        ax.axis("off")

    # Panel 5 in Row 2: Change Legend & Detection Info
    ax_legend = plt.subplot2grid((3, 5), (1, 4))
    ax_legend.set_facecolor("#161925")
    ax_legend.axis("off")
    legend_text = (
        "DETECTION PROTOCOL\n"
        "-------------------------------------\n"
        "Model:      Phase 4 Siam-UNet\n"
        "Inference:  Consecutive Bi-Temporal\n"
        "Threshold:  Sigmoid Prob > 0.50\n"
        "GSD:        10.0 m/pixel (Native)\n"
        "Patch Area: 655.36 ha (6.55 km²)\n"
        "Legend:\n"
        "  ■ Red:   Detected Land Change\n"
        "  ■ Dark:  Unchanged Surface"
    )
    ax_legend.text(
        0.08, 0.5, legend_text,
        fontsize=10,
        fontfamily="monospace",
        color="#94a3b8",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#1e2235", edgecolor="#334155")
    )

    # ----------------------------------------------------
    # ROW 3: Analytical Charts (3 Panels)
    # ----------------------------------------------------
    # Chart 1: Annualized Change Rates Bar Chart (Columns 0 & 1)
    ax_bar = plt.subplot2grid((3, 5), (2, 0), colspan=2)
    ax_bar.set_facecolor("#161925")
    
    period_labels = [f"P{p['period_index']}\n({p['start_date'][:4]}-{p['end_date'][:4]})" for p in periods]
    rates = [p["annualized_change_rate_pct_per_year"] for p in periods]
    bar_colors = ["#38bdf8", "#818cf8", "#c084fc", "#f472b6"]
    
    bars = ax_bar.bar(period_labels, rates, color=bar_colors, width=0.55, edgecolor="#ffffff", linewidth=0.5)
    ax_bar.set_title("Annualized Change Rate (% / Year) Across Periods", fontsize=12, color="#ffffff", fontweight="bold", pad=10)
    ax_bar.set_ylabel("Change Rate (% / Year)", fontsize=10, color="#94a3b8")
    ax_bar.tick_params(colors="#94a3b8", labelsize=9.5)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.2, color="#94a3b8")
    
    # Value annotations on bars
    for bar, rate in zip(bars, rates):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{rate:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#ffffff",
            fontweight="bold"
        )
    ax_bar.set_ylim(0, max(rates) * 1.25)
    for spine in ax_bar.spines.values():
        spine.set_color("#334155")

    # Chart 2: Cumulative Changed Area Trajectory Line Chart (Columns 2 & 3)
    ax_line = plt.subplot2grid((3, 5), (2, 2), colspan=2)
    ax_line.set_facecolor("#161925")
    
    years_plot = [datetime.strptime(p["end_date"], "%Y-%m-%d").year for p in periods]
    cum_ha = trend_summary["cumulative_trajectory_hectares"]
    
    ax_line.plot(years_plot, cum_ha, color="#34d399", marker="o", linewidth=2.5, markersize=8, label="Cumulative Detected Change (ha)")
    
    # Fit linear trendline
    z = np.polyfit(years_plot, cum_ha, 1)
    p_fit = np.poly1d(z)
    ax_line.plot(years_plot, p_fit(years_plot), color="#fbbf24", linestyle="--", linewidth=1.8, label=f"Trendline: +{z[0]:.2f} ha/yr")
    
    ax_line.set_title("Cumulative Detected Change Trajectory (Hectares)\n[Not Unique Land Transformed]", fontsize=11, color="#ffffff", fontweight="bold", pad=10)
    ax_line.set_ylabel("Cumulative Detected Area (ha)", fontsize=10, color="#94a3b8")
    ax_line.set_xlabel("Observation Year", fontsize=10, color="#94a3b8")
    ax_line.tick_params(colors="#94a3b8", labelsize=9.5)
    ax_line.grid(True, linestyle="--", alpha=0.2, color="#94a3b8")
    ax_line.legend(facecolor="#1e2235", edgecolor="#334155", labelcolor="#e2e8f0", loc="upper left", fontsize=8.5)
    for spine in ax_line.spines.values():
        spine.set_color("#334155")

    # Chart 3: Executive Summary Card (Column 4)
    ax_card = plt.subplot2grid((3, 5), (2, 4))
    ax_card.set_facecolor("#161925")
    ax_card.axis("off")
    
    summary_card = (
        "TREND SUMMARY REPORT\n"
        "=====================================\n"
        f"OVERALL TREND:  [{trend_summary['overall_trend_direction'].upper()}]\n"
        f"Trend Slope:    {trend_summary['trend_slope_pct_per_year2']:+.3f} %/yr²\n"
        f"Mean Ann. Rate: {trend_summary['mean_annual_change_rate_pct']:.2f} %/yr\n"
        f"Detected Area:  {trend_summary['cumulative_detected_change_hectares']:.2f} ha*\n"
        f"Detected Ratio: {trend_summary['cumulative_detected_change_percentage']:.2f}%\n"
        "*Cumulative detected across periods\n"
        "-------------------------------------\n"
        "ROI GEOGRAPHY:\n"
        "Location:  Las Vegas / Henderson NV\n"
        "Bounds:    [-115.12, 36.00, -115.09, 36.03]\n"
        "UTM Tile:  11SPV / 11SPA\n"
        "GSD:       10.0 m/pixel (100 m²/px)\n"
        "-------------------------------------\n"
        "MODEL LIMITATION:\n"
        "Phase 6 outputs are model-detected\n"
        "estimates; not ground-truth validated."
    )
    ax_card.text(
        0.05, 0.5, summary_card,
        fontsize=9.0,
        fontfamily="monospace",
        color="#e2e8f0",
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#1a1d2e", edgecolor="#3b82f6", linewidth=1.5)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"  Successfully rendered publication-grade visualization to: {output_path}")


def main():
    print("=" * 70)
    print("TerraVision Phase 6: Multi-Temporal Historical Change Trends")
    print("=" * 70)
    
    # 1. Query STAC for genuine multi-temporal observations
    observations = query_stac_multi_temporal_scenes()
    
    # 2. Fetch genuine 256x256 crops
    images = fetch_roi_crops(observations)
    
    # 3. Load pre-trained Phase 4 Siam-UNet
    model, device = load_siam_unet_model()
    
    # 4. Pairwise change detection & period metrics
    periods, masks = run_pairwise_change_detection(images, observations, model, device)
    
    # 5. Trend analysis
    trend_summary = analyze_historical_trends(observations, periods)
    
    # 6. Export machine-readable JSON
    json_path = export_machine_readable_json(observations, periods, trend_summary)
    
    # 7. Render visualization
    render_trend_visualization(images, masks, observations, periods, trend_summary)
    
    print("\n" + "=" * 70)
    print("TerraVision Phase 6 Completed Successfully!")
    print(f"JSON Results:  {json_path}")
    print(f"Visualization: {PROJECT_ROOT / 'results' / 'historical_change_trends.png'}")
    print("=" * 70)


if __name__ == "__main__":
    main()

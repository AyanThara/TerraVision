"""
TerraVision - Phase 9: Global Scaling & Geospatial Integration Pipeline

This module orchestrates arbitrary geographic ROI ingestion, Microsoft Planetary
Computer STAC observation discovery, strict Phase 4 preprocessing, frozen Siam-UNet
inference, and unified multi-phase geospatial analytics (Phase 5 Quantification,
Phase 6 Historical Trends, Phase 7 Future Prediction, and Phase 8 Risk Intelligence).
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats

from src.geo_utils import (
    validate_coordinates,
    validate_bbox,
    point_to_bbox,
    calculate_actual_footprint,
    bbox_to_geojson_polygon,
    GLOBAL_BENCHMARK_ROIS,
    DEFAULT_PATCH_SIZE,
    DEFAULT_GSD_METERS,
)
from src.stac_client import (
    PlanetaryComputerSTACClient,
    DEFAULT_SEARCH_TOLERANCE_DAYS,
    MINIMUM_TEMPORAL_BASELINE_DAYS,
    DEFAULT_STRICT_CLOUD_THRESHOLD,
)
from src.change_detection_model import SiamUNet
from notebooks.verify_quantification import categorize_change_intensity, calculate_quantification_metrics
from notebooks.verify_historical_trends import classify_trend_direction, calculate_linear_trend_slope


class TerraVisionGeospatialPipeline:
    """
    Unified Geospatial Pipeline for arbitrary global geographic coordinates,
    connecting live Sentinel-2 Level-2A STAC observations with the frozen
    TerraVision deep learning change-detection and risk intelligence stack.
    """

    def __init__(
        self,
        model_checkpoint_path: Optional[Path] = None,
        device: Optional[torch.device] = None,
    ):
        if model_checkpoint_path is None:
            model_checkpoint_path = PROJECT_ROOT / "models" / "v3_siam_unet_change_detection.pth"
        self.model_checkpoint_path = model_checkpoint_path
        
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        self.stac_client = PlanetaryComputerSTACClient()
        
        # Exact Phase 4 audited preprocessing transform:
        # PIL RGB -> float [0.0, 1.0] -> ImageNet Normalize
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.model = self._load_frozen_model()

    def _load_frozen_model(self) -> SiamUNet:
        """Loads the pre-trained Phase 4 Siam-UNet checkpoint in frozen eval mode."""
        if not self.model_checkpoint_path.exists():
            raise FileNotFoundError(f"Trained Phase 4 checkpoint not found at: {self.model_checkpoint_path}")
            
        model = SiamUNet(in_channels=6, num_classes=1, init_features=32)
        checkpoint = torch.load(self.model_checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        
        # Ensure weights are strictly frozen
        for param in model.parameters():
            param.requires_grad = False
            
        return model

    def resolve_roi_bbox(
        self,
        roi: Union[str, List[float], Tuple[float, float], Dict[str, Any]],
        patch_size: int = DEFAULT_PATCH_SIZE,
        gsd_meters: float = DEFAULT_GSD_METERS,
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Resolves arbitrary ROI input (benchmark name, point tuple, dict, or bbox)
        into a validated WGS84 bounding box and detailed footprint metadata.
        """
        roi_name = "Custom Geographic ROI"
        
        if isinstance(roi, str):
            roi_lower = roi.lower().strip()
            if roi_lower in GLOBAL_BENCHMARK_ROIS:
                benchmark = GLOBAL_BENCHMARK_ROIS[roi_lower]
                bbox = benchmark["bbox"]
                roi_name = benchmark["name"]
            else:
                raise ValueError(f"Unknown benchmark ROI '{roi}'. Available benchmarks: {list(GLOBAL_BENCHMARK_ROIS.keys())}")
        elif isinstance(roi, (list, tuple)):
            if len(roi) == 2:
                # Point input (lat, lon) -> geodesic buffered bbox
                lat, lon = float(roi[0]), float(roi[1])
                bbox = point_to_bbox(lat, lon, patch_size, gsd_meters)
                roi_name = f"Point ({lat:.4f}°, {lon:.4f}°)"
            elif len(roi) == 4:
                # Explicit bounding box [min_lon, min_lat, max_lon, max_lat]
                bbox = validate_bbox([float(v) for v in roi])
                roi_name = f"Bounding Box [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]"
            else:
                raise ValueError(f"Sequence ROI must have length 2 (lat, lon) or 4 (bbox). Received length {len(roi)}")
        elif isinstance(roi, dict):
            if "bbox" in roi:
                bbox = validate_bbox([float(v) for v in roi["bbox"]])
                roi_name = roi.get("name", roi_name)
            elif "lat" in roi and "lon" in roi:
                lat, lon = float(roi["lat"]), float(roi["lon"])
                bbox = point_to_bbox(lat, lon, patch_size, gsd_meters)
                roi_name = roi.get("name", f"Point ({lat:.4f}°, {lon:.4f}°)")
            elif "latitude" in roi and "longitude" in roi:
                lat, lon = float(roi["latitude"]), float(roi["longitude"])
                bbox = point_to_bbox(lat, lon, patch_size, gsd_meters)
                roi_name = roi.get("name", f"Point ({lat:.4f}°, {lon:.4f}°)")
            else:
                raise ValueError(f"Dictionary ROI must contain 'bbox' or 'lat'/'lon'. Keys received: {list(roi.keys())}")
        else:
            raise TypeError(f"Unsupported ROI type {type(roi)}")
            
        footprint_meta = calculate_actual_footprint(bbox, patch_size, gsd_meters)
        footprint_meta["roi_name"] = roi_name
        footprint_meta["bounding_box_wgs84"] = bbox
        footprint_meta["geojson_geometry"] = bbox_to_geojson_polygon(bbox, {"roi_name": roi_name})
        
        return bbox, footprint_meta

    def _preprocess_pair(
        self,
        img_A: Image.Image,
        img_B: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies exact Phase 4 audited preprocessing to PIL images.
        """
        if img_A.mode != "RGB":
            img_A = img_A.convert("RGB")
        if img_B.mode != "RGB":
            img_B = img_B.convert("RGB")
            
        if img_A.size != (DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE):
            img_A = img_A.resize((DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE), Image.BILINEAR)
        if img_B.size != (DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE):
            img_B = img_B.resize((DEFAULT_PATCH_SIZE, DEFAULT_PATCH_SIZE), Image.BILINEAR)
            
        tensor_A = self.img_transform(img_A).unsqueeze(0).to(self.device)
        tensor_B = self.img_transform(img_B).unsqueeze(0).to(self.device)
        
        return tensor_A, tensor_B

    def run_inference(
        self,
        img_A: Image.Image,
        img_B: Image.Image,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes frozen Siam-UNet forward inference on bi-temporal RGB image pair.
        Returns: (probability_map [0.0, 1.0], binary_mask [0, 1]).
        """
        tensor_A, tensor_B = self._preprocess_pair(img_A, img_B)
        
        with torch.no_grad():
            output = self.model(tensor_A, tensor_B)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
            binary_mask = (prob > 0.5).astype(np.uint8)
            
        return prob, binary_mask

    def analyze_bitemporal(
        self,
        roi: Union[str, List[float], Tuple[float, float], Dict[str, Any]],
        target_date_before: str,
        target_date_after: str,
        search_tolerance_days: int = DEFAULT_SEARCH_TOLERANCE_DAYS,
        min_baseline_days: int = MINIMUM_TEMPORAL_BASELINE_DAYS,
        max_cloud_cover: float = DEFAULT_STRICT_CLOUD_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Performs end-to-end bi-temporal change detection and quantification over arbitrary ROI.
        """
        start_time = time.time()
        bbox, footprint_meta = self.resolve_roi_bbox(roi)
        
        # 1. Discover genuine observations via STAC
        obs_before, obs_after = self.stac_client.search_bitemporal_pair(
            bbox=bbox,
            target_date_before=target_date_before,
            target_date_after=target_date_after,
            tolerance_days=search_tolerance_days,
            min_baseline_days=min_baseline_days,
            max_cloud_cover=max_cloud_cover,
        )
        
        # 2. Fetch 256x256 visual crops concurrently
        crops = self.stac_client.fetch_crops_parallel([obs_before, obs_after], bbox=bbox)
        img_before, img_after = crops[0], crops[1]
        
        # 3. Execute frozen model inference
        prob_map, binary_mask = self.run_inference(img_before, img_after)
        
        # 4. Compute physical quantification metrics (Phase 5 reuse)
        quant_metrics = calculate_quantification_metrics(binary_mask, gsd_meters=DEFAULT_GSD_METERS)
        
        # Calculate actual geodesic area based on geodesic footprint calculation
        actual_area_m2 = footprint_meta["actual_geographic_footprint"]["actual_area_m2"]
        actual_area_ha = footprint_meta["actual_geographic_footprint"]["actual_area_hectares"]
        total_pixels = DEFAULT_PATCH_SIZE * DEFAULT_PATCH_SIZE
        changed_pixels = quant_metrics["changed_pixels"]
        change_pct = quant_metrics["change_percentage"]
        
        actual_changed_m2 = (changed_pixels / total_pixels) * actual_area_m2
        actual_changed_ha = actual_changed_m2 / 10000.0
        
        date_A = datetime.strptime(obs_before["acquisition_date"], "%Y-%m-%d")
        date_B = datetime.strptime(obs_after["acquisition_date"], "%Y-%m-%d")
        delta_days = (date_B - date_A).days
        delta_years = delta_days / 365.25
        annualized_pct_rate = change_pct / delta_years if delta_years > 0 else 0.0
        annualized_nominal_ha = quant_metrics["changed_area_hectares"] / delta_years if delta_years > 0 else 0.0
        annualized_actual_ha = actual_changed_ha / delta_years if delta_years > 0 else 0.0
        
        elapsed_sec = time.time() - start_time
        
        result = {
            "pipeline_metadata": {
                "phase": "Phase 9 - Global Scaling & Geospatial Integration",
                "mode": "bi_temporal_analysis",
                "sensor": "Sentinel-2 MSI Level-2A (BOA Surface Reflectance)",
                "stac_provider": "Microsoft Planetary Computer",
                "model_checkpoint": "v3_siam_unet_change_detection.pth (Frozen)",
                "inference_device": str(self.device),
                "execution_timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_runtime_seconds": round(elapsed_sec, 3),
            },
            "roi_metadata": footprint_meta,
            "observations": {
                "before": obs_before,
                "after": obs_after,
                "temporal_interval": {
                    "delta_days": delta_days,
                    "delta_years": round(delta_years, 3),
                    "engineering_constraints": {
                        "search_tolerance_days": search_tolerance_days,
                        "min_baseline_days": min_baseline_days,
                        "constraint_note": "Search tolerance (+/-30 days) and minimum baseline (>=15 days) are TerraVision project-defined engineering design constraints."
                    }
                }
            },
            "change_metrics": {
                "total_pixels": total_pixels,
                "changed_pixels": changed_pixels,
                "unchanged_pixels": total_pixels - changed_pixels,
                "change_percentage": round(change_pct, 2),
                "unchanged_percentage": round(100.0 - change_pct, 2),
                "change_intensity": quant_metrics["change_intensity"],
                "nominal_spatial_metrics": {
                    "nominal_gsd_meters_per_pixel": DEFAULT_GSD_METERS,
                    "nominal_changed_area_m2": quant_metrics["changed_area_m2"],
                    "nominal_changed_area_hectares": quant_metrics["changed_area_hectares"],
                    "annualized_rate_pct_per_year": round(annualized_pct_rate, 2),
                    "annualized_rate_ha_per_year": round(annualized_nominal_ha, 4),
                },
                "actual_geographic_metrics": {
                    "actual_changed_area_m2": round(actual_changed_m2, 2),
                    "actual_changed_area_hectares": round(actual_changed_ha, 4),
                    "annualized_rate_ha_per_year": round(annualized_actual_ha, 4),
                    "geodesic_basis": "Haversine-based geographic footprint estimate",
                }
            },
            "temporal_capability_gate": {
                "mode": "Mode A: Bi-Temporal Observation Pair (N=2 observations, 1 interval)",
                "supported_analyses": [
                    "bi_temporal_change_detection",
                    "change_quantification",
                    "pairwise_annualized_change_rate"
                ],
                "gated_unsupported_analyses": [
                    "historical_trend_regression",
                    "future_prediction",
                    "risk_intelligence"
                ],
                "gate_rationale": "Bi-temporal analysis provides exactly two observations (N=2, 1 interval). OLS trend prediction and analytical Student's t intervals require multi-temporal observations (df = N - 2 >= 1). With N=2, df=0, rendering Phase 6-8 trend, forecasting, and risk synthesis mathematically invalid. TerraVision enforces this capability gate and does not fabricate historical observations."
            },
            "scientific_disclaimers": {
                "threshold_and_noise_disclaimer": "The 0.5 probability threshold produced the reported change mask; interpretation remains subject to model limitations and requires independent validation.",
                "change_detection_limitation": "Phase 9 outputs are model-detected historical change estimates from the Siam-UNet prototype applied to satellite imagery and are not independently ground-truth validated for these STAC observations.",
                "verified_conversion_disclaimer": "Model-detected change indicates optical surface reflectance discrepancies and does NOT prove ground-truth legal land conversion, zoning status, or confirmed real-world construction.",
                "spatial_area_disclaimer": "Pixel calculations are authoritative. Physical area measurements are spatial approximations presented as both nominal 10m estimates and geodesic footprint estimates.",
                "global_accuracy_disclaimer": "TerraVision does not claim global model accuracy. Benchmark locations are integration test environments across varied biomes, not validated global accuracy sites.",
                "sensor_and_radiometric_note": "Imagery retrieved from Microsoft Planetary Computer represents 8-bit visual true-color composites (B04, B03, B02 BOA). Contrast stretch and atmospheric corrections can influence model sensitivity."
            }
        }
        
        return result, img_before, img_after, binary_mask, prob_map

    def analyze_multitemporal(
        self,
        roi: Union[str, List[float], Tuple[float, float], Dict[str, Any]],
        years: List[int] = [2020, 2021, 2022, 2023, 2024],
        season_start: str = "06-01",
        season_end: str = "08-25",
        max_cloud_cover: float = DEFAULT_STRICT_CLOUD_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multi-temporal historical change trends, future prediction,
        and risk intelligence synthesis across arbitrary global ROI.
        Reuses verified Phase 5, 6, 7, and 8 algorithms with zero formula alterations.
        """
        start_time = time.time()
        if len(years) < 4:
            raise ValueError(
                f"Mode B (Multi-Temporal Analysis) requires at least 4 years/observations (yielding at least 3 intervals, df = N - 2 >= 1). "
                f"Received {len(years)} years: {years}. "
                "For 2 observations, use Mode A (analyze_bitemporal). "
                "OLS trend regression and Student's t prediction intervals (df = N - 2) cannot be evaluated with insufficient historical points. "
                "TerraVision does not invent or fabricate intermediate historical observations."
            )
        bbox, footprint_meta = self.resolve_roi_bbox(roi)
        
        # 1. Discover multi-temporal series
        observations = self.stac_client.search_multitemporal_series(
            bbox=bbox,
            years=years,
            season_start=season_start,
            season_end=season_end,
            max_cloud_cover=max_cloud_cover,
        )
        
        # 2. Fetch crops in parallel
        images = self.stac_client.fetch_crops_parallel(observations, bbox=bbox)
        
        # 3. Pairwise consecutive change detection
        periods = []
        masks = []
        total_pixels = DEFAULT_PATCH_SIZE * DEFAULT_PATCH_SIZE
        nominal_pixel_area_m2 = DEFAULT_GSD_METERS * DEFAULT_GSD_METERS
        actual_area_ha = footprint_meta["actual_geographic_footprint"]["actual_area_hectares"]
        
        for i in range(len(images) - 1):
            obs_A = observations[i]
            obs_B = observations[i + 1]
            img_A = images[i]
            img_B = images[i + 1]
            
            _, binary_mask = self.run_inference(img_A, img_B)
            masks.append(binary_mask)
            
            date_A = datetime.strptime(obs_A["acquisition_date"], "%Y-%m-%d")
            date_B = datetime.strptime(obs_B["acquisition_date"], "%Y-%m-%d")
            delta_days = (date_B - date_A).days
            delta_years = delta_days / 365.25
            
            changed_pixels = int(np.sum(binary_mask > 0))
            change_pct = (changed_pixels / total_pixels) * 100.0
            
            nominal_changed_m2 = changed_pixels * nominal_pixel_area_m2
            nominal_changed_ha = nominal_changed_m2 / 10000.0
            actual_changed_ha = (changed_pixels / total_pixels) * actual_area_ha
            
            annualized_pct_rate = change_pct / delta_years if delta_years > 0 else 0.0
            annualized_ha_rate = nominal_changed_ha / delta_years if delta_years > 0 else 0.0
            
            periods.append({
                "period_index": i + 1,
                "period_name": f"P{i+1}: {obs_A['acquisition_date']} -> {obs_B['acquisition_date']}",
                "start_date": obs_A["acquisition_date"],
                "end_date": obs_B["acquisition_date"],
                "delta_days": delta_days,
                "delta_years": round(delta_years, 3),
                "total_pixels": total_pixels,
                "changed_pixels": changed_pixels,
                "change_percentage": round(change_pct, 2),
                "nominal_changed_area_hectares": round(nominal_changed_ha, 4),
                "actual_changed_area_hectares": round(actual_changed_ha, 4),
                "annualized_change_rate_pct_per_year": round(annualized_pct_rate, 2),
                "annualized_change_rate_ha_per_year": round(annualized_ha_rate, 4),
                "change_intensity": categorize_change_intensity(change_pct),
            })
            
        # 4. Phase 6 Historical Trends (reused logic)
        midpoint_years = []
        annualized_rates_pct = []
        annualized_rates_ha = []
        for p in periods:
            d_start = datetime.strptime(p["start_date"], "%Y-%m-%d")
            d_end = datetime.strptime(p["end_date"], "%Y-%m-%d")
            mid_date = d_start + (d_end - d_start) / 2
            mid_year = mid_date.year + (mid_date.timetuple().tm_yday / 365.25)
            midpoint_years.append(mid_year)
            annualized_rates_pct.append(p["annualized_change_rate_pct_per_year"])
            annualized_rates_ha.append(p["annualized_change_rate_ha_per_year"])
            
        slope_pct_per_year2 = calculate_linear_trend_slope(midpoint_years, annualized_rates_pct)
        slope_ha_per_year2 = calculate_linear_trend_slope(midpoint_years, annualized_rates_ha)
        trend_direction = classify_trend_direction(slope_pct_per_year2, threshold=0.5)
        
        cumulative_ha = []
        cumulative_pct = []
        c_ha, c_pct = 0.0, 0.0
        for p in periods:
            c_ha += p["nominal_changed_area_hectares"]
            c_pct += p["change_percentage"]
            cumulative_ha.append(round(c_ha, 4))
            cumulative_pct.append(round(c_pct, 2))
            
        mean_annual_rate_pct = float(np.mean(annualized_rates_pct))
        mean_annual_rate_ha = float(np.mean(annualized_rates_ha))
        
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
            "cumulative_metric_note": "Cumulative change is strictly cumulative detected change across periods, NOT unique physical land transformed.",
        }
        
        # 5. Phase 7 Future Prediction (reused OLS + Student's t bounds, df = N - 2)
        N = len(periods)
        df = max(1, N - 2)
        x = np.array(midpoint_years, dtype=np.float64)
        y = np.array(annualized_rates_pct, dtype=np.float64)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        SS_xx = np.sum((x - x_mean) ** 2)
        
        beta1 = slope_pct_per_year2
        beta0 = float(y_mean - beta1 * x_mean)
        y_hat = beta0 + beta1 * x
        residuals = y - y_hat
        residual_se = float(np.sqrt(np.sum(residuals ** 2) / df)) if df > 0 and SS_xx > 0 else 0.0
        
        t_90 = float(stats.t.ppf(0.95, df=df))
        t_95 = float(stats.t.ppf(0.975, df=df))
        
        forecast_horizons = [years[-1] + 1, years[-1] + 2, years[-1] + 3]
        forecasts = []
        nom_total_ha = footprint_meta["nominal_specifications"]["nominal_area_hectares"]
        
        for h_year in forecast_horizons:
            x_h = float(h_year) + 0.5
            y_point_raw = beta0 + beta1 * x_h
            y_point_bounded = max(0.0, y_point_raw)  # Physical non-negativity constraint
            
            se_pred = float(residual_se * np.sqrt(1.0 + (1.0 / N) + ((x_h - x_mean) ** 2 / SS_xx))) if SS_xx > 0 else 0.0
            
            ci_90_lower = max(0.0, y_point_raw - t_90 * se_pred)
            ci_90_upper = max(0.0, y_point_raw + t_90 * se_pred)
            ci_95_lower = max(0.0, y_point_raw - t_95 * se_pred)
            ci_95_upper = max(0.0, y_point_raw + t_95 * se_pred)
            
            ci_95_upper_ha = (ci_95_upper / 100.0) * nom_total_ha
            
            forecasts.append({
                "horizon_year": h_year,
                "predicted_annualized_rate_pct": round(y_point_bounded, 2),
                "prediction_interval_95_pct": {
                    "lower_bound_pct": round(ci_95_lower, 2),
                    "upper_bound_pct": round(ci_95_upper, 2),
                    "upper_bound_ha_per_year": round(ci_95_upper_ha, 4),
                },
                "projected_cumulative_change_hectares": cumulative_ha[-1],
            })
            
        # 6. Phase 8 Risk Intelligence (reused exact formulas)
        cum_pct = cumulative_pct[-1]
        latest_rate_pct = periods[-1]["annualized_change_rate_pct_per_year"]
        fc_2025_point = forecasts[0]["predicted_annualized_rate_pct"]
        fc_2025_95_upper = forecasts[0]["prediction_interval_95_pct"]["upper_bound_pct"]
        
        # Sub-score 1: Change Pressure (0.50 * P_cum + 0.50 * P_recent)
        P_cum = min(100.0, (cum_pct / 10.0) * 100.0)
        P_recent = min(100.0, (latest_rate_pct / 3.0) * 100.0)
        S_pressure = round(0.50 * P_cum + 0.50 * P_recent, 2)
        
        # Sub-score 2: Trend Momentum (50 + (beta1 / 1.5) * 50)
        raw_trend = 50.0 + (beta1 / 1.5) * 50.0
        S_trend = round(float(np.clip(raw_trend, 0.0, 100.0)), 2)
        
        # Sub-score 3: Future Outlook (0.40 * F_point + 0.60 * F_precaution)
        F_point = min(100.0, (fc_2025_point / 3.0) * 100.0)
        F_precaution = min(100.0, (fc_2025_95_upper / 10.0) * 100.0)
        S_forecast = round(0.40 * F_point + 0.60 * F_precaution, 2)
        
        # Composite Risk: R = 0.40 * S_pressure + 0.30 * S_trend + 0.30 * S_forecast
        raw_R = 0.40 * S_pressure + 0.30 * S_trend + 0.30 * S_forecast
        overall_risk = round(float(np.clip(raw_R, 0.0, 100.0)), 2)
        
        if overall_risk < 30.0:
            risk_level = "LOW"
        elif overall_risk < 60.0:
            risk_level = "MODERATE"
        elif overall_risk < 80.0:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
            
        # Evidence Confidence Score
        score_df = 30.0 if df <= 2 else (50.0 if df <= 4 else 70.0)
        dispersion_pct = min(100.0, (fc_2025_95_upper - fc_2025_point) / 10.0 * 100.0)
        score_dispersion = max(0.0, 100.0 - dispersion_pct)
        score_validation = 0.0  # Zero since STAC observations are uncalibrated prototype inferences
        raw_ECS = 0.40 * score_df + 0.40 * score_dispersion + 0.20 * score_validation
        ecs = round(float(np.clip(raw_ECS, 0.0, 100.0)), 2)
        
        if ecs < 40.0:
            confidence_level = "LOW CONFIDENCE"
        elif ecs < 70.0:
            confidence_level = "MODERATE CONFIDENCE"
        else:
            confidence_level = "HIGH CONFIDENCE"
            
        elapsed_sec = time.time() - start_time
        
        return {
            "pipeline_metadata": {
                "phase": "Phase 9 - Global Scaling & Geospatial Integration",
                "mode": "multi_temporal_historical_prediction_risk",
                "sensor": "Sentinel-2 MSI Level-2A (BOA Surface Reflectance)",
                "stac_provider": "Microsoft Planetary Computer",
                "model_checkpoint": "v3_siam_unet_change_detection.pth (Frozen)",
                "inference_device": str(self.device),
                "execution_timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_runtime_seconds": round(elapsed_sec, 3),
            },
            "roi_metadata": footprint_meta,
            "temporal_observations": observations,
            "consecutive_period_changes": periods,
            "historical_trends": trend_summary,
            "future_prediction": {
                "degrees_of_freedom": df,
                "residual_standard_error": round(residual_se, 4),
                "beta1_trend_slope": round(beta1, 4),
                "forecasts": forecasts,
            },
            "risk_intelligence": {
                "overall_analytical_risk_indicator": overall_risk,
                "risk_level": risk_level,
                "evidence_confidence_score": ecs,
                "evidence_confidence_level": confidence_level,
                "sub_scores": {
                    "change_pressure_score": S_pressure,
                    "trend_momentum_score": S_trend,
                    "future_outlook_score": S_forecast,
                }
            },
            "temporal_capability_gate": {
                "mode": "Mode B: Multi-Temporal Sequence (N >= 4 observations)",
                "supported_analyses": [
                    "multi_temporal_pairwise_change",
                    "historical_trend_regression",
                    "future_prediction",
                    "risk_intelligence"
                ],
                "observations_count": len(observations),
                "periods_count": len(periods),
                "degrees_of_freedom": df,
                "gate_rationale": "Sufficient multi-temporal sequence verified (N >= 4, df >= 1). Enables defensible OLS trend slope and Student's t prediction interval estimation."
            },
            "scientific_disclaimers": {
                "change_detection_limitation": "Phase 9 outputs are model-detected historical change estimates from the Siam-UNet prototype applied to satellite imagery and are not independently ground-truth validated for these STAC observations.",
                "verified_conversion_disclaimer": "Model-detected change indicates optical surface reflectance discrepancies and does NOT prove ground-truth legal land conversion, zoning status, or confirmed real-world construction.",
                "prediction_disclaimer": "Forecasts are first-order linear extrapolations of model-detected change, not deterministic guarantees. Epistemic uncertainty is wide due to small sample size (df=2).",
                "risk_disclaimer": "Analytical Risk Indicator is a project-defined heuristic scoring model and does not prove ground-truth municipal zoning or construction activity.",
                "global_accuracy_disclaimer": "TerraVision does not claim global model accuracy. Benchmark locations are integration test environments across varied biomes, not validated global accuracy sites.",
            }
        }, images, masks


def render_bitemporal_summary_visualization(
    img_before: Image.Image,
    img_after: Image.Image,
    binary_mask: np.ndarray,
    prob_map: np.ndarray,
    result_data: Dict[str, Any],
    output_path: Path,
):
    """
    Renders an aesthetic, publication-grade 5-panel visualization for bi-temporal change analysis:
    [Image A (Before) | Image B (After) | Change Mask | Change Overlay | Analytical Summary Box]
    """
    fig, axes = plt.subplots(1, 5, figsize=(24, 5), facecolor="#0e1117")
    fig.suptitle(
        f"TerraVision Phase 9: Global Geospatial Change Detection Analysis\n"
        f"ROI: {result_data['roi_metadata']['roi_name']} (Sentinel-2 Level-2A @ Nominal 10m GSD)",
        fontsize=13,
        fontweight="bold",
        color="white",
        y=1.03,
    )
    
    # 1. Before Image
    axes[0].imshow(img_before)
    obs_A = result_data["observations"]["before"]
    axes[0].set_title(f"Image A (Before)\n{obs_A['acquisition_date']} | Cloud: {obs_A['cloud_cover_percentage']}%", color="white", fontsize=10)
    axes[0].axis("off")
    
    # 2. After Image
    axes[1].imshow(img_after)
    obs_B = result_data["observations"]["after"]
    axes[1].set_title(f"Image B (After)\n{obs_B['acquisition_date']} | Cloud: {obs_B['cloud_cover_percentage']}%", color="white", fontsize=10)
    axes[1].axis("off")
    
    # 3. Binary Change Mask
    axes[2].imshow(binary_mask, cmap="gray")
    cm = result_data["change_metrics"]
    axes[2].set_title(f"Siam-UNet Change Mask\nChanged Pixels: {cm['changed_pixels']:,} / {cm['total_pixels']:,}", color="white", fontsize=10)
    axes[2].axis("off")
    
    # 4. Change Overlay (Red on After Image)
    after_np = np.array(img_after).copy()
    overlay = after_np.copy()
    overlay[binary_mask > 0] = [255, 45, 85]  # Vibrant Coral-Red highlight
    blended = (0.55 * after_np + 0.45 * overlay).astype(np.uint8)
    axes[3].imshow(blended)
    axes[3].set_title(f"Detected Change Overlay\nChange Intensity: [{cm['change_intensity'].upper()}]", color="white", fontsize=10)
    axes[3].axis("off")
    
    # 5. Dedicated Analytical Summary Box
    axes[4].set_facecolor("#161b22")
    axes[4].axis("off")
    
    fp_nom = result_data["roi_metadata"]["nominal_specifications"]
    fp_act = result_data["roi_metadata"]["actual_geographic_footprint"]
    sm_nom = cm["nominal_spatial_metrics"]
    sm_act = cm["actual_geographic_metrics"]
    interval = result_data["observations"]["temporal_interval"]
    
    summary_text = (
        f"GEOSPATIAL ROI METADATA\n"
        f"----------------------------------------\n"
        f"ROI: {result_data['roi_metadata']['roi_name'][:28]}\n"
        f"Patch Size:      256 x 256 pixels\n"
        f"Nominal GSD:     10.0 m/pixel (~2.56 km)\n"
        f"Actual Width:    {fp_act['actual_width_meters']:,.1f} m\n"
        f"Actual Height:   {fp_act['actual_height_meters']:,.1f} m\n"
        f"Actual Area:     {fp_act['actual_area_hectares']:.2f} ha\n"
        f"Effective GSD:   {fp_act['effective_gsd_meters_per_pixel'][0]:.2f}m x {fp_act['effective_gsd_meters_per_pixel'][1]:.2f}m\n\n"
        f"TEMPORAL BASELINE\n"
        f"----------------------------------------\n"
        f"Interval:        {interval['delta_days']} days ({interval['delta_years']:.2f} yrs)\n"
        f"Engineering Tol: +/-{interval['engineering_constraints']['search_tolerance_days']}d (min {interval['engineering_constraints']['min_baseline_days']}d)\n\n"
        f"CHANGE QUANTIFICATION\n"
        f"----------------------------------------\n"
        f"Detected Change: {cm['change_percentage']:.2f}% ({cm['changed_pixels']:,} px)\n"
        f"Intensity Class: {cm['change_intensity']}\n"
        f"Nominal Area:    {sm_nom['nominal_changed_area_hectares']:.2f} ha\n"
        f"Actual Area:     {sm_act['actual_changed_area_hectares']:.2f} ha\n"
        f"Annualized Rate: {sm_nom['annualized_rate_pct_per_year']:.2f} %/yr ({sm_nom['annualized_rate_ha_per_year']:.2f} ha/yr)\n\n"
        f"DISCLAIMER: Model-detected change;\nnot ground-truth legal conversion."
    )
    
    axes[4].text(
        0.05, 0.95, summary_text,
        transform=axes[4].transAxes,
        fontsize=9,
        color="#e6edf3",
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1f242c", edgecolor="#30363d", alpha=0.95)
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()


def main():
    print("=" * 80)
    print("TerraVision Phase 9: Global Scaling & Geospatial Integration Pipeline")
    print("=" * 80)
    
    pipeline = TerraVisionGeospatialPipeline()
    print(f"Initialized TerraVision Geospatial Pipeline on device: {pipeline.device}")
    
    # Run Demonstration on Global Geographic Benchmark ROI: Dubai South Infrastructure Corridor
    benchmark_key = "dubai"
    print(f"\nExecuting Demonstration on Benchmark ROI: '{benchmark_key}'...")
    
    result_data, img_before, img_after, binary_mask, prob_map = pipeline.analyze_bitemporal(
        roi=benchmark_key,
        target_date_before="2021-07-15",
        target_date_after="2024-07-15",
        search_tolerance_days=30,
        min_baseline_days=15,
        max_cloud_cover=5.0,
    )
    
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    
    json_path = results_dir / "global_geospatial_pipeline.json"
    png_path = results_dir / "global_geospatial_pipeline.png"
    
    # Save machine-readable JSON
    with open(json_path, "w") as f:
        json.dump(result_data, f, indent=4)
        
    # Render visual artifact
    render_bitemporal_summary_visualization(
        img_before=img_before,
        img_after=img_after,
        binary_mask=binary_mask,
        prob_map=prob_map,
        result_data=result_data,
        output_path=png_path,
    )
    
    print("\n" + "=" * 80)
    print("Phase 9 Demonstration Completed Successfully!")
    print(f"JSON Report Saved:       {json_path}")
    print(f"Visualization Saved:     {png_path}")
    print(f"Detected Change:         {result_data['change_metrics']['change_percentage']:.2f}%")
    print(f"Nominal Changed Area:    {result_data['change_metrics']['nominal_spatial_metrics']['nominal_changed_area_hectares']:.2f} ha")
    print(f"Actual Changed Area:     {result_data['change_metrics']['actual_geographic_metrics']['actual_changed_area_hectares']:.2f} ha")
    print(f"Change Intensity:        {result_data['change_metrics']['change_intensity']}")
    print("=" * 80)


if __name__ == "__main__":
    main()

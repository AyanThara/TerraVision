"""
TerraVision - Phase 9: Microsoft Planetary Computer STAC Client

This module provides discovery, progressive cloud filtering, metadata extraction,
and windowed RGB visual crop retrieval (B04/B03/B02 BOA surface reflectance)
from Sentinel-2 MSI Level-2A collections on Microsoft Planetary Computer.
"""

import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import requests
from PIL import Image

from src.geo_utils import validate_bbox, bbox_to_geojson_polygon, DEFAULT_PATCH_SIZE


STAC_SEARCH_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
STAC_CROP_URL = "https://planetarycomputer.microsoft.com/api/data/v1/item/crop.png"

# TerraVision Project-Defined Engineering Constraints
# NOTE: These values are project-defined engineering heuristics to balance cloud-free
# scene availability and prevent short-term illumination noise. They are not universal scientific thresholds.
DEFAULT_SEARCH_TOLERANCE_DAYS = 30
MINIMUM_TEMPORAL_BASELINE_DAYS = 15
DEFAULT_STRICT_CLOUD_THRESHOLD = 5.0
MAX_RELAXED_CLOUD_THRESHOLD = 15.0

DEFAULT_TIMEOUT = 25
MAX_RETRIES = 3


class STACError(Exception):
    """Base exception for STAC operations."""
    pass


class NoScenesFoundError(STACError):
    """Raised when no suitable satellite observations match the search criteria."""
    pass


class STACServiceUnavailableError(STACError):
    """Raised when the STAC API endpoint is unreachable or returns persistent errors."""
    pass


class IdenticalObservationsError(STACError):
    """Raised when bi-temporal search resolves to the same observation."""
    pass


class PlanetaryComputerSTACClient:
    """
    Client for querying Microsoft Planetary Computer STAC API and downloading
    calibrated Sentinel-2 Level-2A visual surface reflectance crops.
    """

    def __init__(
        self,
        search_url: str = STAC_SEARCH_URL,
        crop_url: str = STAC_CROP_URL,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.search_url = search_url
        self.crop_url = crop_url
        self.timeout = timeout
        self.session = requests.Session()

    def _execute_search_query(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Executes a STAC search request with exponential backoff retry."""
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.post(self.search_url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                return data.get("features", [])
            except (requests.RequestException, requests.HTTPError) as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    time.sleep(1.5 * attempt)
        raise STACServiceUnavailableError(f"Planetary Computer STAC search failed after {MAX_RETRIES} attempts: {last_error}")

    def search_best_scene(
        self,
        bbox: List[float],
        start_datetime: str,
        end_datetime: str,
        max_cloud_cover: float = DEFAULT_STRICT_CLOUD_THRESHOLD,
        allow_relaxation: bool = True,
    ) -> Dict[str, Any]:
        """
        Searches for the lowest cloud-cover Sentinel-2 L2A scene in the given temporal window.
        If strict cloud threshold returns no results and allow_relaxation is True,
        adaptively expands cloud threshold up to MAX_RELAXED_CLOUD_THRESHOLD (15%).
        """
        bbox = validate_bbox(bbox)
        
        # 1. Primary search with strict cloud threshold
        payload = {
            "collections": ["sentinel-2-l2a"],
            "bbox": bbox,
            "datetime": f"{start_datetime}/{end_datetime}",
            "query": {"eo:cloud_cover": {"lt": max_cloud_cover}},
            "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
            "limit": 5
        }
        
        features = self._execute_search_query(payload)
        was_relaxed = False
        applied_cloud_threshold = max_cloud_cover
        
        # 2. Adaptive cloud relaxation if 0 scenes found
        if not features and allow_relaxation and max_cloud_cover < MAX_RELAXED_CLOUD_THRESHOLD:
            was_relaxed = True
            applied_cloud_threshold = MAX_RELAXED_CLOUD_THRESHOLD
            payload["query"] = {"eo:cloud_cover": {"lt": MAX_RELAXED_CLOUD_THRESHOLD}}
            features = self._execute_search_query(payload)
            
        if not features:
            raise NoScenesFoundError(
                f"No Sentinel-2 Level-2A observations found for bbox {bbox} between {start_datetime} "
                f"and {end_datetime} with cloud cover < {applied_cloud_threshold}%. "
                "Consider widening the temporal search window or selecting a drier seasonal window."
            )
            
        # Select best scene (lowest cloud cover)
        best_item = features[0]
        props = best_item["properties"]
        dt_str = props["datetime"]
        scene_id = best_item["id"]
        cloud_pct = float(props.get("eo:cloud_cover", 0.0))
        
        # Solar elevation extraction with fallback
        sun_elev = props.get("view:sun_elevation")
        if sun_elev is None:
            solar_zenith = props.get("s2:mean_solar_zenith")
            sun_elev = (90.0 - float(solar_zenith)) if solar_zenith is not None else 0.0
        else:
            sun_elev = float(sun_elev)
            
        scene_meta = {
            "scene_id": scene_id,
            "acquisition_datetime": dt_str,
            "acquisition_date": dt_str[:10],
            "cloud_cover_percentage": round(cloud_pct, 2),
            "platform": props.get("platform", "sentinel-2"),
            "constellation": props.get("constellation", "Sentinel-2"),
            "sun_elevation_degrees": round(sun_elev, 2),
            "cloud_relaxed_applied": was_relaxed,
            "applied_cloud_threshold": applied_cloud_threshold,
            "stac_item_href": best_item.get("links", [{}])[0].get("href", ""),
        }
        
        if was_relaxed:
            scene_meta["warning"] = (
                f"Strict cloud threshold ({max_cloud_cover}%) exceeded. Adaptive relaxation applied "
                f"up to {applied_cloud_threshold}%. Scene cloud cover: {cloud_pct:.2f}%."
            )
            
        return scene_meta

    def search_bitemporal_pair(
        self,
        bbox: List[float],
        target_date_before: str,
        target_date_after: str,
        tolerance_days: int = DEFAULT_SEARCH_TOLERANCE_DAYS,
        min_baseline_days: int = MINIMUM_TEMPORAL_BASELINE_DAYS,
        max_cloud_cover: float = DEFAULT_STRICT_CLOUD_THRESHOLD,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Discovers a verified bi-temporal observation pair for the exact same geography.
        Enforces distinct acquisition dates and a minimum temporal baseline.
        """
        d_before = datetime.strptime(target_date_before, "%Y-%m-%d")
        d_after = datetime.strptime(target_date_after, "%Y-%m-%d")
        
        if d_before >= d_after:
            raise ValueError(f"target_date_before ({target_date_before}) must be strictly earlier than target_date_after ({target_date_after})")
            
        start_before = (d_before - timedelta(days=tolerance_days)).strftime("%Y-%m-%dT00:00:00Z")
        end_before = (d_before + timedelta(days=tolerance_days)).strftime("%Y-%m-%dT23:59:59Z")
        
        start_after = (d_after - timedelta(days=tolerance_days)).strftime("%Y-%m-%dT00:00:00Z")
        end_after = (d_after + timedelta(days=tolerance_days)).strftime("%Y-%m-%dT23:59:59Z")
        
        obs_before = self.search_best_scene(bbox, start_before, end_before, max_cloud_cover)
        obs_after = self.search_best_scene(bbox, start_after, end_after, max_cloud_cover)
        
        # Enforce distinct observations
        if obs_before["scene_id"] == obs_after["scene_id"] or obs_before["acquisition_date"] == obs_after["acquisition_date"]:
            raise IdenticalObservationsError(
                f"Both target dates resolved to the identical satellite scene ({obs_before['scene_id']}). "
                "Ensure a greater temporal separation between target dates."
            )
            
        # Check temporal baseline constraint
        act_before = datetime.strptime(obs_before["acquisition_date"], "%Y-%m-%d")
        act_after = datetime.strptime(obs_after["acquisition_date"], "%Y-%m-%d")
        actual_delta_days = (act_after - act_before).days
        
        if actual_delta_days < min_baseline_days:
            raise ValueError(
                f"Actual temporal separation between observations ({actual_delta_days} days: "
                f"{obs_before['acquisition_date']} to {obs_after['acquisition_date']}) is less than the "
                f"TerraVision engineering minimum baseline ({min_baseline_days} days). "
                "Short temporal baselines are susceptible to transient illumination and sun-angle discrepancies."
            )
            
        obs_before["temporal_role"] = "Image A (Before)"
        obs_after["temporal_role"] = "Image B (After)"
        obs_before["engineering_constraints"] = {
            "search_tolerance_days": tolerance_days,
            "min_baseline_days": min_baseline_days,
            "actual_delta_days": actual_delta_days,
        }
        obs_after["engineering_constraints"] = {
            "search_tolerance_days": tolerance_days,
            "min_baseline_days": min_baseline_days,
            "actual_delta_days": actual_delta_days,
        }
        
        # Detect seasonal discrepancy warning
        month_diff = abs(act_after.month - act_before.month)
        if month_diff > 3 and month_diff < 9:
            seasonal_warn = (
                f"Observations are captured in different seasonal quarters (Month {act_before.month} vs Month {act_after.month}). "
                "Detected change may include natural phenological vegetation variation rather than physical land conversion."
            )
            obs_before["seasonal_warning"] = seasonal_warn
            obs_after["seasonal_warning"] = seasonal_warn
            
        return obs_before, obs_after

    def search_multitemporal_series(
        self,
        bbox: List[float],
        years: List[int],
        season_start: str = "06-01",
        season_end: str = "08-25",
        max_cloud_cover: float = DEFAULT_STRICT_CLOUD_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """
        Discovers a multi-temporal sequence of observations across years for historical trend analysis.
        Uses a consistent seasonal window to minimize phenological false positive detections.
        """
        observations = []
        for idx, yr in enumerate(years):
            start_dt = f"{yr}-{season_start}T00:00:00Z"
            end_dt = f"{yr}-{season_end}T23:59:59Z"
            obs = self.search_best_scene(bbox, start_dt, end_dt, max_cloud_cover)
            obs["temporal_index"] = idx
            obs["target_year"] = yr
            observations.append(obs)
            
        return observations

    def fetch_crop(
        self,
        scene_id: str,
        bbox: List[float],
        patch_size: int = DEFAULT_PATCH_SIZE,
    ) -> Image.Image:
        """
        Retrieves a windowed RGB surface reflectance crop (B04, B03, B02 BOA)
        from Microsoft Planetary Computer crop endpoint at 10m nominal GSD.
        Enforces exact 256x256 dimensions and RGB mode.
        """
        feature_geom = bbox_to_geojson_polygon(bbox)
        params = {
            "collection": "sentinel-2-l2a",
            "item": scene_id,
            "assets": "visual",
            "width": patch_size,
            "height": patch_size,
        }
        
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.post(
                    self.crop_url,
                    params=params,
                    json=feature_geom,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content))
                if img.mode != "RGB":
                    img = img.convert("RGB")
                if img.size != (patch_size, patch_size):
                    img = img.resize((patch_size, patch_size), Image.BILINEAR)
                return img
            except (requests.RequestException, requests.HTTPError, IOError) as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    time.sleep(1.5 * attempt)
                    
        raise STACServiceUnavailableError(f"Failed to fetch visual crop for scene {scene_id} after {MAX_RETRIES} attempts: {last_error}")

    def fetch_crops_parallel(
        self,
        observations: List[Dict[str, Any]],
        bbox: List[float],
        patch_size: int = DEFAULT_PATCH_SIZE,
        max_workers: int = 4,
    ) -> List[Image.Image]:
        """
        Fetches multiple crops concurrently via ThreadPoolExecutor while preserving chronological order.
        """
        results: Dict[int, Image.Image] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.fetch_crop, obs["scene_id"], bbox, patch_size): i
                for i, obs in enumerate(observations)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                
        return [results[i] for i in range(len(observations))]

"""
TerraVision - Phase 9: Geospatial Utilities & ROI Handler

This module provides coordinate validation, bounding box normalization,
geodesic footprint scaling, and GeoJSON polygon generation for arbitrary
geographic Regions of Interest (ROIs). It enforces Sentinel-2 terrestrial
coverage constraints, provides actual geodesic footprint calculations,
and maintains strict 256x256 pixel patch dimensions at nominal 10 m GSD.
"""

import math
from typing import List, Dict, Any, Tuple, Union


# Geographic and Sensor Constants
WGS84_EARTH_RADIUS_METERS = 6371000.0  # Mean volumetric radius for Haversine
METERS_PER_DEGREE_LAT = 111320.0       # Equatorial approximation for grid expansion

# Sentinel-2 systematic acquisition limits (terrestrial land surfaces and coastal waters)
SENTINEL2_MIN_LAT = -56.0
SENTINEL2_MAX_LAT = 84.0
SENTINEL2_MIN_LON = -180.0
SENTINEL2_MAX_LON = 180.0

DEFAULT_PATCH_SIZE = 256
DEFAULT_GSD_METERS = 10.0


class InvalidCoordinatesError(ValueError):
    """Raised when geographic coordinates fall outside standard WGS84 boundaries."""
    pass


class TerritorialCoverageError(ValueError):
    """Raised when coordinates fall outside systematic Sentinel-2 MSI terrestrial orbit coverage."""
    pass


def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes the great-circle distance between two points on the WGS84 sphere using Haversine formula.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return WGS84_EARTH_RADIUS_METERS * c


def validate_coordinates(lat: float, lon: float) -> Tuple[float, float]:
    """
    Validates WGS84 geographic coordinates and Sentinel-2 systematic orbit coverage.
    
    Args:
        lat: Latitude in decimal degrees [-90.0, 90.0]
        lon: Longitude in decimal degrees [-180.0, 180.0]
        
    Returns:
        Tuple of validated (latitude, longitude)
    """
    if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
        raise InvalidCoordinatesError(f"Coordinates must be numeric. Received lat={lat}, lon={lon}")
        
    if not (-90.0 <= lat <= 90.0):
        raise InvalidCoordinatesError(f"Latitude must be within [-90.0, 90.0]. Received lat={lat}")
        
    if not (-180.0 <= lon <= 180.0):
        raise InvalidCoordinatesError(f"Longitude must be within [-180.0, 180.0]. Received lon={lon}")
        
    # Check Sentinel-2 systematic acquisition envelope
    if not (SENTINEL2_MIN_LAT <= lat <= SENTINEL2_MAX_LAT):
        raise TerritorialCoverageError(
            f"Latitude {lat:.4f}° is outside systematic Sentinel-2 MSI terrestrial acquisition "
            f"envelope [{SENTINEL2_MIN_LAT}°, +{SENTINEL2_MAX_LAT}°]. Sentinel-2 systematic Level-2A "
            "coverage does not include polar ice sheets or Antarctica interior."
        )
        
    return float(lat), float(lon)


def validate_bbox(bbox: List[float]) -> List[float]:
    """
    Validates a WGS84 bounding box [min_lon, min_lat, max_lon, max_lat].
    
    Args:
        bbox: List of 4 floats [min_lon, min_lat, max_lon, max_lat]
        
    Returns:
        Validated bounding box list
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise InvalidCoordinatesError(f"Bounding box must be a 4-element sequence [min_lon, min_lat, max_lon, max_lat]. Received {bbox}")
        
    min_lon, min_lat, max_lon, max_lat = [float(v) for v in bbox]
    
    if min_lon >= max_lon:
        raise InvalidCoordinatesError(f"Invalid longitude order: min_lon ({min_lon}) must be strictly less than max_lon ({max_lon})")
        
    if min_lat >= max_lat:
        raise InvalidCoordinatesError(f"Invalid latitude order: min_lat ({min_lat}) must be strictly less than max_lat ({max_lat})")
        
    validate_coordinates(min_lat, min_lon)
    validate_coordinates(max_lat, max_lon)
    
    return [round(min_lon, 6), round(min_lat, 6), round(max_lon, 6), round(max_lat, 6)]


def point_to_bbox(
    lat: float,
    lon: float,
    patch_size: int = DEFAULT_PATCH_SIZE,
    gsd_meters: float = DEFAULT_GSD_METERS,
) -> List[float]:
    """
    Calculates a square geographic bounding box centered at (lat, lon) matching
    the nominal ground footprint for the requested patch size at nominal GSD.
    Applies geodesic cosine scaling to adjust longitude delta for latitude convergence.
    
    Nominal footprint = patch_size * gsd_meters (default: 256 * 10m = 2,560 meters).
    
    Args:
        lat: Center latitude in decimal degrees
        lon: Center longitude in decimal degrees
        patch_size: Square patch dimension in pixels (default 256)
        gsd_meters: Nominal ground sample distance in meters per pixel (default 10.0m)
        
    Returns:
        Bounding box [min_lon, min_lat, max_lon, max_lat] formatted in WGS84
    """
    lat, lon = validate_coordinates(lat, lon)
    
    total_extent_meters = patch_size * gsd_meters
    half_extent_meters = total_extent_meters / 2.0
    
    # Delta latitude
    delta_lat = half_extent_meters / METERS_PER_DEGREE_LAT
    
    # Delta longitude (adjusted for latitude convergence)
    lat_rad = math.radians(lat)
    cos_lat = max(math.cos(lat_rad), 0.01)  # Safeguard against extreme latitudes
    meters_per_degree_lon = METERS_PER_DEGREE_LAT * cos_lat
    delta_lon = half_extent_meters / meters_per_degree_lon
    
    min_lon = lon - delta_lon
    max_lon = lon + delta_lon
    min_lat = lat - delta_lat
    max_lat = lat + delta_lat
    
    return validate_bbox([min_lon, min_lat, max_lon, max_lat])


def calculate_actual_footprint(
    bbox: List[float],
    patch_size: int = DEFAULT_PATCH_SIZE,
    nominal_gsd: float = DEFAULT_GSD_METERS,
) -> Dict[str, Any]:
    """
    Calculates both the nominal spatial metadata and the actual geodesic ground footprint
    (width, height, and surface area) for a given WGS84 bounding box.
    """
    bbox = validate_bbox(bbox)
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    
    # Geodesic width (distance along center latitude between min_lon and max_lon)
    actual_width_meters = haversine_distance_meters(center_lat, min_lon, center_lat, max_lon)
    
    # Geodesic height (distance along center longitude between min_lat and max_lat)
    actual_height_meters = haversine_distance_meters(min_lat, center_lon, max_lat, center_lon)
    
    actual_area_m2 = actual_width_meters * actual_height_meters
    actual_area_ha = actual_area_m2 / 10000.0
    actual_area_km2 = actual_area_m2 / 1000000.0
    
    nominal_extent_meters = patch_size * nominal_gsd
    nominal_area_m2 = nominal_extent_meters * nominal_extent_meters
    nominal_area_ha = nominal_area_m2 / 10000.0
    nominal_area_km2 = nominal_area_m2 / 1000000.0
    
    effective_gsd_x = actual_width_meters / patch_size
    effective_gsd_y = actual_height_meters / patch_size
    
    return {
        "nominal_specifications": {
            "patch_dimensions_pixels": [patch_size, patch_size],
            "nominal_gsd_meters": nominal_gsd,
            "nominal_extent_km": [round(nominal_extent_meters / 1000.0, 3), round(nominal_extent_meters / 1000.0, 3)],
            "nominal_area_m2": round(nominal_area_m2, 2),
            "nominal_area_hectares": round(nominal_area_ha, 4),
            "nominal_area_km2": round(nominal_area_km2, 4),
        },
        "actual_geographic_footprint": {
            "actual_width_meters": round(actual_width_meters, 2),
            "actual_height_meters": round(actual_height_meters, 2),
            "actual_area_m2": round(actual_area_m2, 2),
            "actual_area_hectares": round(actual_area_ha, 4),
            "actual_area_km2": round(actual_area_km2, 4),
            "effective_gsd_meters_per_pixel": [round(effective_gsd_x, 3), round(effective_gsd_y, 3)],
            "calculation_method": "Haversine-based geographic footprint estimate",
        }
    }


def bbox_to_geojson_polygon(bbox: List[float], properties: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Converts a WGS84 bounding box into a standard GeoJSON Feature with Polygon geometry.
    
    Args:
        bbox: [min_lon, min_lat, max_lon, max_lat]
        properties: Optional properties dictionary to embed
        
    Returns:
        GeoJSON Feature dictionary suitable for STAC search and Leaflet/Mapbox rendering.
    """
    bbox = validate_bbox(bbox)
    min_lon, min_lat, max_lon, max_lat = bbox
    
    coordinates = [[
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat],
    ]]
    
    feature_props = {
        "bbox_wgs84": bbox,
        "center_lat": round((min_lat + max_lat) / 2.0, 6),
        "center_lon": round((min_lon + max_lon) / 2.0, 6),
    }
    if properties:
        feature_props.update(properties)
        
    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": coordinates,
        },
        "properties": feature_props
    }


# Global Geographic Benchmark / Integration Test Locations
# NOTE: These are integration test and demonstration benchmarks to evaluate pipeline functionality
# across varied global biomes, NOT independently ground-truth validated ML benchmark test sets.
GLOBAL_BENCHMARK_ROIS: Dict[str, Dict[str, Any]] = {
    "las_vegas": {
        "name": "Las Vegas / Henderson Urban Expansion Zone",
        "country": "United States",
        "biome": "Arid / Desert Urban Expansion",
        "bbox": [-115.120, 36.000, -115.090, 36.030],
        "center": {"lat": 36.015, "lon": -115.105},
        "role": "Global geographic benchmark / integration test location (Phase 6 Baseline)",
        "notes": "Desert terrain with rapid residential expansion; established multi-temporal baseline."
    },
    "dubai": {
        "name": "Dubai South Infrastructure Corridor",
        "country": "United Arab Emirates",
        "biome": "Hyper-Arid / Megacity Infrastructure",
        "bbox": [55.150, 24.960, 55.175, 24.985],
        "center": {"lat": 24.9725, "lon": 55.1625},
        "role": "Global geographic benchmark / integration test location",
        "notes": "Large-scale grading and logistics infrastructure development in desert terrain."
    },
    "rondonia": {
        "name": "Rondônia Deforestation Frontier",
        "country": "Brazil",
        "biome": "Tropical Rainforest / Agricultural Edge",
        "bbox": [-62.900, -10.200, -62.875, -10.175],
        "center": {"lat": -10.1875, "lon": -62.8875},
        "role": "Global geographic benchmark / integration test location",
        "notes": "Tropical canopy clearing and pasture conversion along the Amazon agricultural frontier."
    },
    "munich": {
        "name": "Munich North Urban Development",
        "country": "Germany",
        "biome": "Temperate Europe / Commercial & Farmland Transition",
        "bbox": [11.560, 48.180, 11.585, 48.205],
        "center": {"lat": 48.1925, "lon": 11.5725},
        "role": "Global geographic benchmark / integration test location",
        "notes": "Temperate climate with seasonal agricultural cycles and peri-urban construction."
    },
    "lake_mead": {
        "name": "Lake Mead Shoreline & Marina",
        "country": "United States",
        "biome": "Hydrological / Reservoir Shoreline Recession",
        "bbox": [-114.420, 36.010, -114.395, 36.035],
        "center": {"lat": 36.0225, "lon": -114.4075},
        "role": "Global geographic benchmark / integration test location",
        "notes": "Water-to-land boundary dynamics and exposed reservoir lakebed."
    }
}

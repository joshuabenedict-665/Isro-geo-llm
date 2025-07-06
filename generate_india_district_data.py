import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import json
from collections import Counter
from shapely.geometry import mapping
from tqdm import tqdm

# Paths
districts_path = "data/geojson/india.geojson"
dem_raster_path = "data/dem/india_dem.tif"
lulc_raster_path = "data/lulc/LULC_2005.tif"
output_path = "data/district_data.json"

# Optional: LULC classification map (customize per your raster codes)
LULC_CLASS_MAP = {
    1: "Agricultural Land",
    2: "Forest",
    3: "Water Body",
    4: "Built-up Area",
    5: "Wasteland",
    6: "Others"
}

def get_lulc_distribution(geometry, src):
    try:
        out_image, _ = rasterio.mask.mask(src, [mapping(geometry)], crop=True)
        data = out_image[0]
        data = data[data != src.nodata]
        if data.size == 0:
            return []
        counts = Counter(data.flatten())
        total = sum(counts.values())
        return [
            {
                "class_name": LULC_CLASS_MAP.get(k, f"Unknown ({k})"),
                "percentage": round((v / total) * 100, 2)
            }
            for k, v in counts.items()
        ]
    except Exception as e:
        return []

def get_mean_elevation(geometry, src):
    try:
        out_image, _ = rasterio.mask.mask(src, [mapping(geometry)], crop=True)
        data = out_image[0]
        data = data[data != src.nodata]
        if data.size == 0:
            return None
        return round(float(np.mean(data)), 2)
    except Exception:
        return None

# Load districts GeoJSON
gdf = gpd.read_file(districts_path)

# Ensure CRS matches raster
with rasterio.open(lulc_raster_path) as lulc_src:
    gdf = gdf.to_crs(lulc_src.crs)

# Open rasters
with rasterio.open(lulc_raster_path) as lulc_src, rasterio.open(dem_raster_path) as dem_src:
    results = []
    for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="🏞️ Processing LULC + DEM per district"):
        district_name = row.get("dtname") or row.get("DISTRICT") or "Unknown"
        geom = row.geometry

        lulc_stats = get_lulc_distribution(geom, lulc_src)
        elevation = get_mean_elevation(geom, dem_src)

        results.append({
            "district": district_name,
            "average_elevation": elevation if elevation else 0,
            "lulc_classes": lulc_stats
        })

# Save to JSON
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print("✅ All-India district_data.json generated.")

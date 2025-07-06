import geopandas as gpd
import rasterio
import rasterio.mask
import json
import numpy as np
from collections import Counter
from shapely.geometry import mapping
from tqdm import tqdm
import os

# ---- Config ----
districts_path = "data/tamilnadu_districts.geojson"
dem_raster = "data/dem/dem_tamilnadu.tif"
lulc_raster = "data/lulc/LULC_2005.tif"
json_output = "data/district_data.json"

# ---- Class map ----
LULC_CLASS_MAP = {
    1: "Agricultural Land",
    2: "Forest",
    3: "Water Body",
    4: "Built-up Area",
    5: "Wasteland",
    6: "Others"
}

def get_lulc_distribution(geometry, raster_path):
    with rasterio.open(raster_path) as src:
        try:
            out_image, _ = rasterio.mask.mask(src, [geometry], crop=True)
            data = out_image[0]
            data = data[data != src.nodata]
            if data.size == 0:
                return []
            counts = Counter(data.flatten())
            total = sum(counts.values())
            return [
                {"class_name": LULC_CLASS_MAP.get(val, f"Unknown ({val})"), "percentage": round((cnt / total) * 100, 2)}
                for val, cnt in counts.items()
            ]
        except Exception as e:
            print(f"⚠️ Error processing district: {e}")
            return []

def get_mean_elevation(geometry, raster_path):
    with rasterio.open(raster_path) as src:
        try:
            out_image, _ = rasterio.mask.mask(src, [geometry], crop=True)
            data = out_image[0]
            data = data[data != src.nodata]
            return float(np.mean(data)) if data.size > 0 else 0
        except Exception as e:
            print(f"⚠️ Elevation error: {e}")
            return 0

# ---- Process ----
gdf = gpd.read_file(districts_path)

# Ensure CRS matches raster CRS
with rasterio.open(lulc_raster) as sample_raster:
    gdf = gdf.to_crs(sample_raster.crs)

results = []

print("🏞️ Processing LULC + DEM per district:")
for _, row in tqdm(gdf.iterrows(), total=len(gdf)):
    name = row["dtname"]
    geom = row.geometry
    lulc = get_lulc_distribution(geom, lulc_raster)
    elevation = get_mean_elevation(geom, dem_raster)

    results.append({
        "district": name,
        "average_elevation": elevation,
        "lulc_classes": lulc
    })

# Save
os.makedirs(os.path.dirname(json_output), exist_ok=True)
with open(json_output, "w") as f:
    json.dump(results, f, indent=2)

print("✅ All done: district_data.json generated.")

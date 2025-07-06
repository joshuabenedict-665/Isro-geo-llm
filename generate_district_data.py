import json
import geopandas as gpd
import rasterio
from rasterio.mask import mask

# Paths
districts_path = "data/tamilnadu_districts.geojson"
dem_path = "data/dem/dem_tamilnadu.tif"
lulc_path = "data/lulc/LULC_2005.tif"


# Load district boundaries
districts = gpd.read_file(districts_path)
districts["centroid"] = districts.geometry.centroid
districts["lon"] = districts.centroid.x
districts["lat"] = districts.centroid.y

# Output list
output = []

# Open DEM and LULC rasters
with rasterio.open(dem_path) as dem_raster, rasterio.open(lulc_path) as lulc_raster:
    for _, row in districts.iterrows():
        geom = [row["geometry"]]

        # Mask DEM
        try:
            dem_data, _ = mask(dem_raster, geom, crop=True)
            dem_mean = float(dem_data[dem_data > 0].mean())
        except:
            dem_mean = None

        # Mask LULC
        try:
            lulc_data, _ = mask(lulc_raster, geom, crop=True)
            lulc_unique = list(map(int, set(lulc_data[lulc_data > 0].flatten())))
        except:
            lulc_unique = []

        district_name = row["dtname"]  # from your geojson columns

        output.append({
            "district": district_name,
            "latitude": row["lat"],
            "longitude": row["lon"],
            "average_elevation": dem_mean,
            "lulc_classes": lulc_unique,
        })

# Save as JSON
with open("data/district_data.json", "w") as f:
    json.dump(output, f, indent=2)

print("✅ District data saved to data/district_data.json")

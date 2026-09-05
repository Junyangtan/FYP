import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from matplotlib.path import Path

# =====================================
# STEP 1: READ POLYGON FROM KML
# =====================================
kml_file = "border.kml"

tree = ET.parse(kml_file)
root = tree.getroot()
ns = {'kml': 'http://www.opengis.net/kml/2.2'}

coords_text = root.find(".//kml:Polygon//kml:coordinates", ns).text.strip()

coords = []
for line in coords_text.split():
    lon, lat, _ = line.split(",")
    coords.append((float(lat), float(lon)))

polygon_latlon = np.array(coords)

# =====================================
# STEP 2: CONVERT TO METERS
# =====================================
base_lat, base_lon = polygon_latlon[0]

meters_per_deg_lat = 111320
meters_per_deg_lon = 111320 * np.cos(np.radians(base_lat))

polygon_m = []
for lat, lon in polygon_latlon:
    x = (lon - base_lon) * meters_per_deg_lon
    y = (lat - base_lat) * meters_per_deg_lat
    polygon_m.append([x, y])

polygon_m = np.array(polygon_m)
boundary = Path(polygon_m)

# =====================================
# STEP 3: GENERATE TREES (2D ONLY)
# =====================================
spacing = 18

min_x, min_y = polygon_m.min(axis=0)
max_x, max_y = polygon_m.max(axis=0)

x_range = np.arange(min_x, max_x, spacing)
y_range = np.arange(min_y, max_y, (np.sqrt(3)/2)*spacing)

trees = []

for i, y_m in enumerate(y_range):
    for j, x_m in enumerate(x_range):

        if i % 2:
            x_m += spacing / 2

        if boundary.contains_point((x_m, y_m)):

            # small jitter
            x_m += np.random.uniform(-0.15, 0.15)
            y_m += np.random.uniform(-0.15, 0.15)

            # convert to lat/lon
            lat = base_lat + (y_m / meters_per_deg_lat)
            lon = base_lon + (x_m / meters_per_deg_lon)

            trees.append([lat, lon])

# =====================================
# STEP 4: SAVE OUTPUT
# =====================================
df = pd.DataFrame(
    trees,
    columns=["lat", "lon"]
)

df.to_csv("2D_map.csv", index=False)

print(f"Generated {len(df)} 2D tree points")
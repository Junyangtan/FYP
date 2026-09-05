import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from matplotlib.path import Path
from noise import pnoise2   # pip install noise


# =====================================
# CONFIGURATION
# =====================================

kml_file = "map2_border.kml"
map_name = "Map2_3D_map"
spacing = 18          # tree spacing (m)
tree_height = 12      # tree height (m)



# =====================================
# STEP 1: READ ALL POLYGONS FROM KML
# =====================================

tree = ET.parse(kml_file)

root = tree.getroot()

ns = {
    "kml": "http://www.opengis.net/kml/2.2"
}


polygons = root.findall(
    ".//kml:Polygon",
    ns
)


print(
    f"Detected {len(polygons)} polygons"
)



# =====================================
# STEP 2: TREE GENERATION FUNCTION
# =====================================

def generate_trees(coords_text):


    # -----------------------------
    # Extract coordinates
    # -----------------------------

    coords = []


    for line in coords_text.split():

        lon, lat, _ = line.split(",")

        coords.append(
            (
                float(lat),
                float(lon)
            )
        )


    polygon_latlon = np.array(
        coords
    )



    # -----------------------------
    # Convert lat/lon to meters
    # -----------------------------

    base_lat, base_lon = polygon_latlon[0]


    meters_per_deg_lat = 111320


    meters_per_deg_lon = (
        111320 *
        np.cos(
            np.radians(base_lat)
        )
    )



    polygon_m = []



    for lat, lon in polygon_latlon:


        x = (
            lon - base_lon
        ) * meters_per_deg_lon


        y = (
            lat - base_lat
        ) * meters_per_deg_lat


        polygon_m.append(
            [
                x,
                y
            ]
        )



    polygon_m = np.array(
        polygon_m
    )


    boundary = Path(
        polygon_m
    )



    # -----------------------------
    # Boundary range
    # -----------------------------

    min_x, min_y = polygon_m.min(
        axis=0
    )


    max_x, max_y = polygon_m.max(
        axis=0
    )



    x_range = np.arange(
        min_x,
        max_x,
        spacing
    )


    y_range = np.arange(
        min_y,
        max_y,
        (np.sqrt(3)/2)*spacing
    )



    trees = []



    # =====================================
    # TRIANGULAR TREE GRID
    # =====================================

    for i, y_m in enumerate(y_range):


        for x_m in x_range:



            if i % 2:

                x_m += spacing / 2



            if boundary.contains_point(
                (
                    x_m,
                    y_m
                )
            ):



                # Small random variation

                x_m += np.random.uniform(
                    -0.15,
                    0.15
                )


                y_m += np.random.uniform(
                    -0.15,
                    0.15
                )



                # Convert back

                lat = (

                    base_lat +

                    y_m /
                    meters_per_deg_lat

                )


                lon = (

                    base_lon +

                    x_m /
                    meters_per_deg_lon

                )



                # =====================================
                # TERRAIN MODEL
                # =====================================

                scale = 500.0



                noise_val = (

                    pnoise2(
                        x_m / scale,
                        y_m / scale,
                        octaves=3
                    )

                    +

                    0.5 *

                    pnoise2(
                        (x_m+120)/scale,
                        (y_m+80)/scale,
                        octaves=2
                    )

                )



                noise_norm = (
                    noise_val + 0.5
                )



                # slope

                x_norm = (

                    x_m - min_x

                ) / (

                    max_x - min_x

                )



                slope = (
                    -0.6 *
                    x_norm
                )



                ground_z = (

                    1.4

                    +

                    0.2 *
                    noise_norm

                    +

                    slope

                )



                # soft compression

                ground_z = (

                    1.0

                    +

                    (
                        ground_z - 1.0
                    )

                    /

                    (
                        1 +

                        abs(
                            ground_z - 1.0
                        )

                    )

                )



                obstacle_z = (

                    ground_z

                    +

                    tree_height

                )



                trees.append(

                    [
                        lat,
                        lon,
                        obstacle_z
                    ]

                )


    return trees




# =====================================
# STEP 3: PROCESS ALL POLYGONS
# =====================================

all_trees = []



for index, polygon in enumerate(
    polygons,
    start=1
):


    coords_element = polygon.find(
        ".//kml:coordinates",
        ns
    )



    if coords_element is None:

        continue



    coords_text = (
        coords_element.text.strip()
    )



    polygon_trees = generate_trees(
        coords_text
    )



    print(
        f"Polygon {index}: {len(polygon_trees)} trees"
    )



    all_trees.extend(
        polygon_trees
    )




# =====================================
# STEP 4: SAVE CSV
# =====================================

df = pd.DataFrame(
    all_trees,
    columns=[
        "lat",
        "lon",
        "alt"
    ]
)



df.to_csv(
    f"{map_name}.csv",
    index=False
)



print("==============================")
print(
    f"Generated {len(df)} trees"
)
print(
    f"Saved: {map_name}.csv"
)
print("==============================")
"""Drone route viewer and matched algorithm comparison dashboard.

Place this file at Dashboard/dashboard.py and run:
    streamlit run Dashboard/dashboard.py
Analysis outputs belong in Output/Analysis_Matched.
Optional environment variables: DRONE_PROJECT_ROOT, DRONE_ROUTE_OUTPUT.
"""
import io
import os
import re
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import streamlit as st
import folium
import xyzservices.providers as xyz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from streamlit_folium import st_folium

st.set_page_config(page_title="Drone Path Planning Dashboard", page_icon="🚁", layout="wide")

DASHBOARD_FOLDER = Path(__file__).resolve().parent
PROJECT_FOLDER = Path(os.environ.get("DRONE_PROJECT_ROOT", str(DASHBOARD_FOLDER.parent)))
OUTPUT_FOLDER = PROJECT_FOLDER / "Output"
DATA_FILE = OUTPUT_FOLDER / "analysis_data2.csv"
ANALYSIS_FOLDER = OUTPUT_FOLDER / "Analysis_Matched"
ROUTE_OUTPUT_FOLDER = Path(os.environ.get("DRONE_ROUTE_OUTPUT", str(OUTPUT_FOLDER)))
if not ROUTE_OUTPUT_FOLDER.is_absolute():
    ROUTE_OUTPUT_FOLDER = PROJECT_FOLDER / ROUTE_OUTPUT_FOLDER

ALGORITHMS = ["ACO", "GA", "NSGAII"]
METRICS = ["Distance", "Turning", "Elevation"]
UNITS = {"Distance": "m", "Turning": "rad", "Elevation": "m"}
LABELS = {"ACO": "ACO", "GA": "GA", "NSGAII": "NSGA-II"}
COLORS = {"ACO": "#167d9a", "GA": "#d58627", "NSGAII": "#8661a9"}
ALPHA = 0.05

MAP_FILES = {m: next((PROJECT_FOLDER / name for name in [f"{m}_3D_map.csv", f"Maps{m[-1]}_3D_map.csv"]
                     if (PROJECT_FOLDER / name).exists()), PROJECT_FOLDER / f"{m}_3D_map.csv")
             for m in ["Map1", "Map2"]}
ROAD_FILES = {m: PROJECT_FOLDER / f"road_{m.lower()}.kml" for m in MAP_FILES}
BOUNDARY_FILES = {m: PROJECT_FOLDER / f"boundary_{m.lower()}.kml" for m in MAP_FILES}
ALGORITHM_OUTPUT_FOLDERS = {a: ROUTE_OUTPUT_FOLDER / f"{a}_3D" for a in ALGORITHMS}


def read_csv_optional(path):
    """Read each result independently so optional files do not block the viewer."""
    if not Path(path).is_file():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.warning(f"Could not load {Path(path).name}: {error}")
        return None


df = read_csv_optional(DATA_FILE)
if df is None:
    st.error("analysis_data2.csv could not be loaded. Place it in the project's Output folder.")
    st.stop()
required = ["Map", "Cluster", "Algorithm", "Run"] + METRICS
if df.empty or not set(required).issubset(df.columns):
    st.error("The analysis CSV must contain Map, Cluster, Algorithm, Run, Distance, Turning and Elevation.")
    st.stop()
for col in ["Map", "Algorithm"]:
    df[col] = df[col].astype("string").str.strip()
df["Algorithm"] = df["Algorithm"].replace({"NSGA-II": "NSGAII"})
for col in ["Cluster", "Run"] + METRICS:
    df[col] = pd.to_numeric(df[col], errors="coerce")
if df[required].isna().any().any() or not np.isfinite(df[["Cluster", "Run"] + METRICS]).all().all():
    st.error("The raw data contains missing identifiers or nonfinite metric values. Correct the source CSV first.")
    st.stop()
if any((df[c] % 1 != 0).any() for c in ["Cluster", "Run"]):
    st.error("Cluster and Run must be integer identifiers.")
    st.stop()
df[["Cluster", "Run"]] = df[["Cluster", "Run"]].astype(int)
if df.duplicated(["Map", "Cluster", "Algorithm", "Run"]).any():
    st.error("Duplicate map/cluster/algorithm/run records were found. Resolve them before comparing results.")
    st.stop()
if not set(df.Algorithm).issubset(ALGORITHMS) or not set(df.Map).issubset(MAP_FILES):
    st.error("Supported names are ACO, GA, NSGAII and Map1, Map2.")
    st.stop()


def load_tree_map(map_name):

    map_file = MAP_FILES.get(
        map_name
    )

    if map_file is None:
        return None

    if not os.path.exists(
        map_file
    ):
        return None

    return read_csv_optional(map_file)


# =====================================================
# GET LATEST 20 FILES
# =====================================================

def get_map_run_file(folder, extension, map_name, run_number):
    """Match explicit map/run filenames or their corresponding summary metadata.

    Supports routes_Map2_1_20260918_142707.kml and legacy timestamp-only
    filenames when Summary/summary_<same timestamp>.txt contains Map and Run.
    Never infer map/run from the file's position in a sorted list.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return None
    matches = []
    pattern = re.compile(r"(?:^|[_-])(Map\d+)[_-](?:Run[_-]?)?(\d+)(?:[_-]|$)", re.I)
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() != extension.lower():
            continue
        found = pattern.search(path.stem)
        matched_map = matched_run = None
        if found:
            matched_map, matched_run = found.group(1), int(found.group(2))
        else:
            suffix = path.stem.split("_", 1)[-1]
            summary_path = folder.parent / "Summary" / f"summary_{suffix}.txt"
            if summary_path.is_file():
                try:
                    content = summary_path.read_text(encoding="utf-8-sig")
                    mm = re.search(r"^Map:\s*(\S+)", content, re.M | re.I)
                    rr = re.search(r"^Run:\s*(\d+)", content, re.M | re.I)
                    if mm and rr:
                        matched_map, matched_run = mm.group(1), int(rr.group(1))
                except (OSError, UnicodeError):
                    continue
        if matched_map and matched_map.lower() == str(map_name).lower() and matched_run == int(run_number):
            stamp = re.search(r"(\d{8}_\d{6}(?:_\d{6})?)$", path.stem)
            matches.append((stamp.group(1) if stamp else "", path.stat().st_mtime_ns, path))
    return str(max(matches, key=lambda item: (item[0], item[1]))[2]) if matches else None


def get_kml_file(
    map_name,
    algorithm,
    run_number
):

    algorithm_folder = (
        ALGORITHM_OUTPUT_FOLDERS.get(
            algorithm
        )
    )


    if algorithm_folder is None:
        return None


    kml_folder = os.path.join(
        algorithm_folder,
        "KML"
    )


    return get_map_run_file(
        kml_folder,
        ".kml",
        map_name,
        run_number
    )


# =====================================================
# GET METRIC FILE FOR SELECTED RUN
# =====================================================

def get_metric_file(
    map_name,
    algorithm,
    run_number
):

    algorithm_folder = (
        ALGORITHM_OUTPUT_FOLDERS.get(
            algorithm
        )
    )


    if algorithm_folder is None:
        return None


    metric_folder = os.path.join(
        algorithm_folder,
        "Metrics"
    )


    return get_map_run_file(
        metric_folder,
        ".csv",
        map_name,
        run_number
    )


# =====================================================
# PARSE KML COORDINATES
# =====================================================

def parse_coordinates(
    coordinate_text
):

    points = []


    if not coordinate_text:
        return points


    coordinate_text = (
        coordinate_text.strip()
    )


    items = (
        coordinate_text.split()
    )


    for item in items:

        parts = item.split(
            ","
        )


        if len(parts) >= 2:

            try:

                lon = float(
                    parts[0]
                )

                lat = float(
                    parts[1]
                )


                if len(parts) >= 3:

                    alt = float(
                        parts[2]
                    )

                else:

                    alt = 0.0


                points.append(
                    (
                        lat,
                        lon,
                        alt
                    )
                )


            except ValueError:

                continue


    return points


# =====================================================
# EXTRACT PLACEMARK BY NAME
# =====================================================

def extract_placemark_coordinates_by_name(
    kml_file,
    target_name
):

    try:
        tree = ET.parse(kml_file)
    except (ET.ParseError, OSError):
        return []

    root = tree.getroot()


    for placemark in root.iter():

        if (
            placemark.tag.split("}")[-1]
            != "Placemark"
        ):

            continue


        placemark_name = None
        coordinates_text = None


        for element in placemark.iter():

            element_tag = (
                element.tag.split("}")[-1]
            )


            if (
                element_tag == "name"
                and
                placemark_name is None
            ):

                if element.text:

                    placemark_name = (
                        element.text.strip()
                    )


            if (
                element_tag
                == "coordinates"
            ):

                if element.text:

                    coordinates_text = (
                        element.text
                    )


        if (
            placemark_name
            == target_name
        ):

            if coordinates_text is None:

                return []


            return parse_coordinates(
                coordinates_text
            )


    return []


# =====================================================
# EXTRACT FIXED BOUNDARY FROM KML
# =====================================================

def extract_boundary_from_kml(
    boundary_file
):

    if boundary_file is None:
        return []


    if not os.path.exists(
        boundary_file
    ):
        return []


    try:

        tree = ET.parse(
            boundary_file
        )

        root = tree.getroot()


        # =============================================
        # FIRST TRY POLYGON
        # =============================================

        for element in root.iter():

            if (
                element.tag.split("}")[-1]
                != "Polygon"
            ):
                continue


            for child in element.iter():

                if (
                    child.tag.split("}")[-1]
                    == "coordinates"
                ):

                    if child.text:

                        points = (
                            parse_coordinates(
                                child.text
                            )
                        )


                        if points:

                            return points


        # =============================================
        # FALLBACK TO LINESTRING
        # =============================================

        for element in root.iter():

            if (
                element.tag.split("}")[-1]
                != "LineString"
            ):
                continue


            for child in element.iter():

                if (
                    child.tag.split("}")[-1]
                    == "coordinates"
                ):

                    if child.text:

                        points = (
                            parse_coordinates(
                                child.text
                            )
                        )


                        if points:

                            return points


    except (ET.ParseError, OSError):

        return []


    return []


# =====================================================
# EXTRACT FIXED ROAD FROM KML
# =====================================================

def extract_road_from_kml(
    road_file
):

    road_paths = []


    if road_file is None:
        return road_paths


    if not os.path.exists(
        road_file
    ):
        return road_paths


    try:

        tree = ET.parse(
            road_file
        )

        root = tree.getroot()


        # Find every LineString
        for element in root.iter():

            if (
                element.tag.split("}")[-1]
                != "LineString"
            ):

                continue


            for child in element.iter():

                if (
                    child.tag.split("}")[-1]
                    == "coordinates"
                ):

                    if child.text:

                        points = (
                            parse_coordinates(
                                child.text
                            )
                        )


                        if points:

                            road_paths.append(
                                points
                            )


    except (ET.ParseError, OSError):

        return []


    return road_paths


# =====================================================
# EXTRACT SELECTED CLUSTER ROUTE
# =====================================================

def extract_cluster_route(
    kml_file,
    cluster_number
):

    target_name = (
        f"Cluster "
        f"{int(cluster_number)} "
        f"3D Drone Path"
    )


    return (
        extract_placemark_coordinates_by_name(
            kml_file,
            target_name
        )
    )


# =====================================================
# GET ALL KML PLACEMARK NAMES
# =====================================================

def get_placemark_names(
    kml_file
):

    try:
        tree = ET.parse(kml_file)
    except (ET.ParseError, OSError):
        return []

    root = tree.getroot()

    names = []


    for placemark in root.iter():

        if (
            placemark.tag.split("}")[-1]
            != "Placemark"
        ):

            continue


        placemark_name = None


        for element in placemark:

            if (
                element.tag.split("}")[-1]
                == "name"
            ):

                if element.text:

                    placemark_name = (
                        element.text.strip()
                    )

                break


        if placemark_name:

            names.append(
                placemark_name
            )


    return names


# =====================================================
# TITLE
# =====================================================

st.title(
    "🚁 Automated Drone Path Planning Dashboard"
)

st.write(
    "Oil Palm Plantation Pesticide Spraying Route Optimization"
)

st.divider()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header(
    "Route Selection"
)


# =====================================================
# MAP SELECTION
# =====================================================

maps_available = sorted(
    df[
        "Map"
    ]
    .dropna()
    .unique()
)


selected_map = (
    st.sidebar.selectbox(
        "Select Map",
        maps_available
    )
)


# =====================================================
# ALGORITHM SELECTION
# =====================================================

algorithms_available = sorted(
    df[
        df[
            "Map"
        ]
        == selected_map
    ][
        "Algorithm"
    ]
    .dropna()
    .unique()
)


selected_algorithm = (
    st.sidebar.selectbox(
        "Select Algorithm",
        algorithms_available
    )
)


# =====================================================
# RUN SELECTION
# =====================================================

runs_available = sorted(
    df[
        (
            df[
                "Map"
            ]
            == selected_map
        )
        &
        (
            df[
                "Algorithm"
            ]
            == selected_algorithm
        )
    ][
        "Run"
    ]
    .dropna()
    .unique()
)


selected_run = (
    st.sidebar.selectbox(
        "Select Run",
        runs_available
    )
)


# =====================================================
# CLUSTER SELECTION
# =====================================================

clusters_available = sorted(
    df[
        (
            df[
                "Map"
            ]
            == selected_map
        )
        &
        (
            df[
                "Algorithm"
            ]
            == selected_algorithm
        )
        &
        (
            df[
                "Run"
            ]
            == selected_run
        )
    ][
        "Cluster"
    ]
    .dropna()
    .unique()
)


selected_cluster = (
    st.sidebar.selectbox(
        "Select Cluster",
        clusters_available
    )
)


# =====================================================
# GET SELECTED FILES
# =====================================================

kml_file = get_kml_file(
    selected_map,
    selected_algorithm,
    selected_run
)


metric_file = get_metric_file(
    selected_map,
    selected_algorithm,
    selected_run
)


if kml_file is not None:
    export_id = Path(kml_file).stem.removeprefix("routes_")
    exact_metrics = Path(ALGORITHM_OUTPUT_FOLDERS[selected_algorithm]) / "Metrics" / f"metrics_{export_id}.csv"
    metric_file = str(exact_metrics) if exact_metrics.is_file() else None
    st.sidebar.caption(f"Route export: {Path(kml_file).name}")
    st.sidebar.download_button("Download run KML", Path(kml_file).read_bytes(),
                               file_name=Path(kml_file).name, mime="application/vnd.google-earth.kml+xml")

road_file = ROAD_FILES.get(
    selected_map
)


boundary_file = BOUNDARY_FILES.get(
    selected_map
)


# =====================================================
# GET FIXED MAP FEATURES
# =====================================================

plantation_boundary = []

road_paths = []


# =====================================================
# FIXED BOUNDARY
# =====================================================

if (
    boundary_file is not None
    and
    os.path.exists(
        boundary_file
    )
):

    plantation_boundary = (
        extract_boundary_from_kml(
            boundary_file
        )
    )


# =====================================================
# FIXED ROAD
# =====================================================

if (
    road_file is not None
    and
    os.path.exists(
        road_file
    )
):

    road_paths = (
        extract_road_from_kml(
            road_file
        )
    )


# =====================================================
# CURRENT SELECTION
# =====================================================

st.subheader(
    "Current Selection"
)


col1, col2, col3, col4 = (
    st.columns(
        4
    )
)


with col1:

    st.metric(
        "Map",
        selected_map
    )


with col2:

    display_algorithm = (
        "NSGA-II"
        if selected_algorithm == "NSGAII"
        else selected_algorithm
    )

    st.metric(
        "Algorithm",
        display_algorithm
    )


with col3:

    st.metric(
        "Run",
        f"Run {int(selected_run)}"
    )


with col4:

    st.metric(
        "Cluster",
        f"Cluster {int(selected_cluster)}"
    )


st.divider()


# =====================================================
# PLANTATION OVERVIEW
# =====================================================

st.subheader(
    f"Plantation Overview - "
    f"{selected_map}"
)


tree_data = (
    load_tree_map(
        selected_map
    )
)


if tree_data is not None:

    required_columns = {
        "lat",
        "lon",
        "alt"
    }


    if required_columns.issubset(
        tree_data.columns
    ):

        center_lat = (
            tree_data[
                "lat"
            ].mean()
        )

        center_lon = (
            tree_data[
                "lon"
            ].mean()
        )


        # =============================================
        # CREATE MAP
        # =============================================

        plantation_map = (
            folium.Map(
                location=[
                    center_lat,
                    center_lon
                ],
                zoom_start=17,
                tiles=None,
                control_scale=True
            )
        )


        # =============================================
        # SATELLITE
        # =============================================

        folium.TileLayer(
            tiles=xyz.Esri.WorldImagery,
            name="Satellite",
            overlay=False,
            control=True,
            show=True
        ).add_to(
            plantation_map
        )


        # =============================================
        # STREET MAP
        # =============================================

        folium.TileLayer(
            tiles=xyz.OpenStreetMap.Mapnik,
            name="Street Map",
            overlay=False,
            control=True,
            show=False
        ).add_to(
            plantation_map
        )


        # =============================================
        # TREE LAYER
        # =============================================

        tree_layer = (
            folium.FeatureGroup(
                name="Oil Palm Trees",
                show=True
            )
        )


        for (
            index,
            tree
        ) in tree_data.iterrows():

            folium.CircleMarker(
                location=[
                    tree[
                        "lat"
                    ],
                    tree[
                        "lon"
                    ]
                ],
                radius=2.5,
                color="blue",
                weight=1,
                fill=True,
                fill_color="blue",
                fill_opacity=0.9,
                tooltip=(
                    f"Tree {index + 1}"
                    f"<br>"
                    f"Elevation: "
                    f"{tree['alt']:.2f} m"
                )
            ).add_to(
                tree_layer
            )


        tree_layer.add_to(
            plantation_map
        )


        # =============================================
        # PLANTATION BOUNDARY
        # =============================================

        if plantation_boundary:

            boundary_2d = [
                (
                    lat,
                    lon
                )
                for (
                    lat,
                    lon,
                    alt
                )
                in plantation_boundary
            ]


            boundary_layer = (
                folium.FeatureGroup(
                    name="Plantation Boundary",
                    show=True
                )
            )


            folium.Polygon(
                locations=boundary_2d,
                color="grey",
                weight=4,
                opacity=1.0,
                fill=False,
                tooltip="Plantation Boundary"
            ).add_to(
                boundary_layer
            )


            boundary_layer.add_to(
                plantation_map
            )


        # =============================================
        # PLANTATION ROAD
        # =============================================

        if road_paths:

            road_layer = (
                folium.FeatureGroup(
                    name="Plantation Road",
                    show=True
                )
            )


            for road_path in road_paths:

                road_2d = [
                    (
                        lat,
                        lon
                    )
                    for (
                        lat,
                        lon,
                        alt
                    )
                    in road_path
                ]


                if len(road_2d) >= 2:

                    folium.PolyLine(
                        locations=road_2d,
                        color="orange",
                        weight=5,
                        opacity=1.0,
                        tooltip="Plantation Road"
                    ).add_to(
                        road_layer
                    )


            road_layer.add_to(
                plantation_map
            )


        # =============================================
        # FIT MAP
        # =============================================

        if plantation_boundary:

            plantation_map.fit_bounds(
                [
                    [
                        min(
                            p[0]
                            for p
                            in plantation_boundary
                        ),

                        min(
                            p[1]
                            for p
                            in plantation_boundary
                        )
                    ],

                    [
                        max(
                            p[0]
                            for p
                            in plantation_boundary
                        ),

                        max(
                            p[1]
                            for p
                            in plantation_boundary
                        )
                    ]
                ],

                padding=(
                    50,
                    50
                )
            )


        else:

            plantation_map.fit_bounds(
                [
                    [
                        tree_data[
                            "lat"
                        ].min(),

                        tree_data[
                            "lon"
                        ].min()
                    ],

                    [
                        tree_data[
                            "lat"
                        ].max(),

                        tree_data[
                            "lon"
                        ].max()
                    ]
                ],

                padding=(
                    50,
                    50
                )
            )


        # =============================================
        # LAYER CONTROL
        # =============================================

        folium.LayerControl(
            collapsed=False
        ).add_to(
            plantation_map
        )


        # =============================================
        # LEGEND
        # =============================================

        plantation_legend = """
        <div style="
            position: fixed;
            bottom: 35px;
            left: 35px;
            width: 215px;
            background-color: rgba(255,255,255,0.95);
            border: 2px solid #666;
            z-index: 9999;
            font-size: 14px;
            color: black;
            padding: 12px;
            border-radius: 6px;
        ">

            <div style="
                font-weight: bold;
                font-size: 15px;
                margin-bottom: 8px;
            ">
                Legend
            </div>


            <div style="
                display: flex;
                align-items: center;
                margin-bottom: 6px;
            ">

                <span style="
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    background: blue;
                    margin-right: 8px;
                ">
                </span>

                Oil Palm Tree

            </div>


            <div style="
                display: flex;
                align-items: center;
                margin-bottom: 6px;
            ">

                <span style="
                    display: inline-block;
                    width: 25px;
                    height: 4px;
                    background: grey;
                    margin-right: 8px;
                ">
                </span>

                Plantation Boundary

            </div>


            <div style="
                display: flex;
                align-items: center;
            ">

                <span style="
                    display: inline-block;
                    width: 25px;
                    height: 4px;
                    background: orange;
                    margin-right: 8px;
                ">
                </span>

                Plantation Road

            </div>

        </div>
        """


        plantation_map.get_root().html.add_child(
            folium.Element(
                plantation_legend
            )
        )


        # =============================================
        # DISPLAY
        # =============================================

        st_folium(
            plantation_map,
            height=550,
            use_container_width=True
        )


        st.caption(
            f"Total trees in "
            f"{selected_map}: "
            f"{len(tree_data)}"
        )


        if not plantation_boundary:

            st.warning(
                f"No boundary was found in "
                f"{os.path.basename(boundary_file)}."
            )


        if not road_paths:

            st.warning(
                f"No road LineString was found in "
                f"{os.path.basename(road_file)}."
            )


    else:

        st.error(
            "Map CSV must contain "
            "'lat', 'lon', and 'alt' columns."
        )


else:

    st.warning(
        f"Could not load map data "
        f"for {selected_map}."
    )


st.divider()


# =====================================================
# =====================================================
# SELECTED ROUTE PERFORMANCE (ONE RUN, ONE CLUSTER)
# =====================================================
st.subheader("Selected route performance")
selected_records = df[(df.Map == selected_map) & (df.Algorithm == selected_algorithm)
                      & (df.Run == selected_run) & (df.Cluster == selected_cluster)]
row = selected_records.iloc[0]
route_metrics = None
if metric_file is not None:
    metric_data = read_csv_optional(metric_file)
    if metric_data is not None and {"Cluster", *METRICS}.issubset(metric_data.columns):
        selection = metric_data[pd.to_numeric(metric_data.Cluster, errors="coerce") == int(selected_cluster)]
        if len(selection) == 1:
            numeric = pd.to_numeric(selection.iloc[0][METRICS], errors="coerce")
            if np.isfinite(numeric.to_numpy(float)).all():
                route_metrics = selection.iloc[0]
                if not np.allclose(numeric.to_numpy(float), row[METRICS].to_numpy(float), rtol=1e-5, atol=1e-6):
                    st.warning("This route export differs from the corresponding raw analysis record. Route cards show the export; experiment comparisons below use analysis_data2.csv.")
if route_metrics is not None:
    row = route_metrics
else:
    st.caption("Route cards use the selected record in analysis_data2.csv; a corresponding route metric export was not available.")
metric_columns = st.columns(4)
trees_value = pd.to_numeric(row.get("Trees", np.nan), errors="coerce")
metric_columns[0].metric("Trees", str(int(trees_value)) if pd.notna(trees_value) else "Unavailable")
for column, metric in zip(metric_columns[1:], METRICS):
    column.metric(metric, f"{float(row[metric]):.3f} {UNITS[metric]}")
st.caption(f"{selected_map} · {LABELS[selected_algorithm]} · Run {int(selected_run)} · Cluster {int(selected_cluster)}. These are individual route values, not experiment averages.")
st.divider()

# ROUTE VISUALIZATION
# =====================================================

st.subheader(
    f"Route Visualization - "
    f"Cluster "
    f"{int(selected_cluster)}"
)


if (
    kml_file is not None
    and
    os.path.exists(
        kml_file
    )
):

    route_points = (
        extract_cluster_route(
            kml_file,
            selected_cluster
        )
    )


    if route_points:

        st.caption("Top-down route view. Download the KML to inspect altitude in a 3D viewer.")

        route_2d = [
            (
                lat,
                lon
            )
            for (
                lat,
                lon,
                alt
            )
            in route_points
        ]


        start_lat = (
            route_points[
                0
            ][0]
        )

        start_lon = (
            route_points[
                0
            ][1]
        )


        # =============================================
        # CREATE ROUTE MAP
        # =============================================

        route_map = (
            folium.Map(
                location=[
                    start_lat,
                    start_lon
                ],
                zoom_start=17,
                tiles=None,
                control_scale=True
            )
        )


        # =============================================
        # SATELLITE
        # =============================================

        folium.TileLayer(
            tiles=xyz.Esri.WorldImagery,
            name="Satellite",
            overlay=False,
            control=True,
            show=True
        ).add_to(
            route_map
        )


        # =============================================
        # STREET MAP
        # =============================================

        folium.TileLayer(
            tiles=xyz.OpenStreetMap.Mapnik,
            name="Street Map",
            overlay=False,
            control=True,
            show=False
        ).add_to(
            route_map
        )


        # =============================================
        # BOUNDARY
        # =============================================

        if plantation_boundary:

            boundary_2d = [
                (
                    lat,
                    lon
                )
                for (
                    lat,
                    lon,
                    alt
                )
                in plantation_boundary
            ]


            boundary_layer = (
                folium.FeatureGroup(
                    name="Plantation Boundary",
                    show=True
                )
            )


            folium.Polygon(
                locations=boundary_2d,
                color="grey",
                weight=4,
                opacity=1.0,
                fill=False,
                tooltip="Plantation Boundary"
            ).add_to(
                boundary_layer
            )


            boundary_layer.add_to(
                route_map
            )


        # =============================================
        # ROAD
        # =============================================

        if road_paths:

            road_layer = (
                folium.FeatureGroup(
                    name="Plantation Road",
                    show=True
                )
            )


            for road_path in road_paths:

                road_2d = [
                    (
                        lat,
                        lon
                    )
                    for (
                        lat,
                        lon,
                        alt
                    )
                    in road_path
                ]


                if len(road_2d) >= 2:

                    folium.PolyLine(
                        locations=road_2d,
                        color="orange",
                        weight=5,
                        opacity=1.0,
                        tooltip="Plantation Road"
                    ).add_to(
                        road_layer
                    )


            road_layer.add_to(
                route_map
            )


        # =============================================
        # DRONE ROUTE
        # =============================================

        route_layer = (
            folium.FeatureGroup(
                name=(
                    f"Cluster "
                    f"{int(selected_cluster)} Route"
                ),
                show=True
            )
        )


        folium.PolyLine(
            route_2d,
            color="deepskyblue",
            weight=3,
            opacity=0.9,
            tooltip=(
                f"Cluster "
                f"{int(selected_cluster)} "
                f"Drone Route"
            )
        ).add_to(
            route_layer
        )


        route_layer.add_to(
            route_map
        )


        # =============================================
        # START POINT
        # =============================================

        folium.Marker(
            route_2d[
                0
            ],
            tooltip="Start Point",
            popup=(
                f"Cluster "
                f"{int(selected_cluster)} "
                f"Start Point"
            ),
            icon=folium.Icon(
                color="green",
                icon="play"
            )
        ).add_to(
            route_map
        )


        # =============================================
        # END POINT
        # =============================================

        folium.Marker(
            route_2d[
                -1
            ],
            tooltip="End Point",
            popup=(
                f"Cluster "
                f"{int(selected_cluster)} "
                f"End Point"
            ),
            icon=folium.Icon(
                color="red",
                icon="stop"
            )
        ).add_to(
            route_map
        )


        # =============================================
        # FIT ROUTE MAP
        # =============================================

        if plantation_boundary:

            route_map.fit_bounds(
                [
                    [
                        min(
                            p[0]
                            for p
                            in plantation_boundary
                        ),

                        min(
                            p[1]
                            for p
                            in plantation_boundary
                        )
                    ],

                    [
                        max(
                            p[0]
                            for p
                            in plantation_boundary
                        ),

                        max(
                            p[1]
                            for p
                            in plantation_boundary
                        )
                    ]
                ],

                padding=(
                    50,
                    50
                )
            )


        else:

            route_map.fit_bounds(
                route_2d,
                padding=(
                    80,
                    80
                )
            )


        # =============================================
        # MAP CONTROL
        # =============================================

        folium.LayerControl(
            collapsed=False
        ).add_to(
            route_map
        )


        # =============================================
        # ROUTE LEGEND
        # =============================================

        route_legend = """
        <div style="
            position: fixed;
            bottom: 35px;
            left: 35px;
            width: 215px;
            background-color: rgba(255,255,255,0.95);
            border: 2px solid #666;
            z-index: 9999;
            font-size: 14px;
            color: black;
            padding: 12px;
            border-radius: 6px;
        ">

            <div style="
                font-weight: bold;
                font-size: 15px;
                margin-bottom: 8px;
            ">
                Legend
            </div>


            <div style="
                display: flex;
                align-items: center;
                margin-bottom: 6px;
            ">

                <span style="
                    display: inline-block;
                    width: 25px;
                    height: 4px;
                    background: deepskyblue;
                    margin-right: 8px;
                ">
                </span>

                Drone Route

            </div>


            <div style="
                display: flex;
                align-items: center;
                margin-bottom: 6px;
            ">

                <span style="
                    display: inline-block;
                    width: 25px;
                    height: 4px;
                    background: orange;
                    margin-right: 8px;
                ">
                </span>

                Plantation Road

            </div>


            <div style="
                display: flex;
                align-items: center;
                margin-bottom: 6px;
            ">

                <span style="
                    display: inline-block;
                    width: 25px;
                    height: 4px;
                    background: grey;
                    margin-right: 8px;
                ">
                </span>

                Plantation Boundary

            </div>


            <div style="
                display: flex;
                align-items: center;
                margin-bottom: 6px;
            ">

                <span style="
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    background: green;
                    margin-right: 8px;
                ">
                </span>

                Start Point

            </div>


            <div style="
                display: flex;
                align-items: center;
            ">

                <span style="
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    background: red;
                    margin-right: 8px;
                ">
                </span>

                End Point

            </div>

        </div>
        """


        route_map.get_root().html.add_child(
            folium.Element(
                route_legend
            )
        )


        # =============================================
        # DISPLAY
        # =============================================

        st_folium(
            route_map,
            height=650,
            use_container_width=True
        )


    else:

        st.error(
            f"Could not find "
            f"'Cluster "
            f"{int(selected_cluster)} "
            f"3D Drone Path' "
            f"inside this KML file."
        )


        with st.expander(
            "Show KML Placemark Names"
        ):

            placemark_names = (
                get_placemark_names(
                    kml_file
                )
            )


            if placemark_names:

                for name in placemark_names:

                    st.write(
                        name
                    )


            else:

                st.write(
                    "No Placemark names found."
                )


else:

    algorithm_folder = (
        ALGORITHM_OUTPUT_FOLDERS.get(
            selected_algorithm
        )
    )


    expected_folder = (
        os.path.join(
            algorithm_folder,
            "KML"
        )
        if algorithm_folder
        else "Unknown"
    )


    st.error(
        "KML file could not be loaded."
    )


    st.write(
        "Checked folder:"
    )


    st.code(
        expected_folder
    )


    st.info(
        "No route file could be matched to this map and run. "
        "Use filenames containing the map and run, or keep the matching "
        "Summary/summary_<timestamp>.txt file for legacy exports."
    )


# =====================================================


# =====================================================
# MATCHED ALGORITHM PERFORMANCE ANALYSIS
# =====================================================


def p_label(value):
    if pd.isna(value):
        return "Unavailable"
    return f"{float(value):.3e}"


def scope_rows(frame, scope, metric=None):
    if frame is None or not {"Scope", "Metric"}.issubset(frame.columns):
        return pd.DataFrame()
    mask = frame.Scope.eq(scope)
    if metric is not None:
        mask &= frame.Metric.eq(metric)
    return frame.loc[mask].copy()


def table_download(frame, filename, key):
    st.download_button("Download CSV", frame.to_csv(index=False, float_format="%.12g").encode("utf-8"),
                       file_name=filename, mime="text/csv", key=key)


def display_table(frame):
    """Keep source numbers numeric; format only the presentation copy."""
    shown = frame.copy()
    for col in shown:
        if col.lower().startswith("p_") or col == "Adjusted p-value":
            shown[col] = pd.to_numeric(shown[col], errors="coerce").map(p_label)
    st.dataframe(shown, width="stretch", hide_index=True)


def build_descriptive(cluster_means):
    rows = []
    for scope in ["Overall"] + sorted(cluster_means.Map.unique().tolist()):
        scoped = cluster_means if scope == "Overall" else cluster_means[cluster_means.Map.eq(scope)]
        for metric in METRICS:
            for algorithm, group in scoped.groupby("Algorithm"):
                values = group[metric]
                rows.append({"Scope": scope, "Metric": metric, "Algorithm": algorithm,
                             "N_Blocks": len(values), "Mean": values.mean(),
                             "SD_Across_Cluster_Means": values.std(ddof=1)})
    return pd.DataFrame(rows)


def same_table_values(actual, expected, keys, values):
    if actual is None or not set(keys + values).issubset(actual.columns) or actual.duplicated(keys).any():
        return False
    if len(actual) != len(expected):
        return False
    a = actual.set_index(keys).sort_index()
    b = expected.set_index(keys).sort_index()
    # Pandas CSV reads and StringDtype grouping can produce different index
    # dtypes even when the identifiers are identical. Compare key values.
    if a.index.tolist() != b.index.tolist():
        return False
    try:
        return np.allclose(a[values].to_numpy(float), b[values].to_numpy(float),
                           rtol=1e-8, atol=1e-9, equal_nan=True)
    except (ValueError, TypeError):
        return False


def validate_statistics(cm, saved_cm, anova, pairs):
    """Check matching/data identity before displaying saved significance claims.

    This verifies keys, means, pair differences and SEs, not the original
    test implementation or the physical identity of each cluster's trees.
    """
    if not same_table_values(saved_cm, cm, ["Map", "Cluster", "Algorithm"], METRICS):
        return False, "The saved cluster averages are missing or do not match the raw data. Rerun ANOVA_analysis.py."
    counts = cm.groupby(["Map", "Cluster"]).Algorithm.nunique()
    if not counts.eq(len(ALGORITHMS)).all():
        return False, "Some clusters do not contain all three algorithms. Matched statistics are unavailable."
    for _, group in df.groupby(["Map", "Cluster"]):
        sets = [set(group.loc[group.Algorithm.eq(a), "Run"]) for a in ALGORITHMS]
        if any(s != sets[0] or len(s) != 10 for s in sets):
            return False, "The saved analysis expects ten matching run identifiers per algorithm and cluster."
    needed_a = ["Scope", "Metric", "N_Blocks", "F_value", "p_GG_Holm_All_Scopes"]
    needed_p = ["Scope", "Metric", "Group1", "Group2", "N_Blocks",
                "Mean_Difference_Group2_minus_Group1", "SE_Difference", "p_Holm_All_Scopes",
                "CI_Family95_Lower_Bonferroni", "CI_Family95_Upper_Bonferroni"]
    if anova is None or pairs is None or not set(needed_a).issubset(anova) or not set(needed_p).issubset(pairs):
        return False, "The matched ANOVA or paired-comparison file is missing or has an unexpected format."
    if anova.duplicated(["Scope", "Metric"]).any() or pairs.duplicated(["Scope", "Metric", "Group1", "Group2"]).any():
        return False, "Duplicate statistical comparisons were found. Rerun the analysis."
    for table, columns in [(anova, needed_a[2:]), (pairs, needed_p[4:])]:
        try:
            if not np.isfinite(table[columns].to_numpy(float)).all():
                return False, "Some statistical estimates are undefined. Inspect the analysis output."
        except (ValueError, TypeError):
            return False, "Statistical estimates must be numeric. Rerun the analysis."
    if not anova.p_GG_Holm_All_Scopes.between(0, 1).all() or not pairs.p_Holm_All_Scopes.between(0, 1).all():
        return False, "Adjusted p-values must be between zero and one."
    scopes = ["Overall"] + sorted(cm.Map.unique().tolist())
    if len(anova) != len(scopes)*3 or len(pairs) != len(scopes)*9:
        return False, "The statistical outputs do not contain every scope, metric and algorithm pair."
    expected_pairs = {(ALGORITHMS[i], ALGORITHMS[j]) for i in range(3) for j in range(i+1, 3)}
    for scope in scopes:
        scoped = cm if scope == "Overall" else cm[cm.Map.eq(scope)]
        for metric in METRICS:
            wide = scoped.pivot(index=["Map", "Cluster"], columns="Algorithm", values=metric)
            ar, pr = scope_rows(anova, scope, metric), scope_rows(pairs, scope, metric)
            if len(ar) != 1 or set(zip(pr.Group1, pr.Group2)) != expected_pairs:
                return False, "Some scope/metric comparisons are missing."
            if ar.iloc[0].N_Blocks != len(wide):
                return False, "ANOVA block counts do not match the data."
            for row in pr.itertuples():
                delta = wide[row.Group2] - wide[row.Group1]
                if row.N_Blocks != len(wide) or not np.allclose(
                        [delta.mean(), delta.std(ddof=1)/np.sqrt(len(delta))],
                        [row.Mean_Difference_Group2_minus_Group1, row.SE_Difference], rtol=1e-7, atol=1e-9):
                    return False, "Saved pairwise differences do not match the data. Rerun the analysis."
                if not row.CI_Family95_Lower_Bonferroni <= row.Mean_Difference_Group2_minus_Group1 <= row.CI_Family95_Upper_Bonferroni:
                    return False, "A saved confidence interval is invalid."
    return True, "Saved cluster averages, comparison counts, differences and standard errors match the raw data."


def comparison_table(pairs, scope, metric):
    rows = []
    for r in scope_rows(pairs, scope, metric).itertuples():
        difference = r.Mean_Difference_Group2_minus_Group1
        significant = r.p_Holm_All_Scopes < ALPHA
        lower = r.Group1 if difference > 0 else r.Group2
        interpretation = f"{LABELS[lower]} significantly lower" if significant else "No significant difference"
        rows.append({"Comparison": f"{LABELS[r.Group2]} − {LABELS[r.Group1]}",
                     f"Mean difference ({UNITS[metric]})": difference,
                     "Adjusted p-value": r.p_Holm_All_Scopes, "Significant?": "Yes" if significant else "No",
                     "Interpretation": interpretation})
    return pd.DataFrame(rows)


def performance_figure(desc, scope, metric, pairs=None):
    part = scope_rows(desc, scope, metric).set_index("Algorithm").reindex(ALGORITHMS)
    fig, ax = plt.subplots(figsize=(8, 5.8), constrained_layout=True)
    for i, a in enumerate(ALGORITHMS):
        row = part.loc[a]
        if pd.isna(row.Mean):
            ax.text(i, 0, "Unavailable", ha="center", rotation=90)
            continue
        sd = row.SD_Across_Cluster_Means
        ax.bar(i, row.Mean, yerr=sd if pd.notna(sd) else None, color=COLORS[a], capsize=5, width=.6)
        ax.annotate(f"{row.Mean:.4f}" if metric == "Elevation" else f"{row.Mean:.2f}",
                    (i, row.Mean + (sd if pd.notna(sd) else 0)), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(range(3), [LABELS[a] for a in ALGORITHMS])
    ax.set_ylabel(f"{metric} ({UNITS[metric]})")
    ax.set_title(f"{scope}: mean {metric.lower()} per cluster")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=.2)
    ax.set_axisbelow(True)
    if pairs is not None:
        pr = scope_rows(pairs, scope, metric)
        top = (part.Mean + part.SD_Across_Cluster_Means.fillna(0)).max()
        step = max(float(top)*.11, .0001)
        for level, row in enumerate(pr.itertuples()):
            i, j = ALGORITHMS.index(row.Group1), ALGORITHMS.index(row.Group2)
            y = top + step*(level+1)
            ax.plot([i, i, j, j], [y, y+step*.15, y+step*.15, y], color="#555", lw=1)
            ax.text((i+j)/2, y+step*.2, f"p adj = {p_label(row.p_Holm_All_Scopes)}", ha="center", fontsize=9)
        ax.set_ylim(top=top+step*(len(pr)+1.8))
    return fig


def differences_figure(pairs, scope, metric):
    part = scope_rows(pairs, scope, metric)
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for i, row in enumerate(part.itertuples()):
        mean = row.Mean_Difference_Group2_minus_Group1
        lo, hi = row.CI_Family95_Lower_Bonferroni, row.CI_Family95_Upper_Bonferroni
        ax.errorbar(mean, i, xerr=[[mean-lo], [hi-mean]], fmt="o", color="#167d9a", capsize=6)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(part)), [f"{LABELS[r.Group2]} − {LABELS[r.Group1]}" for r in part.itertuples()])
    ax.set_ylim(len(part)-.5, -.5)
    ax.set_xlabel(f"Mean difference in {metric.lower()} ({UNITS[metric]})")
    ax.set_title(f"{scope}: matched cluster differences\n95% simultaneous Bonferroni intervals (family of {len(pairs)} comparisons)")
    ax.grid(axis="x", alpha=.2)
    return fig


def render_figure(fig, filename, key):
    st.pyplot(fig, width="stretch")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    st.download_button("Download chart", buffer.getvalue(), file_name=filename, mime="image/png", key=key)
    plt.close(fig)


def finding_rows(desc, pairs, scope, inference_ok):
    findings = []
    for metric in METRICS:
        values = scope_rows(desc, scope, metric).dropna(subset=["Mean"])
        if values.empty:
            continue
        best = values.loc[values.Mean.idxmin()]
        tied = values.loc[np.isclose(values.Mean, best.Mean, rtol=1e-10, atol=1e-12), "Algorithm"].tolist()
        evidence = "Descriptive result only"
        comparisons = []
        if inference_ok and len(tied) == 1:
            pr = scope_rows(pairs, scope, metric)
            wins = []
            for row in pr.itertuples():
                d = row.Mean_Difference_Group2_minus_Group1
                lower = row.Group1 if d > 0 else row.Group2
                if lower == best.Algorithm and row.p_Holm_All_Scopes < ALPHA and d != 0:
                    wins.append(row.Group2 if lower == row.Group1 else row.Group1)
            evidence = "Significantly lower than " + ", ".join(LABELS[a] for a in wins) if wins else "Lowest mean; no significant advantage established"
        for row in values.itertuples():
            if row.Algorithm not in tied and row.Mean != 0:
                percent = 100*(row.Mean-best.Mean)/row.Mean
                comparisons.append(f"{percent:.2f}% below {LABELS[row.Algorithm]}")
        findings.append({"Metric": metric, "Lowest mean algorithm": ", ".join(LABELS[a] for a in tied),
                         "Mean": best.Mean, "Unit": UNITS[metric], "Difference in mean": "; ".join(comparisons),
                         "Statistical evidence": evidence})
    return pd.DataFrame(findings)


st.divider()
st.header("Algorithm Performance Analysis")
st.caption("Experiment averages are separate from the selected route above. Lower values are preferred for all three metrics.")

cluster_means = df.groupby(["Map", "Cluster", "Algorithm"], as_index=False)[METRICS].mean()
derived_desc = build_descriptive(cluster_means)
saved_cm = read_csv_optional(ANALYSIS_FOLDER / "cluster_means.csv")
saved_desc = read_csv_optional(ANALYSIS_FOLDER / "descriptive_statistics.csv")
anova_df = read_csv_optional(ANALYSIS_FOLDER / "matched_anova.csv")
paired_df = read_csv_optional(ANALYSIS_FOLDER / "paired_comparisons.csv")
friedman_df = read_csv_optional(ANALYSIS_FOLDER / "friedman_sensitivity.csv")
inference_ok, validation_note = validate_statistics(cluster_means, saved_cm, anova_df, paired_df)

desc_values = ["N_Blocks", "Mean", "SD_Across_Cluster_Means"]
desc_matches = same_table_values(saved_desc, derived_desc, ["Scope", "Metric", "Algorithm"], desc_values)
descriptive = saved_desc if desc_matches else derived_desc
if saved_desc is not None and not desc_matches:
    st.warning("The saved descriptive statistics differ from the raw data. Charts use freshly calculated cluster averages.")
if not inference_ok:
    st.warning(validation_note + " Descriptive charts remain available; saved significance claims are hidden.")

run_counts = df.groupby(["Map", "Cluster", "Algorithm"]).Run.nunique()
cards = st.columns(4)
cards[0].metric("Maps", df.Map.nunique())
cards[1].metric("Complete matched clusters", int(cluster_means.groupby(["Map", "Cluster"]).Algorithm.nunique().eq(3).sum()))
cards[2].metric("Runs per algorithm / cluster", str(run_counts.iloc[0]) if run_counts.nunique() == 1 else f"{run_counts.min()}–{run_counts.max()}")
cards[3].metric("Raw experiment records", len(df))

c1, c2 = st.columns(2)
selected_metric = c1.selectbox("Performance metric", METRICS, key="analysis_metric")
selected_scope = c2.selectbox("Analysis scope", ["Overall"] + sorted(df.Map.unique().tolist()), key="analysis_scope")
st.caption("Each cluster contributes its mean across runs. Overall results weight all clusters equally; Map1 and Map2 need not contribute equally. Error bars show SD across cluster averages, not run-to-run variability.")

st.subheader("Performance overview")
summary = scope_rows(descriptive, selected_scope, selected_metric).sort_values("Mean")
if not summary.empty:
    minimum = summary.Mean.min()
    leaders = summary.loc[np.isclose(summary.Mean, minimum, rtol=1e-10, atol=1e-12), "Algorithm"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Lowest mean algorithm", ", ".join(LABELS[a] for a in leaders))
    c2.metric(f"Mean {selected_metric.lower()}", f"{minimum:.4f} {UNITS[selected_metric]}")
    c3.metric("Clusters per algorithm", str(int(summary.N_Blocks.iloc[0])) if summary.N_Blocks.nunique() == 1 else "Unequal counts")
render_figure(performance_figure(descriptive, selected_scope, selected_metric, paired_df if inference_ok else None),
              f"performance_{selected_scope}_{selected_metric}.png", "download_performance")
with st.expander("Performance summary data"):
    table = summary.rename(columns={"SD_Across_Cluster_Means": "SD across cluster means", "N_Blocks": "Clusters"})
    display_table(table)
    table_download(summary, f"performance_{selected_scope}_{selected_metric}.csv", "download_summary")

st.subheader("Performance by map")
st.caption("Two bars per algorithm compare map averages descriptively. The matched tests compare algorithms within a map; they do not test Map1 versus Map2.")
fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
map_names = sorted(df.Map.unique().tolist())
width = .75 / len(map_names)
for j, map_name in enumerate(map_names):
    part = scope_rows(descriptive, map_name, selected_metric).set_index("Algorithm").reindex(ALGORITHMS)
    positions = np.arange(3) + (j-(len(map_names)-1)/2)*width
    valid = part.Mean.notna()
    ax.bar(positions[valid], part.loc[valid, "Mean"], width=width,
           yerr=part.loc[valid, "SD_Across_Cluster_Means"].fillna(0), capsize=5, label=map_name)
ax.set_xticks(range(3), [LABELS[a] for a in ALGORITHMS])
ax.set_ylabel(f"{selected_metric} ({UNITS[selected_metric]})")
ax.set_title(f"{selected_metric}: mean ± SD across cluster averages")
ax.set_ylim(bottom=0)
ax.legend()
ax.grid(axis="y", alpha=.2)
ax.set_axisbelow(True)
render_figure(fig, f"by_map_{selected_metric}.png", "download_map_plot")
with st.expander("Map comparison data"):
    map_table = descriptive[descriptive.Scope.ne("Overall") & descriptive.Metric.eq(selected_metric)].copy()
    display_table(map_table)
    table_download(map_table, f"by_map_{selected_metric}.csv", "download_map_table")

st.subheader("Algorithm comparisons")
if inference_ok:
    pair_table = comparison_table(paired_df, selected_scope, selected_metric)
    display_table(pair_table)
    table_download(pair_table, f"comparisons_{selected_scope}_{selected_metric}.csv", "download_pairs")
    st.caption(f"Two-sided paired t-tests; Holm adjustment across all {len(paired_df)} comparisons. Adjusted p < 0.05 indicates a significant difference. A smaller p-value does not indicate a larger benefit.")
    arow = scope_rows(anova_df, selected_scope, selected_metric).iloc[0]
    st.write(f"Matched ANOVA: **adjusted p = {p_label(arow.p_GG_Holm_All_Scopes)}**. "
             + ("Evidence that at least one algorithm mean differs." if arow.p_GG_Holm_All_Scopes < ALPHA else "No overall algorithm difference established by this test."))

    st.subheader("Matched differences")
    render_figure(differences_figure(paired_df, selected_scope, selected_metric),
                  f"matched_differences_{selected_scope}_{selected_metric}.png", "download_differences")
    st.write("The dot is the average paired difference. The line is its confidence interval. "
             "For GA − ACO, a positive value means ACO has the lower cost; a negative value means GA has the lower cost.")
    st.caption("An interval including zero does not establish a difference using that interval. These simultaneous Bonferroni intervals are more conservative than the Holm tests, so borderline results can differ between the chart and table. Filtering does not recalculate the correction family.")
else:
    st.info("Matched comparisons and confidence intervals will appear when the matching analysis outputs are available.")

st.subheader(f"Findings — {selected_scope}")
findings = finding_rows(descriptive, paired_df, selected_scope, inference_ok)
display_table(findings)
st.write("Choose according to the metric that matters for the application. No universal winner is inferred from these three costs. Runtime and battery consumption are not measured in this analysis.")
st.caption("Interpretation assumes identical trees, endpoints and constraints for each matched cluster and reasonably independent clusters and runs. Conclusions describe the tested maps; they do not prove globally shortest routes or generalize to all plantations.")

with st.expander("Detailed statistics and sensitivity checks"):
    if inference_ok:
        st.markdown("**Repeated-measures ANOVA**")
        st.caption(f"Greenhouse–Geisser correction, followed by Holm adjustment across {len(anova_df)} omnibus tests.")
        display_table(scope_rows(anova_df, selected_scope, selected_metric))
        table_download(anova_df, "matched_anova.csv", "download_anova")
        st.markdown("**Paired estimates and confidence intervals**")
        display_table(scope_rows(paired_df, selected_scope, selected_metric))
        table_download(paired_df, "paired_comparisons.csv", "download_all_pairs")
        if friedman_df is not None and {"Scope", "Metric", "p_Friedman_Holm_All_Scopes"}.issubset(friedman_df):
            st.markdown("**Friedman sensitivity analysis**")
            display_table(scope_rows(friedman_df, selected_scope, selected_metric))
            st.caption("Friedman and the Wilcoxon columns are rank-based sensitivity checks; the primary analysis is unchanged.")
    else:
        st.info(validation_note)

with st.expander("Method and data checks"):
    st.write(validation_note)
    st.write("Means and SDs use one ten-run average per map/cluster/algorithm. Cluster counts are not counts of independent plantations. A nonsignificant result does not prove equivalence.")
    st.caption("Consistency checks verify the saved means and paired standard errors, not the physical identity of the routes or every statistical assumption. Keep all output files from the same analysis execution.")
    table_download(cluster_means, "cluster_means.csv", "download_clusters")

st.divider()
st.caption("Final Year Project — Automated Path Planning Framework for Pesticide Spraying in Oil Palm Plantations")

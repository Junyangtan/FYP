import os
import xml.etree.ElementTree as ET

import pandas as pd
import streamlit as st
import folium
import xyzservices.providers as xyz
import matplotlib.pyplot as plt

from streamlit_folium import st_folium


# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="Drone Path Planning Dashboard",
    page_icon="🚁",
    layout="wide"
)


# =====================================================
# PROJECT PATHS
# =====================================================

DASHBOARD_FOLDER = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_FOLDER = os.path.abspath(
    os.path.join(
        DASHBOARD_FOLDER,
        ".."
    )
)

OUTPUT_FOLDER = os.path.join(
    PROJECT_FOLDER,
    "Output"
)


# =====================================================
# ANALYSIS FILES
# =====================================================

DATA_FILE = os.path.join(
    OUTPUT_FOLDER,
    "analysis_data1.csv"
)

ANALYSIS_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "Analysis"
)

ANOVA_FILE = os.path.join(
    ANALYSIS_FOLDER,
    "anova_results.csv"
)

TUKEY_FILE = os.path.join(
    ANALYSIS_FOLDER,
    "tukey_results.csv"
)


# =====================================================
# MAP FILES
# =====================================================

MAP_FILES = {

    "Map1": os.path.join(
        PROJECT_FOLDER,
        "Map1_3D_map.csv"
    ),

    "Map2": os.path.join(
        PROJECT_FOLDER,
        "Map2_3D_map.csv"
    )

}


# =====================================================
# ROAD FILES
# =====================================================

ROAD_FILES = {

    "Map1": os.path.join(
        PROJECT_FOLDER,
        "road_map1.kml"
    ),

    "Map2": os.path.join(
        PROJECT_FOLDER,
        "road_map2.kml"
    )

}


# =====================================================
# BOUNDARY FILES
# =====================================================

BOUNDARY_FILES = {

    "Map1": os.path.join(
        PROJECT_FOLDER,
        "boundary_map1.kml"
    ),

    "Map2": os.path.join(
        PROJECT_FOLDER,
        "boundary_map2.kml"
    )

}


# =====================================================
# ALGORITHM OUTPUT FOLDERS
# =====================================================

ALGORITHM_OUTPUT_FOLDERS = {

    "ACO": os.path.join(
        OUTPUT_FOLDER,
        "ACO_3D"
    ),

    "GA": os.path.join(
        OUTPUT_FOLDER,
        "GA_3D"
    ),

    "NSGAII": os.path.join(
        OUTPUT_FOLDER,
        "NSGAII_3D"
    )

}


# =====================================================
# LOAD ANALYSIS DATA
# =====================================================

if not os.path.exists(DATA_FILE):

    st.error(
        "analysis_data1.csv could not be found."
    )

    st.code(
        DATA_FILE
    )

    st.stop()


df = pd.read_csv(
    DATA_FILE
)


# =====================================================
# LOAD ANOVA RESULTS
# =====================================================

if os.path.exists(
    ANOVA_FILE
):

    anova_df = pd.read_csv(
        ANOVA_FILE
    )

else:

    anova_df = None


# =====================================================
# LOAD TUKEY RESULTS
# =====================================================

if os.path.exists(
    TUKEY_FILE
):

    tukey_df = pd.read_csv(
        TUKEY_FILE
    )

else:

    tukey_df = None


# =====================================================
# LOAD TREE MAP
# =====================================================

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

    return pd.read_csv(
        map_file
    )


# =====================================================
# GET LATEST 20 FILES
# =====================================================

def get_latest_20_files(
    folder,
    extension
):

    if not os.path.exists(
        folder
    ):
        return []


    files = [

        file

        for file in os.listdir(
            folder
        )

        if file.lower().endswith(
            extension.lower()
        )

    ]


    # Timestamp filenames sort chronologically
    files.sort()


    # Keep latest 20 files only
    files = files[
        -20:
    ]


    return files


# =====================================================
# GET FILE FOR MAP AND RUN
# =====================================================

def get_map_run_file(
    folder,
    extension,
    map_name,
    run_number
):

    files = get_latest_20_files(
        folder,
        extension
    )


    if len(files) < 20:
        return None


    # Latest 20 files:
    #
    # File 1-10  = Map1 Run1-Run10
    # File 11-20 = Map2 Run1-Run10

    if map_name == "Map1":

        map_files = files[
            0:10
        ]


    elif map_name == "Map2":

        map_files = files[
            10:20
        ]


    else:

        return None


    index = (
        int(run_number)
        - 1
    )


    if (
        index < 0
        or
        index >= len(map_files)
    ):

        return None


    return os.path.join(
        folder,
        map_files[
            index
        ]
    )


# =====================================================
# GET KML FILE FOR SELECTED RUN
# =====================================================

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

    tree = ET.parse(
        kml_file
    )

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


    except ET.ParseError:

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


    except ET.ParseError:

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

    tree = ET.parse(
        kml_file
    )

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
# ROUTE PERFORMANCE
# =====================================================

st.subheader(
    "Route Performance"
)


if (
    metric_file is not None
    and
    os.path.exists(
        metric_file
    )
):

    metric_data = pd.read_csv(
        metric_file
    )


    cluster_metric = metric_data[
        metric_data[
            "Cluster"
        ]
        ==
        int(
            selected_cluster
        )
    ]


    if not cluster_metric.empty:

        row = cluster_metric.iloc[
            0
        ]


        trees = int(
            row[
                "Trees"
            ]
        )

        distance = float(
            row[
                "Distance"
            ]
        )

        turning = float(
            row[
                "Turning"
            ]
        )

        elevation = float(
            row[
                "Elevation"
            ]
        )


        col1, col2, col3, col4 = (
            st.columns(
                4
            )
        )


        with col1:

            st.metric(
                "Trees",
                trees
            )


        with col2:

            st.metric(
                "Distance",
                f"{distance:.2f} m"
            )


        with col3:

            st.metric(
                "Turning",
                f"{turning:.3f} rad"
            )


        with col4:

            st.metric(
                "Elevation",
                f"{elevation:.3f} m"
            )


        st.caption(
            f"Performance for "
            f"Run {int(selected_run)} "
            f"and Cluster "
            f"{int(selected_cluster)}."
        )


    else:

        st.warning(
            f"No metric result found "
            f"for Cluster "
            f"{int(selected_cluster)}."
        )


else:

    st.warning(
        "Metric file could not be loaded "
        "for the selected run."
    )


st.divider()


# =====================================================
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
        "The dashboard requires at least "
        "20 KML files in this folder. "
        "The latest 20 files are used: "
        "first 10 for Map1 and next 10 for Map2."
    )


# =====================================================
# ALGORITHM PERFORMANCE ANALYSIS
# =====================================================

st.divider()

st.header(
    "Algorithm Performance Analysis"
)

st.write(
    "Comparison of ACO, GA and NSGA-II "
    "based on Distance, Turning and Elevation."
)


# =====================================================
# METRIC SELECTION
# =====================================================

selected_metric = (
    st.selectbox(
        "Select Performance Metric",
        [
            "Distance",
            "Turning",
            "Elevation"
        ],
        key="analysis_metric"
    )
)


# =====================================================
# METRIC UNIT
# =====================================================

if selected_metric == "Distance":

    metric_unit = "m"

elif selected_metric == "Turning":

    metric_unit = "rad"

else:

    metric_unit = "m"


algorithm_order = [
    "ACO",
    "GA",
    "NSGAII"
]

algorithm_labels = [
    "ACO",
    "GA",
    "NSGA-II"
]

map_order = [
    "Map1",
    "Map2"
]


# =====================================================
# COMPARISON BETWEEN MAPS
# =====================================================

st.subheader(
    "Comparison Between Maps"
)

st.write(
    f"Comparison of {selected_metric.lower()} "
    f"performance between Map1 and Map2 "
    f"for each algorithm."
)


map_means = {
    "Map1": [],
    "Map2": []
}

map_stds = {
    "Map1": [],
    "Map2": []
}


for map_name in map_order:

    for algorithm in algorithm_order:

        values = df[
            (
                df[
                    "Map"
                ]
                == map_name
            )
            &
            (
                df[
                    "Algorithm"
                ]
                == algorithm
            )
        ][
            selected_metric
        ].dropna().astype(
            float
        )


        if len(values) > 0:

            map_means[
                map_name
            ].append(
                values.mean()
            )

            map_stds[
                map_name
            ].append(
                values.std(
                    ddof=1
                )
            )

        else:

            map_means[
                map_name
            ].append(
                0
            )

            map_stds[
                map_name
            ].append(
                0
            )


# =====================================================
# MAP COMPARISON GRAPH
# =====================================================

fig_map, ax_map = plt.subplots(
    figsize=(
        10,
        6
    )
)


x = list(
    range(
        len(
            algorithm_order
        )
    )
)

width = 0.35


x_map1 = [
    value - width / 2
    for value in x
]

x_map2 = [
    value + width / 2
    for value in x
]


ax_map.bar(
    x_map1,
    map_means[
        "Map1"
    ],
    width,
    yerr=map_stds[
        "Map1"
    ],
    capsize=6,
    label="Map1"
)


ax_map.bar(
    x_map2,
    map_means[
        "Map2"
    ],
    width,
    yerr=map_stds[
        "Map2"
    ],
    capsize=6,
    label="Map2"
)


ax_map.set_xticks(
    x
)

ax_map.set_xticklabels(
    algorithm_labels
)

ax_map.set_xlabel(
    "Algorithm"
)

ax_map.set_ylabel(
    f"{selected_metric} "
    f"({metric_unit})"
)

ax_map.set_title(
    f"{selected_metric} "
    f"Comparison by Map"
)

ax_map.legend()

ax_map.grid(
    axis="y",
    alpha=0.25
)

fig_map.tight_layout()


st.pyplot(
    fig_map,
    use_container_width=True
)

plt.close(
    fig_map
)


# =====================================================
# MAP COMPARISON TABLE
# =====================================================

map_table_data = []


for i, algorithm in enumerate(
    algorithm_order
):

    map_table_data.append(
        {
            "Algorithm": (
                "NSGA-II"
                if algorithm == "NSGAII"
                else algorithm
            ),

            "Map1 Mean": (
                map_means[
                    "Map1"
                ][i]
            ),

            "Map2 Mean": (
                map_means[
                    "Map2"
                ][i]
            ),

            "Map1 SD": (
                map_stds[
                    "Map1"
                ][i]
            ),

            "Map2 SD": (
                map_stds[
                    "Map2"
                ][i]
            )
        }
    )


map_comparison_df = (
    pd.DataFrame(
        map_table_data
    )
)


with st.expander(
    "Show Map Comparison Data"
):

    st.dataframe(
        map_comparison_df,
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# OVERALL ALGORITHM COMPARISON
# =====================================================

st.divider()

st.subheader(
    "Overall Algorithm Comparison"
)

st.write(
    f"Overall {selected_metric.lower()} "
    f"performance of ACO, GA and NSGA-II "
    f"across both maps."
)


overall_means = []

overall_stds = []

overall_counts = []


for algorithm in algorithm_order:

    values = df[
        df[
            "Algorithm"
        ]
        == algorithm
    ][
        selected_metric
    ].dropna().astype(
        float
    )


    if len(values) > 0:

        overall_means.append(
            values.mean()
        )

        overall_stds.append(
            values.std(
                ddof=1
            )
        )

        overall_counts.append(
            len(
                values
            )
        )

    else:

        overall_means.append(
            0
        )

        overall_stds.append(
            0
        )

        overall_counts.append(
            0
        )


# =====================================================
# OVERALL GRAPH
# =====================================================

fig_overall, ax_overall = plt.subplots(
    figsize=(
        9,
        6
    )
)


bars = ax_overall.bar(
    algorithm_labels,
    overall_means,
    yerr=overall_stds,
    capsize=7
)


ax_overall.set_xlabel(
    "Algorithm"
)

ax_overall.set_ylabel(
    f"{selected_metric} "
    f"({metric_unit})"
)

ax_overall.set_title(
    f"Overall "
    f"{selected_metric} "
    f"Comparison"
)

ax_overall.grid(
    axis="y",
    alpha=0.25
)


# =====================================================
# VALUE LABELS
# =====================================================

for bar, value in zip(
    bars,
    overall_means
):

    ax_overall.text(
        (
            bar.get_x()
            +
            bar.get_width() / 2
        ),

        bar.get_height(),

        f"{value:.2f}",

        ha="center",
        va="bottom"
    )


fig_overall.tight_layout()


st.pyplot(
    fig_overall,
    use_container_width=True
)

plt.close(
    fig_overall
)


# =====================================================
# OVERALL SUMMARY
# =====================================================

if overall_means:

    best_index = (
        overall_means.index(
            min(
                overall_means
            )
        )
    )


    best_algorithm = (
        algorithm_labels[
            best_index
        ]
    )


    best_value = (
        overall_means[
            best_index
        ]
    )


    col1, col2, col3 = (
        st.columns(
            3
        )
    )


    with col1:

        st.metric(
            "Lowest Mean Algorithm",
            best_algorithm
        )


    with col2:

        st.metric(
            f"Lowest Mean {selected_metric}",
            f"{best_value:.3f} "
            f"{metric_unit}"
        )


    with col3:

        st.metric(
            "Total Data Points",
            len(
                df[
                    selected_metric
                ].dropna()
            )
        )


# =====================================================
# OVERALL TABLE
# =====================================================

overall_table = (
    pd.DataFrame(
        {
            "Algorithm": (
                algorithm_labels
            ),

            f"Mean {selected_metric}": (
                overall_means
            ),

            "Standard Deviation": (
                overall_stds
            ),

            "Data Points": (
                overall_counts
            )
        }
    )
)


with st.expander(
    "Show Overall Comparison Data"
):

    st.dataframe(
        overall_table,
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# STATISTICAL ANALYSIS
# =====================================================

st.divider()

st.subheader(
    "Statistical Analysis"
)

st.write(
    "ANOVA determines whether a statistically "
    "significant difference exists between the "
    "three path planning algorithms. "
    "Tukey HSD identifies which specific "
    "algorithm pairs are significantly different."
)


# =====================================================
# ANOVA RESULTS
# =====================================================

st.markdown(
    "### ANOVA Results"
)


if anova_df is not None:

    if (
        "Metric"
        in anova_df.columns
    ):

        selected_anova = (
            anova_df[
                anova_df[
                    "Metric"
                ]
                .astype(
                    str
                )
                .str.strip()
                .str.lower()
                ==
                selected_metric.lower()
            ]
        )

    else:

        selected_anova = (
            anova_df
        )


    if not selected_anova.empty:

        st.dataframe(
            selected_anova,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            f"No ANOVA result found "
            f"for {selected_metric}."
        )


else:

    st.warning(
        "anova_results.csv "
        "could not be loaded."
    )


# =====================================================
# TUKEY HSD RESULTS
# =====================================================

st.markdown(
    "### Tukey HSD Results"
)


if tukey_df is not None:

    if (
        "Metric"
        in tukey_df.columns
    ):

        selected_tukey = (
            tukey_df[
                tukey_df[
                    "Metric"
                ]
                .astype(
                    str
                )
                .str.strip()
                .str.lower()
                ==
                selected_metric.lower()
            ]
        )

    else:

        selected_tukey = (
            tukey_df
        )


    if not selected_tukey.empty:

        st.dataframe(
            selected_tukey,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            f"No Tukey HSD result found "
            f"for {selected_metric}."
        )


else:

    st.warning(
        "tukey_results.csv "
        "could not be loaded."
    )


# =====================================================
# ANALYSIS FILE INFORMATION
# =====================================================

with st.expander(
    "Analysis File Information"
):

    st.write(
        "**Raw Analysis Data**"
    )

    st.code(
        DATA_FILE
    )

    st.success(
        f"{len(df)} rows loaded "
        f"from analysis_data1.csv."
    )


    st.write(
        "**ANOVA Results**"
    )

    st.code(
        ANOVA_FILE
    )


    if anova_df is not None:

        st.success(
            f"{len(anova_df)} "
            f"ANOVA result rows loaded."
        )

    else:

        st.warning(
            "ANOVA file not found."
        )


    st.write(
        "**Tukey HSD Results**"
    )

    st.code(
        TUKEY_FILE
    )


    if tukey_df is not None:

        st.success(
            f"{len(tukey_df)} "
            f"Tukey result rows loaded."
        )

    else:

        st.warning(
            "Tukey file not found."
        )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "Final Year Project - 2026S1-3580 | "
    "An Automated Path Planning Framework "
    "for Pesticide Spraying in "
    "Oil Palm Plantations"
)
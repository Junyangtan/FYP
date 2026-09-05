import numpy as np
import csv
import os
import math
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


# ============================================================
# CONFIGURATION
# ============================================================

# Keep this the SAME as the other algorithms for fair analysis.
NUM_RUNS = 2

OUTPUT_FOLDER = os.path.join("Output", "ACO_3D")
ROUTES_FOLDER = os.path.join(OUTPUT_FOLDER, "Routes")
METRICS_FOLDER = os.path.join(OUTPUT_FOLDER, "Metrics")
KML_FOLDER = os.path.join(OUTPUT_FOLDER, "KML")
SUMMARY_FOLDER = os.path.join(OUTPUT_FOLDER, "Summary")
BASES_FOLDER = os.path.join(OUTPUT_FOLDER, "Bases")

for folder in [
    ROUTES_FOLDER,
    METRICS_FOLDER,
    KML_FOLDER,
    SUMMARY_FOLDER,
    BASES_FOLDER,
]:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# ACO PARAMETERS
# ============================================================

NUM_ANTS = 30
ITERATIONS = 80
ALPHA = 1.0
BETA = 3.0
EVAPORATION = 0.5

# Weighted single-objective ACO cost.
DISTANCE_WEIGHT = 1.0
TURN_WEIGHT = 10.0
ELEVATION_COST_WEIGHT = 2.0

# Maximum trees covered in one cluster.
MAX_TREES = 80

# Warning only. No trees are discarded.
MIN_CLUSTER_SIZE = 5

# Same 3D K-means elevation weighting as the NSGA-II template.
ELEVATION_WEIGHT = 5.0


# ============================================================
# ROAD / DRONE PARAMETERS
# ============================================================

# Generated road is 5 m outside the plantation/tree boundary.
ROAD_OFFSET = 5.0

# Drone height above each tree/terrain altitude in the 3D KML.
DRONE_OFFSET = 3.0

ALGORITHM_NAME = "ACO_3D_v2"


# ============================================================
# ANALYSIS DATA SETTINGS
# ============================================================

# Change this when running the second plantation map.
MAP_NAME = "Map1"

# Keep this simple and consistent across all experiment files.
ANALYSIS_ALGORITHM = "ACO"

# All algorithms append cluster results to the same analysis file.
ANALYSIS_DATA_FILE = os.path.join(
    "Output",
    "analysis_data.csv"
)

# Existing take-off/reference point.
# It is NOT included in the optimized coverage distance.
# It is used only to choose where Cluster 1 starts around the road.
MISSION_REFERENCE_LAT = 2.70425
MISSION_REFERENCE_LON = 101.633375

R = 6371000.0


# ============================================================
# COORDINATE CONVERSION
# ============================================================

def latlon_to_xy(lat, lon, lat0, lon0):
    """
    Convert latitude/longitude to local XY in metres.

    Latitude/longitude differences must be converted from degrees
    to radians before multiplication by Earth radius.
    """
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    lat0_rad = np.radians(lat0)

    x = dlon * np.cos(lat0_rad) * R
    y = dlat * R

    return float(x), float(y)


def xy_to_latlon(x, y, lat0, lon0):
    """
    Inverse local XY conversion for plantation-scale maps.
    """
    lat = lat0 + np.degrees(y / R)

    denom = R * np.cos(np.radians(lat0))
    lon = lon0 + np.degrees(x / denom)

    return float(lat), float(lon)


def drone_point(tree):
    """
    tree = (x, y, altitude, lat, lon)
    """
    return np.array(
        [tree[0], tree[1], tree[2] + DRONE_OFFSET],
        dtype=float
    )


# ============================================================
# COVERAGE METRICS
# ============================================================

def path_distance(order):
    """
    SUPERVISOR REQUIREMENT:
    Count ONLY:
        first oil palm -> ... -> last oil palm

    Do NOT count:
        road/base -> first palm
        last palm -> road/base
        original drone take-off point -> cluster
    """
    if len(order) < 2:
        return 0.0

    pts = [drone_point(trees[i]) for i in order]

    return float(
        sum(
            np.linalg.norm(pts[i + 1] - pts[i])
            for i in range(len(pts) - 1)
        )
    )


def turn_penalty(order):
    """
    Turning penalty is calculated only inside the oil-palm coverage route.
    Road/base access legs are excluded.
    """
    if len(order) < 3:
        return 0.0

    pts = [drone_point(trees[i]) for i in order]

    penalty = 0.0

    for i in range(1, len(pts) - 1):
        A = pts[i - 1]
        B = pts[i]
        C = pts[i + 1]

        BA = A - B
        BC = C - B

        denom = np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-9
        cosine = np.dot(BA, BC) / denom

        penalty += np.arccos(
            np.clip(cosine, -1.0, 1.0)
        )

    return float(penalty)


def elevation_cost(order):
    """
    Elevation cost is calculated only from first palm to last palm.
    """
    if len(order) < 2:
        return 0.0

    return float(
        sum(
            abs(trees[order[i + 1]][2] - trees[order[i]][2])
            for i in range(len(order) - 1)
        )
    )


def total_cost(order):
    """
    ACO remains a weighted single-objective optimizer.

    IMPORTANT:
    This cost uses the same palm-to-palm-only distance, turning,
    and elevation definitions used for the analysis outputs.
    """
    return float(
        DISTANCE_WEIGHT * path_distance(order)
        + TURN_WEIGHT * turn_penalty(order)
        + ELEVATION_COST_WEIGHT * elevation_cost(order)
    )


def access_distance_2d(base_xy, tree_id):
    """
    Access distance is stored separately for reference.
    It is NOT included in ACO coverage distance or ACO total cost.
    """
    tree_xy = np.array(
        [trees[tree_id][0], trees[tree_id][1]],
        dtype=float
    )

    return float(
        np.linalg.norm(
            np.asarray(base_xy, dtype=float) - tree_xy
        )
    )


# ============================================================
# PLANTATION BOUNDARY + 5 m OFFSET ROAD
# ============================================================

def cross_2d(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def convex_hull(points):
    """
    Build a convex tree-boundary polygon from XY tree positions.

    This does NOT move any tree or modify 3D_map.csv.
    """
    pts = sorted(
        set(
            (float(p[0]), float(p[1]))
            for p in points
        )
    )

    if len(pts) < 3:
        raise ValueError(
            "At least 3 non-collinear tree points are required "
            "to create the plantation boundary."
        )

    def orientation(o, a, b):
        return (
            (a[0] - o[0]) * (b[1] - o[1])
            -
            (a[1] - o[1]) * (b[0] - o[0])
        )

    lower = []
    for p in pts:
        while (
            len(lower) >= 2
            and orientation(lower[-2], lower[-1], p) <= 0
        ):
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while (
            len(upper) >= 2
            and orientation(upper[-2], upper[-1], p) <= 0
        ):
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]

    if len(hull) < 3:
        raise ValueError("Tree boundary is degenerate.")

    return [
        np.array(p, dtype=float)
        for p in hull
    ]


def polygon_signed_area(poly):
    area2 = 0.0

    for i in range(len(poly)):
        p = poly[i]
        q = poly[(i + 1) % len(poly)]
        area2 += p[0] * q[1] - q[0] * p[1]

    return 0.5 * area2


def line_intersection(p, r, q, s):
    denominator = cross_2d(r, s)

    if abs(denominator) < 1e-12:
        return None

    t = cross_2d(q - p, s) / denominator

    return p + t * r


def offset_convex_polygon(poly, offset):
    """
    Create an outward offset of a convex polygon.

    The returned polygon is the generated road centreline.
    """
    poly = [
        np.array(p, dtype=float)
        for p in poly
    ]

    if polygon_signed_area(poly) < 0:
        poly.reverse()

    n = len(poly)
    road = []

    for i in range(n):
        prev_v = poly[(i - 1) % n]
        curr_v = poly[i]
        next_v = poly[(i + 1) % n]

        prev_edge = curr_v - prev_v
        next_edge = next_v - curr_v

        prev_len = np.linalg.norm(prev_edge)
        next_len = np.linalg.norm(next_edge)

        if prev_len < 1e-12 or next_len < 1e-12:
            raise ValueError("Boundary contains duplicate vertices.")

        prev_normal = np.array(
            [prev_edge[1], -prev_edge[0]],
            dtype=float
        ) / prev_len

        next_normal = np.array(
            [next_edge[1], -next_edge[0]],
            dtype=float
        ) / next_len

        p1 = curr_v + offset * prev_normal
        p2 = curr_v + offset * next_normal

        intersection = line_intersection(
            p1,
            prev_edge,
            p2,
            next_edge
        )

        if intersection is None:
            normal = prev_normal + next_normal

            if np.linalg.norm(normal) < 1e-12:
                normal = prev_normal

            normal = normal / np.linalg.norm(normal)
            intersection = curr_v + offset * normal

        road.append(
            np.array(intersection, dtype=float)
        )

    return road


def polyline_segment_lengths(poly):
    return np.array(
        [
            np.linalg.norm(
                poly[(i + 1) % len(poly)] - poly[i]
            )
            for i in range(len(poly))
        ],
        dtype=float
    )


def road_total_length(road):
    return float(
        np.sum(
            polyline_segment_lengths(road)
        )
    )


def project_point_to_road(point, road):
    """
    Find the closest point anywhere on the closed generated road.

    Returns:
        projected_xy
        distance_along_road_s
        perpendicular_distance
    """
    point = np.asarray(point, dtype=float)

    lengths = polyline_segment_lengths(road)

    best_q = None
    best_s = None
    best_distance = float("inf")

    accumulated = 0.0

    for i in range(len(road)):
        A = road[i]
        B = road[(i + 1) % len(road)]

        AB = B - A
        denom = np.dot(AB, AB)

        if denom < 1e-12:
            accumulated += lengths[i]
            continue

        t = np.dot(point - A, AB) / denom
        t = np.clip(t, 0.0, 1.0)

        q = A + t * AB
        d = np.linalg.norm(point - q)

        if d < best_distance:
            best_distance = float(d)
            best_q = q
            best_s = float(
                accumulated + t * lengths[i]
            )

        accumulated += lengths[i]

    return (
        np.array(best_q, dtype=float),
        best_s,
        best_distance
    )


def interpolate_road(road, s):
    """
    Return XY point at distance s along the closed road.
    """
    lengths = polyline_segment_lengths(road)
    total = float(np.sum(lengths))

    s = float(s % total)

    accumulated = 0.0

    for i, seg_len in enumerate(lengths):
        if s <= accumulated + seg_len or i == len(lengths) - 1:
            if seg_len < 1e-12:
                return np.array(road[i], dtype=float)

            t = (s - accumulated) / seg_len

            return (
                road[i]
                +
                t * (
                    road[(i + 1) % len(road)]
                    -
                    road[i]
                )
            )

        accumulated += seg_len

    return np.array(road[0], dtype=float)


def circular_s_distance(a, b, total):
    delta = abs(a - b) % total
    return min(delta, total - delta)


# ============================================================
# CLUSTERING
# ============================================================

def make_capacity_safe_clusters():
    """
    Same clustering rule as the NSGA-II template.

    Start with the minimum K needed for MAX_TREES capacity.
    If K-means still creates a cluster larger than MAX_TREES,
    increase K and run again.

    No trees are discarded.
    """
    coords = np.array(
        [
            (
                t[0],
                t[1],
                t[2] * ELEVATION_WEIGHT
            )
            for t in trees
        ],
        dtype=float
    )

    N = len(coords)

    if N == 0:
        raise ValueError("No trees loaded.")

    k = max(
        1,
        int(math.ceil(N / MAX_TREES))
    )

    while k <= N:
        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            random_state=0
        ).fit(coords)

        clusters = [
            []
            for _ in range(k)
        ]

        for tree_id, label in enumerate(kmeans.labels_):
            clusters[int(label)].append(tree_id)

        max_size = max(
            len(c)
            for c in clusters
        )

        if max_size <= MAX_TREES:
            return clusters

        k += 1

    raise RuntimeError(
        "Unable to produce capacity-safe clusters."
    )


def cluster_centroid(cluster):
    return np.mean(
        np.array(
            [
                [trees[i][0], trees[i][1]]
                for i in cluster
            ],
            dtype=float
        ),
        axis=0
    )


def order_clusters_and_generate_shared_bases(clusters, road):
    """
    Same cluster/base logic as the NSGA-II template.

    1. Project each cluster centroid onto the road.
    2. Order clusters by position along the road.
    3. Generate transition/base points halfway between adjacent
       cluster road projections.
    4. Each transition point is shared:
           end base of Cluster i
           =
           start base of Cluster i+1
    """
    total = road_total_length(road)

    cluster_info = []

    for cluster in clusters:
        centroid = cluster_centroid(cluster)

        projected, s, distance_to_road = project_point_to_road(
            centroid,
            road
        )

        cluster_info.append(
            {
                "cluster": cluster,
                "centroid": centroid,
                "road_projection": projected,
                "anchor_s": s,
                "centroid_to_road": distance_to_road,
            }
        )

    cluster_info.sort(
        key=lambda item: item["anchor_s"]
    )

    n = len(cluster_info)

    if n == 1:
        only_s = cluster_info[0]["anchor_s"]
        boundaries_s = [only_s]

    else:
        anchors = [
            item["anchor_s"]
            for item in cluster_info
        ]

        boundaries_s = []

        previous_unwrapped = anchors[-1] - total
        boundaries_s.append(
            (
                previous_unwrapped + anchors[0]
            ) / 2.0 % total
        )

        for i in range(1, n):
            boundaries_s.append(
                (
                    anchors[i - 1] + anchors[i]
                ) / 2.0
            )

    mission_xy = np.array(
        latlon_to_xy(
            MISSION_REFERENCE_LAT,
            MISSION_REFERENCE_LON,
            lat0,
            lon0
        ),
        dtype=float
    )

    _, mission_s, _ = project_point_to_road(
        mission_xy,
        road
    )

    start_index = min(
        range(n),
        key=lambda i: circular_s_distance(
            boundaries_s[i],
            mission_s,
            total
        )
    )

    cluster_info = (
        cluster_info[start_index:]
        +
        cluster_info[:start_index]
    )

    boundaries_s = (
        boundaries_s[start_index:]
        +
        boundaries_s[:start_index]
    )

    bases = [
        interpolate_road(road, s)
        for s in boundaries_s
    ]

    ordered_clusters = [
        item["cluster"]
        for item in cluster_info
    ]

    return ordered_clusters, bases, cluster_info


# ============================================================
# START / END PALM SELECTION
# ============================================================

def choose_endpoint_trees(cluster, start_base_xy, end_base_xy):
    """
    Start tree = closest tree in the cluster to the start road base.
    End tree   = closest DIFFERENT tree to the end road base.
    """
    if len(cluster) < 2:
        raise ValueError(
            "A route cluster requires at least 2 trees."
        )

    start_base_xy = np.asarray(
        start_base_xy,
        dtype=float
    )

    end_base_xy = np.asarray(
        end_base_xy,
        dtype=float
    )

    start_tree = min(
        cluster,
        key=lambda i: np.linalg.norm(
            np.array(
                [trees[i][0], trees[i][1]]
            )
            -
            start_base_xy
        )
    )

    remaining = [
        i
        for i in cluster
        if i != start_tree
    ]

    end_tree = min(
        remaining,
        key=lambda i: np.linalg.norm(
            np.array(
                [trees[i][0], trees[i][1]]
            )
            -
            end_base_xy
        )
    )

    return start_tree, end_tree


# ============================================================
# ACO SOLVER FOR ONE CLUSTER
# ============================================================

def solve_cluster(cluster, start_tree, end_tree):
    """
    ACO optimizes only the middle palms.

    The first and last palms are fixed using the same shared-road-base
    rule as the NSGA-II template.

    The returned route is:
        start_tree -> optimized middle trees -> end_tree
    """
    middle_trees = [
        tree_id
        for tree_id in cluster
        if tree_id not in (start_tree, end_tree)
    ]

    # Nothing to optimize.
    if len(middle_trees) == 0:
        route = [start_tree, end_tree]
        return route, total_cost(route)

    if len(middle_trees) == 1:
        route = [start_tree, middle_trees[0], end_tree]
        return route, total_cost(route)

    n = len(middle_trees)

    # Pheromone matrix includes:
    # 0 ... n-1 : middle trees
    # n         : fixed start tree
    # n+1       : fixed end tree
    start_node = n
    end_node = n + 1
    pheromone = np.ones((n + 2, n + 2), dtype=float)

    def tree_id_from_node(node):
        if node == start_node:
            return start_tree
        if node == end_node:
            return end_tree
        return middle_trees[node]

    def spatial_distance(node_a, node_b):
        """
        Correct ACO heuristic distance using actual 3D tree positions.
        """
        a = drone_point(trees[tree_id_from_node(node_a)])
        b = drone_point(trees[tree_id_from_node(node_b)])

        return float(
            np.linalg.norm(a - b) + 1e-9
        )

    best_route = None
    best_cost = float("inf")

    for _ in range(ITERATIONS):
        all_ant_results = []

        for _ in range(NUM_ANTS):
            unvisited = list(range(n))
            current = start_node
            route_nodes = []

            while unvisited:
                desirabilities = []

                for candidate in unvisited:
                    tau = pheromone[current, candidate] ** ALPHA
                    eta = (1.0 / spatial_distance(current, candidate)) ** BETA
                    desirabilities.append(tau * eta)

                desirabilities = np.asarray(
                    desirabilities,
                    dtype=float
                )

                total_desirability = float(
                    np.sum(desirabilities)
                )

                if (
                    not np.isfinite(total_desirability)
                    or total_desirability <= 0.0
                ):
                    probabilities = np.ones(
                        len(unvisited),
                        dtype=float
                    ) / len(unvisited)
                else:
                    probabilities = (
                        desirabilities / total_desirability
                    )

                next_node = int(
                    np.random.choice(
                        unvisited,
                        p=probabilities
                    )
                )

                route_nodes.append(next_node)
                unvisited.remove(next_node)
                current = next_node

            real_route = (
                [start_tree]
                +
                [middle_trees[node] for node in route_nodes]
                +
                [end_tree]
            )

            cost = total_cost(real_route)

            all_ant_results.append(
                (route_nodes, cost)
            )

            if cost < best_cost:
                best_cost = cost
                best_route = list(real_route)

        # Evaporation.
        pheromone *= (1.0 - EVAPORATION)

        # Deposit pheromone along every complete ant route,
        # including the fixed start and end connections.
        for route_nodes, cost in all_ant_results:
            node_route = (
                [start_node]
                +
                route_nodes
                +
                [end_node]
            )

            deposit = 1.0 / (cost + 1e-9)

            for i in range(len(node_route) - 1):
                a = node_route[i]
                b = node_route[i + 1]

                pheromone[a, b] += deposit
                pheromone[b, a] += deposit

    return best_route, float(best_cost)

# ============================================================
# FIND NEXT ANALYSIS RUN NUMBER
# ============================================================

def get_next_run_number():
    """
    Find the next cumulative run number for the current
    Map + Algorithm combination.

    Example:
        Existing:
        Map1, ACO, Run 1
        Map1, ACO, Run 2
        Map1, ACO, Run 3

        Next execution starts from Run 4.

    Run numbering is independent for:
        - each map
        - each algorithm
    """

    if not os.path.isfile(ANALYSIS_DATA_FILE):
        return 1

    max_run = 0

    try:
        with open(
            ANALYSIS_DATA_FILE,
            "r",
            newline="",
            encoding="utf-8"
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:

                if (
                    row.get("Map") == MAP_NAME
                    and
                    row.get("Algorithm") == ANALYSIS_ALGORITHM
                ):

                    try:
                        run_value = int(row["Run"])

                        if run_value > max_run:
                            max_run = run_value

                    except (ValueError, TypeError):
                        pass

    except Exception as e:
        print(
            f"WARNING: Could not read previous run numbers: {e}"
        )

    return max_run + 1

# ============================================================
# LOAD 3D MAP
# ============================================================

base_dir = os.path.dirname(
    os.path.abspath(__file__)
)

candidate_paths = [
    os.path.join(
        base_dir,
        "..",
        "3D_map.csv"
    ),
    os.path.join(
        base_dir,
        "3D_map.csv"
    ),
]

file_path = None

for candidate in candidate_paths:
    if os.path.isfile(candidate):
        file_path = candidate
        break

if file_path is None:
    raise FileNotFoundError(
        "3D_map.csv was not found. "
        "Place it either beside this script or one folder above it."
    )

gps_points = []

with open(file_path, newline="") as f:
    reader = csv.DictReader(f)

    required = {
        "lat",
        "lon",
        "alt"
    }

    if not required.issubset(
        set(reader.fieldnames or [])
    ):
        raise ValueError(
            "3D_map.csv must contain columns: lat, lon, alt"
        )

    for row in reader:
        gps_points.append(
            (
                float(row["lat"]),
                float(row["lon"]),
                float(row["alt"])
            )
        )

if len(gps_points) < 3:
    raise ValueError(
        "At least 3 tree points are required."
    )

lat0, lon0, _ = gps_points[0]

trees = []

for lat, lon, alt in gps_points:
    x, y = latlon_to_xy(
        lat,
        lon,
        lat0,
        lon0
    )

    trees.append(
        (
            x,
            y,
            alt,
            lat,
            lon
        )
    )


# ============================================================
# DISTANCE SANITY CHECK
# ============================================================

tree_xy = np.array(
    [
        [t[0], t[1]]
        for t in trees
    ],
    dtype=float
)

if len(tree_xy) >= 2:
    nn = NearestNeighbors(
        n_neighbors=2
    ).fit(tree_xy)

    distances, _ = nn.kneighbors(
        tree_xy
    )

    median_tree_spacing = float(
        np.median(
            distances[:, 1]
        )
    )
else:
    median_tree_spacing = 0.0

print(
    f"Estimated median tree spacing: "
    f"{median_tree_spacing:.2f} m"
)

if not (
    5.0
    <=
    median_tree_spacing
    <=
    40.0
):
    print(
        "WARNING: Tree spacing looks unusual. "
        "Check the GPS data and coordinate units."
    )


# ============================================================
# CREATE TREE BOUNDARY + 5 m GENERATED ROAD
# ============================================================

plantation_boundary = convex_hull(
    tree_xy
)

generated_road = offset_convex_polygon(
    plantation_boundary,
    ROAD_OFFSET
)

print(
    f"Generated road offset: "
    f"{ROAD_OFFSET:.2f} m"
)

print(
    f"Generated road perimeter: "
    f"{road_total_length(generated_road):.2f} m"
)


# ============================================================
# MAIN RUN
# ============================================================

START_RUN_NUMBER = get_next_run_number()

print(
    f"\nStarting {ANALYSIS_ALGORITHM} analysis from "
    f"Run {START_RUN_NUMBER}\n"
)

for local_run in range(NUM_RUNS):

    run_number = START_RUN_NUMBER + local_run

    RUN_ID = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    # --------------------------------------------------------
    # CLUSTER
    # --------------------------------------------------------

    raw_clusters = make_capacity_safe_clusters()

    # Warn but NEVER drop small clusters.
    small_clusters = [
        len(c)
        for c in raw_clusters
        if len(c) < MIN_CLUSTER_SIZE
    ]

    if small_clusters:
        print(
            "WARNING: Some clusters are smaller than "
            f"{MIN_CLUSTER_SIZE}: {small_clusters}"
        )

    # --------------------------------------------------------
    # ORDER CLUSTERS ALONG ROAD + SHARED BASE POINTS
    # --------------------------------------------------------

    (
        ordered_clusters,
        shared_bases,
        ordered_cluster_info
    ) = order_clusters_and_generate_shared_bases(
        raw_clusters,
        generated_road
    )

    n_clusters = len(
        ordered_clusters
    )

    # --------------------------------------------------------
    # SOLVE EACH CLUSTER
    # --------------------------------------------------------

    best_routes = []
    best_details = []
    cluster_access_info = []

    total_aco_cost = 0.0

    for cluster_index, cluster in enumerate(
        ordered_clusters
    ):

        start_base_id = cluster_index

        # Closed generated road:
        # End of current cluster = start of next cluster.
        end_base_id = (
            cluster_index + 1
        ) % n_clusters

        start_base_xy = shared_bases[
            start_base_id
        ]

        end_base_xy = shared_bases[
            end_base_id
        ]

        start_tree, end_tree = choose_endpoint_trees(
            cluster,
            start_base_xy,
            end_base_xy
        )

        route, aco_cost = solve_cluster(
            cluster,
            start_tree,
            end_tree
        )

        if route is None:
            raise RuntimeError(
                f"Cluster {cluster_index + 1} "
                "returned no ACO solution."
            )

        distance = path_distance(route)
        turn = turn_penalty(route)
        elevation = elevation_cost(route)

        start_access = access_distance_2d(
            start_base_xy,
            start_tree
        )

        end_access = access_distance_2d(
            end_base_xy,
            end_tree
        )

        best_routes.append(
            route
        )

        total_aco_cost += aco_cost

        best_details.append(
            {
                "size": len(cluster),
                "distance": distance,
                "turn": turn,
                "elevation": elevation,
                "aco_cost": aco_cost,
                "start_tree": start_tree,
                "end_tree": end_tree,
                "start_base_id": start_base_id,
                "end_base_id": end_base_id,
                "start_access": start_access,
                "end_access": end_access,
            }
        )

        cluster_access_info.append(
            {
                "cluster": cluster_index + 1,
                "start_base_id": start_base_id,
                "end_base_id": end_base_id,
                "start_tree": start_tree,
                "end_tree": end_tree,
                "start_access": start_access,
                "end_access": end_access,
            }
        )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    total_trees = sum(
        item["size"]
        for item in best_details
    )

    total_distance = sum(
        item["distance"]
        for item in best_details
    )

    total_turn = sum(
        item["turn"]
        for item in best_details
    )

    total_elevation = sum(
        item["elevation"]
        for item in best_details
    )

    total_access = sum(
        item["start_access"]
        +
        item["end_access"]
        for item in best_details
    )

    print(
        "\n========== FINAL RESULT ==========\n"
    )

    print(
        f"Total Trees      : {total_trees}"
    )

    print(
        f"Total Clusters   : {len(best_routes)}"
    )

    print(
        f"Total Distance   : {total_distance:.2f} m"
    )

    print(
        f"Total Turning    : {total_turn:.2f} rad"
    )

    print(
        f"Total Elevation  : {total_elevation:.2f} m"
    )

    print(
        f"Total ACO Cost   : {total_aco_cost:.4f}"
    )

    print(
        f"Access Distance* : {total_access:.2f} m"
    )

    print(
        "*Access distance is NOT included "
        "in optimized Total Distance or ACO Cost.\n"
    )

    for i, item in enumerate(
        best_details
    ):
        print(
            f"Cluster {i + 1}"
        )

        print(
            f"  Trees          : {item['size']}"
        )

        print(
            f"  Coverage Dist. : {item['distance']:.2f} m"
        )

        print(
            f"  Turning        : {item['turn']:.2f} rad"
        )

        print(
            f"  Elevation      : {item['elevation']:.2f} m"
        )

        print(
            f"  ACO Cost       : {item['aco_cost']:.4f}"
        )

        print(
            f"  Start Base     : B{item['start_base_id']}"
        )

        print(
            f"  End Base       : B{item['end_base_id']}"
        )

        print(
            f"  Start Tree     : {item['start_tree']}"
        )

        print(
            f"  End Tree       : {item['end_tree']}"
        )

        print(
            f"  Start Access*  : {item['start_access']:.2f} m"
        )

        print(
            f"  End Access*    : {item['end_access']:.2f} m"
        )

        print(
            "----------------------------------"
        )

    # ========================================================
    # OUTPUT FILE PATHS
    # ========================================================

    summary_file = os.path.join(
        SUMMARY_FOLDER,
        f"summary_{RUN_ID}.txt"
    )

    metrics_file = os.path.join(
        METRICS_FOLDER,
        f"metrics_{RUN_ID}.csv"
    )

    routes_file = os.path.join(
        ROUTES_FOLDER,
        f"routes_{RUN_ID}.csv"
    )

    bases_file = os.path.join(
        BASES_FOLDER,
        f"road_bases_{RUN_ID}.csv"
    )

    access_file = os.path.join(
        BASES_FOLDER,
        f"cluster_access_{RUN_ID}.csv"
    )

    kml_path = os.path.join(
        KML_FOLDER,
        f"routes_{RUN_ID}.kml"
    )

    # ========================================================
    # SUMMARY TXT
    # ========================================================

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "========== FINAL RESULT ==========\n\n"
        )

        f.write(
            f"Algorithm        : {ALGORITHM_NAME}\n"
        )

        f.write(
            f"Road Offset      : {ROAD_OFFSET:.2f} m\n"
        )

        f.write(
            f"Median Tree Gap  : {median_tree_spacing:.2f} m\n"
        )

        f.write(
            f"ACO Weights      : "
            f"Distance={DISTANCE_WEIGHT}, "
            f"Turning={TURN_WEIGHT}, "
            f"Elevation={ELEVATION_COST_WEIGHT}\n\n"
        )

        f.write(
            f"Total Trees      : {total_trees}\n"
        )

        f.write(
            f"Total Clusters   : {len(best_routes)}\n"
        )

        f.write(
            f"Total Distance   : {total_distance:.2f} m\n"
        )

        f.write(
            f"Total Turning    : {total_turn:.2f} rad\n"
        )

        f.write(
            f"Total Elevation  : {total_elevation:.2f} m\n"
        )

        f.write(
            f"Total ACO Cost   : {total_aco_cost:.4f}\n"
        )

        f.write(
            f"Access Distance  : {total_access:.2f} m "
            "(NOT counted in Total Distance or ACO Cost)\n\n"
        )

        f.write(
            "DISTANCE DEFINITION:\n"
        )

        f.write(
            "Total/cluster Distance includes ONLY "
            "the first oil palm to the last oil palm "
            "inside each cluster.\n"
        )

        f.write(
            "Road-base access legs and the original "
            "drone take-off point are excluded.\n\n"
        )

        for i, item in enumerate(
            best_details
        ):
            f.write(
                f"Cluster {i + 1}\n"
            )

            f.write(
                f"  Trees          : {item['size']}\n"
            )

            f.write(
                f"  Distance       : {item['distance']:.2f} m\n"
            )

            f.write(
                f"  Turning        : {item['turn']:.2f} rad\n"
            )

            f.write(
                f"  Elevation      : {item['elevation']:.2f} m\n"
            )

            f.write(
                f"  ACO Cost       : {item['aco_cost']:.4f}\n"
            )

            f.write(
                f"  Start Base     : B{item['start_base_id']}\n"
            )

            f.write(
                f"  End Base       : B{item['end_base_id']}\n"
            )

            f.write(
                f"  Start Tree     : {item['start_tree']}\n"
            )

            f.write(
                f"  End Tree       : {item['end_tree']}\n"
            )

            f.write(
                f"  Start Access   : {item['start_access']:.2f} m "
                "(excluded)\n"
            )

            f.write(
                f"  End Access     : {item['end_access']:.2f} m "
                "(excluded)\n"
            )

            f.write(
                "----------------------------------\n"
            )

    # ========================================================
    # METRICS CSV
    # Same headers as NSGA-II.
    # ========================================================

    with open(
        metrics_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "Cluster",
                "Trees",
                "Distance",
                "Turning",
                "Elevation",
            ]
        )

        for i, item in enumerate(
            best_details
        ):
            writer.writerow(
                [
                    i + 1,
                    item["size"],
                    item["distance"],
                    item["turn"],
                    item["elevation"],
                ]
            )

    # ========================================================
    # ROUTES CSV
    # ========================================================

    with open(
        routes_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        for i, route in enumerate(
            best_routes
        ):
            writer.writerow(
                [i + 1]
                +
                route
            )

    # ========================================================
    # SHARED ROAD BASES CSV
    # ========================================================

    with open(
        bases_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "Base",
                "X_m",
                "Y_m",
                "Lat",
                "Lon",
            ]
        )

        for i, base_xy in enumerate(
            shared_bases
        ):
            lat, lon = xy_to_latlon(
                base_xy[0],
                base_xy[1],
                lat0,
                lon0
            )

            writer.writerow(
                [
                    f"B{i}",
                    base_xy[0],
                    base_xy[1],
                    lat,
                    lon,
                ]
            )

    # ========================================================
    # CLUSTER ACCESS CSV
    # ========================================================

    with open(
        access_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "Cluster",
                "StartBase",
                "EndBase",
                "StartTree",
                "EndTree",
                "StartAccess_m",
                "EndAccess_m",
            ]
        )

        for item in cluster_access_info:
            writer.writerow(
                [
                    item["cluster"],
                    f"B{item['start_base_id']}",
                    f"B{item['end_base_id']}",
                    item["start_tree"],
                    item["end_tree"],
                    item["start_access"],
                    item["end_access"],
                ]
            )

    # ========================================================
    # COMPARISON CSV
    # ========================================================

    comparison_file = os.path.join(
        "Output",
        "comparison_3d.csv"
    )

    os.makedirs(
        os.path.dirname(comparison_file),
        exist_ok=True
    )

    comparison_exists = os.path.isfile(
        comparison_file
    )

    with open(
        comparison_file,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if not comparison_exists:
            writer.writerow(
                [
                    "Algorithm",
                    "ID",
                    "Distance",
                    "TurningPenalty",
                    "Elevation",
                ]
            )

        writer.writerow(
            [
                ALGORITHM_NAME,
                RUN_ID,
                total_distance,
                total_turn,
                total_elevation,
            ]
        )

    # ========================================================
    # ANALYSIS-READY MASTER CSV
    # ========================================================
    # One row = one cluster result from one algorithm run.
    # Use later for ANOVA:
    # Map,Cluster,Algorithm,Run,Distance,Elevation,Turning

    os.makedirs(
        os.path.dirname(ANALYSIS_DATA_FILE),
        exist_ok=True
    )

    analysis_exists = os.path.isfile(
        ANALYSIS_DATA_FILE
    )

    with open(
        ANALYSIS_DATA_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if not analysis_exists:
            writer.writerow(
                [
                    "Map",
                    "Cluster",
                    "Algorithm",
                    "Run",
                    "Distance",
                    "Elevation",
                    "Turning",
                ]
            )

        for i, item in enumerate(
            best_details
        ):
            writer.writerow(
                [
                    MAP_NAME,
                    i + 1,
                    ANALYSIS_ALGORITHM,
                    run_number,
                    item["distance"],
                    item["elevation"],
                    item["turn"],
                ]
            )

    # ========================================================
    # KML EXPORT
    # ========================================================

    avg_lat = float(
        np.mean(
            [p[0] for p in gps_points]
        )
    )

    avg_lon = float(
        np.mean(
            [p[1] for p in gps_points]
        )
    )

    colors = [
        "ff0000ff",
        "ff00ff00",
        "ffff0000",
        "ff00ffff",
        "ffffff00",
        "ff9900ff",
        "ffff00ff",
        "ff0099ff",
    ]

    with open(
        kml_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            '<?xml version="1.0" encoding="UTF-8"?>\n'
        )

        f.write(
            '<kml xmlns="http://www.opengis.net/kml/2.2">\n'
        )

        f.write(
            '<Document>\n'
        )

        f.write(
            f'<LookAt>'
            f'<longitude>{avg_lon}</longitude>'
            f'<latitude>{avg_lat}</latitude>'
            f'<range>500</range>'
            f'<tilt>45</tilt>'
            f'</LookAt>\n'
        )

        # ----------------------------------------------------
        # TREE BOUNDARY
        # ----------------------------------------------------

        f.write(
            '<Folder><name>Plantation Boundary</name>\n'
        )

        f.write(
            '<Placemark><name>Tree Boundary</name>'
        )

        f.write(
            '<Style><LineStyle>'
            '<color>ff777777</color>'
            '<width>3</width>'
            '</LineStyle></Style>'
        )

        f.write(
            '<LineString>'
            '<altitudeMode>clampToGround</altitudeMode>'
            '<coordinates>'
        )

        for point in (
            plantation_boundary
            +
            [plantation_boundary[0]]
        ):
            lat, lon = xy_to_latlon(
                point[0],
                point[1],
                lat0,
                lon0
            )

            f.write(
                f"{lon},{lat},0 "
            )

        f.write(
            '</coordinates>'
            '</LineString>'
            '</Placemark>\n'
        )

        f.write(
            '</Folder>\n'
        )

        # ----------------------------------------------------
        # GENERATED 5 m ROAD
        # ----------------------------------------------------

        f.write(
            '<Folder><name>Generated Road</name>\n'
        )

        f.write(
            f'<Placemark><name>'
            f'{ROAD_OFFSET:.1f} m Offset Road'
            f'</name>'
        )

        f.write(
            '<Style><LineStyle>'
            '<color>ff00aaff</color>'
            '<width>5</width>'
            '</LineStyle></Style>'
        )

        f.write(
            '<LineString>'
            '<altitudeMode>clampToGround</altitudeMode>'
            '<coordinates>'
        )

        for point in (
            generated_road
            +
            [generated_road[0]]
        ):
            lat, lon = xy_to_latlon(
                point[0],
                point[1],
                lat0,
                lon0
            )

            f.write(
                f"{lon},{lat},0 "
            )

        f.write(
            '</coordinates>'
            '</LineString>'
            '</Placemark>\n'
        )

        f.write(
            '</Folder>\n'
        )

        # ----------------------------------------------------
        # SHARED BASE POINTS
        # ----------------------------------------------------

        f.write(
            '<Folder><name>Shared Road Bases</name>\n'
        )

        for i, base_xy in enumerate(
            shared_bases
        ):
            lat, lon = xy_to_latlon(
                base_xy[0],
                base_xy[1],
                lat0,
                lon0
            )

            f.write(
                f'<Placemark>'
                f'<name>B{i}</name>'
                f'<Point>'
                f'<coordinates>{lon},{lat},0</coordinates>'
                f'</Point>'
                f'</Placemark>\n'
            )

        f.write(
            '</Folder>\n'
        )

        # ----------------------------------------------------
        # 2D COVERAGE ROUTES
        # Only palm-to-palm coverage is colored.
        # ----------------------------------------------------

        f.write(
            '<Folder><name>2D Coverage Routes</name>\n'
        )

        for i, route in enumerate(
            best_routes
        ):
            color = colors[
                i % len(colors)
            ]

            f.write(
                f'<Placemark>'
                f'<name>Cluster {i + 1} Coverage</name>'
            )

            f.write(
                f'<Style><LineStyle>'
                f'<color>{color}</color>'
                f'<width>4</width>'
                f'</LineStyle></Style>'
            )

            f.write(
                '<LineString>'
                '<altitudeMode>clampToGround</altitudeMode>'
                '<coordinates>'
            )

            for tree_id in route:
                lat, lon, _ = gps_points[
                    tree_id
                ]

                f.write(
                    f"{lon},{lat},0 "
                )

            f.write(
                '</coordinates>'
                '</LineString>'
                '</Placemark>\n'
            )

        f.write(
            '</Folder>\n'
        )

        # ----------------------------------------------------
        # ACCESS LEGS
        # Visible, but NOT counted.
        # ----------------------------------------------------

        f.write(
            '<Folder><name>Access Legs - Not Counted</name>\n'
        )

        for i, item in enumerate(
            best_details
        ):
            start_base = shared_bases[
                item["start_base_id"]
            ]

            end_base = shared_bases[
                item["end_base_id"]
            ]

            start_base_lat, start_base_lon = xy_to_latlon(
                start_base[0],
                start_base[1],
                lat0,
                lon0
            )

            end_base_lat, end_base_lon = xy_to_latlon(
                end_base[0],
                end_base[1],
                lat0,
                lon0
            )

            start_tree_lat, start_tree_lon, _ = gps_points[
                item["start_tree"]
            ]

            end_tree_lat, end_tree_lon, _ = gps_points[
                item["end_tree"]
            ]

            # Start access
            f.write(
                f'<Placemark>'
                f'<name>Cluster {i + 1} Start Access - Excluded</name>'
            )

            f.write(
                '<Style><LineStyle>'
                '<color>ffaaaaaa</color>'
                '<width>2</width>'
                '</LineStyle></Style>'
            )

            f.write(
                '<LineString>'
                '<altitudeMode>clampToGround</altitudeMode>'
                '<coordinates>'
            )

            f.write(
                f"{start_base_lon},{start_base_lat},0 "
                f"{start_tree_lon},{start_tree_lat},0 "
            )

            f.write(
                '</coordinates>'
                '</LineString>'
                '</Placemark>\n'
            )

            # End access
            f.write(
                f'<Placemark>'
                f'<name>Cluster {i + 1} End Access - Excluded</name>'
            )

            f.write(
                '<Style><LineStyle>'
                '<color>ffaaaaaa</color>'
                '<width>2</width>'
                '</LineStyle></Style>'
            )

            f.write(
                '<LineString>'
                '<altitudeMode>clampToGround</altitudeMode>'
                '<coordinates>'
            )

            f.write(
                f"{end_tree_lon},{end_tree_lat},0 "
                f"{end_base_lon},{end_base_lat},0 "
            )

            f.write(
                '</coordinates>'
                '</LineString>'
                '</Placemark>\n'
            )

        f.write(
            '</Folder>\n'
        )

        # ----------------------------------------------------
        # START / END PALM MARKERS
        # ----------------------------------------------------

        f.write(
            '<Folder><name>Cluster Start End Palms</name>\n'
        )

        for i, item in enumerate(
            best_details
        ):
            start_lat, start_lon, _ = gps_points[
                item["start_tree"]
            ]

            end_lat, end_lon, _ = gps_points[
                item["end_tree"]
            ]

            f.write(
                f'<Placemark>'
                f'<name>Cluster {i + 1} Start Palm</name>'
                f'<Point>'
                f'<coordinates>{start_lon},{start_lat},0</coordinates>'
                f'</Point>'
                f'</Placemark>\n'
            )

            f.write(
                f'<Placemark>'
                f'<name>Cluster {i + 1} End Palm</name>'
                f'<Point>'
                f'<coordinates>{end_lon},{end_lat},0</coordinates>'
                f'</Point>'
                f'</Placemark>\n'
            )

        f.write(
            '</Folder>\n'
        )

        # ----------------------------------------------------
        # 3D COVERAGE ROUTES
        # ----------------------------------------------------

        f.write(
            '<Folder><name>3D Coverage Routes</name>\n'
        )

        for i, route in enumerate(
            best_routes
        ):
            color = colors[
                i % len(colors)
            ]

            f.write(
                f'<Placemark>'
                f'<name>Cluster {i + 1} 3D Coverage</name>'
            )

            f.write(
                f'<Style><LineStyle>'
                f'<color>{color}</color>'
                f'<width>4</width>'
                f'</LineStyle></Style>'
            )

            f.write(
                '<LineString>'
                '<altitudeMode>absolute</altitudeMode>'
                '<coordinates>'
            )

            for tree_id in route:
                lat, lon, alt = gps_points[
                    tree_id
                ]

                f.write(
                    f"{lon},{lat},{alt + DRONE_OFFSET} "
                )

            f.write(
                '</coordinates>'
                '</LineString>'
                '</Placemark>\n'
            )

        f.write(
            '</Folder>\n'
        )

        f.write(
            '</Document>\n'
            '</kml>\n'
        )

    print(
        f"\nACO RUN COMPLETED: {RUN_ID}"
    )

    print(
        f"Summary : {summary_file}"
    )

    print(
        f"Metrics : {metrics_file}"
    )

    print(
        f"Routes  : {routes_file}"
    )

    print(
        f"Bases   : {bases_file}"
    )

    print(
        f"Access  : {access_file}"
    )

    print(
        f"KML     : {kml_path}"
    )

    print(
        f"Analysis: {ANALYSIS_DATA_FILE}"
    )
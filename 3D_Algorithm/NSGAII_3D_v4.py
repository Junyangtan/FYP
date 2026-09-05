import numpy as np
import csv
import os
from datetime import datetime
from math import ceil, sqrt
from deap import base, creator, tools, algorithms
from sklearn.cluster import KMeans

# =============================
# RUN SETTINGS
# =============================
NUM_RUNS = 1
RANDOM_SEED = 0

# =============================
# OUTPUT FOLDERS
# =============================
OUTPUT_FOLDER = os.path.join("Output", "NSGAII_3D")
ROUTES_FOLDER = os.path.join(OUTPUT_FOLDER, "Routes")
METRICS_FOLDER = os.path.join(OUTPUT_FOLDER, "Metrics")
KML_FOLDER = os.path.join(OUTPUT_FOLDER, "KML")
SUMMARY_FOLDER = os.path.join(OUTPUT_FOLDER, "Summary")
PARETO_FOLDER = os.path.join(OUTPUT_FOLDER, "Pareto")

for folder in [ROUTES_FOLDER, METRICS_FOLDER, KML_FOLDER, SUMMARY_FOLDER, PARETO_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# =============================
# NSGA-II PARAMETERS
# =============================
POP_SIZE = 200
NGEN = 300
CX_PROB = 0.7
MUT_PROB = 0.3

# =============================
# CLUSTER / PLANTATION PARAMETERS
# =============================
MAX_TREES = 80
MIN_CLUSTER_SIZE = 1
EXPECTED_TREE_SPACING = 18.0  # metres

# For triangular planting, perpendicular row spacing is approximately s*sin(60°)
TRIANGULAR_ROW_SPACING = EXPECTED_TREE_SPACING * sqrt(3.0) / 2.0
SWEEP_BAND_TOLERANCE = EXPECTED_TREE_SPACING * 0.45

# Elevation is still used in 3D K-means, but it should not dominate XY distance.
ELEVATION_WEIGHT = 0.5

DRONE_OFFSET = 3.0
ALGORITHM_NAME = "NSGAII_3D_v4"

BASE_LAT = 2.70425
BASE_LON = 101.633375
BASE_ALT = 0.0

# Earth radius in metres
R = 6371000.0

# =============================
# COORDINATE CONVERSION
# =============================
def latlon_to_xy(lat, lon, lat0, lon0):
    """
    Convert latitude/longitude to local Cartesian x/y in metres.

    IMPORTANT FIX:
    Latitude/longitude differences are converted from degrees to radians
    before multiplying by Earth's radius. The previous code omitted this,
    which inflated distances by about 57.3 times.
    """
    lat_rad = np.radians(lat)
    lat0_rad = np.radians(lat0)
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    mean_lat = 0.5 * (lat_rad + lat0_rad)

    x = R * dlon * np.cos(mean_lat)
    y = R * dlat
    return x, y


def drone_point(tree):
    return (tree[0], tree[1], tree[2] + DRONE_OFFSET)


def drone_start():
    return (base_xy[0], base_xy[1], BASE_ALT + DRONE_OFFSET)


# =============================
# OBJECTIVE FUNCTIONS
# =============================
def path_distance(order):
    """3D route distance in metres, including base -> route -> base."""
    pts = [drone_start()] + [drone_point(trees[i]) for i in order] + [drone_start()]

    return sum(
        np.linalg.norm(np.array(pts[i + 1]) - np.array(pts[i]))
        for i in range(len(pts) - 1)
    )


def turn_penalty(order):
    """Total turning angle in radians."""
    pts = [drone_start()] + [drone_point(trees[i]) for i in order] + [drone_start()]
    penalty = 0.0

    for i in range(1, len(pts) - 1):
        A = np.array(pts[i - 1])
        B = np.array(pts[i])
        C = np.array(pts[i + 1])

        BA = A - B
        BC = C - B

        cos_value = np.dot(BA, BC) / (
            np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-9
        )

        penalty += np.arccos(np.clip(cos_value, -1.0, 1.0))

    return penalty


def elevation_cost(order):
    """Accumulated absolute vertical movement in metres."""
    pts = [drone_start()] + [drone_point(trees[i]) for i in order] + [drone_start()]

    return sum(
        abs(pts[i + 1][2] - pts[i][2])
        for i in range(len(pts) - 1)
    )


def select_balanced_solution(pareto_list):
    """Choose an equal-weight balanced solution from one Pareto front."""
    if not pareto_list:
        return None

    d_list = [p["distance"] for p in pareto_list]
    t_list = [p["turn"] for p in pareto_list]
    e_list = [p["elevation"] for p in pareto_list]

    d_min, d_max = min(d_list), max(d_list)
    t_min, t_max = min(t_list), max(t_list)
    e_min, e_max = min(e_list), max(e_list)

    best = None
    best_score = float("inf")

    for p in pareto_list:
        d_norm = (p["distance"] - d_min) / (d_max - d_min + 1e-9)
        t_norm = (p["turn"] - t_min) / (t_max - t_min + 1e-9)
        e_norm = (p["elevation"] - e_min) / (e_max - e_min + 1e-9)

        score = d_norm + t_norm + e_norm

        if score < best_score:
            best_score = score
            best = p

    return best


# =============================
# LOAD DATA
# =============================
base_dir = os.path.dirname(os.path.abspath(__file__))

candidate_paths = [
    os.path.join(base_dir, "..", "3D_map.csv"),
    os.path.join(base_dir, "3D_map.csv"),
]

file_path = next((p for p in candidate_paths if os.path.isfile(p)), None)

if file_path is None:
    raise FileNotFoundError(
        "3D_map.csv was not found. Put it either in the same folder as this script "
        "or one folder above it."
    )

gps_points = []
with open(file_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        gps_points.append(
            (
                float(row["lat"]),
                float(row["lon"]),
                float(row["alt"]),
            )
        )

if not gps_points:
    raise ValueError("3D_map.csv contains no tree points.")

# Use the drone base as the local XY origin.
lat0 = BASE_LAT
lon0 = BASE_LON

trees = []
for lat, lon, alt in gps_points:
    x, y = latlon_to_xy(lat, lon, lat0, lon0)
    trees.append((x, y, alt, lat, lon))

base_xy = (0.0, 0.0)


# =============================
# DISTANCE SANITY CHECK
# =============================
def estimate_nearest_neighbour_spacing():
    xy = np.array([(t[0], t[1]) for t in trees], dtype=float)

    if len(xy) < 2:
        return 0.0

    nearest = []
    for i in range(len(xy)):
        delta = xy - xy[i]
        dist = np.linalg.norm(delta, axis=1)
        dist[i] = np.inf
        nearest.append(np.min(dist))

    return float(np.median(nearest))


estimated_spacing = estimate_nearest_neighbour_spacing()
print(f"Estimated nearest-neighbour spacing: {estimated_spacing:.2f} m")

if estimated_spacing > 0:
    lower = EXPECTED_TREE_SPACING * 0.5
    upper = EXPECTED_TREE_SPACING * 1.5
    if not (lower <= estimated_spacing <= upper):
        print(
            "WARNING: estimated spacing differs strongly from the expected "
            f"{EXPECTED_TREE_SPACING:.1f} m. Check GPS data/coordinate units."
        )


# =============================
# DEAP SETUP
# =============================
if "FitnessMulti" not in creator.__dict__:
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))

if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMulti)


# =============================
# ROUTE INITIALIZATION HELPERS
# =============================
def nearest_neighbour_local(cluster):
    """Return LOCAL cluster indices in nearest-neighbour order."""
    n = len(cluster)
    remaining = set(range(n))

    current = min(
        remaining,
        key=lambda local_i: np.linalg.norm(
            np.array(trees[cluster[local_i]][:2]) - np.array(base_xy)
        ),
    )

    route = []

    while remaining:
        route.append(current)
        remaining.remove(current)

        if not remaining:
            break

        current_xy = np.array(trees[cluster[current]][:2])
        current = min(
            remaining,
            key=lambda local_i: np.linalg.norm(
                current_xy - np.array(trees[cluster[local_i]][:2])
            ),
        )

    return route


def sweep_route_local(cluster, angle_deg):
    """
    Create a serpentine/sweep route for staggered triangular planting.
    The chromosome contains LOCAL indices 0..len(cluster)-1.
    """
    n = len(cluster)
    if n <= 1:
        return list(range(n))

    xy = np.array(
        [[trees[cluster[i]][0], trees[cluster[i]][1]] for i in range(n)],
        dtype=float,
    )

    center = np.mean(xy, axis=0)
    centered = xy - center

    theta = np.radians(angle_deg)
    along_axis = np.array([np.cos(theta), np.sin(theta)])
    across_axis = np.array([-np.sin(theta), np.cos(theta)])

    along = centered @ along_axis
    across = centered @ across_axis

    # Build sweep bands. The expected triangular row spacing is ~15.6 m
    # when tree-to-tree spacing is 18 m.
    min_across = np.min(across)
    band_spacing = max(TRIANGULAR_ROW_SPACING, 1e-6)
    band_id = np.rint((across - min_across) / band_spacing).astype(int)

    bands = {}
    for i in range(n):
        bands.setdefault(int(band_id[i]), []).append(i)

    route = []
    for band_number, key in enumerate(sorted(bands)):
        band = bands[key]
        band.sort(key=lambda i: along[i])

        if band_number % 2 == 1:
            band.reverse()

        route.extend(band)

    # Safety: ensure every local index appears exactly once.
    if sorted(route) != list(range(n)):
        return list(range(n))

    return route


def build_structured_initial_routes(cluster):
    """Generate several sensible coverage candidates for NSGA-II initialization."""
    candidates = []

    # Triangular grids have useful lattice directions separated by ~60 degrees.
    for angle in [0, 30, 60, 90, 120, 150]:
        route = sweep_route_local(cluster, angle)
        candidates.append(route)
        candidates.append(route[::-1])

    nn = nearest_neighbour_local(cluster)
    candidates.append(nn)
    candidates.append(nn[::-1])

    # Remove duplicates while preserving order.
    unique = []
    seen = set()
    for route in candidates:
        key = tuple(route)
        if key not in seen:
            seen.add(key)
            unique.append(route)

    return unique


def local_route_mutation(individual):
    """Small local changes that preserve route smoothness better than full shuffling."""
    n = len(individual)
    if n < 2:
        return individual,

    if np.random.rand() < 0.5:
        # Reverse a short local segment.
        start = np.random.randint(0, n - 1)
        max_end = min(n, start + 7)
        end = np.random.randint(start + 1, max_end + 1)
        individual[start:end] = reversed(individual[start:end])
    else:
        # Swap nearby positions rather than arbitrary distant ones.
        i = np.random.randint(0, n)
        low = max(0, i - 5)
        high = min(n, i + 6)
        choices = [j for j in range(low, high) if j != i]
        if choices:
            j = int(np.random.choice(choices))
            individual[i], individual[j] = individual[j], individual[i]

    return individual,


# =============================
# NSGA-II CLUSTER SOLVER
# =============================
def solve_cluster(cluster):
    if not cluster:
        return None

    if len(cluster) == 1:
        route = [cluster[0]]
        return [
            {
                "route": route,
                "distance": path_distance(route),
                "turn": turn_penalty(route),
                "elevation": elevation_cost(route),
            }
        ]

    local_indices = list(range(len(cluster)))
    structured_routes = build_structured_initial_routes(cluster)
    toolbox = base.Toolbox()

    def convert(ind):
        # Local chromosome index -> global tree index.
        return [cluster[i] for i in ind]

    def init_individual():
        # 90% structured coverage routes; 10% random for exploration.
        if np.random.rand() < 0.90:
            route = list(structured_routes[np.random.randint(len(structured_routes))])

            # Add a small perturbation to increase population diversity.
            if np.random.rand() < 0.60:
                temp = creator.Individual(route)
                local_route_mutation(temp)
                route = list(temp)

            return creator.Individual(route)

        return creator.Individual(list(np.random.permutation(local_indices)))

    def evaluate(ind):
        route = convert(ind)
        return (
            path_distance(route),
            turn_penalty(route),
            elevation_cost(route),
        )

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", local_route_mutation)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=POP_SIZE)

    pop, _ = algorithms.eaMuPlusLambda(
        pop,
        toolbox,
        mu=POP_SIZE,
        lambda_=POP_SIZE * 2,
        cxpb=CX_PROB,
        mutpb=MUT_PROB,
        ngen=NGEN,
        verbose=False,
    )

    front = tools.sortNondominated(
        pop,
        len(pop),
        first_front_only=True,
    )[0]

    pareto = []
    for ind in front:
        route = convert(ind)
        pareto.append(
            {
                "route": route,
                "distance": path_distance(route),
                "turn": turn_penalty(route),
                "elevation": elevation_cost(route),
            }
        )

    return pareto


# =============================
# CAPACITY-SAFE CLUSTER SPLITTING
# =============================
def split_cluster_by_capacity(cluster):
    """Spatially split oversized K-means clusters so every trip has <= MAX_TREES."""
    if len(cluster) <= MAX_TREES:
        return [cluster]

    n_sub = int(ceil(len(cluster) / MAX_TREES))

    local_coords = np.array(
        [
            (
                trees[i][0],
                trees[i][1],
                trees[i][2] * ELEVATION_WEIGHT,
            )
            for i in cluster
        ]
    )

    labels = KMeans(
        n_clusters=n_sub,
        init="k-means++",
        n_init=20,
        random_state=RANDOM_SEED,
    ).fit_predict(local_coords)

    result = []
    for label in range(n_sub):
        sub = [cluster[j] for j in range(len(cluster)) if labels[j] == label]
        if not sub:
            continue

        if len(sub) > MAX_TREES:
            result.extend(split_cluster_by_capacity(sub))
        else:
            result.append(sub)

    return result


# =============================
# GLOBAL SEARCH
# =============================
for run_number in range(NUM_RUNS):
    np.random.seed(RANDOM_SEED + run_number)

    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

    coords = np.array(
        [
            (
                t[0],
                t[1],
                t[2] * ELEVATION_WEIGHT,
            )
            for t in trees
        ]
    )

    N = len(coords)
    minimum_k = max(1, int(ceil(N / MAX_TREES)))
    k_min = max(1, minimum_k - 1)
    k_max = min(N, minimum_k + 3)

    candidate_solutions = []

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=20,
            random_state=RANDOM_SEED,
        ).fit(coords)

        initial_clusters = [[] for _ in range(k)]
        for i, label in enumerate(kmeans.labels_):
            initial_clusters[int(label)].append(i)

        final_clusters = []
        for cluster in initial_clusters:
            if cluster:
                final_clusters.extend(split_cluster_by_capacity(cluster))

        # Do not discard small clusters; every tree must still be visited.
        routes = []
        details = []
        cluster_pareto_all = []

        valid = True

        for cluster in final_clusters:
            pareto = solve_cluster(cluster)
            if not pareto:
                valid = False
                break

            cluster_pareto_all.append(pareto)
            best_sol = select_balanced_solution(pareto)

            if best_sol is None:
                valid = False
                break

            routes.append(best_sol["route"])
            details.append(
                {
                    "size": len(cluster),
                    "distance": best_sol["distance"],
                    "turn": best_sol["turn"],
                    "elevation": best_sol["elevation"],
                }
            )

        if not valid or not details:
            continue

        total_trees_candidate = sum(c["size"] for c in details)
        if total_trees_candidate != N:
            print(
                f"WARNING: k={k} covers {total_trees_candidate}/{N} trees; skipping candidate."
            )
            continue

        candidate_solutions.append(
            {
                "k": k,
                "routes": routes,
                "details": details,
                "pareto": cluster_pareto_all,
                "distance": sum(c["distance"] for c in details),
                "turn": sum(c["turn"] for c in details),
                "elevation": sum(c["elevation"] for c in details),
            }
        )

    if not candidate_solutions:
        print("No valid NSGA-II solution found.")
        continue

    # Choose the global K/cluster configuration using the same three objectives.
    all_d = [c["distance"] for c in candidate_solutions]
    all_t = [c["turn"] for c in candidate_solutions]
    all_e = [c["elevation"] for c in candidate_solutions]

    d_min, d_max = min(all_d), max(all_d)
    t_min, t_max = min(all_t), max(all_t)
    e_min, e_max = min(all_e), max(all_e)

    for candidate in candidate_solutions:
        d_norm = (candidate["distance"] - d_min) / (d_max - d_min + 1e-9)
        t_norm = (candidate["turn"] - t_min) / (t_max - t_min + 1e-9)
        e_norm = (candidate["elevation"] - e_min) / (e_max - e_min + 1e-9)
        candidate["score"] = d_norm + t_norm + e_norm

    best_candidate = min(candidate_solutions, key=lambda c: c["score"])

    best_total = best_candidate["score"]
    best_routes = best_candidate["routes"]
    best_details = best_candidate["details"]
    best_pareto_all = best_candidate["pareto"]

    # =============================
    # FINAL RESULT
    # =============================
    total_trees = sum(c["size"] for c in best_details)
    total_distance = sum(c["distance"] for c in best_details)
    total_turn = sum(c["turn"] for c in best_details)
    total_elevation = sum(c["elevation"] for c in best_details)

    print("\n========== FINAL RESULT ==========\n")
    print(f"Total Trees      : {total_trees}")
    print(f"Total Clusters   : {len(best_routes)}")
    print(f"Total Distance   : {total_distance:.2f} m")
    print(f"Total Turning    : {total_turn:.2f} rad")
    print(f"Total Elevation  : {total_elevation:.2f} m")
    print(f"Total Cost       : {best_total:.4f}\n")

    for i, c in enumerate(best_details):
        print(f"Cluster {i + 1}")
        print(f"  Trees      : {c['size']}")
        print(f"  Distance   : {c['distance']:.2f} m")
        print(f"  Turning    : {c['turn']:.2f} rad")
        print(f"  Elevation  : {c['elevation']:.2f} m")
        print("----------------------------------")

    # =============================
    # SAVE FILES
    # =============================
    summary_file = os.path.join(SUMMARY_FOLDER, f"summary_{RUN_ID}.txt")
    metrics_file = os.path.join(METRICS_FOLDER, f"metrics_{RUN_ID}.csv")
    routes_file = os.path.join(ROUTES_FOLDER, f"routes_{RUN_ID}.csv")
    pareto_file = os.path.join(PARETO_FOLDER, f"pareto_{RUN_ID}.csv")
    kml_path = os.path.join(KML_FOLDER, f"routes_{RUN_ID}.kml")

    # Summary
    with open(summary_file, "w") as f:
        f.write("========== FINAL RESULT ==========\n\n")
        f.write(f"Total Trees      : {total_trees}\n")
        f.write(f"Total Clusters   : {len(best_routes)}\n")
        f.write(f"Total Distance   : {total_distance:.2f} m\n")
        f.write(f"Total Turning    : {total_turn:.2f} rad\n")
        f.write(f"Total Elevation  : {total_elevation:.2f} m\n")
        f.write(f"Total Cost       : {best_total:.4f}\n")
        f.write(f"Estimated Spacing: {estimated_spacing:.2f} m\n\n")
        f.write("----------------------------------\n")

        for i, c in enumerate(best_details):
            f.write(f"Cluster {i + 1}\n")
            f.write(f"  Trees      : {c['size']}\n")
            f.write(f"  Distance   : {c['distance']:.2f} m\n")
            f.write(f"  Turning    : {c['turn']:.2f} rad\n")
            f.write(f"  Elevation  : {c['elevation']:.2f} m\n")
            f.write("----------------------------------\n")

    # Metrics
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster", "Trees", "Distance", "Turning", "Elevation"])
        for i, c in enumerate(best_details):
            writer.writerow(
                [i + 1, c["size"], c["distance"], c["turn"], c["elevation"]]
            )

    # Routes
    with open(routes_file, "w", newline="") as f:
        writer = csv.writer(f)
        for i, route in enumerate(best_routes):
            writer.writerow([i + 1] + route)

    # Pareto
    with open(pareto_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster", "Solution", "Distance", "Turning", "Elevation"])
        for i, pareto in enumerate(best_pareto_all):
            for j, sol in enumerate(pareto):
                writer.writerow(
                    [
                        i + 1,
                        j + 1,
                        sol["distance"],
                        sol["turn"],
                        sol["elevation"],
                    ]
                )

    # Comparison file
    comparison_file = os.path.join("Output", "comparison_3d.csv")
    os.makedirs("Output", exist_ok=True)
    exists = os.path.isfile(comparison_file)

    with open(comparison_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                ["Algorithm", "ID", "Distance", "TurningPenalty", "Elevation"]
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

    # =============================
    # KML EXPORT (2D + 3D FOLDERS)
    # =============================
    avg_lat = np.mean([p[0] for p in gps_points])
    avg_lon = np.mean([p[1] for p in gps_points])

    colors = [
        "ff0000ff",
        "ff00ff00",
        "ffff0000",
        "ff00ffff",
        "ffffff00",
        "ff9900ff",
    ]

    with open(kml_path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n<Document>\n')

        # Camera view
        f.write(
            f"<LookAt><longitude>{avg_lon}</longitude>"
            f"<latitude>{avg_lat}</latitude><range>300</range>"
            f"<tilt>45</tilt></LookAt>\n"
        )

        # Base point
        f.write(
            f"<Placemark><name>Base</name><Point><coordinates>"
            f"{BASE_LON},{BASE_LAT},{BASE_ALT}</coordinates>"
            f"</Point></Placemark>\n"
        )

        # 2D routes
        f.write("<Folder><name>2D Routes</name>\n")

        for i, route in enumerate(best_routes):
            color = colors[i % len(colors)]

            f.write(f"<Placemark><name>Route {i + 1}</name>")
            f.write(
                f"<Style><LineStyle><color>{color}</color><width>4</width>"
                f"</LineStyle></Style>"
            )
            f.write(
                "<LineString><altitudeMode>clampToGround</altitudeMode>"
                "<coordinates>"
            )

            f.write(f"{BASE_LON},{BASE_LAT},0 ")
            for idx in route:
                lat, lon, _ = gps_points[idx]
                f.write(f"{lon},{lat},0 ")
            f.write(f"{BASE_LON},{BASE_LAT},0 ")

            f.write("</coordinates></LineString></Placemark>\n")

        f.write("</Folder>\n")

        # 3D routes
        f.write("<Folder><name>3D Routes</name>\n")

        for i, route in enumerate(best_routes):
            color = colors[i % len(colors)]

            f.write(f"<Placemark><name>Route {i + 1}</name>")
            f.write(
                f"<Style><LineStyle><color>{color}</color><width>4</width>"
                f"</LineStyle></Style>"
            )
            f.write(
                "<LineString><altitudeMode>relativeToGround</altitudeMode>"
                "<coordinates>"
            )

            f.write(f"{BASE_LON},{BASE_LAT},{DRONE_OFFSET} ")
            for idx in route:
                lat, lon, alt = gps_points[idx]
                f.write(f"{lon},{lat},{alt + DRONE_OFFSET} ")
            f.write(f"{BASE_LON},{BASE_LAT},{DRONE_OFFSET} ")

            f.write("</coordinates></LineString></Placemark>\n")

        f.write("</Folder>\n")
        f.write("</Document></kml>")

    print(f"\n✅ RUN COMPLETED: {RUN_ID}")
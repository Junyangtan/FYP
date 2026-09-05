import numpy as np
import csv
import os
from datetime import datetime
from deap import base, creator, tools, algorithms
from sklearn.cluster import KMeans

# =============================
NUM_RUNS = 1
# =============================

# =============================
# CONFIG
# =============================
OUTPUT_FOLDER = os.path.join("Output", "NSGAII_3D")
ROUTES_FOLDER = os.path.join(OUTPUT_FOLDER, "Routes")
METRICS_FOLDER = os.path.join(OUTPUT_FOLDER, "Metrics")
KML_FOLDER = os.path.join(OUTPUT_FOLDER, "KML")
SUMMARY_FOLDER = os.path.join(OUTPUT_FOLDER, "Summary")
PARETO_FOLDER = os.path.join(OUTPUT_FOLDER, "Pareto")

for folder in [ROUTES_FOLDER, METRICS_FOLDER, KML_FOLDER, SUMMARY_FOLDER, PARETO_FOLDER]:
    os.makedirs(folder, exist_ok=True)

POP_SIZE = 200
NGEN = 300
MAX_TREES = 80
MIN_CLUSTER_SIZE = 5   # 🔥 FIX
DRONE_OFFSET = 3
ALGORITHM_NAME = "NSGAII_3D_v3"

# Row-aware initialization
ROW_GAP = 5.0
HEURISTIC_RATIO = 0.8

BASE_LAT = 2.70425
BASE_LON = 101.633375
BASE_ALT = 0

R = 6371000

# =============================
# FUNCTIONS
# =============================
def latlon_to_xy(lat, lon, lat0, lon0):
    x = (lon - lon0) * np.cos(np.radians(lat0)) * R
    y = (lat - lat0) * R
    return x, y

def drone_point(tree):
    return (tree[0], tree[1], tree[2] + DRONE_OFFSET)

def drone_start():
    return (base_xy[0], base_xy[1], BASE_ALT)

def path_distance(order):
    pts = [drone_start()] + [drone_point(trees[i]) for i in order] + [drone_start()]
    return sum(np.linalg.norm(np.array(pts[i+1]) - np.array(pts[i])) for i in range(len(pts)-1))

def turn_penalty(order):
    pts = [drone_start()] + [drone_point(trees[i]) for i in order] + [drone_start()]
    penalty = 0
    for i in range(1, len(pts)-1):
        A, B, C = np.array(pts[i-1]), np.array(pts[i]), np.array(pts[i+1])
        BA, BC = A-B, C-B
        cos = np.dot(BA, BC)/(np.linalg.norm(BA)*np.linalg.norm(BC)+1e-6)
        penalty += np.arccos(np.clip(cos, -1, 1))
    return penalty

def elevation_cost(order):
    pts = [drone_start()] + [drone_point(trees[i]) for i in order] + [drone_start()]
    return sum(abs(pts[i+1][2]-pts[i][2]) for i in range(len(pts)-1))

def select_balanced_solution(pareto_list):
    d_list = [p["distance"] for p in pareto_list]
    t_list = [p["turn"] for p in pareto_list]
    e_list = [p["elevation"] for p in pareto_list]

    best = None
    best_score = float("inf")

    for p in pareto_list:
        d_norm = (p["distance"] - min(d_list)) / (max(d_list)-min(d_list)+1e-6)
        t_norm = (p["turn"] - min(t_list)) / (max(t_list)-min(t_list)+1e-6)
        e_norm = (p["elevation"] - min(e_list)) / (max(e_list)-min(e_list)+1e-6)

        score = d_norm + t_norm + e_norm

        if score < best_score:
            best_score = score
            best = p

    return best

# =============================
# LOAD DATA
# =============================
base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "..", "3D_map.csv")

gps_points = []
with open(file_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        gps_points.append((float(row["lat"]), float(row["lon"]), float(row["alt"])))

lat0, lon0, _ = gps_points[0]

trees = []
for lat, lon, alt in gps_points:
    x, y = latlon_to_xy(lat, lon, lat0, lon0)
    trees.append((x, y, alt, lat, lon))

base_xy = latlon_to_xy(BASE_LAT, BASE_LON, lat0, lon0)

# =============================
# DEAP SETUP
# =============================
if "FitnessMulti" not in creator.__dict__:
    creator.create("FitnessMulti", base.Fitness, weights=(-1,-1,-1))

if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

def solve_cluster(cluster):
    if len(cluster) < 2:
        return None

    idx = list(range(len(cluster)))

    def convert(ind):
        return [cluster[i] for i in ind]

    # --------------------------------------------------
    # Row-based / Serpentine Initialization
    # --------------------------------------------------

    def serpentine_init():

        # XY coordinates of trees in this cluster
        xy = np.array([
            [
                trees[cluster[i]][0],
                trees[cluster[i]][1]
            ]
            for i in idx
        ])

        # Centre coordinates
        center = np.mean(xy, axis=0)
        centered = xy - center

        # PCA: estimate dominant plantation/row orientation
        covariance = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        main_axis = eigenvectors[:, np.argmax(eigenvalues)]

        # Axis perpendicular to the rows
        row_axis = np.array([
            -main_axis[1],
            main_axis[0]
        ])

        # Project each tree along/across the estimated rows
        along_row = centered @ main_axis
        across_row = centered @ row_axis

        # Sort trees from one side of the plantation to the other
        sorted_indices = sorted(
            idx,
            key=lambda i: across_row[i]
        )

        # Group trees into rows
        rows = []
        current_row = []
        row_center = None

        for i in sorted_indices:
            value = across_row[i]

            if not current_row:
                current_row = [i]
                row_center = value

            elif abs(value - row_center) <= ROW_GAP:
                current_row.append(i)

                # Update row centre as points are added
                row_center = np.mean([
                    across_row[x]
                    for x in current_row
                ])

            else:
                rows.append(current_row)
                current_row = [i]
                row_center = value

        if current_row:
            rows.append(current_row)

        # Create back-and-forth (serpentine) route
        route = []

        for row_number, row in enumerate(rows):

            row = sorted(
                row,
                key=lambda i: along_row[i]
            )

            # Reverse every second row
            if row_number % 2 == 1:
                row.reverse()

            route.extend(row)

        return route


    def init():

        # 80% row-aware heuristic individuals
        if np.random.rand() < HEURISTIC_RATIO:

            route = serpentine_init()

            # Reverse the complete route sometimes for diversity
            if np.random.rand() < 0.5:
                route = route[::-1]

            return creator.Individual(route)

        # 20% random individuals preserve exploration
        return creator.Individual(
            list(np.random.permutation(idx))
        )

    if hasattr(toolbox, "individual"):
        toolbox.unregister("individual")
    toolbox.register(
        "individual",
        init
    )

    if hasattr(toolbox, "population"):
        toolbox.unregister("population")
    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.individual
    )

    def evaluate(ind):
        route = convert(ind)
        return path_distance(route), turn_penalty(route), elevation_cost(route)

    if hasattr(toolbox, "evaluate"):
        toolbox.unregister("evaluate")
    toolbox.register("evaluate", evaluate)
    if hasattr(toolbox, "mate"):
        toolbox.unregister("mate")
    toolbox.register("mate", tools.cxOrdered)
    if hasattr(toolbox, "mutate"):
        toolbox.unregister("mutate")
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    if hasattr(toolbox, "select"):
        toolbox.unregister("select")
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=POP_SIZE)

    pop, _ = algorithms.eaMuPlusLambda(
        pop, toolbox,
        mu=POP_SIZE,
        lambda_=POP_SIZE*2,
        cxpb=0.7,
        mutpb=0.3,
        ngen=NGEN,
        verbose=False
    )

    front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

    pareto = []
    for ind in front:
        route = convert(ind)
        pareto.append({
            "route": route,
            "distance": path_distance(route),
            "turn": turn_penalty(route),
            "elevation": elevation_cost(route)
        })

    return pareto

# =============================
# GLOBAL SEARCH
# =============================
for _ in range(NUM_RUNS):

    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Elevation importance
    ELEVATION_WEIGHT = 5.0
    coords = np.array([
    (
        t[0],
        t[1],
        t[2] * ELEVATION_WEIGHT
    )
    for t in trees
    ])
    N = len(coords)

    k_center = N // MAX_TREES
    k_max_limit = N // MIN_CLUSTER_SIZE

    k_min = max(2, k_center - 3)
    k_max = min(k_center + 3, k_max_limit)

    best_total = float("inf")

    for k in range(k_min, k_max+1):

        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=0).fit(coords)

        clusters = [[] for _ in range(k)]
        for i, label in enumerate(kmeans.labels_):
            clusters[label].append(i)

        final_clusters = []
        for c in clusters:
            for i in range(0, len(c), MAX_TREES):
                sub = c[i:i+MAX_TREES]
                if len(sub) < MIN_CLUSTER_SIZE:
                    continue
                final_clusters.append(sub)

        routes, details = [], []
        cluster_pareto_all = []

        d_list, t_list, e_list = [], [], []

        for c in final_clusters:

            pareto = solve_cluster(c)
            if pareto is None:
                continue

            cluster_pareto_all.append(pareto)

            best_sol = select_balanced_solution(pareto)

            r = best_sol["route"]
            routes.append(r)

            d = best_sol["distance"]
            t = best_sol["turn"]
            e = best_sol["elevation"]

            d_list.append(d)
            t_list.append(t)
            e_list.append(e)

            details.append({
                "size": len(c),
                "distance": d,
                "turn": t,
                "elevation": e
            })

        if len(details) == 0:
            continue

        total = sum(
            (d_list[i]-min(d_list))/(max(d_list)-min(d_list)+1e-6) +
            (t_list[i]-min(t_list))/(max(t_list)-min(t_list)+1e-6) +
            (e_list[i]-min(e_list))/(max(e_list)-min(e_list)+1e-6)
            for i in range(len(details))
        )

        if total < best_total:
            best_total = total
            best_routes = routes
            best_details = details
            best_pareto_all = cluster_pareto_all

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
    print(f"Total Distance   : {total_distance:.2f}")
    print(f"Total Turning    : {total_turn:.2f}")
    print(f"Total Elevation  : {total_elevation:.2f}")
    print(f"Total Cost       : {best_total:.4f}\n")

    for i, c in enumerate(best_details):
        print(f"Cluster {i+1}")
        print(f"  Trees      : {c['size']}")
        print(f"  Distance   : {c['distance']:.2f}")
        print(f"  Turning    : {c['turn']:.2f}")
        print(f"  Elevation  : {c['elevation']:.2f}")
        print("----------------------------------")
    # =============================
    # SAVE FILES
    # =============================
    summary_file = os.path.join(SUMMARY_FOLDER, f"summary_{RUN_ID}.txt")
    metrics_file = os.path.join(METRICS_FOLDER, f"metrics_{RUN_ID}.csv")
    routes_file = os.path.join(ROUTES_FOLDER, f"routes_{RUN_ID}.csv")
    pareto_file = os.path.join(PARETO_FOLDER, f"pareto_{RUN_ID}.csv")
    kml_path = os.path.join(KML_FOLDER, f"routes_{RUN_ID}.kml")

    # summary
    with open(summary_file, "w") as f:
        f.write("========== FINAL RESULT ==========\n\n")

        f.write(f"Total Trees      : {total_trees}\n")
        f.write(f"Total Clusters   : {len(best_routes)}\n")
        f.write(f"Total Distance   : {total_distance:.2f}\n")
        f.write(f"Total Turning    : {total_turn:.2f}\n")
        f.write(f"Total Elevation  : {total_elevation:.2f}\n")
        f.write(f"Total Cost       : {best_total:.4f}\n\n")

        f.write("----------------------------------\n")
    # metrics
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster","Trees","Distance","Turning","Elevation"])
        for i,c in enumerate(best_details):
            writer.writerow([i+1,c["size"],c["distance"],c["turn"],c["elevation"]])

    # routes
    with open(routes_file, "w", newline="") as f:
        writer = csv.writer(f)
        for i, r in enumerate(best_routes):
            writer.writerow([i+1] + r)

    # pareto
    with open(pareto_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster","Solution","Distance","Turning","Elevation"])
        for i, pareto in enumerate(best_pareto_all):
            for j, sol in enumerate(pareto):
                writer.writerow([i+1,j+1,sol["distance"],sol["turn"],sol["elevation"]])

    # comparison
    comparison_file = os.path.join("Output","comparison_3d.csv")
    exists = os.path.isfile(comparison_file)

    with open(comparison_file,"a",newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["Algorithm","ID","Distance","TurningPenalty","Elevation"])
        writer.writerow([ALGORITHM_NAME,RUN_ID,total_distance,total_turn,total_elevation])
        
    # =============================
    # KML EXPORT (FIXED: 2D + 3D FOLDERS)
    # =============================
    avg_lat = np.mean([p[0] for p in gps_points])
    avg_lon = np.mean([p[1] for p in gps_points])

    colors = ["ff0000ff","ff00ff00","ffff0000","ff00ffff","ffffff00","ff9900ff"]

    with open(kml_path,"w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n<Document>\n')

        # Camera view
        f.write(f'<LookAt><longitude>{avg_lon}</longitude><latitude>{avg_lat}</latitude><range>300</range><tilt>45</tilt></LookAt>\n')

        # Base
        f.write(f'<Placemark><name>Base</name><Point><coordinates>{BASE_LON},{BASE_LAT},0</coordinates></Point></Placemark>\n')

        # =============================
        # 📁 2D ROUTES
        # =============================
        f.write('<Folder><name>2D Routes</name>\n')

        for i, r in enumerate(best_routes):
            color = colors[i % len(colors)]

            f.write(f'<Placemark><name>Route {i+1}</name>')
            f.write(f'<Style><LineStyle><color>{color}</color><width>4</width></LineStyle></Style>')
            f.write('<LineString><altitudeMode>clampToGround</altitudeMode><coordinates>')

            f.write(f"{BASE_LON},{BASE_LAT},0 ")
            for idx in r:
                lat, lon, _ = gps_points[idx]
                f.write(f"{lon},{lat},0 ")
            f.write(f"{BASE_LON},{BASE_LAT},0 ")

            f.write('</coordinates></LineString></Placemark>\n')

        f.write('</Folder>\n')

        # =============================
        # 📁 3D ROUTES
        # =============================
        f.write('<Folder><name>3D Routes</name>\n')

        for i, r in enumerate(best_routes):
            color = colors[i % len(colors)]

            f.write(f'<Placemark><name>Route {i+1}</name>')
            f.write(f'<Style><LineStyle><color>{color}</color><width>4</width></LineStyle></Style>')
            f.write('<LineString><altitudeMode>relativeToGround</altitudeMode><coordinates>')

            f.write(f"{BASE_LON},{BASE_LAT},{DRONE_OFFSET} ")
            for idx in r:
                lat, lon, alt = gps_points[idx]
                f.write(f"{lon},{lat},{alt + DRONE_OFFSET} ")
            f.write(f"{BASE_LON},{BASE_LAT},{DRONE_OFFSET} ")

            f.write('</coordinates></LineString></Placemark>\n')

        f.write('</Folder>\n')

        f.write('</Document></kml>')

    print(f"\n✅ RUN COMPLETED: {RUN_ID}")
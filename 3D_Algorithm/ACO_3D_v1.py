import numpy as np
import csv
import os
from datetime import datetime
from sklearn.cluster import KMeans

# =============================
NUM_RUNS = 5
# =============================

# =============================
# CONFIG
# =============================
OUTPUT_FOLDER = os.path.join("Output", "ACO_3D")

ROUTES_FOLDER = os.path.join(OUTPUT_FOLDER, "Routes")
METRICS_FOLDER = os.path.join(OUTPUT_FOLDER, "Metrics")
KML_FOLDER = os.path.join(OUTPUT_FOLDER, "KML")
SUMMARY_FOLDER = os.path.join(OUTPUT_FOLDER, "Summary")

os.makedirs(ROUTES_FOLDER, exist_ok=True)
os.makedirs(METRICS_FOLDER, exist_ok=True)
os.makedirs(KML_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

MAX_TREES = 80
MIN_CLUSTER_SIZE = 5   # 🔥 FIX
DRONE_OFFSET = 3
ALGORITHM_NAME = "ACO_3D_v1"

BASE_LAT = 2.70425
BASE_LON = 101.633375
BASE_ALT = 0

R = 6371000

NUM_ANTS = 30
ITERATIONS = 80
ALPHA = 1
BETA = 3
EVAPORATION = 0.5

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

def distance_3d(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def path_distance(order):
    pts = [drone_start()] + [drone_point(trees[i]) for i in order] + [drone_start()]
    return sum(distance_3d(pts[i], pts[i+1]) for i in range(len(pts)-1))

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

def total_cost(order):
    return path_distance(order) + 10*turn_penalty(order) + 2*elevation_cost(order)

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
# ACO SOLVER
# =============================
def solve_cluster(cluster):

    if len(cluster) < 2:
        return None

    n = len(cluster)
    pheromone = np.ones((n, n))
    coords = [cluster[i] for i in range(n)]

    def dist(i, j):
        return np.linalg.norm(np.array(coords[i]) - np.array(coords[j])) + 1e-6

    best_route = None
    best_cost = float("inf")

    for _ in range(ITERATIONS):

        all_routes = []

        for _ in range(NUM_ANTS):

            unvisited = list(range(n))
            current = np.random.choice(unvisited)
            route = [current]
            unvisited.remove(current)

            while unvisited:
                probs = []
                for j in unvisited:
                    tau = pheromone[current][j] ** ALPHA
                    eta = (1.0 / dist(current, j)) ** BETA
                    probs.append(tau * eta)

                probs = np.array(probs)
                probs /= probs.sum()

                next_node = np.random.choice(unvisited, p=probs)
                route.append(next_node)
                unvisited.remove(next_node)
                current = next_node

            real_route = [cluster[i] for i in route]
            cost = total_cost(real_route)

            all_routes.append((route, cost))

            if cost < best_cost:
                best_cost = cost
                best_route = real_route

        pheromone *= (1 - EVAPORATION)

        for route, cost in all_routes:
            for i in range(len(route)-1):
                pheromone[route[i]][route[i+1]] += 1.0 / (cost + 1e-6)

    return best_route

# =============================
# GLOBAL SEARCH
# =============================
for _ in range(NUM_RUNS):

    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

    coords = np.array([(t[0], t[1], t[2]) for t in trees])
    N = len(coords)

    k_center = N // MAX_TREES
    k_max_limit = N // MIN_CLUSTER_SIZE

    k_min = max(2, k_center - 3)
    k_max = min(k_center + 3, k_max_limit)

    best_total = float("inf")

    for k in range(k_min, k_max+1):

        kmeans = KMeans(n_clusters=k, random_state=0).fit(coords)

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
        total = 0

        for c in final_clusters:

            r = solve_cluster(c)
            if r is None:
                continue

            routes.append(r)

            d = path_distance(r)
            t = turn_penalty(r)
            e = elevation_cost(r)

            cost = d + 10*t + 2*e
            total += cost

            details.append({
                "size": len(c),
                "distance": d,
                "turn": t,
                "elevation": e
            })

        if len(details) == 0:
            continue

        if total < best_total:
            best_total = total
            best_routes = routes
            best_details = details

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
    kml_path = os.path.join(KML_FOLDER, f"optimal_routes_{RUN_ID}.kml")

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
        for i,r in enumerate(best_routes):
            writer.writerow([i+1] + r)

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

    print(f"\n✅ ACO RUN COMPLETED: {RUN_ID}")
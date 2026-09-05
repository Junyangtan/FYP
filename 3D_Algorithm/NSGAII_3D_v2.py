import numpy as np
import csv
import os
from datetime import datetime
from deap import base as deap_base, creator, tools, algorithms
from sklearn.cluster import KMeans

NUM_RUNS = 1

OUTPUT_FOLDER = os.path.join("Output", "NSGAII_3D")
ROUTES_FOLDER = os.path.join(OUTPUT_FOLDER, "Routes")
METRICS_FOLDER = os.path.join(OUTPUT_FOLDER, "Metrics")
KML_FOLDER = os.path.join(OUTPUT_FOLDER, "KML")
SUMMARY_FOLDER = os.path.join(OUTPUT_FOLDER, "Summary")
PARETO_FOLDER = os.path.join(OUTPUT_FOLDER, "Pareto")

for folder in [ROUTES_FOLDER, METRICS_FOLDER, KML_FOLDER, SUMMARY_FOLDER, PARETO_FOLDER]:
    os.makedirs(folder, exist_ok=True)

POP_SIZE = 120
NGEN = 100
MAX_TREES = 80
MIN_CLUSTER_SIZE = 40
DRONE_OFFSET = 3
ALGORITHM_NAME = "NSGAII_3D_v2"

BASE_LAT = 2.70425
BASE_LON = 101.633375
BASE_ALT = 0
R = 6371000


def latlon_to_xy(lat, lon, lat0, lon0):
    x = (lon - lon0) * np.cos(np.radians(lat0)) * R
    y = (lat - lat0) * R
    return x, y


def drone_point(tree):
    return (tree[0], tree[1], tree[2] + DRONE_OFFSET)


def drone_start():
    return (base_xy[0], base_xy[1], BASE_ALT)


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
# PRECOMPUTE DISTANCE
# =============================
N = len(trees)
dist_matrix = np.zeros((N, N))
base_dist = np.zeros(N)

base_point = np.array(drone_start())

for i in range(N):
    pi = np.array(drone_point(trees[i]))
    base_dist[i] = np.linalg.norm(pi - base_point)
    for j in range(N):
        pj = np.array(drone_point(trees[j]))
        dist_matrix[i][j] = np.linalg.norm(pi - pj)


def fast_distance(order):
    d = base_dist[order[0]]
    for i in range(len(order) - 1):
        d += dist_matrix[order[i]][order[i + 1]]
    d += base_dist[order[-1]]
    return d


def fast_turn(order):
    pts = [drone_start()] + [drone_point(trees[i]) for i in order] + [drone_start()]
    penalty = 0
    for i in range(1, len(pts) - 1):
        A, B, C = np.array(pts[i - 1]), np.array(pts[i]), np.array(pts[i + 1])
        BA, BC = A - B, C - B
        cos = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
        penalty += (1 - cos)
    return penalty


def elevation_cost(order):
    pts = [drone_start()] + [drone_point(trees[i]) for i in order] + [drone_start()]
    return sum(abs(pts[i + 1][2] - pts[i][2]) for i in range(len(pts) - 1))


def two_opt(route):
    best = route[:]
    best_dist = fast_distance(best)
    for i in range(len(route)):
        for j in range(i + 1, len(route)):
            new = route[:]
            new[i:j] = reversed(new[i:j])
            d = fast_distance(new)
            if d < best_dist:
                best = new
                best_dist = d
    return best


# =============================
# DEAP SETUP
# =============================
if "FitnessMulti" not in creator.__dict__:
    creator.create("FitnessMulti", deap_base.Fitness, weights=(-1, -1, -1))

if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = deap_base.Toolbox()


def mut_inversion(ind):
    a, b = sorted(np.random.choice(len(ind), 2, replace=False))
    ind[a:b] = reversed(ind[a:b])
    return ind,


# =============================
# SOLVE CLUSTER
# =============================
def solve_cluster(cluster):
    if len(cluster) < 2:
        return None

    idx = list(range(len(cluster)))

    def convert(ind):
        return [cluster[i] for i in ind]

    def init():
        return np.random.permutation(idx)

    toolbox.register("indices", init)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        route = convert(ind)
        return fast_distance(route), fast_turn(route), elevation_cost(route)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", mut_inversion)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=POP_SIZE)

    pop, _ = algorithms.eaMuPlusLambda(
        pop, toolbox,
        mu=POP_SIZE,
        lambda_=POP_SIZE * 2,
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
            "distance": fast_distance(route),
            "turn": fast_turn(route),
            "elevation": elevation_cost(route)
        })

    return sorted(pareto, key=lambda x: x["distance"])[:3]


# =============================
# MAIN
# =============================
def main():

    for run in range(NUM_RUNS):

        RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"\n=== RUN {run+1} ===")

        coords = np.array([
            (t[0], t[1], t[2]*2,
             np.linalg.norm(np.array([t[0], t[1]]) - np.array(base_xy)))
            for t in trees
        ])

        k_center = len(coords) // MAX_TREES
        k_max_limit = len(coords) // MIN_CLUSTER_SIZE

        k_min = max(2, k_center - 3)
        k_max = min(k_center + 3, k_max_limit)

        best_total = float("inf")

        for k in range(k_min, k_max + 1):

            kmeans = KMeans(n_clusters=k, random_state=0).fit(coords)

            clusters = [[] for _ in range(k)]
            for i, label in enumerate(kmeans.labels_):
                clusters[label].append(i)

            # =============================
            # 🔥 CLUSTER REDISTRIBUTION FIX
            # =============================
            final_clusters = []

            for c in clusters:
                subs = [c[i:i+MAX_TREES] for i in range(0, len(c), MAX_TREES)]

                for i in range(len(subs)):
                    if len(subs[i]) < MIN_CLUSTER_SIZE:
                        largest_idx = max(range(len(subs)), key=lambda x: len(subs[x]))

                        if largest_idx != i and len(subs[largest_idx]) > MIN_CLUSTER_SIZE:
                            needed = MIN_CLUSTER_SIZE - len(subs[i])

                            move = subs[largest_idx][:needed]
                            subs[i].extend(move)
                            subs[largest_idx] = subs[largest_idx][needed:]

                final_clusters.extend(subs)

            # =============================

            results = list(map(solve_cluster, final_clusters))

            routes, details = [], []

            for pareto in results:
                if pareto is None:
                    continue

                best = pareto[0]
                opt_route = two_opt(best["route"])

                d = fast_distance(opt_route)
                t = fast_turn(opt_route)
                e = elevation_cost(opt_route)

                routes.append(opt_route)
                details.append({
                    "size": len(opt_route),
                    "distance": d,
                    "turn": t,
                    "elevation": e
                })

            if len(details) == 0:
                continue

            total = sum(d["distance"] + d["turn"] + d["elevation"] for d in details)

            if total < best_total:
                best_total = total
                best_routes = routes
                best_details = details
                best_pareto_all = results

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

        print("\n✅ V2 FIXED COMPLETED")


if __name__ == "__main__":
    main()
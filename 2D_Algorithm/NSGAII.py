import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from deap import base, creator, tools, algorithms

# -----------------------------
# Create Output Folder
# -----------------------------
output_folder = os.path.join("Output", "NSGAII")
os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# Convert lat/lon → meters
# -----------------------------
def latlon_to_xy(lat, lon, ref_lat):
    R = 6371000  # Earth radius (meters)
    x = np.radians(lon) * R * np.cos(np.radians(ref_lat))
    y = np.radians(lat) * R
    return x, y

# -----------------------------
# Load Tree Coordinates (LAT/LON)
# -----------------------------
trees = []

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, "..", "2D_map.csv")  # your file

lat_ref = None

with open(file_path) as file:
    reader = csv.DictReader(file)

    for row in reader:
        lat = float(row["lat"])
        lon = float(row["lon"])

        if lat_ref is None:
            lat_ref = lat  # reference latitude

        x, y = latlon_to_xy(lat, lon, lat_ref)
        trees.append((x, y))

NUM_TREES = len(trees)

# start point (you can also change to first tree if needed)
start = trees[0]

route_results = []
metric_results = []

# -----------------------------
# Fitness Setup
# -----------------------------
if "FitnessMulti" not in creator.__dict__:
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))

if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

toolbox.register("indices", np.random.permutation, NUM_TREES)

toolbox.register("individual",
                 tools.initIterate,
                 creator.Individual,
                 toolbox.indices)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# -----------------------------
# Distance
# -----------------------------
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def path_distance(order):
    points = [start]

    for i in order:
        points.append(trees[i])

    dist = 0

    for i in range(len(points) - 1):
        dist += distance(points[i], points[i + 1])

    return dist


def turn_penalty(order):
    points = [start]

    for i in order:
        points.append(trees[i])

    penalty = 0

    for i in range(1, len(points) - 1):
        A = np.array(points[i - 1])
        B = np.array(points[i])
        C = np.array(points[i + 1])

        BA = A - B
        BC = C - B

        cos_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        penalty += angle

    return penalty

# -----------------------------
# Fitness
# -----------------------------
def evaluate(individual):
    dist = path_distance(individual)
    turn = turn_penalty(individual)
    return dist, turn


toolbox.register("evaluate", evaluate)

toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

# -----------------------------
# Run NSGA-II
# -----------------------------
population = toolbox.population(n=200)

population, logbook = algorithms.eaMuPlusLambda(
    population,
    toolbox,
    mu=200,
    lambda_=400,
    cxpb=0.7,
    mutpb=0.3,
    ngen=200,
    verbose=True
)

# -----------------------------
# Save Population Results
# -----------------------------
for ind in population:
    route_results.append(list(ind))
    metric_results.append(list(ind.fitness.values))

# -----------------------------
# Save CSV Files
# -----------------------------
with open(os.path.join(output_folder, "routes.csv"), "w", newline="") as file:
    writer = csv.writer(file)
    header = [f"Step{i+1}" for i in range(NUM_TREES)]
    writer.writerow(header)

    for r in route_results:
        writer.writerow(r)

with open(os.path.join(output_folder, "metrics.csv"), "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Distance", "TurningPenalty"])

    for m in metric_results:
        writer.writerow(m)

# -----------------------------
# Pareto Front
# -----------------------------
pareto_front = tools.sortNondominated(population, len(population), True)[0]

shortest_route = min(pareto_front, key=lambda ind: ind.fitness.values[0])
smoothest_route = min(pareto_front, key=lambda ind: ind.fitness.values[1])
balanced_route = min(pareto_front, key=lambda ind: sum(ind.fitness.values))

# -----------------------------
# Save Representative Routes
# -----------------------------
with open(os.path.join(output_folder, "representative_routes.csv"), "w", newline="") as file:
    writer = csv.writer(file)
    header = ["Type"] + [f"Step{i+1}" for i in range(NUM_TREES)] + ["Distance", "TurningPenalty"]
    writer.writerow(header)

    for name, route in [
        ("Shortest Distance", shortest_route),
        ("Smoothest Route", smoothest_route),
        ("Balanced Route", balanced_route)
    ]:
        dist, turn = route.fitness.values
        writer.writerow([name] + [int(i) for i in route] + [dist, turn])

# -----------------------------
# Best Solution
# -----------------------------
best = tools.selBest(population, 1)[0]

best_distance, best_turn = best.fitness.values

print("\n===== NSGA-II RESULT =====")
print("Best route:", [int(i) for i in best])
print("Distance (m):", round(best_distance, 2))
print("Turn:", round(best_turn, 2))

# -----------------------------
# Save Comparison Table
# -----------------------------
comparison_file = os.path.join("Output", "comparison.csv")

file_exists = os.path.isfile(comparison_file)

with open(comparison_file, "a", newline="") as file:
    writer = csv.writer(file)

    if not file_exists:
        writer.writerow(["Algorithm", "Distance", "TurningPenalty"])

    writer.writerow(["NSGAII", round(best_distance, 2), round(best_turn, 2)])
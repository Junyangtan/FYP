import numpy as np
import csv
import os
from deap import base, creator, tools, algorithms

# -----------------------------
# Output Folder
# -----------------------------
output_folder = os.path.join("Output", "GA")
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
file_path = os.path.join(base_dir, "..", "2D_map.csv")

lat_ref = None

with open(file_path) as file:
    reader = csv.DictReader(file)

    for row in reader:
        lat = float(row["lat"])
        lon = float(row["lon"])

        if lat_ref is None:
            lat_ref = lat

        x, y = latlon_to_xy(lat, lon, lat_ref)
        trees.append((x, y))

NUM_TREES = len(trees)

# Start point (same logic as NSGA-II)
start = trees[0]

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
# Fitness Setup
# -----------------------------
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

if "IndividualGA" not in creator.__dict__:
    creator.create("IndividualGA", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

toolbox.register("indices", np.random.permutation, NUM_TREES)

toolbox.register("individual",
                 tools.initIterate,
                 creator.IndividualGA,
                 toolbox.indices)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# -----------------------------
# Fitness Function
# -----------------------------
def evaluate(individual):
    dist = path_distance(individual)
    turn = turn_penalty(individual)

    return (dist + 10 * turn,)  # weighted objective


toolbox.register("evaluate", evaluate)

toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# -----------------------------
# Run GA
# -----------------------------
population = toolbox.population(n=200)

population, logbook = algorithms.eaSimple(
    population,
    toolbox,
    cxpb=0.7,
    mutpb=0.3,
    ngen=200,
    verbose=True
)

# -----------------------------
# Best Solution
# -----------------------------
best = tools.selBest(population, 1)[0]

best_distance = path_distance(best)
best_turn = turn_penalty(best)

print("\n===== GA RESULT =====")
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

    writer.writerow(["GA", round(best_distance, 2), round(best_turn, 2)])
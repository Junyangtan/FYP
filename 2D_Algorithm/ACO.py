import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import random

# -----------------------------
# Create Output Folder
# -----------------------------
output_folder = os.path.join("Output", "ACO")
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

# same as GA & NSGA-II
start = trees[0]

# -----------------------------
# Distance
# -----------------------------
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# distance matrix
dist_matrix = np.zeros((NUM_TREES, NUM_TREES))

for i in range(NUM_TREES):
    for j in range(NUM_TREES):
        dist_matrix[i][j] = distance(trees[i], trees[j])

# -----------------------------
# ACO Parameters
# -----------------------------
num_ants = 50
iterations = 200

alpha = 1
beta = 5
evaporation = 0.5

pheromone = np.ones((NUM_TREES, NUM_TREES))

best_route = None
best_distance = float("inf")

route_results = []

# -----------------------------
# Route distance
# -----------------------------
def route_distance(route):
    dist = distance(start, trees[route[0]])

    for i in range(len(route) - 1):
        dist += dist_matrix[route[i]][route[i + 1]]

    return dist

# -----------------------------
# ACO Main Loop
# -----------------------------
for it in range(iterations):

    for ant in range(num_ants):

        unvisited = list(range(NUM_TREES))
        route = []

        current = random.choice(unvisited)

        route.append(current)
        unvisited.remove(current)

        while unvisited:

            probs = []

            for j in unvisited:
                tau = pheromone[current][j] ** alpha
                eta = (1 / (dist_matrix[current][j] + 1e-6)) ** beta
                probs.append(tau * eta)

            probs = np.array(probs)
            probs = probs / np.sum(probs)

            next_city = np.random.choice(unvisited, p=probs)

            route.append(int(next_city))
            unvisited.remove(next_city)

            current = next_city

        dist = route_distance(route)
        route_results.append(route)

        if dist < best_distance:
            best_distance = dist
            best_route = route

    pheromone *= (1 - evaporation)

# -----------------------------
# Turning penalty (same as others)
# -----------------------------
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

best_turn = turn_penalty(best_route)

# -----------------------------
# Save Routes CSV
# -----------------------------
with open(os.path.join(output_folder, "routes.csv"), "w", newline="") as file:
    writer = csv.writer(file)
    header = [f"Step{i+1}" for i in range(NUM_TREES)]
    writer.writerow(header)

    for r in route_results:
        writer.writerow(r)

# -----------------------------
# Save Metrics CSV
# -----------------------------
with open(os.path.join(output_folder, "metrics.csv"), "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Distance", "TurningPenalty"])

    for r in route_results:
        d = route_distance(r)
        t = turn_penalty(r)
        writer.writerow([d, t])

# -----------------------------
# Print Result
# -----------------------------
print("\n===== ACO RESULT =====")
print("Best route:", [int(i) for i in best_route])
print("Distance (m):", round(best_distance, 2))
print("Turn:", round(best_turn, 2))

# -----------------------------
# Plot Best Route
# -----------------------------
plt.figure(figsize=(8, 8))

x_tree = [t[0] for t in trees]
y_tree = [t[1] for t in trees]

plt.scatter(x_tree, y_tree, label="Trees")
plt.scatter(start[0], start[1], s=120, label="Start")

route = [start]
for i in best_route:
    route.append(trees[i])

x = [p[0] for p in route]
y = [p[1] for p in route]

plt.plot(x, y, linewidth=1.5, label="ACO Route")

plt.legend()
plt.grid()

plt.savefig(os.path.join(output_folder, "best_route.png"))
plt.show()

# -----------------------------
# Update Comparison Table
# -----------------------------
comparison_file = os.path.join("Output", "comparison.csv")

file_exists = os.path.isfile(comparison_file)

with open(comparison_file, "a", newline="") as file:
    writer = csv.writer(file)

    if not file_exists:
        writer.writerow(["Algorithm", "Distance", "TurningPenalty"])

    writer.writerow(["ACO", round(best_distance, 2), round(best_turn, 2)])
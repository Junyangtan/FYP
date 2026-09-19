# Updated route comparison: corrected turning + identical bounded local search.
# Set MAP_NAME and NUM_RUNS below, then run this standalone script.
# Outputs go to Output_Improved so old turning results are not mixed in.
# Native search budget: 40,000 objective slots/cluster in each algorithm.
# Cleanup fixes endpoints and visits each tree once; it is not a global solver.
# See ROUTE_IMPROVEMENTS.md for trade-offs, audit columns and comparison protocol.

import time
import numpy as np
import csv
import os
import math
import random
from datetime import datetime
from sklearn.cluster import KMeans
BASE_SEED = 20260918
LOCAL_SEARCH_MAX_CANDIDATES = 250000
LOCAL_SEARCH_MAX_MOVES = 100
LOCAL_SEARCH_WEIGHTS = np.array([1.0, 10.0, 2.0])
LOCAL_SEARCH_EPS = 1e-08
NUM_RUNS = 10
MAP_NAME = 'Map2'
ALGORITHM_NAME = 'NSGAII_3D_v7'
ANALYSIS_ALGORITHM = 'NSGAII'
OUTPUT_FOLDER = 'Output'
NSGA_FOLDER = os.path.join(OUTPUT_FOLDER, 'NSGAII_3D')
ROUTES_FOLDER = os.path.join(NSGA_FOLDER, 'Routes')
METRICS_FOLDER = os.path.join(NSGA_FOLDER, 'Metrics')
SUMMARY_FOLDER = os.path.join(NSGA_FOLDER, 'Summary')
BASE_FOLDER = os.path.join(NSGA_FOLDER, 'Bases')
KML_FOLDER = os.path.join(NSGA_FOLDER, 'KML')
PARETO_FOLDER = os.path.join(NSGA_FOLDER, 'Pareto')
ANALYSIS_FILE = os.path.join(OUTPUT_FOLDER, 'analysis_data2.csv')
POPULATION_SIZE = 200
GENERATIONS = 199
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.05
MAX_TREES = 80
ELEVATION_WEIGHT = 5.0
ROAD_OFFSET = 5.0
DRONE_OFFSET = 3.0
MAP_CONFIG = {'Map1': {'csv_file': 'Map1_3D_map.csv', 'base_lat': 2.70425, 'base_lon': 101.633375}, 'Map2': {'csv_file': 'Map2_3D_map.csv', 'base_lat': 2.7055, 'base_lon': 101.6348}}
SELECTED_MAP = MAP_CONFIG[MAP_NAME]
EARTH_RADIUS = 6371000.0
trees = []
gps_points = []
lat0 = None
lon0 = None
plantation_boundary = []
generated_road = []

def latlon_to_xy(lat, lon, ref_lat, ref_lon):
    dlat = np.radians(lat - ref_lat)
    dlon = np.radians(lon - ref_lon)
    ref = np.radians(ref_lat)
    x = dlon * np.cos(ref) * EARTH_RADIUS
    y = dlat * EARTH_RADIUS
    return (float(x), float(y))

def xy_to_latlon(x, y, ref_lat, ref_lon):
    lat = ref_lat + np.degrees(y / EARTH_RADIUS)
    lon = ref_lon + np.degrees(x / (EARTH_RADIUS * np.cos(np.radians(ref_lat))))
    return (float(lat), float(lon))

def drone_point(tree_id):
    tree = trees[tree_id]
    return np.array([tree[0], tree[1], tree[2] + DRONE_OFFSET], dtype=float)

def evaluate_route(route):
    return tuple(route_metrics(route))

def cross_2d(a, b):
    return a[0] * b[1] - a[1] * b[0]

def convex_hull(points):
    pts = sorted(set(((float(p[0]), float(p[1])) for p in points)))

    def orientation(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and orientation(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and orientation(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return [np.array(p, dtype=float) for p in hull]

def polygon_area(poly):
    area = 0.0
    for i in range(len(poly)):
        p = poly[i]
        q = poly[(i + 1) % len(poly)]
        area += p[0] * q[1] - q[0] * p[1]
    return area / 2

def line_intersection(p, r, q, s):
    denominator = cross_2d(r, s)
    if abs(denominator) < 1e-12:
        return None
    t = cross_2d(q - p, s) / denominator
    return p + t * r

def offset_convex_polygon(polygon, offset):
    poly = [np.array(p) for p in polygon]
    if polygon_area(poly) < 0:
        poly.reverse()
    road = []
    for i in range(len(poly)):
        prev = poly[(i - 1) % len(poly)]
        curr = poly[i]
        nxt = poly[(i + 1) % len(poly)]
        e1 = curr - prev
        e2 = nxt - curr
        n1 = np.array([e1[1], -e1[0]])
        n2 = np.array([e2[1], -e2[0]])
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)
        p1 = curr + offset * n1
        p2 = curr + offset * n2
        intersection = line_intersection(p1, e1, p2, e2)
        if intersection is None:
            normal = n1 + n2
            normal /= np.linalg.norm(normal)
            intersection = curr + offset * normal
        road.append(intersection)
    return road

def segment_lengths(poly):
    return np.array([np.linalg.norm(poly[(i + 1) % len(poly)] - poly[i]) for i in range(len(poly))])

def road_length(road):
    return float(np.sum(segment_lengths(road)))

def project_to_road(point, road):
    point = np.asarray(point)
    lengths = segment_lengths(road)
    best_distance = float('inf')
    best_point = None
    best_s = 0
    travelled = 0
    for i in range(len(road)):
        A = road[i]
        B = road[(i + 1) % len(road)]
        AB = B - A
        denominator = np.dot(AB, AB)
        if denominator == 0:
            continue
        t = np.dot(point - A, AB) / denominator
        t = np.clip(t, 0, 1)
        q = A + t * AB
        d = np.linalg.norm(point - q)
        if d < best_distance:
            best_distance = d
            best_point = q
            best_s = travelled + t * lengths[i]
        travelled += lengths[i]
    return (best_point, best_s, best_distance)

def interpolate_road(road, s):
    lengths = segment_lengths(road)
    total = np.sum(lengths)
    s = s % total
    travelled = 0
    for i, length in enumerate(lengths):
        if s <= travelled + length:
            ratio = (s - travelled) / length
            return road[i] + ratio * (road[(i + 1) % len(road)] - road[i])
        travelled += length
    return road[0]

def circular_distance(a, b, total):
    d = abs(a - b) % total
    return min(d, total - d)

def create_clusters():
    data = np.array([[t[0], t[1], t[2] * ELEVATION_WEIGHT] for t in trees])
    number = max(1, math.ceil(len(trees) / MAX_TREES))
    while True:
        model = KMeans(n_clusters=number, n_init=20, random_state=0)
        labels = model.fit_predict(data)
        clusters = [[] for _ in range(number)]
        for i, label in enumerate(labels):
            clusters[label].append(i)
        if max((len(c) for c in clusters)) <= MAX_TREES:
            return clusters
        number += 1

def cluster_centroid(cluster):
    return np.mean([[trees[i][0], trees[i][1]] for i in cluster], axis=0)

def order_clusters_and_create_bases(clusters, road):
    total = road_length(road)
    info = []
    for cluster in clusters:
        centroid = cluster_centroid(cluster)
        _, s, _ = project_to_road(centroid, road)
        info.append({'cluster': cluster, 's': s})
    info.sort(key=lambda x: x['s'])
    bases = []
    for i in range(len(info)):
        s1 = info[i]['s']
        s2 = info[(i + 1) % len(info)]['s']
        if i == len(info) - 1:
            s2 += total
        middle = (s1 + s2) / 2
        bases.append(interpolate_road(road, middle))
    ordered = [x['cluster'] for x in info]
    return (ordered, bases)

def choose_start_end_tree(cluster, start_base, end_base):
    if not cluster:
        raise ValueError('Empty cluster')
    if len(cluster) == 1:
        return (cluster[0], cluster[0])
    start = min(cluster, key=lambda i: np.linalg.norm(np.array([trees[i][0], trees[i][1]]) - start_base))
    remaining = [x for x in cluster if x != start]
    end = min(remaining, key=lambda i: np.linalg.norm(np.array([trees[i][0], trees[i][1]]) - end_base))
    return (start, end)

def create_individual(middle_trees):
    chromosome = list(range(len(middle_trees)))
    random.shuffle(chromosome)
    return chromosome

def decode_route(chromosome, middle_trees, start_tree, end_tree):
    return [start_tree] + [middle_trees[i] for i in chromosome] + [end_tree]

def evaluate_population(population, middle_trees, start_tree, end_tree):
    result = []
    for chromosome in population:
        route = decode_route(chromosome, middle_trees, start_tree, end_tree)
        result.append(evaluate_route(route))
    return result

def dominates(a, b):
    return all((x <= y for x, y in zip(a, b))) and any((x < y for x, y in zip(a, b)))

def fast_non_dominated_sort(objectives):
    domination_count = [0 for _ in objectives]
    dominated = [[] for _ in objectives]
    fronts = [[]]
    for p in range(len(objectives)):
        for q in range(len(objectives)):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                dominated[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]

def calculate_crowding_distance(front, objectives):
    distance = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            distance[i] = float('inf')
        return distance
    number_objectives = len(objectives[0])
    for obj in range(number_objectives):
        sorted_front = sorted(front, key=lambda i: objectives[i][obj])
        distance[sorted_front[0]] = float('inf')
        distance[sorted_front[-1]] = float('inf')
        minimum = objectives[sorted_front[0]][obj]
        maximum = objectives[sorted_front[-1]][obj]
        if maximum - minimum == 0:
            continue
        for i in range(1, len(sorted_front) - 1):
            previous = objectives[sorted_front[i - 1]][obj]
            following = objectives[sorted_front[i + 1]][obj]
            distance[sorted_front[i]] += (following - previous) / (maximum - minimum)
    return distance

def tournament_selection(population, rank, crowding):
    a, b = random.sample(range(len(population)), 2)
    if rank[a] < rank[b]:
        return population[a].copy()
    if rank[b] < rank[a]:
        return population[b].copy()
    if crowding[a] > crowding[b]:
        return population[a].copy()
    return population[b].copy()

def ordered_crossover(parent1, parent2):
    size = len(parent1)
    if size < 2:
        return (parent1.copy(), parent2.copy())
    a, b = sorted(random.sample(range(size), 2))
    child1 = [None] * size
    child2 = [None] * size
    child1[a:b] = parent1[a:b]
    child2[a:b] = parent2[a:b]
    fill1 = [x for x in parent2 if x not in child1]
    fill2 = [x for x in parent1 if x not in child2]
    index1 = 0
    index2 = 0
    for i in range(size):
        if child1[i] is None:
            child1[i] = fill1[index1]
            index1 += 1
        if child2[i] is None:
            child2[i] = fill2[index2]
            index2 += 1
    return (child1, child2)

def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        a, b = sorted(random.sample(range(len(chromosome)), 2))
        chromosome[a:b] = reversed(chromosome[a:b])
    return chromosome

def _solve_cluster_raw(cluster, start_tree, end_tree):
    middle_trees = [i for i in cluster if i not in (start_tree, end_tree)]
    if len(middle_trees) == 0:
        return [start_tree, end_tree]
    if len(middle_trees) == 1:
        return [start_tree, middle_trees[0], end_tree]
    population = [create_individual(middle_trees) for _ in range(POPULATION_SIZE)]
    objectives = evaluate_population(population, middle_trees, start_tree, end_tree)
    for generation in range(GENERATIONS):
        fronts = fast_non_dominated_sort(objectives)
        rank = [0] * len(population)
        crowding = [0] * len(population)
        for r, front in enumerate(fronts):
            crowd = calculate_crowding_distance(front, objectives)
            for index in front:
                rank[index] = r
                crowding[index] = crowd[index]
        offspring = []
        while len(offspring) < POPULATION_SIZE:
            parent1 = tournament_selection(population, rank, crowding)
            parent2 = tournament_selection(population, rank, crowding)
            if random.random() < CROSSOVER_RATE:
                child1, child2 = ordered_crossover(parent1, parent2)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()
            offspring.append(mutate(child1))
            if len(offspring) < POPULATION_SIZE:
                offspring.append(mutate(child2))
        combined = population + offspring
        combined_objectives = objectives + evaluate_population(offspring, middle_trees, start_tree, end_tree)
        fronts = fast_non_dominated_sort(combined_objectives)
        new_population = []
        new_objectives = []
        for front in fronts:
            if len(new_population) + len(front) <= POPULATION_SIZE:
                new_population.extend([combined[i] for i in front])
                new_objectives.extend([combined_objectives[i] for i in front])
            else:
                crowd = calculate_crowding_distance(front, combined_objectives)
                sorted_front = sorted(front, key=lambda x: crowd[x], reverse=True)
                remaining = POPULATION_SIZE - len(new_population)
                new_population.extend([combined[i] for i in sorted_front[:remaining]])
                new_objectives.extend([combined_objectives[i] for i in sorted_front[:remaining]])
                break
        population = new_population
        objectives = new_objectives
    final_objectives = objectives
    fronts = fast_non_dominated_sort(final_objectives)
    pareto_front = fronts[0]
    best = min(pareto_front, key=lambda i: float(np.asarray(final_objectives[i]) @ LOCAL_SEARCH_WEIGHTS))
    return decode_route(population[best], middle_trees, start_tree, end_tree)

def load_map():
    global gps_points
    global trees
    global lat0
    global lon0
    global plantation_boundary
    global generated_road
    filename = SELECTED_MAP['csv_file']
    possible_paths = [filename, os.path.join('Maps', filename), os.path.join('..', filename)]
    map_path = None
    for path in possible_paths:
        if os.path.isfile(path):
            map_path = path
            break
    if map_path is None:
        raise FileNotFoundError(f'Cannot find {filename}')
    with open(map_path, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            gps_points.append((float(row['lat']), float(row['lon']), float(row['alt'])))
    lat0 = gps_points[0][0]
    lon0 = gps_points[0][1]
    for lat, lon, alt in gps_points:
        x, y = latlon_to_xy(lat, lon, lat0, lon0)
        trees.append((x, y, alt, lat, lon))
    tree_xy = np.array([[t[0], t[1]] for t in trees])
    plantation_boundary = convex_hull(tree_xy)
    generated_road = offset_convex_polygon(plantation_boundary, ROAD_OFFSET)
    print(f'Loaded {len(trees)} trees')
    print(f'Road length: {road_length(generated_road):.2f} m')

def get_start_run_number():
    if not os.path.isfile(ANALYSIS_FILE):
        return 1
    max_run = 0
    with open(ANALYSIS_FILE, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Map'] == MAP_NAME and row['Algorithm'] == ANALYSIS_ALGORITHM:
                max_run = max(max_run, int(row['Run']))
    return max_run + 1

def save_analysis_data(details, run_number):
    file_exists = os.path.isfile(ANALYSIS_FILE)
    with open(ANALYSIS_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Map', 'Cluster', 'Algorithm', 'Run', 'Distance', 'Elevation', 'Turning'])
        for item in details:
            writer.writerow([MAP_NAME, item['cluster'], ANALYSIS_ALGORITHM, run_number, item['distance'], item['elevation'], item['turning']])

def metrics_from_points(points):
    """Open path: 3D distance (m), 3D heading change (rad), |delta-z| (m)."""
    points = np.asarray(points, dtype=float)
    if not np.all(np.isfinite(points)):
        raise ValueError('Map contains non-finite coordinates.')
    if len(points) < 2:
        return np.zeros(3, dtype=float)
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError('Consecutive trees have identical 3D coordinates; check map data.')
    turning = 0.0
    if len(vectors) > 1:
        unit = vectors / lengths[:, None]
        cosine = np.sum(unit[:-1] * unit[1:], axis=1)
        turning = float(np.arccos(np.clip(cosine, -1.0, 1.0)).sum())
    return np.array([lengths.sum(), turning, np.abs(vectors[:, 2]).sum()])

def route_metrics(route):
    return metrics_from_points([trees[i][:3] for i in route])

def route_distance(route):
    return float(route_metrics(route)[0])

def route_turning(route):
    return float(route_metrics(route)[1])

def route_elevation(route):
    return float(route_metrics(route)[2])

def validate_route(route, cluster, start_tree, end_tree):
    if len(route) != len(cluster) or len(set(route)) != len(route) or set(route) != set(cluster):
        raise ValueError('Route must visit every cluster tree exactly once.')
    if not route or route[0] != start_tree or route[-1] != end_tree:
        raise ValueError('Route endpoints changed.')

def improve_route(route):
    """Identical bounded 2-opt + Or-opt(1,2,3) cleanup for all algorithms.

    Fix endpoints. Accept only strict distance reductions that do not increase
    D + 10*T + 2*E. Individual T or E can rise: export all before/after metrics.
    Alternates full best-improvement 2-opt scans with block relocation scans.
    No global-optimum or crossing-free guarantee under these constraints.
    """
    original = list(route)
    points = np.asarray([trees[i][:3] for i in original], dtype=float)
    n = len(original)
    order = list(range(n))
    current = metrics_from_points(points)
    distance_matrix = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    examined = moves = 0
    phase = '2opt'
    reason = 'local_optimum'
    while n >= 4 and moves < LOCAL_SEARCH_MAX_MOVES:
        best_order = None
        best_metrics = current
        best_distance = current[0]
        score_limit = float(current @ LOCAL_SEARCH_WEIGHTS)
        budget_hit = False
        if phase == '2opt':
            candidates = ((i, j, 0) for i in range(1, n - 2) for j in range(i + 1, n - 1))
        else:
            candidates = ((i, j, length) for length in (1, 2, 3) for i in range(1, n - length) for j in range(1, n - length) if j != i)
        for i, j, length in candidates:
            if examined >= LOCAL_SEARCH_MAX_CANDIDATES:
                budget_hit = True
                break
            examined += 1
            if length == 0:
                a, b, c, d = (order[i - 1], order[i], order[j], order[j + 1])
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                if current[0] + delta >= best_distance - LOCAL_SEARCH_EPS:
                    continue
                candidate = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
            else:
                block = order[i:i + length]
                remaining = order[:i] + order[i + length:]
                a, b, c, d = (order[i - 1], block[0], block[-1], order[i + length])
                u, v = (remaining[j - 1], remaining[j])
                delta = distance_matrix[a, d] - distance_matrix[a, b] - distance_matrix[c, d] + distance_matrix[u, b] + distance_matrix[c, v] - distance_matrix[u, v]
                if current[0] + delta >= best_distance - LOCAL_SEARCH_EPS:
                    continue
                candidate = remaining[:j] + block + remaining[j:]
            candidate_metrics = metrics_from_points(points[candidate])
            if candidate_metrics[0] < best_distance - LOCAL_SEARCH_EPS and float(candidate_metrics @ LOCAL_SEARCH_WEIGHTS) <= score_limit + LOCAL_SEARCH_EPS:
                best_order, best_metrics = (candidate, candidate_metrics)
                best_distance = candidate_metrics[0]
        if best_order is not None:
            order, current = (best_order, best_metrics)
            moves += 1
            phase = '2opt'
        elif not budget_hit and phase == '2opt':
            phase = 'oropt'
            continue
        else:
            reason = 'candidate_budget' if budget_hit else 'local_optimum'
            break
        if budget_hit:
            reason = 'candidate_budget'
            break
    else:
        if n >= 4 and moves >= LOCAL_SEARCH_MAX_MOVES:
            reason = 'move_budget'
    result = [original[i] for i in order]
    validate_route(result, original, original[0], original[-1])
    return (result, {'candidates': examined, 'moves': moves, 'stop': reason})

def solve_cluster(cluster, start_tree, end_tree):
    if not cluster or len(set(cluster)) != len(cluster):
        raise ValueError('Empty cluster or duplicate tree IDs.')
    if len(cluster) == 1:
        route = [start_tree]
    elif len(cluster) == 2:
        route = [start_tree, end_tree]
    else:
        route = _solve_cluster_raw(cluster, start_tree, end_tree)
    route = [int(i) for i in route]
    validate_route(route, cluster, start_tree, end_tree)
    return route

def solver_objective_slots(cluster_size):
    """Report this algorithm's scored slots; trivial clusters bypass search."""
    return 0 if cluster_size <= 3 else POPULATION_SIZE * (GENERATIONS + 1)

def process_cluster(cluster, start_tree, end_tree, cluster_id, run_number):
    seed = BASE_SEED + 100000 * run_number + cluster_id
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    started = time.perf_counter()
    raw_route = solve_cluster(cluster, start_tree, end_tree)
    raw_seconds = time.perf_counter() - started
    before = route_metrics(raw_route)
    started = time.perf_counter()
    route, stats = improve_route(raw_route)
    cleanup_seconds = time.perf_counter() - started
    validate_route(route, cluster, start_tree, end_tree)
    after = route_metrics(route)
    print(f"Cluster {cluster_id + 1}: {before[0]:.2f} -> {after[0]:.2f} m; cleanup {stats['stop']}")
    return (route, {'Map': MAP_NAME, 'Algorithm': ANALYSIS_ALGORITHM, 'Run': run_number, 'Cluster': cluster_id + 1, 'Seed': seed, 'Trees': len(cluster), 'StartTree': start_tree, 'EndTree': end_tree, 'ClusterTreeIDs': repr(sorted(cluster)), 'RawRoute': repr(raw_route), 'ImprovedRoute': repr(route), 'RawDistance': before[0], 'RawTurning': before[1], 'RawElevation': before[2], 'ImprovedDistance': after[0], 'ImprovedTurning': after[1], 'ImprovedElevation': after[2], 'DistanceReductionPercent': 100 * (before[0] - after[0]) / before[0] if before[0] else 0.0, 'RawScore': float(before @ LOCAL_SEARCH_WEIGHTS), 'ImprovedScore': float(after @ LOCAL_SEARCH_WEIGHTS), 'SolverSeconds': raw_seconds, 'CleanupSeconds': cleanup_seconds, 'CleanupCandidates': stats['candidates'], 'CleanupMoves': stats['moves'], 'CleanupStop': stats['stop'], 'SolverObjectiveSlots': solver_objective_slots(len(cluster)), 'CleanupCandidateLimit': LOCAL_SEARCH_MAX_CANDIDATES, 'CleanupMoveLimit': LOCAL_SEARCH_MAX_MOVES, 'CleanupWeights': repr(LOCAL_SEARCH_WEIGHTS.tolist()), 'TurningDefinition': '3D_forward_heading_change_rad'})

def save_improvement_audit(rows, run_id):
    if not rows:
        return
    audit_file = os.path.join(SUMMARY_FOLDER, f'improvement_{run_id}.csv')
    with open(audit_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    for folder in [ROUTES_FOLDER, METRICS_FOLDER, SUMMARY_FOLDER, BASE_FOLDER, KML_FOLDER, PARETO_FOLDER]:
        os.makedirs(folder, exist_ok=True)
    load_map()
    clusters = create_clusters()
    ordered_clusters, bases = order_clusters_and_create_bases(clusters, generated_road)
    START_RUN = get_start_run_number()
    for run_index in range(NUM_RUNS):
        RUN_NUMBER = START_RUN + run_index
        RUN_ID = f'{MAP_NAME}_{RUN_NUMBER}_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        print('\n======================')
        print('NSGA-II RUN', RUN_NUMBER)
        print('======================')
        details = []
        improvement_rows = []
        routes = []
        pareto_output = []
        for cluster_id, cluster in enumerate(ordered_clusters):
            start_base = bases[cluster_id]
            end_base = bases[(cluster_id + 1) % len(bases)]
            start_tree, end_tree = choose_start_end_tree(cluster, start_base, end_base)
            route, audit = process_cluster(cluster, start_tree, end_tree, cluster_id, RUN_NUMBER)
            improvement_rows.append(audit)
            routes.append(route)
            distance = route_distance(route)
            turning = route_turning(route)
            elevation = route_elevation(route)
            details.append({'cluster': cluster_id + 1, 'trees': len(cluster), 'route': route, 'distance': distance, 'turning': turning, 'elevation': elevation, 'start_tree': start_tree, 'end_tree': end_tree, 'start_base': cluster_id, 'end_base': (cluster_id + 1) % len(bases)})
            pareto_output.append([cluster_id + 1, distance, turning, elevation])
        total_distance = sum((x['distance'] for x in details))
        total_turning = sum((x['turning'] for x in details))
        total_elevation = sum((x['elevation'] for x in details))
        pareto_file = os.path.join(PARETO_FOLDER, f'pareto_{RUN_ID}.csv')
        with open(pareto_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Cluster', 'Distance', 'Turning', 'Elevation'])
            writer.writerows(pareto_output)
        metrics_file = os.path.join(METRICS_FOLDER, f'metrics_{RUN_ID}.csv')
        with open(metrics_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Cluster', 'Trees', 'Distance', 'Turning', 'Elevation'])
            for item in details:
                writer.writerow([item['cluster'], item['trees'], item['distance'], item['turning'], item['elevation']])
        routes_file = os.path.join(ROUTES_FOLDER, f'routes_{RUN_ID}.csv')
        with open(routes_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Cluster', 'Route'])
            for item in details:
                writer.writerow([item['cluster'], item['route']])
        save_analysis_data(details, RUN_NUMBER)
        save_improvement_audit(improvement_rows, RUN_ID)
        print('Distance:', total_distance)
        print('Turning:', total_turning)
        print('Elevation:', total_elevation)
        kml_file = os.path.join(KML_FOLDER, f'routes_{RUN_ID}.kml')
        avg_lat = float(np.mean([p[0] for p in gps_points]))
        avg_lon = float(np.mean([p[1] for p in gps_points]))
        colors = ['ff0000ff', 'ff00ff00', 'ffff0000', 'ff00ffff', 'ffffff00', 'ff9900ff', 'ffff00ff', 'ff0099ff', 'ff800000', 'ff008080', 'ff808000', 'ff800080', 'ff0080ff', 'ffff8000', 'ff808080', 'ff00ff99']
        with open(kml_file, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<kml xmlns="http://www.opengis.net/kml/2.2">\n')
            f.write('<Document>\n')
            f.write(f'\n\n<LookAt>\n\n<longitude>{avg_lon}</longitude>\n\n<latitude>{avg_lat}</latitude>\n\n<range>500</range>\n\n<tilt>45</tilt>\n\n</LookAt>\n\n')
            for i, c in enumerate(colors):
                f.write(f'\n\n<Style id="cluster{i}">\n\n<LineStyle>\n\n<color>{c}</color>\n\n<width>4</width>\n\n</LineStyle>\n\n</Style>\n\n')
            f.write('\n\n<Style id="roadStyle">\n\n<LineStyle>\n\n<color>ff00aaff</color>\n\n<width>5</width>\n\n</LineStyle>\n\n</Style>\n\n\n<Style id="accessStyle">\n\n<LineStyle>\n\n<color>ffaaaaaa</color>\n\n<width>2</width>\n\n</LineStyle>\n\n</Style>\n\n\n<Style id="startStyle">\n\n<IconStyle>\n\n<color>ff00ffff</color>\n\n<scale>1.2</scale>\n\n</IconStyle>\n\n</Style>\n\n\n<Style id="endStyle">\n\n<IconStyle>\n\n<color>ff0000ff</color>\n\n<scale>1.2</scale>\n\n</IconStyle>\n\n</Style>\n\n')
            f.write('\n\n<Folder>\n\n<name>Plantation Boundary</name>\n\n<Placemark>\n\n<name>Tree Boundary</name>\n\n<LineString>\n\n<altitudeMode>clampToGround</altitudeMode>\n\n<coordinates>\n\n')
            for p in plantation_boundary + [plantation_boundary[0]]:
                lat, lon = xy_to_latlon(p[0], p[1], lat0, lon0)
                f.write(f'{lon},{lat},0 ')
            f.write('\n\n</coordinates>\n\n</LineString>\n\n</Placemark>\n\n</Folder>\n\n')
            f.write('\n\n<Folder>\n\n<name>Generated Road</name>\n\n<Placemark>\n\n<styleUrl>#roadStyle</styleUrl>\n\n<name>5m Offset Road</name>\n\n<LineString>\n\n<altitudeMode>clampToGround</altitudeMode>\n\n<coordinates>\n\n')
            for p in generated_road + [generated_road[0]]:
                lat, lon = xy_to_latlon(p[0], p[1], lat0, lon0)
                f.write(f'{lon},{lat},0 ')
            f.write('\n\n</coordinates>\n\n</LineString>\n\n</Placemark>\n\n</Folder>\n\n')
            f.write('\n\n<Folder>\n\n<name>Shared Road Bases</name>\n\n')
            for i, b in enumerate(bases):
                lat, lon = xy_to_latlon(b[0], b[1], lat0, lon0)
                f.write(f'\n\n<Placemark>\n\n<name>B{i}</name>\n\n<Point>\n\n<coordinates>\n\n{lon},{lat},0\n\n</coordinates>\n\n</Point>\n\n</Placemark>\n\n')
            f.write('</Folder>\n')
            f.write('\n\n<Folder>\n\n<name>2D Coverage Routes</name>\n\n')
            for item in details:
                colour_id = (item['cluster'] - 1) % len(colors)
                f.write(f"\n\n<Placemark>\n\n<styleUrl>#cluster{colour_id}</styleUrl>\n\n<name>\n\nCluster {item['cluster']} Coverage\n\n</name>\n\n\n<LineString>\n\n<altitudeMode>clampToGround</altitudeMode>\n\n<coordinates>\n\n")
                for tree_id in item['route']:
                    tree = trees[tree_id]
                    lat, lon = xy_to_latlon(tree[0], tree[1], lat0, lon0)
                    f.write(f'{lon},{lat},0 ')
                f.write('\n\n</coordinates>\n\n</LineString>\n\n</Placemark>\n\n')
            f.write('</Folder>\n')
            f.write('\n\n<Folder>\n\n<name>Access Legs - Not Counted</name>\n\n')
            for item in details:
                start_base = bases[item['start_base']]
                end_base = bases[item['end_base']]
                start_tree = trees[item['start_tree']]
                end_tree = trees[item['end_tree']]
                for name, points in [('Base To Start Palm', [start_base, start_tree]), ('End Palm To Base', [end_tree, end_base])]:
                    f.write(f'\n\n<Placemark>\n\n<styleUrl>#accessStyle</styleUrl>\n\n<name>\n\n{name}\n\n</name>\n\n\n<LineString>\n\n<altitudeMode>clampToGround</altitudeMode>\n\n<coordinates>\n\n')
                    for p in points:
                        if len(p) == 2:
                            lat, lon = xy_to_latlon(p[0], p[1], lat0, lon0)
                        else:
                            lat, lon = xy_to_latlon(p[0], p[1], lat0, lon0)
                        f.write(f'{lon},{lat},0 ')
                    f.write('\n\n</coordinates>\n\n</LineString>\n\n</Placemark>\n\n')
            f.write('\n\n</Folder>\n\n')
            f.write('\n\n<Folder>\n\n<name>Cluster Start End Palms</name>\n\n')
            for item in details:
                for label, tree_id, style in [('Start Palm', item['start_tree'], 'startStyle'), ('End Palm', item['end_tree'], 'endStyle')]:
                    tree = trees[tree_id]
                    lat, lon = xy_to_latlon(tree[0], tree[1], lat0, lon0)
                    f.write(f"\n\n<Placemark>\n\n<styleUrl>#{style}</styleUrl>\n\n<name>\n\nCluster {item['cluster']} {label}\n\n</name>\n\n\n<Point>\n\n<coordinates>\n\n{lon},{lat},{tree[2]}\n\n</coordinates>\n\n</Point>\n\n</Placemark>\n\n")
            f.write('\n\n</Folder>\n\n')
            f.write('\n\n<Folder>\n\n<name>3D Coverage Routes</name>\n\n')
            for item in details:
                colour_id = (item['cluster'] - 1) % len(colors)
                f.write(f"\n\n<Placemark>\n\n<styleUrl>#cluster{colour_id}</styleUrl>\n\n\n<name>\n\nCluster {item['cluster']} 3D Drone Path\n\n</name>\n\n\n<LineString>\n\n\n<tessellate>0</tessellate>\n\n\n<altitudeMode>absolute</altitudeMode>\n\n\n<coordinates>\n\n")
                for tree_id in item['route']:
                    tree = trees[tree_id]
                    lat, lon = xy_to_latlon(tree[0], tree[1], lat0, lon0)
                    altitude = tree[2] + DRONE_OFFSET
                    f.write(f'{lon},{lat},{altitude} ')
                f.write('\n\n</coordinates>\n\n</LineString>\n\n</Placemark>\n\n')
            f.write('\n\n</Folder>\n\n')
            f.write('\n\n</Document>\n\n</kml>\n\n')
        print()
        print('KML saved:', kml_file)
    print('\nNSGA-II RUN COMPLETED:', RUN_NUMBER)
if __name__ == '__main__':
    main()

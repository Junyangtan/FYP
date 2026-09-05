# ============================================================
# NSGAII_3D.py
# Automated Drone Path Planning
# NSGA-II Multi-objective Optimisation
#
# Output:
#   - Metrics CSV
#   - Routes CSV
#   - Summary TXT
#   - Pareto CSV
#   - KML
#   - analysis_data.csv
#
# ============================================================


import numpy as np
import csv
import os
import math
import random

from datetime import datetime

from sklearn.cluster import KMeans




# ============================================================
# USER CONFIGURATION
# ============================================================


NUM_RUNS = 10


MAP_NAME = "Map2"



ALGORITHM_NAME = "NSGAII_3D_v6"

ANALYSIS_ALGORITHM = "NSGAII"





# ============================================================
# OUTPUT FOLDERS
# ============================================================


OUTPUT_FOLDER = "Output"


NSGA_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "NSGAII_3D"
)


ROUTES_FOLDER = os.path.join(
    NSGA_FOLDER,
    "Routes"
)


METRICS_FOLDER = os.path.join(
    NSGA_FOLDER,
    "Metrics"
)


SUMMARY_FOLDER = os.path.join(
    NSGA_FOLDER,
    "Summary"
)


BASE_FOLDER = os.path.join(
    NSGA_FOLDER,
    "Bases"
)


KML_FOLDER = os.path.join(
    NSGA_FOLDER,
    "KML"
)


PARETO_FOLDER = os.path.join(
    NSGA_FOLDER,
    "Pareto"
)



for folder in [

    ROUTES_FOLDER,

    METRICS_FOLDER,

    SUMMARY_FOLDER,

    BASE_FOLDER,

    KML_FOLDER,

    PARETO_FOLDER

]:

    os.makedirs(
        folder,
        exist_ok=True
    )






# ============================================================
# ANALYSIS FILE
# ============================================================


ANALYSIS_FILE = os.path.join(

    OUTPUT_FOLDER,

    "analysis_data1.csv"

)







# ============================================================
# NSGA-II PARAMETERS
# ============================================================


POPULATION_SIZE = 200


GENERATIONS = 200


CROSSOVER_RATE = 0.8


MUTATION_RATE = 0.05







# ============================================================
# CLUSTER PARAMETERS
# ============================================================


MAX_TREES = 80


ELEVATION_WEIGHT = 5.0






# ============================================================
# ROAD PARAMETERS
# ============================================================


ROAD_OFFSET = 5.0


DRONE_OFFSET = 3.0







# ============================================================
# MAP CONFIGURATION
# ============================================================


MAP_CONFIG = {


    "Map1": {


        "csv_file":
        "Map1_3D_map.csv",


        "base_lat":
        2.70425,


        "base_lon":
        101.633375

    },



    "Map2": {


        "csv_file":
        "Map2_3D_map.csv",


        "base_lat":
        2.70550,


        "base_lon":
        101.634800

    }

}



SELECTED_MAP = MAP_CONFIG[MAP_NAME]




# ============================================================
# EARTH CONSTANT
# ============================================================


EARTH_RADIUS = 6371000.0







# ============================================================
# GLOBAL DATA
# ============================================================


trees = []


gps_points = []


lat0 = None


lon0 = None


plantation_boundary = []


generated_road = []







# ============================================================
# COORDINATE CONVERSION
# ============================================================


def latlon_to_xy(

    lat,

    lon,

    ref_lat,

    ref_lon

):


    dlat = np.radians(

        lat - ref_lat

    )


    dlon = np.radians(

        lon - ref_lon

    )


    ref = np.radians(

        ref_lat

    )



    x = (

        dlon

        *

        np.cos(ref)

        *

        EARTH_RADIUS

    )



    y = (

        dlat

        *

        EARTH_RADIUS

    )


    return (

        float(x),

        float(y)

    )







def xy_to_latlon(

    x,

    y,

    ref_lat,

    ref_lon

):


    lat = (

        ref_lat

        +

        np.degrees(

            y / EARTH_RADIUS

        )

    )


    lon = (

        ref_lon

        +

        np.degrees(

            x

            /

            (

                EARTH_RADIUS

                *

                np.cos(

                    np.radians(ref_lat)

                )

            )

        )

    )


    return (

        float(lat),

        float(lon)

    )







def drone_point(tree_id):


    tree = trees[tree_id]


    return np.array(

        [

            tree[0],

            tree[1],

            tree[2]

            +

            DRONE_OFFSET

        ],

        dtype=float

    )








# ============================================================
# OBJECTIVE FUNCTIONS
# ============================================================


def route_distance(route):


    total = 0.0



    for i in range(

        len(route)-1

    ):


        p1 = drone_point(

            route[i]

        )


        p2 = drone_point(

            route[i+1]

        )



        total += np.linalg.norm(

            p2-p1

        )



    return float(total)







def route_turning(route):


    if len(route)<3:

        return 0.0



    total=0.0



    points=[

        drone_point(i)

        for i in route

    ]



    for i in range(

        1,

        len(points)-1

    ):


        a = points[i-1]-points[i]


        b = points[i+1]-points[i]



        cosine=(

            np.dot(a,b)

            /

            (

                np.linalg.norm(a)

                *

                np.linalg.norm(b)

                +

                1e-9

            )

        )



        total += np.arccos(

            np.clip(

                cosine,

                -1,

                1

            )

        )



    return float(total)








def route_elevation(route):


    total=0.0



    for i in range(

        len(route)-1

    ):


        total += abs(

            trees[route[i+1]][2]

            -

            trees[route[i]][2]

        )



    return float(total)








def evaluate_route(route):


    return (

        route_distance(route),

        route_turning(route),

        route_elevation(route)

    )
# ============================================================
# GEOMETRY FUNCTIONS
# ============================================================


def cross_2d(a,b):

    return (

        a[0]*b[1]

        -

        a[1]*b[0]

    )







def convex_hull(points):


    pts = sorted(

        set(

            (

                float(p[0]),

                float(p[1])

            )

            for p in points

        )

    )



    def orientation(o,a,b):

        return (

            (a[0]-o[0])*(b[1]-o[1])

            -

            (a[1]-o[1])*(b[0]-o[0])

        )



    lower=[]



    for p in pts:


        while (

            len(lower)>=2

            and

            orientation(

                lower[-2],

                lower[-1],

                p

            ) <= 0

        ):

            lower.pop()



        lower.append(p)





    upper=[]



    for p in reversed(pts):


        while (

            len(upper)>=2

            and

            orientation(

                upper[-2],

                upper[-1],

                p

            ) <= 0

        ):

            upper.pop()



        upper.append(p)





    hull=(

        lower[:-1]

        +

        upper[:-1]

    )



    return [

        np.array(

            p,

            dtype=float

        )

        for p in hull

    ]









def polygon_area(poly):


    area=0.0



    for i in range(len(poly)):


        p=poly[i]

        q=poly[(i+1)%len(poly)]



        area += (

            p[0]*q[1]

            -

            q[0]*p[1]

        )



    return area/2







def line_intersection(
    p,
    r,
    q,
    s
):


    denominator=cross_2d(

        r,

        s

    )



    if abs(denominator)<1e-12:

        return None



    t=(

        cross_2d(

            q-p,

            s

        )

        /

        denominator

    )



    return p+t*r








def offset_convex_polygon(
    polygon,
    offset
):


    poly=[

        np.array(p)

        for p in polygon

    ]



    if polygon_area(poly)<0:

        poly.reverse()



    road=[]



    for i in range(len(poly)):


        prev=poly[

            (i-1)%len(poly)

        ]


        curr=poly[i]


        nxt=poly[

            (i+1)%len(poly)

        ]



        e1=curr-prev

        e2=nxt-curr



        n1=np.array(

            [

                e1[1],

                -e1[0]

            ]

        )



        n2=np.array(

            [

                e2[1],

                -e2[0]

            ]

        )



        n1/=np.linalg.norm(n1)


        n2/=np.linalg.norm(n2)



        p1=curr+offset*n1


        p2=curr+offset*n2



        intersection=line_intersection(

            p1,

            e1,

            p2,

            e2

        )



        if intersection is None:


            normal=n1+n2


            normal/=np.linalg.norm(normal)


            intersection=(

                curr

                +

                offset*normal

            )



        road.append(intersection)



    return road








def segment_lengths(poly):


    return np.array(

        [

            np.linalg.norm(

                poly[(i+1)%len(poly)]

                -

                poly[i]

            )

            for i in range(len(poly))

        ]

    )







def road_length(road):


    return float(

        np.sum(

            segment_lengths(road)

        )

    )







def project_to_road(
    point,
    road
):


    point=np.asarray(point)



    lengths=segment_lengths(
        road
    )



    best_distance=float("inf")


    best_point=None


    best_s=0



    travelled=0



    for i in range(len(road)):


        A=road[i]


        B=road[(i+1)%len(road)]



        AB=B-A



        denominator=np.dot(

            AB,

            AB

        )



        if denominator==0:

            continue



        t=np.dot(

            point-A,

            AB

        )/denominator



        t=np.clip(

            t,

            0,

            1

        )



        q=A+t*AB



        d=np.linalg.norm(

            point-q

        )



        if d < best_distance:


            best_distance=d


            best_point=q


            best_s=(

                travelled

                +

                t*lengths[i]

            )



        travelled+=lengths[i]



    return (

        best_point,

        best_s,

        best_distance

    )







def interpolate_road(
    road,
    s
):


    lengths=segment_lengths(
        road
    )


    total=np.sum(lengths)



    s=s%total



    travelled=0



    for i,length in enumerate(lengths):


        if s <= travelled+length:


            ratio=(

                s-travelled

            )/length



            return (

                road[i]

                +

                ratio*(

                    road[(i+1)%len(road)]

                    -

                    road[i]

                )

            )



        travelled+=length



    return road[0]







def circular_distance(
    a,
    b,
    total
):


    d=abs(a-b)%total


    return min(

        d,

        total-d

    )









# ============================================================
# CLUSTERING
# ============================================================


def create_clusters():


    data=np.array(

        [

            [

                t[0],

                t[1],

                t[2]*ELEVATION_WEIGHT

            ]

            for t in trees

        ]

    )



    number=max(

        1,

        math.ceil(

            len(trees)/MAX_TREES

        )

    )



    while True:


        model=KMeans(

            n_clusters=number,

            n_init=20,

            random_state=0

        )



        labels=model.fit_predict(
            data
        )



        clusters=[

            []

            for _ in range(number)

        ]



        for i,label in enumerate(labels):


            clusters[label].append(i)



        if max(

            len(c)

            for c in clusters

        ) <= MAX_TREES:


            return clusters



        number+=1







def cluster_centroid(cluster):


    return np.mean(

        [

            [

                trees[i][0],

                trees[i][1]

            ]

            for i in cluster

        ],

        axis=0

    )







def order_clusters_and_create_bases(
    clusters,
    road
):


    total=road_length(road)



    info=[]



    for cluster in clusters:


        centroid=cluster_centroid(cluster)



        _,s,_=project_to_road(

            centroid,

            road

        )



        info.append(

            {

                "cluster":cluster,

                "s":s

            }

        )



    info.sort(

        key=lambda x:x["s"]

    )



    bases=[]



    for i in range(len(info)):


        s1=info[i]["s"]


        s2=info[

            (i+1)%len(info)

        ]["s"]



        if i==len(info)-1:

            s2+=total



        middle=(

            s1+s2

        )/2



        bases.append(

            interpolate_road(

                road,

                middle

            )

        )



    ordered=[

        x["cluster"]

        for x in info

    ]



    return ordered,bases







def choose_start_end_tree(
    cluster,
    start_base,
    end_base
):


    start=min(

        cluster,

        key=lambda i:

        np.linalg.norm(

            np.array(

                [

                    trees[i][0],

                    trees[i][1]

                ]

            )

            -

            start_base

        )

    )



    remaining=[

        x

        for x in cluster

        if x!=start

    ]



    end=min(

        remaining,

        key=lambda i:

        np.linalg.norm(

            np.array(

                [

                    trees[i][0],

                    trees[i][1]

                ]

            )

            -

            end_base

        )

    )



    return start,end
# ============================================================
# NSGA-II OPERATORS
# ============================================================


def create_individual(
    middle_trees
):

    chromosome=list(

        range(

            len(middle_trees)

        )

    )


    random.shuffle(
        chromosome
    )


    return chromosome







def decode_route(
    chromosome,
    middle_trees,
    start_tree,
    end_tree
):


    return (

        [

            start_tree

        ]

        +

        [

            middle_trees[i]

            for i in chromosome

        ]

        +

        [

            end_tree

        ]

    )









# ============================================================
# OBJECTIVE EVALUATION
# ============================================================


def evaluate_population(
    population,
    middle_trees,
    start_tree,
    end_tree
):


    result=[]



    for chromosome in population:


        route=decode_route(

            chromosome,

            middle_trees,

            start_tree,

            end_tree

        )



        result.append(

            evaluate_route(

                route

            )

        )



    return result







# ============================================================
# NON-DOMINATION
# ============================================================


def dominates(
    a,
    b
):


    return (

        all(

            x <= y

            for x,y in zip(a,b)

        )

        and

        any(

            x < y

            for x,y in zip(a,b)

        )

    )







def fast_non_dominated_sort(
    objectives
):


    domination_count=[

        0

        for _ in objectives

    ]



    dominated=[

        []

        for _ in objectives

    ]



    fronts=[[]]



    for p in range(

        len(objectives)

    ):


        for q in range(

            len(objectives)

        ):


            if p==q:

                continue



            if dominates(

                objectives[p],

                objectives[q]

            ):


                dominated[p].append(q)



            elif dominates(

                objectives[q],

                objectives[p]

            ):


                domination_count[p]+=1





        if domination_count[p]==0:


            fronts[0].append(p)





    i=0



    while len(fronts[i])>0:


        next_front=[]



        for p in fronts[i]:


            for q in dominated[p]:


                domination_count[q]-=1



                if domination_count[q]==0:


                    next_front.append(q)



        i+=1


        fronts.append(

            next_front

        )



    return fronts[:-1]









# ============================================================
# CROWDING DISTANCE
# ============================================================


def calculate_crowding_distance(
    front,
    objectives
):


    distance={

        i:0.0

        for i in front

    }



    if len(front)<=2:


        for i in front:

            distance[i]=float(
                "inf"
            )


        return distance





    number_objectives=len(

        objectives[0]

    )



    for obj in range(

        number_objectives

    ):



        sorted_front=sorted(

            front,

            key=lambda i:

            objectives[i][obj]

        )



        distance[

            sorted_front[0]

        ]=float(

            "inf"

        )


        distance[

            sorted_front[-1]

        ]=float(

            "inf"

        )



        minimum=objectives[

            sorted_front[0]

        ][obj]



        maximum=objectives[

            sorted_front[-1]

        ][obj]



        if maximum-minimum==0:

            continue





        for i in range(

            1,

            len(sorted_front)-1

        ):


            previous=objectives[

                sorted_front[i-1]

            ][obj]



            following=objectives[

                sorted_front[i+1]

            ][obj]



            distance[

                sorted_front[i]

            ] += (

                following

                -

                previous

            )/(

                maximum

                -

                minimum

            )



    return distance








# ============================================================
# TOURNAMENT SELECTION
# ============================================================


def tournament_selection(
    population,
    rank,
    crowding
):


    a,b=random.sample(

        range(

            len(population)

        ),

        2

    )



    if rank[a] < rank[b]:


        return population[a].copy()



    if rank[b] < rank[a]:


        return population[b].copy()



    if crowding[a] > crowding[b]:


        return population[a].copy()



    return population[b].copy()








# ============================================================
# ORDER CROSSOVER
# ============================================================


def ordered_crossover(
    parent1,
    parent2
):


    size=len(parent1)



    if size<2:


        return (

            parent1.copy(),

            parent2.copy()

        )





    a,b=sorted(

        random.sample(

            range(size),

            2

        )

    )



    child1=[None]*size


    child2=[None]*size



    child1[a:b]=parent1[a:b]


    child2[a:b]=parent2[a:b]





    fill1=[

        x

        for x in parent2

        if x not in child1

    ]



    fill2=[

        x

        for x in parent1

        if x not in child2

    ]





    index1=0


    index2=0



    for i in range(size):


        if child1[i] is None:


            child1[i]=fill1[index1]

            index1+=1




        if child2[i] is None:


            child2[i]=fill2[index2]

            index2+=1



    return (

        child1,

        child2

    )









# ============================================================
# MUTATION
# ============================================================


def mutate(
    chromosome
):


    if random.random()<MUTATION_RATE:


        a,b=sorted(

            random.sample(

                range(

                    len(chromosome)

                ),

                2

            )

        )


        chromosome[a:b]=reversed(

            chromosome[a:b]

        )



    return chromosome










# ============================================================
# NSGA-II CLUSTER SOLVER
# ============================================================


def solve_cluster(
    cluster,
    start_tree,
    end_tree
):


    middle_trees=[

        i

        for i in cluster

        if i not in (

            start_tree,

            end_tree

        )

    ]




    if len(middle_trees)==0:


        return [

            start_tree,

            end_tree

        ]





    if len(middle_trees)==1:


        return [

            start_tree,

            middle_trees[0],

            end_tree

        ]







    population=[

        create_individual(

            middle_trees

        )

        for _ in range(

            POPULATION_SIZE

        )

    ]






    for generation in range(

        GENERATIONS

    ):



        objectives=evaluate_population(

            population,

            middle_trees,

            start_tree,

            end_tree

        )



        fronts=fast_non_dominated_sort(

            objectives

        )




        rank=[0]*len(population)


        crowding=[0]*len(population)





        for r,front in enumerate(fronts):


            crowd=calculate_crowding_distance(

                front,

                objectives

            )



            for index in front:


                rank[index]=r


                crowding[index]=crowd[index]





        offspring=[]



        while len(offspring)<POPULATION_SIZE:


            parent1=tournament_selection(

                population,

                rank,

                crowding

            )


            parent2=tournament_selection(

                population,

                rank,

                crowding

            )





            if random.random()<CROSSOVER_RATE:


                child1,child2=ordered_crossover(

                    parent1,

                    parent2

                )

            else:


                child1=parent1.copy()

                child2=parent2.copy()



            offspring.append(

                mutate(child1)

            )



            if len(offspring)<POPULATION_SIZE:


                offspring.append(

                    mutate(child2)

                )







        combined=(

            population

            +

            offspring

        )



        combined_objectives=evaluate_population(

            combined,

            middle_trees,

            start_tree,

            end_tree

        )



        fronts=fast_non_dominated_sort(

            combined_objectives

        )



        new_population=[]




        for front in fronts:


            if (

                len(new_population)

                +

                len(front)

                <=

                POPULATION_SIZE

            ):


                new_population.extend(

                    [

                        combined[i]

                        for i in front

                    ]

                )



            else:


                crowd=calculate_crowding_distance(

                    front,

                    combined_objectives

                )


                sorted_front=sorted(

                    front,

                    key=lambda x:

                    crowd[x],

                    reverse=True

                )



                remaining=(

                    POPULATION_SIZE

                    -

                    len(new_population)

                )



                new_population.extend(

                    [

                        combined[i]

                        for i in sorted_front[:remaining]

                    ]

                )


                break



        population=new_population






    # ========================================================
    # FINAL PARETO SELECTION
    # ========================================================


    final_objectives=evaluate_population(

        population,

        middle_trees,

        start_tree,

        end_tree

    )



    fronts=fast_non_dominated_sort(

        final_objectives

    )



    pareto_front=fronts[0]




    best=min(

        pareto_front,

        key=lambda i:

        sum(

            final_objectives[i]

        )

    )





    return decode_route(

        population[best],

        middle_trees,

        start_tree,

        end_tree

    )
# ============================================================
# LOAD MAP DATA
# ============================================================


def load_map():


    global gps_points
    global trees
    global lat0
    global lon0
    global plantation_boundary
    global generated_road



    filename=SELECTED_MAP["csv_file"]



    possible_paths=[

        filename,

        os.path.join(
            "Maps",
            filename
        ),

        os.path.join(
            "..",
            filename
        )

    ]



    map_path=None



    for path in possible_paths:


        if os.path.isfile(path):

            map_path=path

            break




    if map_path is None:


        raise FileNotFoundError(

            f"Cannot find {filename}"

        )





    with open(

        map_path,

        "r",

        newline=""

    ) as file:


        reader=csv.DictReader(file)



        for row in reader:


            gps_points.append(

                (

                    float(row["lat"]),

                    float(row["lon"]),

                    float(row["alt"])

                )

            )





    lat0=gps_points[0][0]

    lon0=gps_points[0][1]





    for lat,lon,alt in gps_points:


        x,y=latlon_to_xy(

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





    tree_xy=np.array(

        [

            [

                t[0],

                t[1]

            ]

            for t in trees

        ]

    )



    plantation_boundary=convex_hull(

        tree_xy

    )



    generated_road=offset_convex_polygon(

        plantation_boundary,

        ROAD_OFFSET

    )





    print(

        f"Loaded {len(trees)} trees"

    )


    print(

        f"Road length: "
        f"{road_length(generated_road):.2f} m"

    )







# ============================================================
# RUN NUMBER
# ============================================================


def get_start_run_number():


    if not os.path.isfile(

        ANALYSIS_FILE

    ):


        return 1



    max_run=0



    with open(

        ANALYSIS_FILE,

        "r"

    ) as file:


        reader=csv.DictReader(
            file
        )



        for row in reader:


            if (

                row["Map"]

                ==

                MAP_NAME

                and

                row["Algorithm"]

                ==

                ANALYSIS_ALGORITHM

            ):


                max_run=max(

                    max_run,

                    int(

                        row["Run"]

                    )

                )



    return max_run+1







# ============================================================
# SAVE ANALYSIS DATA
# ============================================================


def save_analysis_data(

    details,

    run_number

):


    file_exists=os.path.isfile(

        ANALYSIS_FILE

    )



    with open(

        ANALYSIS_FILE,

        "a",

        newline=""

    ) as file:


        writer=csv.writer(file)



        if not file_exists:


            writer.writerow(

                [

                    "Map",

                    "Cluster",

                    "Algorithm",

                    "Run",

                    "Distance",

                    "Elevation",

                    "Turning"

                ]

            )





        for item in details:


            writer.writerow(

                [

                    MAP_NAME,

                    item["cluster"],

                    ANALYSIS_ALGORITHM,

                    run_number,

                    item["distance"],

                    item["elevation"],

                    item["turning"]

                ]

            )








# ============================================================
# MAIN PROGRAM
# ============================================================


load_map()

# ------------------------------------
# Generate clusters
# ------------------------------------


clusters=create_clusters()



ordered_clusters,bases=order_clusters_and_create_bases(

    clusters,

    generated_road

)

START_RUN=get_start_run_number()





for run_index in range(

    NUM_RUNS

):


    RUN_NUMBER=(

        START_RUN

        +

        run_index

    )



    RUN_ID=datetime.now().strftime(

        "%Y%m%d_%H%M%S"

    )



    print("\n======================")

    print(

        "NSGA-II RUN",

        RUN_NUMBER

    )

    print("======================")



    details=[]


    routes=[]


    pareto_output=[]






    # ------------------------------------
    # Optimise every cluster
    # ------------------------------------


    for cluster_id,cluster in enumerate(

        ordered_clusters

    ):


        start_base=bases[

            cluster_id

        ]



        end_base=bases[

            (

                cluster_id+1

            )

            %

            len(bases)

        ]




        start_tree,end_tree=choose_start_end_tree(

            cluster,

            start_base,

            end_base

        )





        route=solve_cluster(

            cluster,

            start_tree,

            end_tree

        )



        routes.append(

            route

        )




        distance=route_distance(

            route

        )


        turning=route_turning(

            route

        )


        elevation=route_elevation(

            route

        )






        details.append(

            {


                "cluster":

                cluster_id+1,



                "trees":

                len(cluster),



                "route":

                route,



                "distance":

                distance,



                "turning":

                turning,



                "elevation":

                elevation,



                "start_tree":

                start_tree,



                "end_tree":

                end_tree,



                "start_base":

                cluster_id,



                "end_base":

                (

                    cluster_id+1

                )

                %

                len(bases)

            }

        )






        pareto_output.append(

            [

                cluster_id+1,

                distance,

                turning,

                elevation

            ]

        )







    total_distance=sum(

        x["distance"]

        for x in details

    )



    total_turning=sum(

        x["turning"]

        for x in details

    )



    total_elevation=sum(

        x["elevation"]

        for x in details

    )







    # ========================================================
    # SAVE PARETO
    # ========================================================


    pareto_file=os.path.join(

        PARETO_FOLDER,

        f"pareto_{RUN_ID}.csv"

    )



    with open(

        pareto_file,

        "w",

        newline=""

    ) as file:


        writer=csv.writer(file)



        writer.writerow(

            [

                "Cluster",

                "Distance",

                "Turning",

                "Elevation"

            ]

        )


        writer.writerows(

            pareto_output

        )






    # ========================================================
    # SAVE METRICS
    # ========================================================


    metrics_file=os.path.join(

        METRICS_FOLDER,

        f"metrics_{RUN_ID}.csv"

    )



    with open(

        metrics_file,

        "w",

        newline=""

    ) as file:


        writer=csv.writer(file)



        writer.writerow(

            [

                "Cluster",

                "Trees",

                "Distance",

                "Turning",

                "Elevation"

            ]

        )



        for item in details:


            writer.writerow(

                [

                    item["cluster"],

                    item["trees"],

                    item["distance"],

                    item["turning"],

                    item["elevation"]

                ]

            )








    # ========================================================
    # SAVE ROUTES
    # ========================================================


    routes_file=os.path.join(

        ROUTES_FOLDER,

        f"routes_{RUN_ID}.csv"

    )



    with open(

        routes_file,

        "w",

        newline=""

    ) as file:


        writer=csv.writer(file)



        writer.writerow(

            [

                "Cluster",

                "Route"

            ]

        )



        for item in details:


            writer.writerow(

                [

                    item["cluster"],

                    item["route"]

                ]

            )







    # ========================================================
    # SAVE ANALYSIS
    # ========================================================


    save_analysis_data(

        details,

        RUN_NUMBER

    )





    print(

        "Distance:",

        total_distance

    )


    print(

        "Turning:",

        total_turning

    )


    print(

        "Elevation:",

        total_elevation

    )
    # ========================================================
    # KML EXPORT (COLOURED + 3D VERSION)
    # ========================================================


    kml_file=os.path.join(

        KML_FOLDER,

        f"routes_{RUN_ID}.kml"

    )



    avg_lat=float(

        np.mean(

            [

                p[0]

                for p in gps_points

            ]

        )

    )


    avg_lon=float(

        np.mean(

            [

                p[1]

                for p in gps_points

            ]

        )

    )




    colors=[

        "ff0000ff",   # red

        "ff00ff00",   # green

        "ffff0000",   # blue

        "ff00ffff",   # yellow

        "ffffff00",   # cyan

        "ff9900ff",   # orange

        "ffff00ff",   # magenta

        "ff0099ff",

        "ff800000",

        "ff008080",

        "ff808000",

        "ff800080",

        "ff0080ff",

        "ffff8000",

        "ff808080",

        "ff00ff99"

    ]





    with open(

        kml_file,

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





        # ----------------------------------------------------
        # CAMERA VIEW
        # ----------------------------------------------------


        f.write(

f"""

<LookAt>

<longitude>{avg_lon}</longitude>

<latitude>{avg_lat}</latitude>

<range>500</range>

<tilt>45</tilt>

</LookAt>

"""

        )







        # ----------------------------------------------------
        # CLUSTER STYLES
        # ----------------------------------------------------


        for i,c in enumerate(colors):


            f.write(

f"""

<Style id="cluster{i}">

<LineStyle>

<color>{c}</color>

<width>4</width>

</LineStyle>

</Style>

"""

            )





        f.write(

"""

<Style id="roadStyle">

<LineStyle>

<color>ff00aaff</color>

<width>5</width>

</LineStyle>

</Style>


<Style id="accessStyle">

<LineStyle>

<color>ffaaaaaa</color>

<width>2</width>

</LineStyle>

</Style>


<Style id="startStyle">

<IconStyle>

<color>ff00ffff</color>

<scale>1.2</scale>

</IconStyle>

</Style>


<Style id="endStyle">

<IconStyle>

<color>ff0000ff</color>

<scale>1.2</scale>

</IconStyle>

</Style>

"""

        )









        # ====================================================
        # PLANTATION BOUNDARY
        # ====================================================


        f.write(

"""

<Folder>

<name>Plantation Boundary</name>

<Placemark>

<name>Tree Boundary</name>

<LineString>

<altitudeMode>clampToGround</altitudeMode>

<coordinates>

"""

        )



        for p in plantation_boundary+[plantation_boundary[0]]:


            lat,lon=xy_to_latlon(

                p[0],

                p[1],

                lat0,

                lon0

            )


            f.write(

                f"{lon},{lat},0 "

            )



        f.write(

"""

</coordinates>

</LineString>

</Placemark>

</Folder>

"""

        )









        # ====================================================
        # GENERATED ROAD
        # ====================================================


        f.write(

"""

<Folder>

<name>Generated Road</name>

<Placemark>

<styleUrl>#roadStyle</styleUrl>

<name>5m Offset Road</name>

<LineString>

<altitudeMode>clampToGround</altitudeMode>

<coordinates>

"""

        )


        for p in generated_road+[generated_road[0]]:


            lat,lon=xy_to_latlon(

                p[0],

                p[1],

                lat0,

                lon0

            )


            f.write(

                f"{lon},{lat},0 "

            )



        f.write(

"""

</coordinates>

</LineString>

</Placemark>

</Folder>

"""

        )









        # ====================================================
        # BASE POINTS
        # ====================================================


        f.write(

"""

<Folder>

<name>Shared Road Bases</name>

"""

        )



        for i,b in enumerate(bases):


            lat,lon=xy_to_latlon(

                b[0],

                b[1],

                lat0,

                lon0

            )


            f.write(

f"""

<Placemark>

<name>B{i}</name>

<Point>

<coordinates>

{lon},{lat},0

</coordinates>

</Point>

</Placemark>

"""

            )



        f.write(

"</Folder>\n"

        )









        # ====================================================
        # 2D ROUTES
        # ====================================================


        f.write(

"""

<Folder>

<name>2D Coverage Routes</name>

"""

        )



        for item in details:


            colour_id=(

                item["cluster"]-1

            ) % len(colors)



            f.write(

f"""

<Placemark>

<styleUrl>#cluster{colour_id}</styleUrl>

<name>

Cluster {item["cluster"]} Coverage

</name>


<LineString>

<altitudeMode>clampToGround</altitudeMode>

<coordinates>

"""

            )



            for tree_id in item["route"]:


                tree=trees[tree_id]


                lat,lon=xy_to_latlon(

                    tree[0],

                    tree[1],

                    lat0,

                    lon0

                )


                f.write(

                    f"{lon},{lat},0 "

                )



            f.write(

"""

</coordinates>

</LineString>

</Placemark>

"""

            )



        f.write(

"</Folder>\n"

        )
        # ====================================================
        # ACCESS LEGS - NOT COUNTED
        # ====================================================


        f.write(

"""

<Folder>

<name>Access Legs - Not Counted</name>

"""

        )



        for item in details:


            start_base=bases[

                item["start_base"]

            ]


            end_base=bases[

                item["end_base"]

            ]



            start_tree=trees[

                item["start_tree"]

            ]


            end_tree=trees[

                item["end_tree"]

            ]





            for name,points in [

                (

                    "Base To Start Palm",

                    [

                        start_base,

                        start_tree

                    ]

                ),


                (

                    "End Palm To Base",

                    [

                        end_tree,

                        end_base

                    ]

                )

            ]:


                f.write(

f"""

<Placemark>

<styleUrl>#accessStyle</styleUrl>

<name>

{name}

</name>


<LineString>

<altitudeMode>clampToGround</altitudeMode>

<coordinates>

"""

                )



                for p in points:


                    if len(p)==2:


                        lat,lon=xy_to_latlon(

                            p[0],

                            p[1],

                            lat0,

                            lon0

                        )


                    else:


                        lat,lon=xy_to_latlon(

                            p[0],

                            p[1],

                            lat0,

                            lon0

                        )



                    f.write(

                        f"{lon},{lat},0 "

                    )



                f.write(

"""

</coordinates>

</LineString>

</Placemark>

"""

                )



        f.write(

"""

</Folder>

"""

        )









        # ====================================================
        # CLUSTER START END PALMS
        # ====================================================


        f.write(

"""

<Folder>

<name>Cluster Start End Palms</name>

"""

        )



        for item in details:


            for label,tree_id,style in [

                (

                    "Start Palm",

                    item["start_tree"],

                    "startStyle"

                ),


                (

                    "End Palm",

                    item["end_tree"],

                    "endStyle"

                )

            ]:


                tree=trees[tree_id]



                lat,lon=xy_to_latlon(

                    tree[0],

                    tree[1],

                    lat0,

                    lon0

                )



                f.write(

f"""

<Placemark>

<styleUrl>#{style}</styleUrl>

<name>

Cluster {item["cluster"]} {label}

</name>


<Point>

<coordinates>

{lon},{lat},{tree[2]}

</coordinates>

</Point>

</Placemark>

"""

                )



        f.write(

"""

</Folder>

"""

        )









        # ====================================================
        # 3D COVERAGE ROUTES
        # ====================================================


        f.write(

"""

<Folder>

<name>3D Coverage Routes</name>

"""

        )



        for item in details:


            colour_id=(

                item["cluster"]-1

            ) % len(colors)



            f.write(

f"""

<Placemark>

<styleUrl>#cluster{colour_id}</styleUrl>


<name>

Cluster {item["cluster"]} 3D Drone Path

</name>


<LineString>


<tessellate>0</tessellate>


<altitudeMode>absolute</altitudeMode>


<coordinates>

"""

            )



            for tree_id in item["route"]:


                tree=trees[tree_id]



                lat,lon=xy_to_latlon(

                    tree[0],

                    tree[1],

                    lat0,

                    lon0

                )


                altitude=(

                    tree[2]

                    +

                    DRONE_OFFSET

                )



                f.write(

                    f"{lon},{lat},{altitude} "

                )



            f.write(

"""

</coordinates>

</LineString>

</Placemark>

"""

            )



        f.write(

"""

</Folder>

"""

        )









        # ====================================================
        # CLOSE KML
        # ====================================================


        f.write(

"""

</Document>

</kml>

"""

        )





    print()

    print(

        "KML saved:",

        kml_file

    )

print(

    "\nNSGA-II RUN COMPLETED:",

    RUN_NUMBER

)


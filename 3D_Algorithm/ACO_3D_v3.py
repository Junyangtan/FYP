# ============================================================
# ACO_3D.py
# Automated Drone Path Planning
# Ant Colony Optimisation
#
# Fixed Version:
# - Multiple runs
# - Cumulative ANOVA data
# - Matching NSGA-II output format
# - Matching KML structure
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



ALGORITHM_NAME = "ACO_3D_v3"


ANALYSIS_ALGORITHM = "ACO"






# ============================================================
# OUTPUT FOLDERS
# ============================================================


OUTPUT_FOLDER = "Output"


ACO_FOLDER = os.path.join(

    OUTPUT_FOLDER,

    "ACO_3D"

)



ROUTES_FOLDER = os.path.join(

    ACO_FOLDER,

    "Routes"

)



METRICS_FOLDER = os.path.join(

    ACO_FOLDER,

    "Metrics"

)



SUMMARY_FOLDER = os.path.join(

    ACO_FOLDER,

    "Summary"

)



BASE_FOLDER = os.path.join(

    ACO_FOLDER,

    "Bases"

)



KML_FOLDER = os.path.join(

    ACO_FOLDER,

    "KML"

)



PARETO_FOLDER = os.path.join(

    ACO_FOLDER,

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
# ACO PARAMETERS
# ============================================================


NUM_ANTS = 50


ITERATIONS = 200



ALPHA = 1.0


BETA = 2.0


EVAPORATION = 0.5


Q = 100







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
# GLOBAL DATA
# ============================================================


trees=[]


gps_points=[]


lat0=None


lon0=None


plantation_boundary=[]


generated_road=[]







EARTH_RADIUS = 6371000.0







# ============================================================
# COORDINATE CONVERSION
# ============================================================


def latlon_to_xy(

    lat,

    lon,

    ref_lat,

    ref_lon

):


    dlat=np.radians(

        lat-ref_lat

    )


    dlon=np.radians(

        lon-ref_lon

    )



    ref=np.radians(

        ref_lat

    )



    x=(

        dlon

        *

        np.cos(ref)

        *

        EARTH_RADIUS

    )



    y=(

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


    lat=(

        ref_lat

        +

        np.degrees(

            y/EARTH_RADIUS

        )

    )



    lon=(

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


    tree=trees[tree_id]



    return np.array(

        [

            tree[0],

            tree[1],

            tree[2]+DRONE_OFFSET

        ],

        dtype=float

    )









# ============================================================
# ROUTE METRICS
# ============================================================


def route_distance(route):


    total=0.0



    for i in range(

        len(route)-1

    ):


        p1=drone_point(

            route[i]

        )


        p2=drone_point(

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


        a=points[i-1]-points[i]


        b=points[i+1]-points[i]



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


    pts=sorted(

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

            (a[0]-o[0])

            *

            (b[1]-o[1])

            -

            (a[1]-o[1])

            *

            (b[0]-o[0])

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

            )

            <=0

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

            )

            <=0

        ):

            upper.pop()



        upper.append(p)





    return [

        np.array(p,dtype=float)

        for p in (

            lower[:-1]

            +

            upper[:-1]

        )

    ]








def polygon_area(poly):


    area=0.0



    for i in range(len(poly)):


        p=poly[i]

        q=poly[

            (i+1)%len(poly)

        ]



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



    result=[]



    for i in range(len(poly)):


        prev=poly[

            (i-1)%len(poly)

        ]


        curr=poly[i]


        nxt=poly[

            (i+1)%len(poly)

        ]



        edge1=curr-prev


        edge2=nxt-curr




        normal1=np.array(

            [

                edge1[1],

                -edge1[0]

            ]

        )



        normal2=np.array(

            [

                edge2[1],

                -edge2[0]

            ]

        )



        normal1/=np.linalg.norm(

            normal1

        )



        normal2/=np.linalg.norm(

            normal2

        )




        p1=curr+offset*normal1


        p2=curr+offset*normal2




        point=line_intersection(

            p1,

            edge1,

            p2,

            edge2

        )



        if point is None:


            normal=normal1+normal2


            normal/=np.linalg.norm(

                normal

            )


            point=curr+offset*normal




        result.append(point)



    return result







def segment_lengths(poly):


    return np.array(

        [

            np.linalg.norm(

                poly[

                    (i+1)%len(poly)

                ]

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



    best_distance=float(

        "inf"

    )


    best_point=None


    best_s=0



    travelled=0



    for i in range(len(road)):


        A=road[i]


        B=road[

            (i+1)%len(road)

        ]



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



        distance=np.linalg.norm(

            point-q

        )



        if distance<best_distance:


            best_distance=distance


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


        if s<=travelled+length:


            ratio=(

                s-travelled

            )/length



            return (

                road[i]

                +

                ratio*(

                    road[

                        (i+1)%len(road)

                    ]

                    -

                    road[i]

                )

            )



        travelled+=length



    return road[0]









# ============================================================
# CLUSTERING
# ============================================================


def create_clusters():


    data=np.array(

        [

            [

                tree[0],

                tree[1],

                tree[2]*ELEVATION_WEIGHT

            ]

            for tree in trees

        ]

    )



    k=max(

        1,

        math.ceil(

            len(trees)

            /

            MAX_TREES

        )

    )



    while True:


        model=KMeans(

            n_clusters=k,

            n_init=20,

            random_state=0

        )



        labels=model.fit_predict(

            data

        )



        clusters=[

            []

            for _ in range(k)

        ]



        for i,label in enumerate(labels):


            clusters[label].append(i)



        if max(

            len(c)

            for c in clusters

        )<=MAX_TREES:


            return clusters



        k+=1








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








def order_clusters_create_bases(
    clusters,
    road
):


    info=[]



    for cluster in clusters:


        centroid=cluster_centroid(

            cluster

        )



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



    total=road_length(

        road

    )



    bases=[]



    for i in range(len(info)):


        s1=info[i]["s"]


        s2=info[

            (i+1)%len(info)

        ]["s"]



        if i==len(info)-1:

            s2+=total



        bases.append(

            interpolate_road(

                road,

                (

                    s1+s2

                )/2

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
# ACO OPTIMISATION
# ============================================================


def calculate_cost(route):


    distance = route_distance(route)


    turning = route_turning(route)


    elevation = route_elevation(route)



    # Same weighted objective idea
    # used for optimisation only

    return (

        distance

        +

        10*turning

        +

        2*elevation

    )








def create_ant_route(
    middle_trees,
    start_tree,
    end_tree,
    pheromone
):


    unvisited=middle_trees.copy()


    route=[

        start_tree

    ]



    current=start_tree





    while len(unvisited)>0:



        probabilities=[]



        for node in unvisited:



            tau=(

                pheromone[current][node]

                **

                ALPHA

            )



            distance=np.linalg.norm(

                drone_point(current)

                -

                drone_point(node)

            )



            eta=(

                1/(distance+1e-9)

            )



            probability=(

                tau

                *

                eta**BETA

            )



            probabilities.append(

                probability

            )





        probabilities=np.array(

            probabilities

        )



        if np.sum(probabilities)==0:


            selected=random.choice(

                unvisited

            )


        else:


            probabilities /= np.sum(

                probabilities

            )


            selected=np.random.choice(

                unvisited,

                p=probabilities

            )



        route.append(

            selected

        )



        unvisited.remove(

            selected

        )


        current=selected





    route.append(

        end_tree

    )


    return route











def update_pheromone(
    pheromone,
    routes,
    costs
):


    # evaporation

    pheromone *= (

        1-EVAPORATION

    )



    for route,cost in zip(

        routes,

        costs

    ):



        deposit=(

            Q

            /

            (

                cost

                +

                1e-9

            )

        )



        for i in range(

            len(route)-1

        ):



            a=route[i]


            b=route[i+1]



            pheromone[a][b]+=deposit


            pheromone[b][a]+=deposit






    return pheromone







def solve_cluster(
    cluster,
    start_tree,
    end_tree
):


    middle=[

        i

        for i in cluster

        if i not in (

            start_tree,

            end_tree

        )

    ]



    if len(middle)==0:


        return [

            start_tree,

            end_tree

        ]




    if len(middle)==1:


        return [

            start_tree,

            middle[0],

            end_tree

        ]







    max_index=len(trees)



    pheromone=np.ones(

        (

            max_index,

            max_index

        )

    )







    best_route=None


    best_cost=float(

        "inf"

    )







    for iteration in range(

        ITERATIONS

    ):



        ant_routes=[]


        costs=[]




        for ant in range(

            NUM_ANTS

        ):



            route=create_ant_route(

                middle,

                start_tree,

                end_tree,

                pheromone

            )



            cost=calculate_cost(

                route

            )



            ant_routes.append(

                route

            )



            costs.append(

                cost

            )




            if cost < best_cost:


                best_cost=cost


                best_route=route.copy()





        pheromone=update_pheromone(

            pheromone,

            ant_routes,

            costs

        )






    return best_route











# ============================================================
# RUN NUMBER HANDLING
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

                    int(row["Run"])

                )



    return max_run+1











# ============================================================
# SAVE ANOVA DATA
# ============================================================


def save_analysis_data(
    details,
    run_number
):


    exists=os.path.isfile(

        ANALYSIS_FILE

    )



    with open(

        ANALYSIS_FILE,

        "a",

        newline=""

    ) as file:


        writer=csv.writer(

            file

        )



        if not exists:


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
# LOAD MAP
# ============================================================


def load_map():


    global gps_points

    global trees

    global lat0

    global lon0

    global plantation_boundary

    global generated_road




    filename=SELECTED_MAP["csv_file"]




    with open(

        filename,

        "r"

    ) as file:


        reader=csv.DictReader(

            file

        )


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




    points=np.array(

        [

            [

                t[0],

                t[1]

            ]

            for t in trees

        ]

    )



    plantation_boundary=convex_hull(

        points

    )


    generated_road=offset_convex_polygon(

        plantation_boundary,

        ROAD_OFFSET

    )






load_map()

# --------------------------------------------------------
# CLUSTER GENERATION
# --------------------------------------------------------


clusters=create_clusters()



ordered_clusters,bases=order_clusters_create_bases(

    clusters,

    generated_road

)

# ============================================================
# MAIN ACO EXECUTION
# ============================================================


START_RUN=get_start_run_number()



for local_run in range(NUM_RUNS):


    RUN_NUMBER=START_RUN+local_run


    RUN_ID=datetime.now().strftime(

        "%Y%m%d_%H%M%S"

    )



    print("\n========================")

    print(

        "ACO RUN",

        RUN_NUMBER

    )

    print("========================")




    details=[]







    # --------------------------------------------------------
    # SOLVE EACH CLUSTER
    # --------------------------------------------------------


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
    # SAVE BASES
    # ========================================================


    base_file=os.path.join(

        BASE_FOLDER,

        f"bases_{RUN_ID}.csv"

    )



    with open(

        base_file,

        "w",

        newline=""

    ) as file:


        writer=csv.writer(file)



        writer.writerow(

            [

                "Base",

                "X",

                "Y",

                "Latitude",

                "Longitude"

            ]

        )




        for i,base in enumerate(bases):


            lat,lon=xy_to_latlon(

                base[0],

                base[1],

                lat0,

                lon0

            )



            writer.writerow(

                [

                    f"B{i}",

                    base[0],

                    base[1],

                    lat,

                    lon

                ]

            )









    # ========================================================
    # SAVE SUMMARY
    # ========================================================


    summary_file=os.path.join(

        SUMMARY_FOLDER,

        f"summary_{RUN_ID}.txt"

    )



    with open(

        summary_file,

        "w"

    ) as file:


        file.write(

            "ACO 3D RESULT\n\n"

        )


        file.write(

            f"Map: {MAP_NAME}\n"

        )


        file.write(

            f"Run: {RUN_NUMBER}\n\n"

        )



        file.write(

            f"Clusters: {len(details)}\n"

        )



        file.write(

            f"Total Distance: "

            f"{sum(x['distance'] for x in details):.3f}\n"

        )


        file.write(

            f"Total Turning: "

            f"{sum(x['turning'] for x in details):.3f}\n"

        )


        file.write(

            f"Total Elevation: "

            f"{sum(x['elevation'] for x in details):.3f}\n"

        )









    # ========================================================
    # SAVE ANOVA DATA
    # ========================================================


    save_analysis_data(

        details,

        RUN_NUMBER

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

    print()

    print(

        "ACO RUN COMPLETED:",

        RUN_NUMBER

    )


print()

print(

    "ALL ACO RUNS COMPLETED"

)

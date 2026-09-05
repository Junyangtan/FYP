import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# =====================================
# 🔥 SELECT MODE HERE
# =====================================
MODE = "3D"   # change to "3D" when needed

# =====================================
# LOAD DATA
# =====================================
if MODE == "2D":
    df = pd.read_csv("2D_map.csv")
else:
    df = pd.read_csv("Map2_3D_map.csv")

lat = df["lat"].values
lon = df["lon"].values

# =====================================
# ================= 2D MODE =================
# =====================================
if MODE == "2D":

    plt.figure(figsize=(6, 6))

    plt.scatter(
        lon, lat,
        s=3,
        alpha=0.6
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("2D Map (Tree Distribution)")
    plt.axis("equal")

    plt.tight_layout()
    plt.show()


# =====================================
# ================= 3D MODE =================
# =====================================
elif MODE == "3D":

    z = df["alt"].values

    # ---- CREATE GRID ----
    grid_res = 120

    lat_grid = np.linspace(lat.min(), lat.max(), grid_res)
    lon_grid = np.linspace(lon.min(), lon.max(), grid_res)

    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

    # ---- INTERPOLATE ----
    z_grid = griddata(
        (lat, lon),
        z,
        (lat_grid, lon_grid),
        method='linear'
    )

    # =========================
    # 🎯 CREATE FIGURE
    # =========================
    fig = plt.figure(figsize=(12, 5))

    # -------------------------
    # 🗺️ TOP VIEW
    # -------------------------
    ax1 = fig.add_subplot(1, 2, 1)

    contour = ax1.contourf(lon_grid, lat_grid, z_grid, alpha=0.85)
    ax1.contour(lon_grid, lat_grid, z_grid, linewidths=0.5)

    ax1.scatter(
        lon, lat,
        s=2,
        c='red',
        alpha=0.4
    )

    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_title("Top View (Height Map)")
    ax1.set_aspect('equal')

    fig.colorbar(contour, ax=ax1, label="Obstacle Height (m)")

    # -------------------------
    # 🌄 3D VIEW
    # -------------------------
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    ax2.plot_surface(
        lon_grid, lat_grid, z_grid,
        alpha=0.8
    )

    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.set_zlabel("Obstacle Height (m)")
    ax2.set_title("3D Surface")

    plt.tight_layout()
    plt.show()

else:
    print("Invalid MODE. Use '2D' or '3D'")
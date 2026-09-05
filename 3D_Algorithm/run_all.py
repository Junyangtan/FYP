import subprocess
import os

base_dir = os.path.dirname(__file__)


files = [
    f"NSGAII_3D_v6.py",
    f"GA_3D_v3.py",
    f"ACO_3D_v3.py"
]

for file in files:
    file_path = os.path.join(base_dir, file)

    if os.path.exists(file_path):
        print(f"Running {file}")
        subprocess.run(["python", file_path])
    else:
        print(f"File not found: {file}")
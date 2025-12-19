
import sys
import os
import time
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append(os.getcwd())

from utils import surface_utils

def main():
    print("Starting reproduction script...")
    
    surf_file = Path(r"d:\vscode_project\Digital-Twin-Brain-Based-on-Neural-Operators\dataset\T103\anat\sub-01_hemi-L_midthickness.surf.gii")
    
    if not surf_file.exists():
        print(f"File not found: {surf_file}")
        return

    print("Loading surface...")
    t0 = time.time()
    vertices, faces = surface_utils.load_surface(str(surf_file))
    print(f"Surface loaded in {time.time() - t0:.4f}s")
    print(f"Vertices shape: {vertices.shape}")
    print(f"Faces shape: {faces.shape}")
    print(f"Faces type: {type(faces)}")
    
    print("Computing adjacency...")
    t0 = time.time()
    adj = surface_utils.get_mesh_adjacency(faces, vertices.shape[0])
    print(f"Adjacency computed in {time.time() - t0:.4f}s")
    print(f"Adjacency shape: {adj.shape}")
    print("Done.")

if __name__ == "__main__":
    main()


import sys
import os
import time
import numpy as np
from scipy import sparse

# Add current directory to path
sys.path.append(os.getcwd())

from utils import surface_utils

def main():
    print("Starting reproduction script with DUMMY data...")
    
    n_vertices = 32492 # Typical number for fsaverage5 or similar
    n_faces = 64980
    
    print(f"Generating dummy mesh with {n_vertices} vertices and {n_faces} faces...")
    
    # Generate random faces
    # Each face has 3 vertex indices
    faces = np.random.randint(0, n_vertices, size=(n_faces, 3))
    
    print("Faces shape:", faces.shape)
    print("Faces type:", type(faces))
    
    print("Computing adjacency...")
    t0 = time.time()
    adj = surface_utils.get_mesh_adjacency(faces, n_vertices)
    print(f"Adjacency computed in {time.time() - t0:.4f}s")
    print(f"Adjacency shape: {adj.shape}")
    print("Done.")

if __name__ == "__main__":
    main()

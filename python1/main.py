import os
import numpy as np
import nibabel as nib
from nibabel.gifti import GiftiImage, GiftiDataArray
from scipy.io import loadmat, savemat
import pyvista as pv
from neuromaps.datasets import fetch_fsaverage
from neuromaps.transforms import mni_to_fsaverage
from neuromaps.nulls import alexander_bloch

# =====================================================
# Setup
# =====================================================
RESULTS_DIR = "Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================
# 1) Load Brainstorm source map (MNI space)
# =====================================================
print("\n[Step 1] Loading Brainstorm source map (.mat) in MNI space...")
mat_file = "Data/MEEG_source_alpha_7Hz_14Hz.mat"
mat = loadmat(mat_file)
source_map = mat["J10K"].squeeze().astype(np.float32)
print(f"[Done] Source map loaded with {len(source_map)} vertices")

# =====================================================
# 2) Save as GIFTI (original Brainstorm MNI space)
# =====================================================
print("\n[Step 2] Saving Brainstorm source map as GIFTI...")
g = GiftiImage()
g.add_gifti_data_array(GiftiDataArray(source_map))
output_file = os.path.join(RESULTS_DIR, "source_map_brainstorm.gii")
nib.save(g, output_file)
print(f"[Done] GIFTI file saved: {output_file}")

# =====================================================
# 3) Load Brainstorm surface mesh (MNI space)
# =====================================================
print("\n[Step 3] Loading Brainstorm cortical surface...")
surface_file = "Data/cortex_concat10K.gii"
surf = nib.load(surface_file)
vertices = surf.darrays[0].data.astype(np.float32)
faces = surf.darrays[1].data.astype(np.int32)

if len(source_map) != vertices.shape[0]:
    raise ValueError(
        f"Vertex mismatch: source_map={len(source_map)}, surface={vertices.shape[0]}"
    )

# Brainstorm J10K: [RH | LH] -> reorder to [LH | RH] to match fsaverage
half = len(source_map) // 2
source_rh = source_map[:half]
source_lh = source_map[half:]
source_ordered = np.concatenate([source_lh, source_rh])  # [LH | RH]

# Convert faces for PyVista
faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int32), faces]).flatten()

# Create PyVista mesh
brainstorm_mesh = pv.PolyData(vertices, faces_pv)
brainstorm_mesh["Activation"] = source_ordered

mesh_file = os.path.join(RESULTS_DIR, "brainstorm_surface_with_activation.vtk")
brainstorm_mesh.save(mesh_file)
print(f"[Done] Brainstorm mesh saved as '{mesh_file}'")

# =====================================================
# 4) Project Brainstorm MNI map to fsaverage32k (proper projection)
# =====================================================
print("\n[Step 4] Projecting Brainstorm MNI map to fsaverage10k (neuromaps)...")
source_fsaverage = mni_to_fsaverage(
    source_map,
    density="10k",
    mni_surface="cortex10k",
    hemi=None  # maps both hemispheres
)
source_fsaverage = source_fsaverage.astype(np.float32)
print(f"[Done] Source map projected to fsaverage10k ({len(source_fsaverage)} vertices)")

# =====================================================
# 5) Generate spatial nulls
# =====================================================
print("\n[Step 5] Generating spatial nulls (fsaverage10k)...")
n_nulls = 1000
nulls = alexander_bloch(
    source_fsaverage,
    atlas="fsaverage",
    density="10k",
    n_perm=n_nulls,
    seed=42
)
print(f"[Done] Nulls generated with shape: {nulls.shape} (vertices x permutations)")

# =====================================================
# 6) Compute z-score map
# =====================================================
print("\n[Step 6] Computing z-score map from null distribution...")
null_mean = np.mean(nulls, axis=1)
null_std = np.std(nulls, axis=1)
z_map = (source_fsaverage - null_mean) / null_std
print(f"[Done] Z-score map computed with {z_map.shape[0]} vertices")

# =====================================================
# 7) Save results
# =====================================================
print("\n[Step 7] Saving projected map, nulls, and z-map...")
np.save(os.path.join(RESULTS_DIR, "source_map_fsaverage.npy"), source_fsaverage)
np.save(os.path.join(RESULTS_DIR, "z_map_fsaverage.npy"), z_map)
np.save(os.path.join(RESULTS_DIR, "nulls_fsaverage.npy"), nulls)

savemat(os.path.join(RESULTS_DIR, "source_map_fsaverage.mat"), {"source_map": source_fsaverage})
savemat(os.path.join(RESULTS_DIR, "z_map_fsaverage.mat"), {"z_map": z_map})
savemat(os.path.join(RESULTS_DIR, "nulls_fsaverage.mat"), {"nulls": nulls})

print("\n? All results saved successfully in 'Results/' folder!")
print("? Pipeline finished successfully!\n")

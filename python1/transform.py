import os
import numpy as np
import nibabel as nib
from nibabel.gifti import GiftiImage, GiftiDataArray
import pyvista as pv
from scipy.spatial import cKDTree

# -----------------------------
# Paths
# -----------------------------
data_dir = "Data"
results_dir = os.path.join(data_dir, "Results")
os.makedirs(results_dir, exist_ok=True)

subject_id = "sub-CBM00008"

# Combined source map from previous step
source_gii_file = os.path.join(results_dir, "source_fsLR32k.gii")

# Subject fs_LR surfaces (very_inflated 32k)
sub_fsLR_dir = os.path.join(data_dir, subject_id, "T1w", "fsaverage_LR32k")
lh_surf_file = os.path.join(sub_fsLR_dir, f"{subject_id}.L.very_inflated.32k_fs_LR.surf.gii")
rh_surf_file = os.path.join(sub_fsLR_dir, f"{subject_id}.R.very_inflated.32k_fs_LR.surf.gii")

# -----------------------------
# 1) Load combined source map
# -----------------------------
g = nib.load(source_gii_file)
data = g.darrays[0].data
print(f"[Info] Loaded combined source map with {len(data)} vertices")

# -----------------------------
# 2) Split into LH/RH
# -----------------------------
n_vert = 32492  # HCP fs_LR 32k
lh_data = data[:n_vert]
rh_data = data[n_vert:]

# Save split GIFTIs
lh_gii_file = os.path.join(results_dir, f"{subject_id}.L.source_fsLR32k.func.gii")
rh_gii_file = os.path.join(results_dir, f"{subject_id}.R.source_fsLR32k.func.gii")

nib.save(GiftiImage(darrays=[GiftiDataArray(lh_data.astype("float32"))]), lh_gii_file)
nib.save(GiftiImage(darrays=[GiftiDataArray(rh_data.astype("float32"))]), rh_gii_file)
print(f"[Saved] Split GIFTIs -> LH: {lh_gii_file}, RH: {rh_gii_file}")

# -----------------------------
# 3) Load fs_LR surfaces
# -----------------------------
lh_surf = nib.load(lh_surf_file)
rh_surf = nib.load(rh_surf_file)

vertices_lh = lh_surf.darrays[0].data.astype(np.float32)
vertices_rh = rh_surf.darrays[0].data.astype(np.float32)
vertices_native = np.vstack([vertices_lh, vertices_rh])

faces_lh = lh_surf.darrays[1].data.astype(np.int32)
faces_rh = rh_surf.darrays[1].data.astype(np.int32) + vertices_lh.shape[0]
faces_native = np.vstack([faces_lh, faces_rh])
faces_pv_native = np.hstack([np.full((faces_native.shape[0], 1), 3, dtype=np.int32), faces_native]).flatten()

native_mesh = pv.PolyData(vertices_native, faces_pv_native)
native_mesh['Activation'] = data.astype(np.float32)

# Optional: save combined PyVista mesh
vtk_file = os.path.join(results_dir, "source_fsLR32k_combined.vtk")
native_mesh.save(vtk_file)
print(f"[Saved] Combined PyVista mesh: {vtk_file}")

# -----------------------------
# 4) Interpolation example (optional)
# -----------------------------
# If you have a different target surface (fs_LR template) to resample onto:
# target_lh_file = "path_to_target_lh.surf.gii"
# target_rh_file = "path_to_target_rh.surf.gii"
# Then you can use cKDTree to interpolate exactly like before:
#
# tree = cKDTree(vertices_native)
# fsLR_vertices = np.vstack([target_lh_vertices, target_rh_vertices])
# _, idx = tree.query(fsLR_vertices)
# source_resampled = data[idx].astype(np.float32)
# Save resampled VTK or GIFTI as needed.

print("[Done] Transform and split pipeline finished successfully!")

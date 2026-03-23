import numpy as np
import nibabel as nib
from nibabel.gifti import GiftiImage, GiftiDataArray
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.io import loadmat
import os

# -----------------------------
# Paths
# -----------------------------
subject_id = "sub-CBM00008"
data_dir = "Data"
subject_dir = os.path.join(data_dir, subject_id)
t1w_dir = os.path.join(subject_dir, "T1w")
templates_dir = os.path.join(data_dir, "zz_templates")  # HCP fs_LR 32k surfaces

# Source map from Brainstorm
source_map_file = os.path.join(data_dir, "MEEG_source_alpha_7Hz_14Hz.mat")

# Output folder
results_dir = os.path.join(data_dir, "Results")
os.makedirs(results_dir, exist_ok=True)

# -----------------------------
# 1) Load Brainstorm source map
# -----------------------------
mat = loadmat(source_map_file)
source_map = mat['J'].squeeze()  # shape: (n_vertices,)
print(f"[Info] Source map loaded: {source_map.shape[0]} vertices")

# -----------------------------
# 2) Load Brainstorm 8k surface for interpolation
# -----------------------------
# Assuming you have the Brainstorm 8k vertices stored in a GIFTI (or use the mat)
# Here we just load from mat; you can replace with actual vertices if needed
# For demonstration, we assume vertices from source_map order corresponds to 8k vertices
# If you have a separate surface file (e.g., cortex_concat_8000V_fix.gii), load it:
brainstorm_surf_file = os.path.join(data_dir, "cortex_concat_8000V_fix.gii")
brainstorm_surf = nib.load(brainstorm_surf_file)
vertices_8k = np.vstack([brainstorm_surf.darrays[0].data.astype(np.float32)])
faces_8k = brainstorm_surf.darrays[1].data.astype(np.int32)
print(f"[Info] Brainstorm 8k surface loaded: {vertices_8k.shape[0]} vertices")

# -----------------------------
# 3) Load subject fs_LR 32k surfaces
# -----------------------------
sub_fsLR_dir = os.path.join(t1w_dir, "fsaverage_LR32k")
lh_surf_file = os.path.join(sub_fsLR_dir, f"{subject_id}.L.very_inflated.32k_fs_LR.surf.gii")
rh_surf_file = os.path.join(sub_fsLR_dir, f"{subject_id}.R.very_inflated.32k_fs_LR.surf.gii")

if not (os.path.exists(lh_surf_file) and os.path.exists(rh_surf_file)):
    print("[Info] Subject-specific fs_LR 32k surfaces not found. Using template surfaces.")
    lh_surf_file = os.path.join(templates_dir, "fs_LR32k", "lh.sphere.32k_fs_LR.surf.gii")
    rh_surf_file = os.path.join(templates_dir, "fs_LR32k", "rh.sphere.32k_fs_LR.surf.gii")

lh_surf = nib.load(lh_surf_file)
rh_surf = nib.load(rh_surf_file)

vertices_lh = lh_surf.darrays[0].data.astype(np.float32)
vertices_rh = rh_surf.darrays[0].data.astype(np.float32)
vertices_32k = np.vstack([vertices_lh, vertices_rh])

faces_lh = lh_surf.darrays[1].data.astype(np.int32)
faces_rh = rh_surf.darrays[1].data.astype(np.int32) + vertices_lh.shape[0]
faces_32k = np.vstack([faces_lh, faces_rh])
faces_pv_32k = np.hstack([np.full((faces_32k.shape[0], 1), 3, dtype=np.int32), faces_32k]).flatten()

fsLR_mesh = pv.PolyData(vertices_32k, faces_pv_32k)

# -----------------------------
# 4) Interpolate Brainstorm 8k map ? fs_LR 32k surface
# -----------------------------
tree = cKDTree(vertices_8k)
_, idx = tree.query(vertices_32k)
source_fsLR = source_map[idx].astype(np.float32)
fsLR_mesh['Activation'] = source_fsLR

# Save interpolated mesh
fsLR_mesh_file = os.path.join(results_dir, "brainstorm_resampled_fsLR32k.vtk")
fsLR_mesh.save(fsLR_mesh_file)
print(f"[Saved] Resampled fs_LR 32k mesh: {fsLR_mesh_file}")

# -----------------------------
# 5) Optional: save as GIFTI
# -----------------------------
g = GiftiImage()
g.add_gifti_data_array(GiftiDataArray(source_fsLR))
gii_file = os.path.join(results_dir, "source_fsLR32k.gii")
nib.save(g, gii_file)
print(f"[Saved] Resampled source map as GIFTI: {gii_file}")

print("[Done] Resampling pipeline finished successfully!")

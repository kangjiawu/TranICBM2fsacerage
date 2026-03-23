import os
import numpy as np
import nibabel as nib
from nibabel.gifti import GiftiImage, GiftiDataArray
from scipy.io import loadmat, savemat
import pyvista as pv
from neuromaps.datasets import fetch_fsaverage

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
mat_file = "E:/Transfer/neuromaps/matlab/results/MEEG_source_alpha_7Hz_14Hz.mat"
mat = loadmat(mat_file)
source_map = mat["J32K"].squeeze().astype(np.float32)
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
surface_file = "Data/cortex_concat.gii"
surf = nib.load(surface_file)
vertices = surf.darrays[0].data.astype(np.float32)
faces = surf.darrays[1].data.astype(np.int32)

if len(source_map) != vertices.shape[0]:
    raise ValueError(
        f"Vertex mismatch: source_map={len(source_map)}, surface={vertices.shape[0]}"
    )

# Brainstorm J32K: [RH | LH] -> reorder to [LH | RH] to match fsaverage
half = len(source_map) // 2
source_rh = source_map[:half]
source_lh = source_map[half:]

L = GiftiImage()
L.add_gifti_data_array(GiftiDataArray(source_lh))
output_file = os.path.join(RESULTS_DIR, "source_map_left.gii")
nib.save(L, output_file)
print(f"[Done] Left GIFTI file saved: {output_file}")

R = GiftiImage()
R.add_gifti_data_array(GiftiDataArray(source_rh))
output_file = os.path.join(RESULTS_DIR, "source_map_right.gii")
nib.save(R, output_file)
print(f"[Done] Right GIFTI file saved: {output_file}")
# =====================================================
# 4) Project Brainstorm MNI map to fsaverage32k
# =====================================================

print("\n[Step 4] Projecting Brainstorm MNI map to fsaverage10k (workbench)...")
# workbench code
# wb_command -metric-resample ^
#   "C:\Users\Administrator\Desktop\neuromaps\data\source_map_right1.gii" ^
#   "C:\Users\Administrator\Desktop\neuromaps\data\R.sphere.32k_fs_LR.surf.gii" ^
#   "C:\Users\Administrator\Desktop\neuromaps\data\fsaverage.R.sphere.10k_fs_LR.surf.gii" ^
#   BARYCENTRIC ^
#   "C:\Users\Administrator\Desktop\neuromaps\results\right_madata_fs10k.func.gii"
#
# wb_command -metric-resample ^
#   "C:\Users\Administrator\Desktop\neuromaps\data\source_map_left1.gii" ^
#   "C:\Users\Administrator\Desktop\neuromaps\data\L.sphere.32k_fs_LR.surf.gii" ^
#   "C:\Users\Administrator\Desktop\neuromaps\data\fsaverage.L.sphere.10k_fs_LR.surf.gii" ^
#   BARYCENTRIC ^
#   "C:\Users\Administrator\Desktop\neuromaps\results\left_madata_fs10k.func.gii"
print(f"[Done] Right GIFTI file saved:left_madata_fs32k.func.gii  right_madata_fs10k.func.gii ")

# =====================================================
# 5) Generate spatial nulls
# =====================================================
print("\n[Step 5] Generating spatial nulls (fsaverage10k)...")
my_annotation_paths = [
    r'C:\Users\Administrator\Desktop\neuromaps\results\left_madata_fs10k.func.gii',
    r'C:\Users\Administrator\Desktop\neuromaps\results\right_madata_fs10k.func.gii'
]

n_nulls = 1000
rotated = alexander_bloch(
    my_annotation_paths,
    atlas="fsaverage",
    density="10k",
    n_perm=n_nulls,
    seed=42
)
print(f"[Done] Nulls generated with shape: {rotated.shape} (vertices x permutations)")

# =====================================================
# 6) Compute z-score map
# =====================================================
from neuromaps import stats
from neuromaps.datasets import fetch_annotation
abagen = fetch_annotation(source='abagen')
corr, pval = stats.compare_images(my_annotation_paths, abagen, nulls=rotated)
print(f'r = {corr:.3f}, p = {pval:.3f}')


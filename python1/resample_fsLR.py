import os
import subprocess

# -----------------------------
# Paths
# -----------------------------
data_dir = "Data"
results_dir = os.path.join(data_dir, "Results")
os.makedirs(results_dir, exist_ok=True)

subject_id = "sub-CBM00008"

# LH and RH functional GIFTIs (scalar data per vertex)
lh_func = os.path.join(results_dir, f"{subject_id}.L.source_fsLR32k.func.gii")
rh_func = os.path.join(results_dir, f"{subject_id}.R.source_fsLR32k.func.gii")

# Template sphere surfaces (from zz_templates/fs_LR32k)
templates_dir = os.path.join(data_dir, "zz_templates", "fs_LR32k")
lh_sphere_template = os.path.join(templates_dir, "L.sphere.32k_fs_LR.surf.gii")
rh_sphere_template = os.path.join(templates_dir, "R.sphere.32k_fs_LR.surf.gii")


# Output resampled functional maps
lh_resampled = os.path.join(results_dir, f"{subject_id}.L.source_fsLR32k_resampled.func.gii")
rh_resampled = os.path.join(results_dir, f"{subject_id}.R.source_fsLR32k_resampled.func.gii")

# -----------------------------
# Resample LH functional map
# -----------------------------
print("[Resampling] Left hemisphere...")
subprocess.run([
    "wb_command", "-metric-resample",
    lh_func,               # functional GIFTI
    lh_sphere_template,    # source surface
    lh_sphere_template,    # target surface (HCP template)
    "ADAP_BARY_AREA",      # interpolation method
    lh_resampled
], check=True)
print(f"[Done] LH resampled: {lh_resampled}")

# -----------------------------
# Resample RH functional map
# -----------------------------
print("[Resampling] Right hemisphere...")
subprocess.run([
    "wb_command", "-metric-resample",
    rh_func,               # functional GIFTI
    rh_sphere_template,    # source surface
    rh_sphere_template,    # target surface (HCP template)
    "ADAP_BARY_AREA",      # interpolation method
    rh_resampled
], check=True)
print(f"[Done] RH resampled: {rh_resampled}")

print("[Success] Resampling pipeline finished!")

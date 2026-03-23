import os
import numpy as np
import nibabel as nib
import pyvista as pv
from neuromaps.datasets import fetch_fsaverage

def visualize_on_fsaverage():
    """
    Visualize a Brainstorm/neuromaps activation map on the fsaverage 10k template.
    """
    print("\n[Visualization] Loading fsaverage10k surface and activation map...")

    RESULTS_DIR = "Results"
    map_file = os.path.join(RESULTS_DIR, "source_map.npy")  # or "source_10k.npy"
    if not os.path.exists(map_file):
        raise FileNotFoundError(f"Activation file not found: {map_file}")

    source_map = np.load(map_file)
    source_map = np.nan_to_num(source_map, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[Done] Loaded activation map with {len(source_map)} vertices")

    # -----------------------------
    # Load fsaverage 10k template surface
    # -----------------------------
    fs_files = fetch_fsaverage(density="10k")
    lh_path, rh_path = fs_files["pial"]

    # Vertices
    lh_vertices = nib.load(lh_path).darrays[0].data.astype(np.float32)
    rh_vertices = nib.load(rh_path).darrays[0].data.astype(np.float32)
    vertices = np.vstack([lh_vertices, rh_vertices])

    # Faces
    lh_faces = nib.load(lh_path).darrays[1].data.astype(np.int32)
    rh_faces = nib.load(rh_path).darrays[1].data.astype(np.int32)
    rh_faces_shifted = rh_faces + lh_vertices.shape[0]
    faces = np.vstack([lh_faces, rh_faces_shifted])

    # Build PyVista surface
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int32), faces]).flatten()
    fs_mesh = pv.PolyData(vertices, faces_pv)
    fs_mesh["Activation"] = source_map

    # -----------------------------
    # Visualize in 3D
    # -----------------------------
    print("[Visualization] Rendering activation map on fsaverage10k surface...")

    plotter = pv.Plotter(window_size=(1200, 800))
    plotter.set_background("white")
    plotter.add_text("Activation Map on fsaverage10k", font_size=14)
    plotter.add_mesh(
        fs_mesh,
        scalars="Activation",
        cmap="hot",
        smooth_shading=True,
        clim=[np.min(source_map), np.max(source_map)],
    )
    plotter.add_scalar_bar(title="Activation", n_labels=5)

    # -----------------------------
    # Remove axes / grids
    # -----------------------------
    plotter.hide_axes()

    # Camera view
    plotter.camera_position = "xy"
    plotter.show()

    print("[Done] Visualization complete.")


if __name__ == "__main__":
    visualize_on_fsaverage()

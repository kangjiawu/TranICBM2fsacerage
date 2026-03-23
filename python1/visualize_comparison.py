import os
import numpy as np
import pyvista as pv
import nibabel as nib
from neuromaps.datasets import fetch_fsaverage
from scipy.spatial import cKDTree

def visualize_comparison():
    print("\n[Visualization] Loading meshes...")
    RESULTS_DIR = "Results"

    # -----------------------------
    # Load Brainstorm mesh (original MNI source map)
    # -----------------------------
    brainstorm_path = os.path.join(RESULTS_DIR, "brainstorm_surface_with_activation.vtk")
    brainstorm_mesh = pv.read(brainstorm_path)
    print(f"[Done] Brainstorm mesh loaded ({brainstorm_mesh.n_points} vertices)")

    # -----------------------------
    # Load projected fsaverage source map (from pipeline)
    # -----------------------------
    source_fsaverage_path = os.path.join(RESULTS_DIR, "source_map_fsaverage.npy")
    source_fsaverage = np.load(source_fsaverage_path)
    source_fsaverage = np.nan_to_num(source_fsaverage, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[Done] Projected source map loaded ({source_fsaverage.shape[0]} vertices)")

    # -----------------------------
    # Load original Brainstorm source map (correct hemispheres)
    # -----------------------------
    source_orig_path = os.path.join(RESULTS_DIR, "source_map_brainstorm.gii")
    g_orig = nib.load(source_orig_path)
    source_orig = g_orig.darrays[0].data.astype(np.float32)
    half = len(source_orig) // 2
    # Reorder to [LH | RH]
    source_ordered = np.concatenate([source_orig[half:], source_orig[:half]])
    print(f"[Done] Original Brainstorm source map loaded with correct hemispheres ({len(source_ordered)} vertices)")

    # -----------------------------
    # Build fsaverage10k mesh
    # -----------------------------
    fs10k_files = fetch_fsaverage(density='10k')
    lh_path, rh_path = fs10k_files['pial']
    lh_gii = nib.load(lh_path)
    rh_gii = nib.load(rh_path)

    # Vertices
    fs10k_lh_vertices = lh_gii.darrays[0].data.astype(np.float32)
    fs10k_rh_vertices = rh_gii.darrays[0].data.astype(np.float32)
    fs10k_vertices = np.vstack([fs10k_lh_vertices, fs10k_rh_vertices])

    # Faces
    fs10k_lh_faces = lh_gii.darrays[1].data.astype(np.int32)
    fs10k_rh_faces = rh_gii.darrays[1].data.astype(np.int32) + fs10k_lh_vertices.shape[0]
    fs10k_faces = np.vstack([fs10k_lh_faces, fs10k_rh_faces])
    faces_pv = np.hstack([np.full((fs10k_faces.shape[0], 1), 3, dtype=np.int32), fs10k_faces]).flatten()

    # -----------------------------
    # Map original Brainstorm source map to fsaverage vertices (nearest neighbor)
    # -----------------------------
    half_vertices = brainstorm_mesh.n_points // 2
    bs_lh_vertices = brainstorm_mesh.points[:half_vertices]  # LH
    bs_rh_vertices = brainstorm_mesh.points[half_vertices:]  # RH
    bs_lh_map = source_ordered[:half_vertices]
    bs_rh_map = source_ordered[half_vertices:]

    tree_lh = cKDTree(bs_lh_vertices)
    tree_rh = cKDTree(bs_rh_vertices)
    idx_lh = tree_lh.query(fs10k_lh_vertices)[1]
    idx_rh = tree_rh.query(fs10k_rh_vertices)[1]

    source_orig_fsaverage = np.concatenate([bs_lh_map[idx_lh], bs_rh_map[idx_rh]]).astype(np.float32)

    # -----------------------------
    # Create PyVista meshes
    # -----------------------------
    fs_mesh_projected = pv.PolyData(fs10k_vertices, faces_pv)
    fs_mesh_projected['Activation'] = source_fsaverage

    fs_mesh_original = pv.PolyData(fs10k_vertices, faces_pv)
    fs_mesh_original['Activation'] = source_orig_fsaverage

    # Center meshes
    brainstorm_mesh.points -= brainstorm_mesh.points.mean(axis=0)
    fs_mesh_projected.points -= fs_mesh_projected.points.mean(axis=0)
    fs_mesh_original.points -= fs_mesh_original.points.mean(axis=0)

    # Scale Brainstorm mesh to match fsaverage
    scale_factor = np.linalg.norm(fs_mesh_projected.points, axis=1).max() / np.linalg.norm(brainstorm_mesh.points, axis=1).max()
    brainstorm_mesh.points *= scale_factor
    brainstorm_mesh.rotate_x(180, inplace=True)

    # -----------------------------
    # Plot side-by-side
    # -----------------------------
    plotter = pv.Plotter(shape=(1,3), window_size=[2400,800])
    plotter.set_background('white')

    plotter.subplot(0,0)
    plotter.add_text("Original Brainstorm Surface", font_size=14)
    plotter.add_mesh(brainstorm_mesh, scalars='Activation', cmap='hot', smooth_shading=True)
    plotter.add_scalar_bar(title='Activation', n_labels=5)
    plotter.hide_axes()

    plotter.subplot(0,1)
    plotter.add_text("fsaverage Surface (Projected)", font_size=14)
    plotter.add_mesh(fs_mesh_projected, scalars='Activation', cmap='hot', smooth_shading=True)
    plotter.add_scalar_bar(title='Activation', n_labels=5)
    plotter.hide_axes()

    plotter.subplot(0,2)
    plotter.add_text("fsaverage Surface (Original Map)", font_size=14)
    plotter.add_mesh(fs_mesh_original, scalars='Activation', cmap='hot', smooth_shading=True)
    plotter.add_scalar_bar(title='Activation', n_labels=5)
    plotter.hide_axes()

    plotter.link_views()
    plotter.camera_position = 'yz'
    plotter.camera.zoom(1.3)
    plotter.show()


if __name__ == "__main__":
    visualize_comparison()

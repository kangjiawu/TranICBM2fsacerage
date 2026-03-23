import numpy as np
import pyvista as pv
import nibabel as nib
from neuromaps.datasets import fetch_fsaverage


def visualize_results():
    print("\n[Visualization] Loading saved files...")

    # -----------------------------
    # Load Brainstorm mesh (original source)
    # -----------------------------
    mesh = pv.read("brainstorm_surface_with_activation.vtk")
    print("[Done] Brainstorm mesh loaded with Activation")

    # Center Brainstorm mesh
    brainstorm_center = mesh.points.mean(axis=0)
    mesh.points -= brainstorm_center

    # Flip X axis to match fsaverage orientation (optional adjustment)
    mesh.points[:, 0] *= -1

    # -----------------------------
    # Load fsaverage10k maps
    # -----------------------------
    source_10k = np.load("source_10k.npy")
    z_map_10k = np.load("z_map_fsaverage10k.npy")

    source_10k = np.nan_to_num(source_10k, nan=0.0, posinf=0.0, neginf=0.0)
    z_map_10k = np.nan_to_num(z_map_10k, nan=0.0, posinf=0.0, neginf=0.0)

    # -----------------------------
    # Build fsaverage10k mesh
    # -----------------------------
    fs10k_files = fetch_fsaverage(density='10k')
    fs10k_lh_path, fs10k_rh_path = fs10k_files['pial']

    # Vertices
    fs10k_lh_vertices = nib.load(fs10k_lh_path).darrays[0].data.astype(np.float32)
    fs10k_rh_vertices = nib.load(fs10k_rh_path).darrays[0].data.astype(np.float32)
    fs10k_vertices = np.vstack([fs10k_lh_vertices, fs10k_rh_vertices])

    # Faces
    fs10k_lh_faces = nib.load(fs10k_lh_path).darrays[1].data.astype(np.int32)
    fs10k_rh_faces = nib.load(fs10k_rh_path).darrays[1].data.astype(np.int32)
    fs10k_rh_faces_shifted = fs10k_rh_faces + fs10k_lh_vertices.shape[0]
    fs10k_faces = np.vstack([fs10k_lh_faces, fs10k_rh_faces_shifted])

    faces_pv = np.hstack(
        [np.full((fs10k_faces.shape[0], 1), 3, dtype=np.int32), fs10k_faces]
    ).flatten()

    fs_mesh = pv.PolyData(fs10k_vertices, faces_pv)
    fs_mesh['SourceMap'] = source_10k
    fs_mesh['ZMap'] = z_map_10k

    # Center fsaverage10k mesh
    fs_center = fs_mesh.points.mean(axis=0)
    fs_mesh.points -= fs_center

    # -----------------------------
    # Plot comparison
    # -----------------------------
    plotter = pv.Plotter(shape=(1, 2), window_size=[1600, 800])
    plotter.set_background('white')

    # Left: Brainstorm mesh with original activation
    plotter.subplot(0, 0)
    plotter.add_text("Brainstorm Source Map (Original Surface)", font_size=14)
    plotter.add_mesh(
        mesh,
        scalars='Activation',
        cmap='hot',
        smooth_shading=True,
        clim=[np.min(mesh['Activation']), np.max(mesh['Activation'])]
    )
    plotter.add_scalar_bar(title='Activation', n_labels=5)

    # Right: fsaverage10k surface with ZMap
    plotter.subplot(0, 1)
    plotter.add_text("fsaverage10k ZMap (Transformed Surface)", font_size=14)
    plotter.add_mesh(
        fs_mesh,
        scalars='ZMap',
        cmap='coolwarm',
        smooth_shading=True,
        clim=[np.min(z_map_10k), np.max(z_map_10k)]
    )
    plotter.add_scalar_bar(title='Z-score', n_labels=5)

    # Link views so you can rotate both together
    plotter.link_views()
    plotter.camera_position = 'yz'
    plotter.camera.zoom(1.3)

    plotter.show()


if __name__ == "__main__":
    visualize_results()

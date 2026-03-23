import pyvista as pv
import numpy as np

# Load the interpolated mesh
fsLR_mesh_file = "Data/Results/brainstorm_resampled_fsLR32k.vtk"
mesh = pv.read(fsLR_mesh_file)

plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars='Activation', cmap='hot', smooth_shading=True)
plotter.add_scalar_bar(title="Activation")
plotter.show()

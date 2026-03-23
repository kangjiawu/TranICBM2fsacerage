import nibabel as nib
from nibabel.gifti import GiftiImage, GiftiDataArray

input_gii = "Data/Results/source_fsLR32k.gii"  # combined Brainstorm source map
lh_out = "Data/Results/sub-CBM00008.L.source_fsLR32k.func.gii"
rh_out = "Data/Results/sub-CBM00008.R.source_fsLR32k.func.gii"

g = nib.load(input_gii)
# Assuming first half is LH, second half is RH
n_vertices = g.darrays[0].data.shape[0] // 2
lh_data = g.darrays[0].data[:n_vertices]
rh_data = g.darrays[0].data[n_vertices:]

# Create new GIFTIs
g_lh = GiftiImage(darrays=[GiftiDataArray(lh_data)])
g_rh = GiftiImage(darrays=[GiftiDataArray(rh_data)])

nib.save(g_lh, lh_out)
nib.save(g_rh, rh_out)
print("LH and RH GIFTIs ready for resampling")

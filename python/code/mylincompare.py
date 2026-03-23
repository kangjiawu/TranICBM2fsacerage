import os
import numpy as np
import nibabel as nib
from nibabel.gifti import GiftiImage, GiftiDataArray
from scipy.io import loadmat, savemat
import pyvista as pv
from neuromaps.datasets import fetch_fsaverage

from neuromaps.nulls import alexander_bloch
from neuromaps import stats
from neuromaps.datasets import fetch_annotation

myelin = fetch_annotation(source='hcps1200', desc='myelinmap', space='fsLR', den='32k')


my_annotation_paths = [
    r'E:\NeuromapPipline\results\AlphaXINet\Alpha_Exponent_lh.func.gii',
    r'E:\NeuromapPipline\results\AlphaXINet\Alpha_Exponent_rh.func.gii'
]

n_nulls = 1000
rotated = alexander_bloch(
    my_annotation_paths,
    atlas="fsLR",
    density="32k",
    n_perm=n_nulls,
    seed=42
)
corr, pval = stats.compare_images(my_annotation_paths, myelin, nulls=rotated)
print(f'r = {corr:.3f}, p = {pval:.3f}')
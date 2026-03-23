from neuromaps.datasets import fetch_annotation
import  neuromaps
import matplotlib.pyplot as plt
annotation = fetch_annotation(source='abagen')
kwargs = {
    'view': 'lateral',
    'cmap': 'viridis',
    'colorbar': True,
    'threshold': 0.5
}

data = (annotation[0], annotation[1])


import matplotlib
from matplotlib import colors as mcolors, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from nilearn.plotting import plot_surf
import numpy as np

from neuromaps.datasets import ALIAS, fetch_atlas
from neuromaps.images import load_gifti
from neuromaps.transforms import _check_hemi

HEMI = dict(L='left', R='right')

matplotlib.colormaps.register(
    mcolors.LinearSegmentedColormap.from_list(
        'caret_blueorange', [
            '#00d2ff', '#009eff', '#006cfe', '#0043fe',
            '#fd4604', '#fe6b01', '#ffd100', '#ffff04'
            ]),
    name="caret_blueorange"
)


def plot_surf_template(data, template, density, surf='inflated',
                       hemi=None, data_dir=None, mask_medial=True, **kwargs):
    """
    Plot `data` on `template` surface.

    Parameters
    ----------
    data : str or os.PathLike or tuple-of-str
        Path to data file(s) to be plotted. If tuple, assumes (left, right)
        hemisphere.
    template : {'civet', 'fsaverage', 'fsLR'}
        Template on which `data` is defined
    density : str
        Resolution of template
    surf : str, optional
        Surface on which `data` should be plotted. Must be valid for specified
        `space`. Default: 'inflated'
    hemi : {'L', 'R'}, optional
        If `data` is not a tuple, which hemisphere it should be plotted on.
        Default: None
    mask_medial : bool, optional
        Whether to mask vertices along the medial wall. Default: True
    kwargs : key-value pairs
        Passed directly to `nilearn.plotting.plot_surf`

    Returns
    -------
    fig : matplotlib.Figure instance
        Plotted figure
    """
    atlas = fetch_atlas(template, density, data_dir=data_dir, verbose=0)
    template = ALIAS.get(template, template)
    if template == 'MNI152':
        raise ValueError('Cannot plot MNI152 on the surface. Try performing '
                         'registration fusion to project data to the surface '
                         'and plotting the projection instead.')

    # 调试：打印 atlas 和 surf 的信息
    print(f"Atlas type: {type(atlas)}")
    print(f"Atlas keys: {list(atlas.keys()) if hasattr(atlas, 'keys') else 'N/A'}")
    print(f"Surf value: {surf}")
    print(f"Atlas[surf] type: {type(atlas[surf])}")
    print(f"Atlas[surf] value: {atlas[surf]}")

    surf_data, medial = atlas[surf], atlas['medial']

    # 检查 surf_data 的类型
    if isinstance(surf_data, str):
        # 如果 surf_data 是字符串，可能是文件路径
        # 我们需要加载这个文件
        surf_data = load_gifti(surf_data).agg_data()
        # 或者，如果 atlas 包含左右半球的数据，可能需要分别处理
        # 这里需要根据实际情况调整
        if hasattr(atlas, 'L') and hasattr(atlas, 'R'):
            surf_data = type('obj', (object,), {'L': atlas.L, 'R': atlas.R})
        else:
            raise ValueError(f"无法处理 atlas[surf] 的类型: {type(surf_data)}")

    opts = dict(alpha=1.0, threshold=np.spacing(1))
    opts.update(**kwargs)
    if kwargs.get('bg_map') is not None and kwargs.get('alpha') is None:
        opts['alpha'] = 'auto'

    # 从 opts 中移除将在 plot_surf 调用中明确指定的参数
    plot_surf_args = ['hemi', 'axes', 'view']
    filtered_opts = {k: v for k, v in opts.items() if k not in plot_surf_args}

    data, hemispheres = zip(*_check_hemi(data, hemi))
    n_surf = len(data)
    fig, axes = plt.subplots(n_surf, 2, subplot_kw={'projection': '3d'})
    axes = (axes,) if n_surf == 1 else axes.T
    for row, hemi, img in zip(axes, hemispheres, data):
        # 使用修正后的 surf_data
        geom = load_gifti(getattr(surf_data, hemi)).agg_data()
        img = load_gifti(img).agg_data().astype('float32')
        # set medial wall to NaN; this will avoid it being plotted
        if mask_medial:
            med = load_gifti(getattr(medial, hemi)).agg_data().astype(bool)
            img[np.logical_not(med)] = np.nan

        for ax, view in zip(row, ['lateral', 'medial']):
            ax.disable_mouse_rotation()
            plot_surf(geom, img, hemi=HEMI[hemi], axes=ax, view=view, **filtered_opts)
            poly = ax.collections[0]
            try:
                poly.set_facecolors(
                    _fix_facecolors(ax, poly._original_facecolor,
                                    *geom, view, hemi)
                )
            except AttributeError:
                pass

    if not opts.get('colorbar', False):
        fig.tight_layout()
        if n_surf == 1:
            fig.subplots_adjust(wspace=-0.1)
        else:
            fig.subplots_adjust(wspace=-0.4, hspace=-0.15)

    return fig



def _fix_facecolors(ax, facecolors, vertices, faces, view, hemi):
    """
    Update `facecolors` to reflect shading of mesh geometry.

    Parameters
    ----------
    ax : plt.Axes3dSubplot
        Axis instance
    facecolors : (F,) array_like
        Original facecolors of plot
    vertices : (V, 3)
        Vertices of surface mesh
    faces : (F, 3)
        Triangles of surface mesh
    view : {'lateral', 'medial'}
        Plotted view of brain

    Returns
    -------
    colors : (F,) array_like
        Updated facecolors with appropriate shading
    """
    hemi_view = {'R': {'lateral': 'medial', 'medial': 'lateral'}}
    views = {
        'lateral': plt.cm.colors.LightSource(azdeg=225, altdeg=19.4712),
        'medial': plt.cm.colors.LightSource(azdeg=45, altdeg=19.4712)
    }

    # reverse medial / lateral views if plotting right hemisphere
    view = hemi_view.get(hemi, {}).get(view, view)
    # re-shade colors
    normals = ax._generate_normals(vertices[faces])
    colors = ax._shade_colors(np.asarray(facecolors), normals, views[view])

    return colors

fig=plot_surf_template(data, 'fsaverage', '10k', surf='inflated',  mask_medial=True, **kwargs)
plt.show()
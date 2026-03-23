# 1. 导入必要的库
from brainspace.datasets import load_parcellation, load_conte69
from brainspace.plotting import plot_hemispheres
import numpy as np
import nibabel as nib

# 2. 获取Schaefer100分区数据
print("正在获取Schaefer100分区数据...")
parcellation = fetch_parcellation(
    template='fslr32k',
    atlas='schaefer',
    n_regions=100,
    seven_networks=True,
    join=False
)

# 获取左右半球的分区标签
lh_labels, rh_labels = parcellation

# 3. 加载你的表面数据
# 如果你有自己的表面数据文件，使用以下代码加载：
# surf_lh = nib.load('你的左脑表面文件.surf.gii')
# surf_rh = nib.load('你的右脑表面文件.surf.gii')

# 如果你还没有自己的表面数据，可以使用BrainSpace内置的FSLR32k表面作为演示
print("正在加载表面数据...")
surf_lh, surf_rh = load_conte69()

# 4. 可视化分区结果
print("正在生成可视化...")
plot_hemispheres(
    surf_lh,
    surf_rh,
    array_name=[lh_labels, rh_labels],
    label_text=['Schaefer100 - 左脑', 'Schaefer100 - 右脑'],
    cmap='prism',  # 使用prism颜色映射，相邻区域颜色区分明显
    embed_nb=True,
    size=(1200, 400),
    color_range='separate'  # 左右脑使用独立的颜色范围
)

# 5. 打印分区信息
print("\n分区信息:")
print(f"左脑分区数: {len(np.unique(lh_labels))}")
print(f"右脑分区数: {len(np.unique(rh_labels))}")
print(f"左脑标签范围: {np.min(lh_labels)} - {np.max(lh_labels)}")
print(f"右脑标签范围: {np.min(rh_labels)} - {np.max(rh_labels)}")
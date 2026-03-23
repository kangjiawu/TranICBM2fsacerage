import numpy as np
from scipy.io import loadmat, savemat
from scipy.spatial import KDTree
import mne
from mne.datasets import fetch_fsaverage
import os.path as op

# =====================================================
# 1) 加载源数据（8003点）
# =====================================================
print("[Step 1] 加载源数据...")
mat_file = "D:/mywork/romaldo_data/Group_Average_xialphanet/Alpha_Bandwidth.mat"
mat_data = loadmat(mat_file)
mat_keys = [k for k in mat_data.keys() if not k.startswith('__')]
source_data = mat_data[mat_keys[0]].flatten()  # (8003,)
print(f"源数据维度: {source_data.shape}")

# 加载源表面（顶点坐标）
surface_file = "E:/XiAlphaNet-master/+templates/Cortex.mat"
sf_raw = loadmat(surface_file)
source_vertices = sf_raw['Vertices']  # (8003, 3)
print(f"源顶点范围 X: [{source_vertices[:,0].min():.1f}, {source_vertices[:,0].max():.1f}]")
if np.abs(source_vertices).max() < 1.0:  # 源顶点很小，可能是米
    print("检测到源顶点单位可能为米，自动乘以 1000 转换为毫米...")
    source_vertices = source_vertices * 1000.0
    print(f"转换后源顶点范围 X: [{source_vertices[:,0].min():.1f}, {source_vertices[:,0].max():.1f}]")
# =====================================================
# 2) 获取目标表面 fsaverage10k（每半球 10242 点）
# =====================================================
print("[Step 2] 获取目标表面 fsaverage32k...")
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# 加载左右半球白色表面（顶点坐标）
surf_lh = mne.read_surface(op.join(fs_dir, 'surf', 'lh.white'))
surf_rh = mne.read_surface(op.join(fs_dir, 'surf', 'rh.white'))
verts_lh = surf_lh[0]  # (10242, 3)
verts_rh = surf_rh[0]  # (10242, 3)

print(f"左半球目标顶点数: {len(verts_lh)}")
print(f"右半球目标顶点数: {len(verts_rh)}")

# =====================================================
# 3) 根据已知索引分离源顶点为左右半球
# =====================================================
print("[Step 3] 根据已知索引分离源半球...")
n_left = 4002
n_right = 4001
assert len(source_data) == n_left + n_right, "总点数不符（应为8003）"

source_left_verts = source_vertices[:n_left]
source_left_data = source_data[:n_left]
source_right_verts = source_vertices[n_left:]
source_right_data = source_data[n_left:]

print(f"左半球源点数: {len(source_left_verts)}")
print(f"右半球源点数: {len(source_right_verts)}")

# =====================================================
# 4) 对每个半球进行最近邻映射
# =====================================================
import numpy as np
from scipy.spatial import KDTree

import numpy as np
from scipy.spatial import KDTree

def interpolate_to_target(source_verts, source_data, target_verts, k=3, power=2, eps=1e-8):
    """
    使用反距离加权插值将源顶点数据映射到目标顶点。

    参数
    ----------
    source_verts : (N, 3) 源顶点坐标
    source_data : (N,) 或 (N, D) 源顶点上的数据（标量或向量）
    target_verts : (M, 3) 目标顶点坐标
    k : int, 近邻数量
    power : float, 距离衰减指数（越大衰减越快）
    eps : float, 避免除以零的小常数

    返回
    -------
    target_data : (M,) 或 (M, D) 插值后的目标数据，与source_data维度一致
    """
    tree = KDTree(source_verts)
    distances, indices = tree.query(target_verts, k=k)

    # 记录原始维度，并将数据统一转为二维进行处理
    original_ndim = source_data.ndim
    if original_ndim == 1:
        source_data_2d = source_data[:, np.newaxis]
    else:
        source_data_2d = source_data

    M = len(target_verts)
    D = source_data_2d.shape[1]
    target_data_2d = np.zeros((M, D))

    for i in range(M):
        d = distances[i]
        idx = indices[i]

        # 如果最近点距离几乎为0，直接使用该点的值
        if d[0] < eps:
            target_data_2d[i] = source_data_2d[idx[0]]
        else:
            weights = 1.0 / (d ** power + eps)
            weights /= weights.sum()  # 归一化
            # 加权平均
            target_data_2d[i] = np.sum(weights[:, np.newaxis] * source_data_2d[idx], axis=0)

    # 恢复原始维度
    if original_ndim == 1:
        return target_data_2d[:, 0]
    else:
        return target_data_2d

# 使用插值代替最近邻
target_left_data = interpolate_to_target(source_left_verts, source_left_data, verts_lh, k=3, power=2)
target_right_data = interpolate_to_target(source_right_verts, source_right_data, verts_rh, k=3, power=2)

# 合并（左半球在前）
target_data = np.concatenate([target_left_data, target_right_data])
print(f"目标数据总点数: {len(target_data)} (应为 {len(verts_lh) + len(verts_rh)})")
# print("[Step 4] 最近邻映射...")
#
# def map_to_target(source_verts, source_data, target_verts):
#     """将源顶点数据映射到目标顶点（最近邻）"""
#     tree = KDTree(source_verts)
#     dist, idx = tree.query(target_verts)
#     return source_data[idx]
#
# target_left_data = map_to_target(source_left_verts, source_left_data, verts_lh)
# target_right_data = map_to_target(source_right_verts, source_right_data, verts_rh)
#
# # 合并（左半球在前）
# target_data = np.concatenate([target_left_data, target_right_data])
# print(f"目标数据总点数: {len(target_data)} (应为 {len(verts_lh) + len(verts_rh)})")

# =====================================================
# 5) 保存结果
# =====================================================
print("[Step 5] 保存结果...")
output_mat = 'Alpha_Bandwidth_fsaverage10k_NN2.mat'
savemat(output_mat, {'Alpha_Bandwidth_fsaverage10k': target_data})
print(f"已保存: {output_mat}")

# =====================================================
# 6) 绘制变换前后对比图（需安装pyvista）
# =====================================================
try:
    import pyvista as pv
    print("生成可视化...")
    target_vertices = np.vstack([verts_lh, verts_rh])
    plotter = pv.Plotter()
    point_cloud = pv.PolyData(target_vertices)
    point_cloud.point_data['Alpha'] = target_data
    plotter.add_points(point_cloud, scalars='Alpha', cmap='viridis', point_size=3)
    plotter.show_grid()
    plotter.show()
except ImportError:
    print("未安装 PyVista，跳过可视化。")
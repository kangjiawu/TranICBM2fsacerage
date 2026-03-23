import numpy as np
import pyvista as pv
from scipy.io import loadmat

# =====================================================
# 1) 加载表面数据 (Cortex.mat) 和 标量数据 (Alpha_Bandwidth.mat)
# =====================================================
print("[Step 1] 加载表面和标量数据...")

# 标量数据文件
mat_file = "D:/mywork/romaldo_data/Group_Average_xialphanet/Alpha_Bandwidth.mat"
mat_data = loadmat(mat_file)
mat_keys = [k for k in mat_data.keys() if not k.startswith('__')]
print("mat文件中的变量名：", mat_keys)
# 选择要显示的组（例如第一组）
values = mat_data[mat_keys[0]].flatten()
print(f"标量数据维度：{values.shape}")

# 表面文件
surface_file = "E:/XiAlphaNet-master/+templates/Cortex.mat"
sf_raw = loadmat(surface_file)
sf_keys = [k for k in sf_raw.keys() if not k.startswith('__')]
print("表面文件中的变量名：", sf_keys)

# 提取顶点和面（MATLAB索引从1开始，转换为0索引）
vertices = sf_raw['Vertices']          # shape (8003, 3)
faces = sf_raw['Faces'] - 1            # shape (15998, 3)

# 检查顶点数与标量数是否一致
if len(values) != vertices.shape[0]:
    raise ValueError(f"顶点数({vertices.shape[0]})与标量数据长度({len(values)})不匹配")

# 打印调试信息
print(f"顶点数组: shape={vertices.shape}, dtype={vertices.dtype}")
print(f"面数组: shape={faces.shape}, dtype={faces.dtype}")
print(f"面索引范围: {faces.min()} ~ {faces.max()} (应介于0~{vertices.shape[0]-1})")

# 确保面索引有效
if faces.min() < 0 or faces.max() >= vertices.shape[0]:
    raise ValueError("面索引超出顶点范围")

# =====================================================
# 2) 创建PyVista网格并设置标量数据
# =====================================================
print("[Step 2] 创建PyVista网格...")

# 将faces转换为VTK CellArray格式：每行 [3, v0, v1, v2] 然后拉平
vtk_faces = np.column_stack((np.full(faces.shape[0], 3, dtype=faces.dtype), faces)).ravel()

# 创建PolyData对象
mesh = pv.PolyData(vertices, vtk_faces)

# 添加标量数据到点
mesh.point_data['Alpha Bandwidth'] = values

# =====================================================
# 3) 可视化
# =====================================================
print("[Step 3] 开始可视化...")

# 创建绘图对象
plotter = pv.Plotter()

# 添加网格
plotter.add_mesh(mesh,
                 scalars='Alpha Bandwidth',
                 cmap='viridis',
                 show_edges=False,
                 smooth_shading=True,
                 lighting=True,
                 scalar_bar_args={'title': 'Alpha Bandwidth', 'n_colors': 256})

# 设置背景色
plotter.set_background('white')

# 显示坐标轴（可选）
plotter.show_grid(color='black')

# 设置初始视角（俯视图）
plotter.view_xy()

# 显示窗口
plotter.show()

# 如需保存截图，取消下面注释
# plotter.screenshot('cortex_alpha.png')
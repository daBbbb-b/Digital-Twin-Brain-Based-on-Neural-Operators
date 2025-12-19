"""
表面工具模块

功能:
    - 从 GIFTI 文件加载表面网格。
    - 从网格计算邻接矩阵。
    - 计算用于表面偏微分方程的拉普拉斯算子。
"""

import numpy as np
from scipy import sparse
import logging

logger = logging.getLogger(__name__)

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    logger.warning("未找到 nibabel。表面加载将失败。")

def load_surface(gii_file):
    """
    从 GIFTI 文件加载表面网格。
    
    参数:
        gii_file: .gii 文件路径
        
    返回:
        vertices: (N, 3) 坐标
        faces: (M, 3) 顶点索引
    """
    if not HAS_NIBABEL:
        raise ImportError("加载 GIFTI 文件需要 nibabel。")
        
    gii = nib.load(gii_file)
    # 通常 darrays[0] 是坐标，darrays[1] 是拓扑结构（三角形）
    # 我们应该检查 intent，但对于标准的 surf.gii 文件，这是常见的。
    vertices = gii.darrays[0].data
    faces = gii.darrays[1].data
    return vertices, faces

def get_mesh_adjacency(faces, n_vertices):
    """
    从网格面计算邻接矩阵。
    
    参数:
        faces: (M, 3) 顶点索引数组
        n_vertices: 顶点总数
        
    返回:
        adj: (N, N) 稀疏 CSR 格式的邻接矩阵
    """
    # 从三角形生成边: (v0, v1), (v1, v2), (v2, v0)
    edges_src = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    edges_dst = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    
    # 添加对称边
    rows = np.concatenate([edges_src, edges_dst])
    cols = np.concatenate([edges_dst, edges_src])
    
    data = np.ones(len(rows), dtype=int)
    
    # 创建稀疏矩阵
    adj = sparse.coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
    
    # 移除重复边和自环（尽管在有效网格中不应存在自环）
    adj = adj.tocsr()
    adj.data[:] = 1 # 二值化
    
    return adj

def compute_surface_laplacian(vertices, faces):
    """
    从表面网格计算图拉普拉斯算子。
    L = D - A
    """
    n_vertices = vertices.shape[0]
    adj = get_mesh_adjacency(faces, n_vertices)
    
    degree = np.array(adj.sum(axis=1)).flatten()
    D = sparse.diags(degree)
    
    L = D - adj
    return L

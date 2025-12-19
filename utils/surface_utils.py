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
import time

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
    logger.info(f"开始构建邻接矩阵: {n_vertices} 顶点, {faces.shape[0]} 面")
    t0 = time.time()
    
    # 确保 faces 是 numpy 数组
    faces = np.asarray(faces)
    
    # 从三角形生成边: (v0, v1), (v1, v2), (v2, v0)
    # 使用 numpy 向量化操作
    v0 = faces[:, 0]
    v1 = faces[:, 1]
    v2 = faces[:, 2]
    
    edges_src = np.concatenate([v0, v1, v2])
    edges_dst = np.concatenate([v1, v2, v0])
    
    # 添加对称边
    rows = np.concatenate([edges_src, edges_dst])
    cols = np.concatenate([edges_dst, edges_src])
    
    data = np.ones(len(rows), dtype=int)
    
    logger.info("创建 COO 矩阵...")
    # 创建稀疏矩阵
    adj = sparse.coo_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
    
    logger.info("转换为 CSR 并二值化...")
    # 移除重复边和自环
    adj = adj.tocsr()
    adj.data[:] = 1 # 二值化
    
    logger.info(f"邻接矩阵构建完成，耗时 {time.time() - t0:.4f}s")
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

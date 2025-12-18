"""
连接性模块

本模块处理大脑的各种连接性数据：
- 结构连接（皮层结构连接）
- 有效连接（基于虚拟扰动方法）
- 白质连接（组平均数据）
"""

from .structural_connectivity import StructuralConnectivity
from .effective_connectivity import EffectiveConnectivity
from .white_matter_connectivity import WhiteMatterConnectivity

__all__ = ['StructuralConnectivity', 'EffectiveConnectivity', 'WhiteMatterConnectivity']

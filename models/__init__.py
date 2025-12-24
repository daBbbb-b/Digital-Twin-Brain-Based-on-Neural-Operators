"""
神经算子模型模块

本模块实现各种神经算子方法，用于学习大脑动力学方程的刺激函数：
- FNO（Fourier Neural Operator）
- DeepONet（Deep Operator Network）
- 其他神经算子变体
"""

from .mlp import MLP
from .deeponet import DeepONet
from .fno import FNO1d, FNO2d

__all__ = ['MLP', 'DeepONet', 'FNO1d', 'FNO2d']
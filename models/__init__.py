"""
神经算子模型模块

本模块实现各种神经算子方法，用于学习大脑动力学方程的刺激函数：
- FNO（Fourier Neural Operator）
- DeepONet（Deep Operator Network）
- 其他神经算子变体
"""

from .base_operator import BaseOperator
from .fno import FNO, FNO1d, FNO2d, FNO3d
from .deeponet import DeepONet, BranchNet, TrunkNet
from .operator_ensemble import OperatorEnsemble

__all__ = ['BaseOperator', 'FNO', 'FNO1d', 'FNO2d', 'FNO3d',
           'DeepONet', 'BranchNet', 'TrunkNet', 'OperatorEnsemble']

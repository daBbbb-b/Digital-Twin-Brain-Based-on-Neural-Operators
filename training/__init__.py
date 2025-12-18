"""
训练和推理模块

本模块负责神经算子的训练、微调和推理：
- 在仿真数据上训练
- 在真实fMRI数据上微调
- 求解刺激函数
"""

from .trainer import Trainer
from .fine_tuner import FineTuner
from .stimulus_solver import StimulusSolver

__all__ = ['Trainer', 'FineTuner', 'StimulusSolver']

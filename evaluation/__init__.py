"""
评估模块

本模块负责模型和结果的评估：
- 仿真数据集上的性能评估
- 真实数据集上的性能评估
- 一致性分析
"""

from .simulation_metrics import SimulationMetrics
from .real_data_metrics import RealDataMetrics
from .consistency_analysis import ConsistencyAnalysis

__all__ = ['SimulationMetrics', 'RealDataMetrics', 'ConsistencyAnalysis']

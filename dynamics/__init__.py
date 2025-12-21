"""
神经动力学模块

本模块实现各种神经动力学方程，包括：
- ODE方程（常微分方程）
- PDE方程（偏微分方程）
- 随机微分方程
- 气球-卷积模型
"""

from .ode_models import ODEModel, EIModel
from .pde_models import PDEModel, WaveEquationModel
from .balloon_model import BalloonModel

__all__ = ['ODEModel', 'EIModel', 'PDEModel', 'WaveEquationModel', 
           'StochasticODE', 'StochasticPDE', 'BalloonModel']

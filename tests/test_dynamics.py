"""
神经动力学模块测试

测试内容：
- ODE模型正确性
- PDE模型正确性
- 随机微分方程
- 气球模型
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from dynamics import EIModel, DiffusionModel, StochasticODE, BalloonModel


class TestODEModels:
    """测试ODE模型"""
    
    def test_ei_model_initialization(self):
        """测试EI模型初始化"""
        model = EIModel(n_nodes=10)
        assert model.n_nodes == 10
        
    def test_ei_model_solve(self):
        """测试EI模型求解"""
        model = EIModel(n_nodes=10)
        initial_state = np.random.randn(20)  # E和I状态
        connectivity = np.random.randn(10, 10)
        
        t, states = model.solve(
            initial_state=initial_state,
            connectivity=connectivity,
            t_span=(0, 10),
            dt=0.1
        )
        
        assert len(t) > 0
        assert states.shape[0] == len(t)


class TestPDEModels:
    """测试PDE模型"""
    
    def test_diffusion_model_initialization(self):
        """测试扩散模型初始化"""
        model = DiffusionModel(n_vertices=100)
        assert model.n_vertices == 100
        
    def test_diffusion_model_solve(self):
        """测试扩散模型求解"""
        # 此处应包含实际的测试代码
        pass


class TestStochasticModels:
    """测试随机模型"""
    
    def test_stochastic_ode(self):
        """测试随机ODE"""
        # 此处应包含实际的测试代码
        pass


class TestBalloonModel:
    """测试气球模型"""
    
    def test_balloon_model(self):
        """测试气球模型"""
        model = BalloonModel()
        
        # 测试神经活动到BOLD信号的转换
        neural_activity = np.random.randn(100, 10)
        bold_signal = model.neural_to_bold(neural_activity)
        
        assert bold_signal.shape[1] == 10  # 保持脑区数量


if __name__ == '__main__':
    pytest.main([__file__])

"""
神经算子模型测试

测试内容：
- FNO模型
- DeepONet模型
- 算子集成
"""

import pytest
import torch
import sys
sys.path.append('..')

from models import FNO, DeepONet, OperatorEnsemble


class TestFNO:
    """测试FNO模型"""
    
    def test_fno1d_initialization(self):
        """测试FNO1d初始化"""
        model = FNO(dim=1, modes=16, width=64, n_layers=4)
        assert model is not None
        
    def test_fno1d_forward(self):
        """测试FNO1d前向传播"""
        model = FNO(dim=1, modes=16, width=64)
        x = torch.randn(2, 100, 3)  # (batch, length, channels)
        
        # 前向传播应该不报错
        output = model(x)
        assert output.shape[0] == 2  # 批次大小保持


class TestDeepONet:
    """测试DeepONet模型"""
    
    def test_deeponet_initialization(self):
        """测试DeepONet初始化"""
        model = DeepONet(
            input_dim=1,
            coord_dim=2,
            n_sensors=50,
            n_basis=100
        )
        assert model is not None
        
    def test_deeponet_forward(self):
        """测试DeepONet前向传播"""
        model = DeepONet(
            input_dim=1,
            coord_dim=2,
            n_sensors=50,
            n_basis=100
        )
        
        u = torch.randn(2, 50, 1)  # 输入函数
        y = torch.randn(2, 100, 2)  # 输出位置
        
        output = model(u, y)
        assert output.shape == (2, 100, 1)


class TestOperatorEnsemble:
    """测试算子集成"""
    
    def test_ensemble_initialization(self):
        """测试集成初始化"""
        model1 = FNO(dim=1)
        model2 = FNO(dim=1)
        
        ensemble = OperatorEnsemble(
            models=[model1, model2],
            ensemble_method='average'
        )
        
        assert len(ensemble.models) == 2


if __name__ == '__main__':
    pytest.main([__file__])

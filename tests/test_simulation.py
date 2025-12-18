"""
仿真模块测试

测试内容：
- ODE仿真器
- PDE仿真器
- 刺激生成器
- 噪声生成器
"""

import pytest
import numpy as np
import sys
sys.path.append('..')

from simulation import ODESimulator, PDESimulator, StimulationGenerator, NoiseGenerator


class TestODESimulator:
    """测试ODE仿真器"""
    
    def test_generate_single_sample(self):
        """测试生成单个样本"""
        simulator = ODESimulator(model_type='EI', n_nodes=10)
        
        connectivity = np.random.randn(10, 10)
        stimulus_params = {
            'stimulus_type': 'pulse',
            'target_regions': [0, 1],
            'amplitude': 1.0,
            'duration': 1.0,
            'onset_time': 5.0
        }
        
        sample = simulator.generate_single_sample(
            connectivity=connectivity,
            stimulus_params=stimulus_params,
            t_span=(0, 20),
            dt=0.1
        )
        
        assert 'timeseries' in sample
        assert sample['timeseries'].shape[1] == 10


class TestStimulationGenerator:
    """测试刺激生成器"""
    
    def test_pulse_stimulus(self):
        """测试脉冲刺激"""
        generator = StimulationGenerator()
        
        t, stimulus = generator.generate_temporal_stimulus(
            stimulus_type='pulse',
            params={'amplitude': 1.0, 'onset_time': 5.0, 'duration': 1.0},
            t_span=(0, 20),
            dt=0.1
        )
        
        assert len(t) == len(stimulus)
        assert np.max(stimulus) > 0


class TestNoiseGenerator:
    """测试噪声生成器"""
    
    def test_white_noise(self):
        """测试白噪声"""
        generator = NoiseGenerator(seed=42)
        
        noise = generator.generate_white_noise(
            shape=(100, 10),
            mean=0.0,
            std=1.0
        )
        
        assert noise.shape == (100, 10)
        assert np.abs(np.mean(noise)) < 0.2  # 均值接近0


if __name__ == '__main__':
    pytest.main([__file__])

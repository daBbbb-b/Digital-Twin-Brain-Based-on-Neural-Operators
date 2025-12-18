# 基于神经算子的数字大脑模型

## 项目简介

本项目是数字孪生脑技术路线之一，核心目标是利用神经算子（Neural Operator）对宏观、介观和微观尺度的大脑进行建模。通过神经算子学习大脑动力学方程的刺激函数，实现从结构到功能的映射。

## 主要特性

- **多尺度建模**：支持宏观、介观、微观多个尺度的大脑建模
- **神经动力学仿真**：基于ODE和PDE的神经动力学方程构建仿真数据集
- **神经算子学习**：使用FNO、DeepONet等方法学习刺激函数
- **真实数据应用**：在103个认知任务数据集上进行刺激图谱预测
- **可视化分析**：使用Brainspace进行大脑结果可视化和一致性分析

## 项目结构

```
Digital-Twin-Brain-Based-on-Neural-Operators/
├── README.md                      # 项目说明文档
├── requirements.txt               # Python依赖包列表
├── setup.py                       # 项目安装配置
├── config/                        # 配置文件目录
│   ├── data_config.yaml          # 数据配置
│   ├── model_config.yaml         # 模型配置
│   └── train_config.yaml         # 训练配置
├── data/                          # 数据模块
│   ├── __init__.py
│   ├── data_loader.py            # 数据加载器
│   ├── preprocessor.py           # 数据预处理
│   └── task_manager.py           # 103任务管理
├── dynamics/                      # 神经动力学模块
│   ├── __init__.py
│   ├── ode_models.py             # ODE方程实现
│   ├── pde_models.py             # PDE方程实现
│   ├── stochastic_models.py      # 随机微分方程
│   └── balloon_model.py          # 气球-卷积模型
├── simulation/                    # 仿真数据生成模块
│   ├── __init__.py
│   ├── ode_simulator.py          # ODE仿真器
│   ├── pde_simulator.py          # PDE仿真器
│   ├── stimulation_generator.py  # 刺激生成器
│   └── noise_generator.py        # 噪声生成器
├── models/                        # 神经算子模型模块
│   ├── __init__.py
│   ├── fno.py                    # Fourier Neural Operator
│   ├── deeponet.py               # DeepONet
│   ├── base_operator.py          # 基础算子类
│   └── operator_ensemble.py      # 算子集成
├── connectivity/                  # 连接性模块
│   ├── __init__.py
│   ├── structural_connectivity.py # 结构连接
│   ├── effective_connectivity.py  # 有效连接
│   └── white_matter_connectivity.py # 白质连接
├── training/                      # 训练和推理模块
│   ├── __init__.py
│   ├── trainer.py                # 训练器
│   ├── fine_tuner.py             # 微调器
│   └── stimulus_solver.py        # 刺激函数求解器
├── visualization/                 # 可视化模块
│   ├── __init__.py
│   ├── brain_visualizer.py       # 大脑可视化
│   ├── stimulus_mapper.py        # 刺激图谱
│   └── comparison_plots.py       # 对比图表
├── evaluation/                    # 评估模块
│   ├── __init__.py
│   ├── simulation_metrics.py     # 仿真评估指标
│   ├── real_data_metrics.py      # 真实数据评估
│   └── consistency_analysis.py   # 一致性分析
├── utils/                         # 工具模块
│   ├── __init__.py
│   ├── math_utils.py             # 数学工具
│   ├── io_utils.py               # 输入输出工具
│   └── logger.py                 # 日志工具
├── examples/                      # 示例脚本
│   ├── 01_generate_simulation_data.py
│   ├── 02_train_on_simulation.py
│   ├── 03_finetune_on_real_data.py
│   └── 04_visualize_results.py
└── tests/                         # 测试模块
    ├── __init__.py
    ├── test_dynamics.py
    ├── test_models.py
    └── test_simulation.py
```

## 技术路线

### 1. 仿真数据生成
- 使用ODE方程（基于EI模型）构建仿真数据
- 使用PDE方程（扩散模型）构建仿真数据
- 包含随机噪声项（随机ODE和随机PDE）
- 覆盖不同刺激脑区、刺激方式、连接模式

### 2. 神经算子训练
- 在仿真数据集上训练FNO或DeepONet
- 学习从连接矩阵到刺激函数的映射
- 支持多尺度刺激（神经递质层面、平均发放率层面）

### 3. 真实数据应用
- 在103个认知任务的fMRI数据上微调模型
- 考虑fMRI时间间隔（2秒）的稀疏监督
- 预测每个任务的刺激图谱

### 4. 评估与分析
- 仿真数据集效果评估
- 真实任务刺激图谱与功能图谱一致性分析
- 多模型对比分析

## 数据说明

### 输入数据
- **结构连接**：皮层结构连接、组平均白质结构连接
- **有效连接**：基于虚拟扰动方法计算的有效连接
- **fMRI数据**：MNI空间的功能磁共振数据（103个任务）
- **解剖数据**：T1图像、分割标签、灰白质surface

### 数据格式
- 连接矩阵：`.npy`或`.mat`格式
- fMRI时间序列：`.nii.gz`格式
- 任务列表：`.tsv`格式

## 使用方法

### 环境配置
```bash
pip install -r requirements.txt
```

### 生成仿真数据
```bash
python examples/01_generate_simulation_data.py
```

### 训练神经算子模型
```bash
python examples/02_train_on_simulation.py
```

### 在真实数据上微调
```bash
python examples/03_finetune_on_real_data.py
```

### 可视化结果
```bash
python examples/04_visualize_results.py
```

## 参考文献

1. Li et al. (2021) - Fourier Neural Operator for Parametric PDEs
2. Lu et al. (2019) - DeepONet
3. Pang et al. (2023) - Geometric constraints on human brain function
4. Luo et al. (2025) - Mapping effective connectivity
5. Friston et al. (2003) - Dynamic causal modelling

## 注意事项

1. fMRI数据时间间隔为2秒，监督信息稀疏
2. 需要考虑随机微分方程视角下的噪声项
3. 任务态刺激基于动态因果模型视角
4. 使用Brainspace库进行大脑可视化

## 许可证

MIT License
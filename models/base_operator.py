"""
神经算子基类模块

功能说明：
    定义神经算子的通用接口和基础功能。

主要类：
    BaseOperator: 所有神经算子的基类

输入：
    - 输入函数/场（如连接矩阵）
    - 参数（如物理参数、初始条件）

输出：
    - 输出函数/场（如刺激函数、解轨迹）

说明：
    神经算子学习函数到函数的映射，用于求解参数化的微分方程。
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union
from abc import ABC, abstractmethod


class BaseOperator(nn.Module, ABC):
    """
    神经算子基类
    
    功能：
        - 定义神经算子的通用接口
        - 提供训练和推理的基础方法
        - 管理模型参数和状态
        
    核心概念：
        神经算子学习算子 G: U → V 的映射
        其中 U 和 V 是函数空间
        
        在本项目中：
        - 输入空间 U: 连接矩阵 C(i,j)
        - 输出空间 V: 刺激函数 s(t, i) 或解轨迹 x(t, i)
        
    属性：
        input_dim (int): 输入维度
        output_dim (int): 输出维度
        hidden_dim (int): 隐藏层维度
        
    方法：
        forward: 前向传播
        loss_function: 损失函数计算
        train_step: 单步训练
        predict: 预测
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 128):
        """
        初始化神经算子基类
        
        参数：
            input_dim (int): 输入维度
            output_dim (int): 输出维度
            hidden_dim (int): 隐藏层维度
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        前向传播（抽象方法，子类必须实现）
        
        参数：
            x (torch.Tensor): 输入张量
            **kwargs: 其他参数
            
        返回：
            torch.Tensor: 输出张量
        """
        pass
    
    def loss_function(self, 
                     pred: torch.Tensor, 
                     target: torch.Tensor,
                     loss_type: str = 'mse') -> torch.Tensor:
        """
        计算损失函数
        
        参数：
            pred (torch.Tensor): 预测值
            target (torch.Tensor): 目标值
            loss_type (str): 损失类型
                - 'mse': 均方误差
                - 'mae': 平均绝对误差
                - 'relative': 相对误差
                
        返回：
            torch.Tensor: 损失值
        """
        pass
    
    def train_step(self,
                  batch: Dict[str, torch.Tensor],
                  optimizer: torch.optim.Optimizer) -> float:
        """
        单步训练
        
        参数：
            batch (Dict[str, torch.Tensor]): 批次数据
                - 'input': 输入数据
                - 'target': 目标数据
            optimizer (torch.optim.Optimizer): 优化器
            
        返回：
            float: 损失值
        """
        pass
    
    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        预测（推理模式）
        
        参数：
            x (torch.Tensor): 输入数据
            **kwargs: 其他参数
            
        返回：
            torch.Tensor: 预测结果
        """
        pass
    
    def count_parameters(self) -> int:
        """
        统计模型参数数量
        
        返回：
            int: 可训练参数总数
        """
        pass
    
    def save_model(self, save_path: str):
        """
        保存模型
        
        参数：
            save_path (str): 保存路径
        """
        pass
    
    def load_model(self, load_path: str):
        """
        加载模型
        
        参数：
            load_path (str): 模型路径
        """
        pass

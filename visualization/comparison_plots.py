"""
对比图表模块

功能说明：
    生成刺激图谱与功能图谱的对比图表。

主要类：
    ComparisonPlots: 对比图表生成器

输入：
    - 预测的刺激图谱
    - 真实的功能激活图谱
    - 评估指标

输出：
    - 对比可视化图表
    - 一致性分析图

说明：
    核心评价标准：刺激图谱与真实功能图谱的一致性分析。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple


class ComparisonPlots:
    """
    对比图表生成器
    
    功能：
        - 对比预测刺激与真实激活
        - 一致性分析可视化
        - 误差分析
        - 跨任务对比
        
    核心目标：
        分析103个任务上预测的刺激图谱与真实功能图谱的一致性
        
    方法：
        plot_stimulus_vs_activation: 刺激vs激活对比
        plot_consistency_analysis: 一致性分析
        plot_error_distribution: 误差分布
        plot_cross_task_comparison: 跨任务对比
    """
    
    def __init__(self):
        """初始化对比图表生成器"""
        pass
    
    def plot_stimulus_vs_activation(self,
                                   predicted_stimulus: np.ndarray,
                                   true_activation: np.ndarray,
                                   task_name: str,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制预测刺激与真实激活的对比
        
        参数：
            predicted_stimulus (np.ndarray): 预测的刺激图谱 (n_regions,)
            true_activation (np.ndarray): 真实的功能激活 (n_regions,)
            task_name (str): 任务名称
            save_path (str, optional): 保存路径
            
        返回：
            plt.Figure: 对比图
        """
        pass
    
    def plot_consistency_analysis(self,
                                  stimuli: Dict[str, np.ndarray],
                                  activations: Dict[str, np.ndarray],
                                  metrics: Dict[str, Dict[str, float]],
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制一致性分析图
        
        分析所有103个任务的一致性
        
        参数：
            stimuli (Dict[str, np.ndarray]): 任务到刺激的映射
            activations (Dict[str, np.ndarray]): 任务到激活的映射
            metrics (Dict[str, Dict[str, float]]): 任务到指标的映射
            save_path (str, optional): 保存路径
            
        返回：
            plt.Figure: 一致性分析图
        """
        pass
    
    def plot_spatial_correlation(self,
                                predicted: np.ndarray,
                                true: np.ndarray,
                                region_names: Optional[List[str]] = None,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制空间相关性散点图
        
        参数：
            predicted (np.ndarray): 预测值
            true (np.ndarray): 真实值
            region_names (List[str], optional): 区域名称（用于标注）
            save_path (str, optional): 保存路径
            
        返回：
            plt.Figure: 散点图
        """
        pass
    
    def plot_error_distribution(self,
                               errors: Dict[str, np.ndarray],
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制误差分布
        
        参数：
            errors (Dict[str, np.ndarray]): 任务到误差的映射
            save_path (str, optional): 保存路径
            
        返回：
            plt.Figure: 误差分布图
        """
        pass
    
    def plot_cross_task_comparison(self,
                                   task_metrics: Dict[str, Dict[str, float]],
                                   metric_name: str = 'correlation',
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制跨任务指标对比
        
        参数：
            task_metrics (Dict[str, Dict[str, float]]): 任务指标
            metric_name (str): 要对比的指标名称
            save_path (str, optional): 保存路径
            
        返回：
            plt.Figure: 对比图
        """
        pass
    
    def plot_task_category_performance(self,
                                      task_categories: Dict[str, List[str]],
                                      task_metrics: Dict[str, Dict[str, float]],
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        按任务类别分析性能
        
        参数：
            task_categories (Dict[str, List[str]]): 类别到任务列表的映射
            task_metrics (Dict[str, Dict[str, float]]): 任务指标
            save_path (str, optional): 保存路径
            
        返回：
            plt.Figure: 类别性能图
        """
        pass

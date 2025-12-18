"""
刺激图谱模块

功能说明：
    绘制和分析刺激图谱。

主要类：
    StimulusMapper: 刺激图谱绘制器

输入：
    - 刺激函数
    - 任务信息
    - 分区数据

输出：
    - 刺激图谱
    - 时空刺激模式图

说明：
    核心评价标准之一：分析刺激图谱与真实功能图谱的一致性。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List


class StimulusMapper:
    """
    刺激图谱绘制器
    
    功能：
        - 绘制各任务的刺激图谱
        - 可视化刺激的时空模式
        - 对比不同任务的刺激
        - 识别刺激热点
        
    应用：
        - 每个任务的刺激图谱可视化
        - 与功能激活图谱的对比
        - 跨任务刺激模式分析
        
    方法：
        plot_stimulus_map: 绘制单个任务的刺激图谱
        plot_stimulus_timecourse: 绘制刺激时间过程
        plot_spatial_pattern: 绘制空间刺激模式
        compare_tasks: 对比多个任务的刺激
    """
    
    def __init__(self):
        """初始化刺激图谱绘制器"""
        pass
    
    def plot_stimulus_map(self,
                         stimulus: np.ndarray,
                         task_name: str,
                         parcellation: Optional[np.ndarray] = None,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制单个任务的刺激图谱
        
        参数：
            stimulus (np.ndarray): 刺激函数 (n_timepoints, n_regions)
            task_name (str): 任务名称
            parcellation (np.ndarray, optional): 分区标签
            save_path (str, optional): 保存路径
            
        返回：
            plt.Figure: 图形对象
        """
        pass
    
    def plot_stimulus_timecourse(self,
                                stimulus: np.ndarray,
                                region_names: Optional[List[str]] = None,
                                highlight_regions: Optional[List[int]] = None,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制刺激的时间过程
        
        参数：
            stimulus (np.ndarray): 刺激函数 (n_timepoints, n_regions)
            region_names (List[str], optional): 脑区名称
            highlight_regions (List[int], optional): 需要高亮的脑区
            save_path (str, optional): 保存路径
            
        返回：
            plt.Figure: 图形对象
        """
        pass
    
    def plot_spatial_pattern(self,
                            stimulus: np.ndarray,
                            time_window: Optional[Tuple[int, int]] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制空间刺激模式
        
        在指定时间窗口内的平均刺激空间分布
        
        参数：
            stimulus (np.ndarray): 刺激函数
            time_window (Tuple[int, int], optional): 时间窗口索引
            save_path (str, optional): 保存路径
            
        返回：
            plt.Figure: 图形对象
        """
        pass
    
    def plot_stimulus_heatmap(self,
                             stimulus: np.ndarray,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制刺激热图（时间×脑区）
        
        参数：
            stimulus (np.ndarray): 刺激函数 (n_timepoints, n_regions)
            save_path (str, optional): 保存路径
            
        返回：
            plt.Figure: 图形对象
        """
        pass
    
    def compare_tasks(self,
                     stimuli: Dict[str, np.ndarray],
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        对比多个任务的刺激模式
        
        参数：
            stimuli (Dict[str, np.ndarray]): 任务名到刺激函数的映射
            save_path (str, optional): 保存路径
            
        返回：
            plt.Figure: 对比图
        """
        pass
    
    def identify_stimulus_hotspots(self,
                                   stimulus: np.ndarray,
                                   percentile: float = 90) -> List[int]:
        """
        识别刺激热点（受刺激最强的脑区）
        
        参数：
            stimulus (np.ndarray): 刺激函数
            percentile (float): 百分位阈值
            
        返回：
            List[int]: 热点脑区索引列表
        """
        pass

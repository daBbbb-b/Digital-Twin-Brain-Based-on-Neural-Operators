"""
任务管理器模块

功能说明：
    管理103个认知任务的元数据、数据加载和组织。

主要类：
    TaskManager: 任务管理器类

输入：
    - 任务列表文件（.tsv格式）
    - 任务ID或索引

输出：
    - 任务元数据
    - 任务数据路径
    - 任务分组信息

使用示例：
    task_manager = TaskManager(task_file='tasks.tsv')
    task_info = task_manager.get_task_info('task001')
    all_tasks = task_manager.get_all_tasks()
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


class TaskManager:
    """
    任务管理器类
    
    功能：
        - 加载和解析103个任务的元数据
        - 提供任务查询接口
        - 管理任务分组和标签
        - 组织任务相关的数据路径
        
    属性：
        task_file (Path): 任务列表文件路径
        tasks_df (pd.DataFrame): 任务信息数据框
        n_tasks (int): 任务总数（应为103）
        
    方法：
        load_tasks: 加载任务列表
        get_task_info: 获取特定任务的信息
        get_all_tasks: 获取所有任务列表
        get_tasks_by_category: 按类别筛选任务
        get_task_data_paths: 获取任务相关数据路径
    """
    
    def __init__(self, task_file: str):
        """
        初始化任务管理器
        
        参数：
            task_file (str): 任务列表tsv文件路径
        """
        pass
    
    def load_tasks(self) -> pd.DataFrame:
        """
        加载任务列表
        
        返回：
            pd.DataFrame: 任务信息数据框，包含以下列：
                - task_id: 任务ID
                - task_name: 任务名称
                - category: 任务类别（如认知、情感、运动等）
                - n_subjects: 被试数量
                - n_timepoints: 时间点数量
                - description: 任务描述
        """
        pass
    
    def get_task_info(self, task_id: str) -> Dict:
        """
        获取特定任务的详细信息
        
        参数：
            task_id (str): 任务ID
            
        返回：
            Dict: 任务信息字典
        """
        pass
    
    def get_all_tasks(self) -> List[str]:
        """
        获取所有任务的ID列表
        
        返回：
            List[str]: 包含103个任务ID的列表
        """
        pass
    
    def get_tasks_by_category(self, category: str) -> List[str]:
        """
        按类别筛选任务
        
        参数：
            category (str): 任务类别
                - 'cognitive': 认知任务
                - 'emotion': 情感任务
                - 'motor': 运动任务
                - 'language': 语言任务
                - 'social': 社交任务
                
        返回：
            List[str]: 该类别下的任务ID列表
        """
        pass
    
    def get_task_data_paths(self, task_id: str) -> Dict[str, Path]:
        """
        获取任务相关的所有数据文件路径
        
        参数：
            task_id (str): 任务ID
            
        返回：
            Dict[str, Path]: 数据路径字典，包含：
                - 'fmri': fMRI数据路径
                - 'effective_connectivity': 有效连接路径
                - 'functional_map': 功能图谱路径
        """
        pass
    
    def get_task_statistics(self) -> Dict:
        """
        获取任务数据集的统计信息
        
        返回：
            Dict: 统计信息字典，包含：
                - total_tasks: 任务总数
                - tasks_by_category: 各类别任务数量
                - avg_timepoints: 平均时间点数
                - avg_subjects: 平均被试数
        """
        pass

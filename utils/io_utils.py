"""
输入输出工具模块

功能说明：
    处理文件读写和数据格式转换。

主要功能：
    - 数据加载和保存
    - 格式转换
    - 路径管理
"""

import numpy as np
import pickle
import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class IOUtils:
    """输入输出工具类"""
    
    @staticmethod
    def load_numpy(file_path: str) -> np.ndarray:
        """
        加载numpy数组
        
        参数：
            file_path (str): 文件路径
            
        返回：
            np.ndarray: 数组
        """
        pass
    
    @staticmethod
    def save_numpy(data: np.ndarray, file_path: str):
        """
        保存numpy数组
        
        参数：
            data (np.ndarray): 数组
            file_path (str): 保存路径
        """
        pass
    
    @staticmethod
    def load_pickle(file_path: str) -> Any:
        """
        加载pickle文件
        
        参数：
            file_path (str): 文件路径
            
        返回：
            Any: 对象
        """
        pass
    
    @staticmethod
    def save_pickle(obj: Any, file_path: str):
        """
        保存pickle文件
        
        参数：
            obj (Any): 对象
            file_path (str): 保存路径
        """
        pass
    
    @staticmethod
    def load_config(config_file: str) -> Dict:
        """
        加载配置文件（支持yaml和json）
        
        参数：
            config_file (str): 配置文件路径
            
        返回：
            Dict: 配置字典
        """
        pass
    
    @staticmethod
    def save_config(config: Dict, save_path: str):
        """
        保存配置文件
        
        参数：
            config (Dict): 配置字典
            save_path (str): 保存路径
        """
        pass
    
    @staticmethod
    def create_directory(dir_path: str):
        """
        创建目录（如果不存在）
        
        参数：
            dir_path (str): 目录路径
        """
        pass
    
    @staticmethod
    def get_file_list(directory: str, extension: str = '*') -> list:
        """
        获取目录下的文件列表
        
        参数：
            directory (str): 目录路径
            extension (str): 文件扩展名（如'.npy'）
            
        返回：
            list: 文件路径列表
        """
        pass

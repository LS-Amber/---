# coding=utf-8
"""
城市信息流通效率分析工具包

该工具包用于分析城市间信息流通效率，基于百度搜索指数数据计算加权特征路径长度，
评估区域间信息流通效率的比较和动态变化。
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# 从各个模块导入主要类
from .data_processor import DataProcessor
from .network_analyzer import NetworkAnalyzer
from .region_manager import RegionManager
from .temporal_analyzer import TemporalAnalyzer
from .visualizer import Visualizer

# 定义公开接口
__all__ = [
    'DataProcessor',
    'NetworkAnalyzer',
    'RegionManager',
    'TemporalAnalyzer',
    'Visualizer'
]

# 模块级别的便捷函数
def get_version():
    """获取包版本号"""
    return __version__

def get_author():
    """获取作者信息"""
    return __author__

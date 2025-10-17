import numpy as np
import pandas as pd


class HierarchyManager:
    """城市层级管理类，负责城市层级定义和分类"""

    def __init__(self):
        # 定义城市层级（示例，应根据实际情况调整）
        self.hierarchy_definitions = {
            '直辖市': ['北京市', '天津市', '上海市', '重庆市'],
            '副省级城市': [
                '沈阳市', '大连市', '长春市', '哈尔滨市', '南京市', '杭州市', '宁波市',
                '厦门市', '济南市', '青岛市', '武汉市', '广州市', '深圳市', '成都市', '西安市'  # 修正了"成都四"
            ],
            '省会城市': [
                '石家庄市', '太原市', '呼和浩特市', '合肥市', '福州市', '南昌市', '郑州市',
                '长沙市', '南宁市', '海口市', '贵阳市', '昆明市', '拉萨市', '兰州市',
                '西宁市', '银川市', '乌鲁木齐市'
            ],
            '普通地级市': []  # 其他城市将自动归类为此类
        }

        self.city_hierarchy_map = {}
        self.hierarchy_names = list(self.hierarchy_definitions.keys())

    def set_hierarchy_definitions(self, definitions):
        """
        设置城市层级定义

        参数:
        definitions: 层级定义字典 {层级名: [城市列表]}
        """
        self.hierarchy_definitions = definitions
        self.hierarchy_names = list(definitions.keys())
        self.city_hierarchy_map = {}  # 重置映射

    def create_city_hierarchy_map(self, all_cities):
        """
        创建城市到层级的映射

        参数:
        all_cities: 所有城市的列表

        返回:
        城市到层级的映射字典
        """
        self.city_hierarchy_map = {}

        # 首先处理明确指定的城市
        for hierarchy, cities in self.hierarchy_definitions.items():
            for city in cities:
                if city in all_cities:
                    self.city_hierarchy_map[city] = hierarchy

        # 剩余城市归类为普通地级市
        for city in all_cities:
            if city not in self.city_hierarchy_map:
                self.city_hierarchy_map[city] = '普通地级市'

        return self.city_hierarchy_map

    def get_hierarchy_stats(self, values, cities):
        """
        获取各层级的统计信息

        参数:
        values: 值列表（与cities顺序一致）
        cities: 城市列表

        返回:
        各层级统计信息字典
        """
        if not self.city_hierarchy_map:
            raise ValueError("请先创建城市层级映射")

        stats = {}

        for hierarchy in self.hierarchy_names:
            hierarchy_cities = [
                city for city in cities
                if self.city_hierarchy_map.get(city) == hierarchy
            ]
            hierarchy_indices = [i for i, city in enumerate(cities) if city in hierarchy_cities]

            if hierarchy_indices:
                hierarchy_values = [values[i] for i in hierarchy_indices]
                stats[hierarchy] = {
                    'count': len(hierarchy_values),
                    'mean': np.mean(hierarchy_values),
                    'median': np.median(hierarchy_values),
                    'std': np.std(hierarchy_values),
                    'min': np.min(hierarchy_values),
                    'max': np.max(hierarchy_values)
                }
            else:
                stats[hierarchy] = {
                    'count': 0,
                    'mean': 0,
                    'median': 0,
                    'std': 0,
                    'min': 0,
                    'max': 0
                }

        return stats

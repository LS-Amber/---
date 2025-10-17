# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


class RegionManager:
    """区域管理类，负责区域定义和区域间分析"""

    def __init__(self, region_definitions):
        """
        初始化区域管理器

        参数:
        region_definitions: 区域定义字典 {区域名: [城市列表]}
        """
        self.regions = region_definitions
        self.region_names = list(region_definitions.keys())
        self.n_regions = len(self.region_names)

        # 创建城市到区域的映射
        self.city_region_map = {}
        for region, city_list in region_definitions.items():
            for city in city_list:
                self.city_region_map[city] = region

    def compute_region_path_length(self, distance_matrix, cities):
        """
        计算区域间平均路径长度

        参数:
        distance_matrix: 城市间距离矩阵
        cities: 城市列表（与距离矩阵的行列顺序一致）

        返回:
        区域间平均路径长度矩阵
        """
        region_matrix = np.zeros((self.n_regions, self.n_regions))
        region_counts = np.zeros((self.n_regions, self.n_regions))

        # 计算区域间路径平均值
        for i, source in enumerate(cities):
            if source not in self.city_region_map:
                continue
            source_region = self.city_region_map[source]
            source_region_idx = self.region_names.index(source_region)

            for j, target in enumerate(cities):
                if target not in self.city_region_map:
                    continue
                target_region = self.city_region_map[target]
                target_region_idx = self.region_names.index(target_region)

                # 累加路径长度和计数
                if distance_matrix.iloc[i, j] < 1e6:  # 忽略不可达路径
                    region_matrix[source_region_idx, target_region_idx] += distance_matrix.iloc[i, j]
                    region_counts[source_region_idx, target_region_idx] += 1

        # 计算平均值（避免除零）
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_region_matrix = np.divide(
                region_matrix,
                region_counts,
                out=np.full_like(region_matrix, fill_value=1e6),
                where=region_counts != 0
            )

        return pd.DataFrame(
            avg_region_matrix,
            index=self.region_names,
            columns=self.region_names
        )

    def get_region_stats(self, distance_matrix, cities):
        """
        获取区域统计信息

        参数:
        distance_matrix: 城市间距离矩阵
        cities: 城市列表

        返回:
        区域统计信息字典
        """
        stats = {}

        for region in self.region_names:
            region_cities = self.regions[region]
            region_indices = [i for i, city in enumerate(cities) if city in region_cities]

            if not region_indices:  # 如果区域内没有城市
                stats[region] = {
                    'intra_region_mean': 1e6,
                    'intra_region_std': 0,
                    'inter_region_mean': 1e6,
                    'inter_region_std': 0,
                }
                continue

            # 提取区域内和区域间的距离
            intra_region_dists = []
            inter_region_dists = []

            for i in region_indices:
                for j in range(len(cities)):
                    if distance_matrix.iloc[i, j] < 1e6:  # 忽略不可达路径
                        if cities[j] in region_cities:
                            intra_region_dists.append(distance_matrix.iloc[i, j])
                        else:
                            inter_region_dists.append(distance_matrix.iloc[i, j])

            stats[region] = {
                'intra_region_mean': np.mean(intra_region_dists) if intra_region_dists else 1e6,
                'intra_region_std': np.std(intra_region_dists) if intra_region_dists else 0,
                'inter_region_mean': np.mean(inter_region_dists) if inter_region_dists else 1e6,
                'inter_region_std': np.std(inter_region_dists) if inter_region_dists else 0,
            }

        return stats
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import networkx as nx


class NetworkAnalyzer:
    """网络分析类，负责构建网络和计算最短路径"""

    def __init__(self, weight_matrix, cities, region_manager=None):
        """
        初始化网络分析器

        参数:
        weight_matrix: 权重矩阵（城市×城市）
        cities: 城市列表
        region_manager: 区域管理器（可选）
        """
        self.weight_matrix = weight_matrix
        self.cities = cities
        self.n_cities = len(cities)
        self.city_to_idx = {city: idx for idx, city in enumerate(cities)}
        self.graph = None
        self.distance_matrix = None
        self.region_manager = region_manager

    def build_graph(self):
        """构建有向加权图"""
        self.graph = nx.DiGraph()

        # 添加节点
        self.graph.add_nodes_from(self.cities)

        # 添加加权边
        for i, source in enumerate(self.cities):
            for j, target in enumerate(self.cities):
                if i != j and self.weight_matrix.iloc[i, j] < 1e6:  # 跳过自身和无穷大权重
                    self.graph.add_edge(
                        source,
                        target,
                        weight=self.weight_matrix.iloc[i, j]
                    )

        return self.graph

    def compute_shortest_paths(self, method='dijkstra'):
        """
        计算所有城市对之间的最短路径长度

        参数:
        method: 计算方法 ('dijkstra' 或 'floyd_warshall')

        返回:
        距离矩阵DataFrame
        """
        if self.graph is None:
            self.build_graph()

        # 初始化距离矩阵（使用大值表示不可达）
        self.distance_matrix = np.full(
            (self.n_cities, self.n_cities),
            fill_value=1e6,  # 表示不可达的大数值
            dtype=np.float32
        )

        if method == 'dijkstra':
            # 使用Dijkstra算法计算最短路径
            for i, source in enumerate(self.cities):
                try:
                    # 计算从源城市到所有其他城市的最短路径长度
                    lengths = nx.single_source_dijkstra_path_length(
                        self.graph, source, weight='weight'
                    )

                    for j, target in enumerate(self.cities):
                        if target in lengths:
                            self.distance_matrix[i, j] = lengths[target]
                except nx.NetworkXNoPath:
                    # 如果不存在路径，保持初始大值
                    pass

        elif method == 'floyd_warshall':
            # 使用Floyd-Warshall算法（适用于稠密图）
            dist_dict = nx.floyd_warshall_numpy(self.graph, weight='weight')
            self.distance_matrix = np.where(
                np.isinf(dist_dict), 1e6, dist_dict
            )

        return pd.DataFrame(
            self.distance_matrix,
            index=self.cities,
            columns=self.cities
        )

    def get_network_properties(self):
        """
        获取网络属性

        返回:
        网络属性字典
        """
        if self.graph is None:
            self.build_graph()

        return {
            'number_of_nodes': self.graph.number_of_nodes(),
            'number_of_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_strongly_connected': nx.is_strongly_connected(self.graph),
            'is_weakly_connected': nx.is_weakly_connected(self.graph),
        }

    def compute_top_shortest_paths_by_region(self, top_n=5):
        """
        计算各区域间城市节点最短路径的前N名

        参数:
        top_n: 返回前N名最短路径

        返回:
        区域间最短路径的前N名字典
        """
        if self.distance_matrix is None or self.region_manager is None:
            return {}

        # 创建城市到区域的映射
        city_to_region = self.region_manager.city_region_map

        # 初始化结果字典
        region_paths = {}

        # 遍历所有城市对
        path_list = []
        for i, source in enumerate(self.cities):
            source_region = city_to_region.get(source, 'Unknown')
            for j, target in enumerate(self.cities):
                if i != j and target in city_to_region:  # 排除自身和未知区域城市
                    target_region = city_to_region.get(target, 'Unknown')
                    distance = self.distance_matrix[i, j]

                    # 只考虑可达路径
                    if distance < 1e6:
                        path_list.append({
                            'source': source,
                            'source_region': source_region,
                            'target': target,
                            'target_region': target_region,
                            'distance': distance
                        })

        # 转换为DataFrame便于处理
        path_df = pd.DataFrame(path_list)

        # 按区域对分组并获取前N名
        if not path_df.empty:
            grouped = path_df.groupby(['source_region', 'target_region'])
            for (src_region, tgt_region), group in grouped:
                # 按距离排序并取前N名
                top_paths = group.nsmallest(top_n, 'distance')

                if (src_region, tgt_region) not in region_paths:
                    region_paths[(src_region, tgt_region)] = []

                for _, row in top_paths.iterrows():
                    region_paths[(src_region, tgt_region)].append({
                        'source': row['source'],
                        'target': row['target'],
                        'distance': row['distance']
                    })

        return region_paths

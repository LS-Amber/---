import numpy as np
import pandas as pd


class ClusteringCalculator:
    """聚类系数计算类，负责计算加权聚类系数"""

    def __init__(self):
        self.clustering_results = {}

    def calculate_node_strength(self, weight_matrix):
        """
        计算节点强度（总信息辐射能力）

        参数:
        weight_matrix: 权重矩阵（城市×城市）

        返回:
        节点强度数组
        """
        return np.sum(weight_matrix.values, axis=1)

    def calculate_weighted_clustering(self, weight_matrix):
        """
        按照标准公式计算加权聚类系数
        公式: C_i = (1 / (s_i * (k_i - 1))) * Σ(j,k) (w_ij * w_jk * w_ki)^(1/3)

        参数:
        weight_matrix: 权重矩阵（城市×城市）

        返回:
        加权聚类系数数组
        """
        W = weight_matrix.values
        n = W.shape[0]
        clustering_coeffs = np.zeros(n)
        strengths = self.calculate_node_strength(weight_matrix)

        # 计算每个节点的度数（非零连接数）
        degrees = np.count_nonzero(W, axis=1)

        for i in range(n):
            # 如果节点强度为0或度数小于2，则聚类系数为0
            if strengths[i] == 0 or degrees[i] < 2:
                clustering_coeffs[i] = 0
                continue

            triple_sum = 0
            # 遍历所有可能的节点j和k
            for j in range(n):
                # 节点j必须是节点i的邻居且不等于i
                if j == i or W[i, j] == 0:
                    continue

                for k in range(n):
                    # 节点k必须是节点i的邻居且不等于i和j
                    # 同时j和k之间必须有连接
                    if k == i or k == j or W[i, k] == 0 or W[j, k] == 0:
                        continue

                    # 计算三元组权重的几何平均
                    geometric_mean = (W[i, j] * W[j, k] * W[k, i]) ** (1 / 3)
                    triple_sum += geometric_mean

            # 按照标准公式计算聚类系数
            # C_i = (1 / (s_i * (k_i - 1))) * Σ(j,k) (w_ij * w_jk * w_ki)^(1/3)
            denominator = strengths[i] * (degrees[i] - 1)
            if denominator > 0:
                clustering_coeffs[i] = triple_sum / denominator
            else:
                clustering_coeffs[i] = 0

        return clustering_coeffs

    def calculate_for_all_years(self, transformed_data):
        """
        计算所有年份的聚类系数

        参数:
        transformed_data: 转换后的权重数据字典

        返回:
        聚类系数结果字典 {year: {cities: [], coefficients: []}}
        """
        self.clustering_results = {}

        for year, weight_matrix in transformed_data.items():
            print(f"计算 {year} 年聚类系数...")

            cities = weight_matrix.index.tolist()
            coefficients = self.calculate_weighted_clustering(weight_matrix)

            self.clustering_results[year] = {
                'cities': cities,
                'coefficients': coefficients
            }

        return self.clustering_results

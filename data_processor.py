# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import stats


class DataProcessor:
    """数据处理类，负责数据加载、清洗和权重转换"""

    def __init__(self):
        self.original_data = None
        self.transformed_data = None

    def transform_weights(self, panel_data, alpha=1, c=3, epsilon=None):
        """
        将搜索指数转换为信息流动阻力

        参数:
        panel_data: 面板数据字典 {year: city×city DataFrame}
        alpha: 缩放参数
        c: 常数
        epsilon: 避免除零错的小常数

        返回:
        转换后的权重数据字典
        """
        if panel_data is None or len(panel_data) == 0:
            raise ValueError("请先加载数据")

        self.original_data = panel_data
        self.transformed_data = {}

        for year, matrix in panel_data.items():
            # 计算epsilon（如果未提供）
            if epsilon is None:
                non_zero_values = matrix.values[matrix.values > 0]
                epsilon = np.min(non_zero_values) / 2 if len(non_zero_values) > 0 else 0.5

            # 应用转换公式
            with np.errstate(divide='ignore', invalid='ignore'):
                transformed_values = alpha / (np.log(matrix.values + epsilon) + c)

                # 处理零搜索量，设置为较大值
                zero_mask = matrix.values == 0
                if np.any(~zero_mask):  # 如果有非零值
                    max_val = np.max(transformed_values[~zero_mask])
                    transformed_values[zero_mask] = 10 * max_val
                else:
                    transformed_values[zero_mask] = 1000

            self.transformed_data[year] = pd.DataFrame(
                transformed_values,
                index=matrix.index,
                columns=matrix.columns
            )

        return self.transformed_data

    def get_descriptive_stats(self, data_type='original'):
        """
        获取描述性统计信息

        参数:
        data_type: 'original' 或 'transformed'

        返回:
        描述性统计信息字典
        """
        if data_type == 'original' and self.original_data is None:
            raise ValueError("原始数据未加载")
        elif data_type == 'transformed' and self.transformed_data is None:
            raise ValueError("转换后的数据未生成")

        stats_dict = {}
        data_source = self.original_data if data_type == 'original' else self.transformed_data

        for year, matrix in data_source.items():
            flat_values = matrix.values.flatten()
            flat_values = flat_values[flat_values < 1e6]  # 排除不可达的大值

            stats_dict[year] = {
                'mean': np.mean(flat_values),
                'median': np.median(flat_values),
                'std': np.std(flat_values),
                'min': np.min(flat_values),
                'max': np.max(flat_values),
                'skewness': stats.skew(flat_values),
                'kurtosis': stats.kurtosis(flat_values)
            }

        return stats_dict


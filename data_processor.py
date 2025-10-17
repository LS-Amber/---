import numpy as np
import pandas as pd
from scipy import stats


class DataProcessor:
    """数据处理类，负责数据加载、清洗和权重转换"""

    def __init__(self):
        self.original_data = None
        self.transformed_data = None

    def load_data(self, data_dir):
        """
        加载CSV数据文件

        参数:
        data_dir: 数据文件目录路径

        返回:
        面板数据字典 {year: city×city DataFrame}
        """
        import glob
        import os

        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

        if not csv_files:
            raise ValueError(f"在目录 {data_dir} 中未找到CSV文件")

        panel_data = {}
        for file_path in csv_files:
            try:
                # 从文件名提取年份
                file_name = os.path.basename(file_path)
                year_str = ''.join(filter(str.isdigit, file_name))
                if not year_str or len(year_str) != 4:
                    print(f"无法从文件名 {file_name} 中提取年份，跳过此文件")
                    continue

                year = int(year_str)

                # 读取CSV文件
                df = pd.read_csv(file_path)

                # 检查必要的列是否存在
                required_columns = ['source_city', 'target_city', 'search_index']
                if not all(col in df.columns for col in required_columns):
                    print(f"文件 {file_name} 缺少必要的列，需要的列: {required_columns}")
                    continue

                # 转换为城市×城市矩阵
                matrix = df.pivot(index='source_city', columns='target_city', values='search_index')
                matrix = matrix.fillna(0)  # 将NaN替换为0

                # 确保所有城市都包含在行和列中
                all_cities = sorted(set(matrix.index) | set(matrix.columns))
                matrix = matrix.reindex(index=all_cities, columns=all_cities, fill_value=0)

                panel_data[year] = matrix
                print(f"成功加载 {year} 年数据，包含 {len(all_cities)} 个城市")

            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {e}")

        if not panel_data:
            raise ValueError("未能加载任何数据")

        self.original_data = panel_data
        return panel_data

    def transform_weights(self, panel_data, alpha=1, beta=1):
        """
        将搜索指数转换为信息强度权重

        参数:
        panel_data: 面板数据字典 {year: city×city DataFrame}
        alpha: 比例系数
        beta: 常数

        返回:
        转换后的权重数据字典
        """
        if panel_data is None or len(panel_data) == 0:
            raise ValueError("请先加载数据")

        self.transformed_data = {}
        self.original_data = panel_data  # 确保原始数据也被设置

        for year, matrix in panel_data.items():
            # 应用转换公式 S_ij = ln(α * SI_ij + β)
            with np.errstate(divide='ignore', invalid='ignore'):
                transformed_matrix = np.log(alpha * matrix + beta)

                # 处理可能出现的无效值
                transformed_matrix = np.where(
                    np.isfinite(transformed_matrix),
                    transformed_matrix,
                    0
                )

            self.transformed_data[year] = pd.DataFrame(
                transformed_matrix,
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
        if data_type == 'original' and (self.original_data is None or len(self.original_data) == 0):
            raise ValueError("原始数据未加载")
        elif data_type == 'transformed' and (self.transformed_data is None or len(self.transformed_data) == 0):
            raise ValueError("转换后的数据未生成")

        stats_dict = {}
        data_source = self.original_data if data_type == 'original' else self.transformed_data

        for year, matrix in data_source.items():
            flat_values = matrix.values.flatten()
            # 排除零值（无连接）
            flat_values = flat_values[flat_values > 0]

            if len(flat_values) > 0:
                stats_dict[year] = {
                    'mean': np.mean(flat_values),
                    'median': np.median(flat_values),
                    'std': np.std(flat_values),
                    'min': np.min(flat_values),
                    'max': np.max(flat_values),
                    'skewness': stats.skew(flat_values) if len(flat_values) > 2 else 0,
                    'kurtosis': stats.kurtosis(flat_values) if len(flat_values) > 3 else 0
                }
            else:
                stats_dict[year] = {
                    'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0,
                    'skewness': 0, 'kurtosis': 0
                }

        return stats_dict

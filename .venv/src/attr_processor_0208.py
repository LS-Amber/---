# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import re
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class RobustCityAttributeProcessor(object):
    """
    鲁棒的城市属性处理器
    专门处理带缺失值的复杂表格数据
    """

    def __init__(self, gamma=1.0, num_years=10):
        """
        初始化

        Parameters:
        -----------
        gamma : float
            指数函数的尺度参数，默认1.0
        num_years : int
            要处理的年份数量，默认10年
        """
        self.gamma = gamma
        self.years = range(2014, 2014 + num_years)  # 动态生成年份范围
        self.num_cities = 296
        self.num_attributes = 24

        # 修正后的属性类型定义
        self.attribute_types = [
            'continuous',  # 1. 常住人口
            'continuous',  # 2. 人口密度
            'continuous',  # 3. 城镇化率
            'continuous',  # 4. GDP总量
            'continuous',  # 5. 人均GDP
            'continuous',  # 6. 第三产业结构比例
            'continuous',  # 7. 规模以上工业企业数
            'continuous',  # 8. 职工平均工资
            'continuous',  # 9. 就业率
            'continuous',  # 10. 公路+铁路里程数
            'continuous',  # 11. 公共交通客运总量
            'continuous',  # 12. 固定资产投资额
            'continuous',  # 13. 医院数量
            'continuous',  # 14. 高等学校数量（修正：改为连续型）
            'continuous',  # 15. 城市绿化率
            'continuous',  # 16. 剧场和影院数量（修正：改为连续型）
            'continuous',  # 17. 每百人公共图书馆藏书
            'continuous',  # 18. 电信业务收入
            'continuous',  # 19. 外资使用情况
            'continuous',  # 20. 专利授权数
            'continuous',  # 21. 金融机构存贷款总额
            'continuous',  # 22. 二氧化碳排放总量
            'continuous',  # 23. PM2.5年浓度
            'continuous',  # 24. 全社会耗电总量
        ]

        # 缺失值标识符
        self.missing_indicators = ['', '-', 'NaN', 'nan', 'N/A', 'n/a', 'null', 'NULL', ' ', '  ']

    def clean_numeric_value(self, value):
        """
        清洗数值数据，处理各种格式问题

        Parameters:
        -----------
        value : str or numeric
            原始数据值

        Returns:
        --------
        clean_value : float or np.nan
            清洗后的数值
        """
        if pd.isna(value):
            return np.nan

        # 转换为字符串处理
        if isinstance(value, (int, float)):
            # 已经是数值类型
            if np.isinf(value):
                return np.nan
            return float(value)

        str_value = str(value).strip()

        # 检查是否为缺失值标识符
        if str_value in self.missing_indicators:
            return np.nan

        # 处理空字符串
        if not str_value:
            return np.nan

        # 处理科学计数法
        if 'e' in str_value.lower():
            try:
                return float(str_value)
            except:
                return np.nan

        # 处理千分位逗号
        if ',' in str_value:
            str_value = str_value.replace(',', '')

        # 处理百分号
        if '%' in str_value:
            str_value = str_value.replace('%', '')
            try:
                return float(str_value) / 100.0
            except:
                return np.nan

        # 处理括号内容（如数值后的说明）
        if '(' in str_value:
            str_value = str_value.split('(')[0].strip()

        # 处理中文数字（如"一万"）
        chinese_digits = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
                          '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
        if any(c in str_value for c in chinese_digits.keys()):
            # 简单处理，可以扩展更复杂的转换
            return np.nan

        # 转换为数值
        try:
            return float(str_value)
        except:
            # 尝试提取数值部分
            numeric_match = re.search(r'[-+]?\d*\.?\d+', str_value)
            if numeric_match:
                try:
                    return float(numeric_match.group())
                except:
                    return np.nan
            else:
                return np.nan

    def load_yearly_data(self, data_dir):
        """
        加载每年的属性数据（仅支持CSV格式）

        Parameters:
        -----------
        data_dir : str
            数据目录路径，包含2014.csv, 2015.csv, ..., 2023.csv

        Returns:
        --------
        yearly_data : dict
            键为年份，值为属性矩阵（24×296）
        """
        yearly_data = {}

        for year in self.years:
            file_path = os.path.join(data_dir, '%d.csv' % year)

            if not os.path.exists(file_path):
                print('警告：%d年数据文件不存在于目录 %s' % (year, data_dir))
                # 创建空矩阵
                yearly_data[year] = np.full((self.num_attributes, self.num_cities), np.nan)
                continue

            print('加载%d年数据: %s' % (year, file_path))

            try:
                # 尝试不同的编码方式读取CSV文件
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', header=None)
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(file_path, encoding='gbk', header=None)
                    except:
                        df = pd.read_csv(file_path, encoding='latin1', header=None)

                # 打印数据形状以便调试
                print('  原始数据形状: %s' % str(df.shape))

                # 查找数据起始行（跳过可能的标题行）
                data_start_row = 0
                for i in range(min(10, df.shape[0])):  # 检查前10行
                    row_vals = df.iloc[i].astype(str).fillna('').tolist()
                    # 检查是否包含"属性id"或类似标识
                    if any('属性id' in str(val) or '属性ID' in str(val) for val in row_vals):
                        data_start_row = i
                        break

                # 提取数值数据（从第3列开始）
                data_matrix = []

                # 遍历每一行（属性）
                for row_idx in range(data_start_row, min(data_start_row + self.num_attributes + 5, df.shape[0])):
                    row = df.iloc[row_idx]

                    # 检查是否还有数据行
                    if len(row) < 3:
                        continue

                    # 尝试提取属性ID
                    try:
                        attr_id = int(self.clean_numeric_value(row[1]))
                        if 1 <= attr_id <= self.num_attributes:
                            # 提取城市数据
                            city_values = []
                            for col_idx in range(2, min(2 + self.num_cities, len(row))):
                                value = self.clean_numeric_value(row[col_idx])
                                city_values.append(value)

                            # 如果城市数量不足，用NaN填充
                            while len(city_values) < self.num_cities:
                                city_values.append(np.nan)

                            data_matrix.append(city_values[:self.num_cities])
                    except:
                        # 如果提取失败，跳过该行
                        continue

                # 转换为numpy数组
                if data_matrix:
                    year_matrix = np.array(data_matrix)

                    # 确保是24×296
                    if year_matrix.shape[0] < self.num_attributes:
                        # 用NaN填充缺失的行
                        padding = np.full((self.num_attributes - year_matrix.shape[0], self.num_cities), np.nan)
                        year_matrix = np.vstack([year_matrix, padding])
                    elif year_matrix.shape[0] > self.num_attributes:
                        # 截断多余的行
                        year_matrix = year_matrix[:self.num_attributes, :]

                    if year_matrix.shape[1] < self.num_cities:
                        # 用NaN填充缺失的列
                        padding = np.full((self.num_attributes, self.num_cities - year_matrix.shape[1]), np.nan)
                        year_matrix = np.hstack([year_matrix, padding])
                    elif year_matrix.shape[1] > self.num_cities:
                        # 截断多余的列
                        year_matrix = year_matrix[:, :self.num_cities]

                    yearly_data[year] = year_matrix
                    print('  处理后的数据形状: %s' % str(year_matrix.shape))
                    print('  缺失值比例: %.2f%%' % (np.isnan(year_matrix).mean() * 100))
                else:
                    print('  警告：%d年未提取到有效数据' % year)
                    yearly_data[year] = np.full((self.num_attributes, self.num_cities), np.nan)

            except Exception as e:
                print('  加载%d年数据时出错: %s' % (year, str(e)))
                yearly_data[year] = np.full((self.num_attributes, self.num_cities), np.nan)

        # 检查加载结果
        valid_years = [year for year in self.years if not np.all(np.isnan(yearly_data.get(year, np.nan)))]
        print('\n成功加载 %d 年的数据: %s' % (len(valid_years), valid_years))

        return yearly_data

    def handle_missing_values(self, yearly_data):
        """
        处理缺失值（简化策略：只使用时间维度和全局填补）

        Parameters:
        -----------
        yearly_data : dict
            原始数据字典

        Returns:
        --------
        filled_data : dict
            填补后的数据字典
        """
        filled_data = {}
        for year in yearly_data:
            filled_data[year] = yearly_data[year].copy()

        # 1. 计算每个属性的全局平均值
        print('\n计算全局平均值...')
        global_means = np.zeros(self.num_attributes)

        for attr_idx in range(self.num_attributes):
            all_values = []
            for year in self.years:
                if year in yearly_data:
                    attr_values = yearly_data[year][attr_idx, :]
                    valid_values = attr_values[~np.isnan(attr_values)]
                    all_values.extend(valid_values.tolist())

            if all_values:
                global_means[attr_idx] = np.nanmean(all_values)
            else:
                global_means[attr_idx] = 0

            # 打印进度
            if (attr_idx + 1) % 5 == 0:
                print('  已处理 %d/24 个属性' % (attr_idx + 1))

        print('全局平均值计算完成')

        # 2. 对每个属性进行处理（按城市）
        print('\n进行缺失值填补...')
        for attr_idx in range(self.num_attributes):
            # 收集该属性所有年份的数据（年份×城市）
            attr_all_years = []
            for year in self.years:
                if year in yearly_data:
                    attr_all_years.append(yearly_data[year][attr_idx, :])
                else:
                    attr_all_years.append(np.full(self.num_cities, np.nan))

            # 转换为数组（年份×城市）
            attr_matrix = np.array(attr_all_years)  # 10×296

            # 对每个城市进行时间维度填补
            for city_idx in range(self.num_cities):
                city_series = attr_matrix[:, city_idx]  # 长度10的时间序列

                # 查找缺失值位置
                nan_indices = np.where(np.isnan(city_series))[0]

                for year_idx in nan_indices:
                    # 尝试时间维度填补
                    filled_value = self._temporal_fill(city_series, year_idx)

                    if filled_value is not None and not np.isnan(filled_value):
                        attr_matrix[year_idx, city_idx] = filled_value
                    else:
                        # 使用全局填补
                        attr_matrix[year_idx, city_idx] = global_means[attr_idx]

            # 更新到filled_data
            for i, year in enumerate(self.years):
                if year in filled_data:
                    filled_data[year][attr_idx, :] = attr_matrix[i, :]

            # 打印进度
            if (attr_idx + 1) % 5 == 0:
                print('  已填补 %d/24 个属性' % (attr_idx + 1))

        # 3. 检查填补效果
        print('\n填补效果检查:')
        for year in self.years[:3]:  # 只检查前3年
            if year in filled_data:
                original_nan = np.isnan(yearly_data.get(year, np.full((24, 296), np.nan))).mean()
                filled_nan = np.isnan(filled_data[year]).mean()
                print('  %d年: 缺失值比例从 %.2f%% 降至 %.2f%%' % (year, original_nan * 100, filled_nan * 100))

        return filled_data

    def _temporal_fill(self, time_series, idx):
        """
        时间维度填补（前后年份平均）

        Parameters:
        -----------
        time_series : np.ndarray
            时间序列数据
        idx : int
            当前缺失值的位置索引

        Returns:
        --------
        filled_value : float or None
            填补的值，如果无法填补返回None
        """
        n = len(time_series)

        # 向前查找最近的非缺失值
        prev_value = None
        for i in range(idx - 1, -1, -1):
            if i >= 0 and not np.isnan(time_series[i]):
                prev_value = time_series[i]
                break

        # 向后查找最近的非缺失值
        next_value = None
        for i in range(idx + 1, n):
            if i < n and not np.isnan(time_series[i]):
                next_value = time_series[i]
                break

        # 填补逻辑
        if prev_value is not None and next_value is not None:
            # 前后都有值，取平均
            return (prev_value + next_value) / 2.0
        elif prev_value is not None:
            # 只有前一年有值
            return prev_value
        elif next_value is not None:
            # 只有后一年有值
            return next_value
        else:
            # 前后都没有值
            return None

    def compute_yearly_similarity(self, filled_data):
        """
        计算每年的属性相似度矩阵（使用指数函数）

        Parameters:
        -----------
        filled_data : dict
            填补后的数据字典

        Returns:
        --------
        similarity_matrices : dict
            每年的相似度矩阵（296×296）
        """
        similarity_matrices = {}

        for year in self.years:
            if year not in filled_data:
                print('警告：%d年数据缺失，跳过' % year)
                continue

            print('计算%d年相似度矩阵...' % year)
            year_data = filled_data[year]  # 24×296

            # 初始化相似度矩阵
            similarity_matrix = np.zeros((self.num_cities, self.num_cities))

            # 计算每个属性的贡献
            for attr_idx in range(self.num_attributes):
                attr_values = year_data[attr_idx, :]  # 长度296

                # 计算标准差（用于标准化）
                valid_values = attr_values[~np.isnan(attr_values)]

                if len(valid_values) > 1 and np.std(valid_values) > 0:
                    # 标准化处理
                    mean_val = np.mean(valid_values)
                    std_val = np.std(valid_values)

                    # 避免除零
                    if std_val > 0:
                        normalized = (attr_values - mean_val) / std_val
                    else:
                        normalized = np.zeros_like(attr_values)

                    # 计算绝对差异矩阵
                    diff_matrix = np.abs(
                        normalized[:, np.newaxis] - normalized[np.newaxis, :]
                    )

                    # 应用指数函数计算相似度
                    attr_similarity = np.exp(-self.gamma * diff_matrix)

                    # 累加到总相似度
                    similarity_matrix += attr_similarity
                else:
                    # 如果所有值都相同或标准差为0，该属性贡献为全1矩阵
                    similarity_matrix += 1.0

            # 平均化
            similarity_matrix /= self.num_attributes

            # 确保对角线为1
            np.fill_diagonal(similarity_matrix, 1.0)

            # 对称化
            similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2.0

            # 确保值在[0,1]范围内
            similarity_matrix = np.clip(similarity_matrix, 0.0, 1.0)

            similarity_matrices[year] = similarity_matrix

            # 打印统计信息
            triu_values = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
            print('  %d年完成: 均值=%.4f, 标准差=%.4f' % (year, np.mean(triu_values), np.std(triu_values)))

        return similarity_matrices

    def save_results(self, similarity_matrices, output_dir):
        """
        保存结果

        Parameters:
        -----------
        similarity_matrices : dict
            相似度矩阵字典
        output_dir : str
            输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 保存每年的相似度矩阵为CSV
        print('\n保存相似度矩阵...')
        for year, matrix in similarity_matrices.items():
            # 创建DataFrame
            # 注意：这里假设有296个城市，实际可能需要城市名称
            df = pd.DataFrame(
                matrix,
                index=['City_%03d' % i for i in range(self.num_cities)],
                columns=['City_%03d' % i for i in range(self.num_cities)]
            )

            # 保存为CSV
            filename = os.path.join(output_dir, 'attribute_similarity_%d.csv' % year)
            df.to_csv(filename, encoding='utf-8')
            print('  已保存: %s' % filename)

        # 保存为numpy格式
        npz_file = os.path.join(output_dir, 'all_years_similarity.npz')
        np.savez(npz_file, **{'year_%d' % year: matrix for year, matrix in similarity_matrices.items()})
        print('已保存为numpy格式: %s' % npz_file)

        # 生成统计摘要
        self._generate_statistics(similarity_matrices, output_dir)

    def _generate_statistics(self, similarity_matrices, output_dir):
        """
        生成统计摘要

        Parameters:
        -----------
        similarity_matrices : dict
            相似度矩阵字典
        output_dir : str
            输出目录
        """
        stats_data = []

        for year, matrix in similarity_matrices.items():
            # 提取上三角部分（不包括对角线）
            triu_indices = np.triu_indices_from(matrix, k=1)
            similarities = matrix[triu_indices]

            stats = {
                'year': year,
                'mean': np.mean(similarities),
                'std': np.std(similarities),
                'min': np.min(similarities),
                'max': np.max(similarities),
                'median': np.median(similarities),
                'q25': np.percentile(similarities, 25),
                'q75': np.percentile(similarities, 75),
                'num_cities': self.num_cities,
            }
            stats_data.append(stats)

        # 创建DataFrame并保存
        stats_df = pd.DataFrame(stats_data)
        stats_file = os.path.join(output_dir, 'similarity_statistics.csv')
        stats_df.to_csv(stats_file, index=False, encoding='utf-8')

        print('\n统计摘要已保存: %s' % stats_file)

        # 打印简要统计
        print('\n' + '=' * 70)
        print('各年份属性相似度统计摘要')
        print('=' * 70)
        print(stats_df.to_string(index=False))

        # 生成趋势分析
        self._generate_trend_analysis(stats_df, similarity_matrices, output_dir)

    def _generate_trend_analysis(self, stats_df, similarity_matrices, output_dir):
        """
        生成趋势分析报告

        Parameters:
        -----------
        stats_df : pd.DataFrame
            统计摘要DataFrame
        similarity_matrices : dict
            相似度矩阵字典
        output_dir : str
            输出目录
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            # 创建趋势图
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            # 1. 均值和中位数趋势
            ax1 = axes[0, 0]
            years = stats_df['year'].astype(str)
            ax1.plot(years, stats_df['mean'], marker='o', linewidth=2, label='均值', color='blue')
            ax1.plot(years, stats_df['median'], marker='s', linewidth=2, label='中位数', color='red')
            ax1.fill_between(years, stats_df['q25'], stats_df['q75'],
                             alpha=0.2, color='gray', label='25-75百分位')
            ax1.set_xlabel('年份', fontsize=12)
            ax1.set_ylabel('相似度', fontsize=12)
            ax1.set_title('城市属性相似度年度趋势', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            plt.setp(ax1.get_xticklabels(), rotation=45)

            # 2. 相似度分布箱线图
            ax2 = axes[0, 1]
            box_data = []
            for year in stats_df['year']:
                if year in similarity_matrices:
                    matrix = similarity_matrices[year]
                    triu_values = matrix[np.triu_indices_from(matrix, k=1)]
                    box_data.append(triu_values)

            box_plot = ax2.boxplot(box_data, labels=years, patch_artist=True)
            # 设置颜色
            colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)

            ax2.set_xlabel('年份', fontsize=12)
            ax2.set_ylabel('相似度分布', fontsize=12)
            ax2.set_title('相似度分布年度对比', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.get_xticklabels(), rotation=45)

            # 3. 标准差趋势
            ax3 = axes[1, 0]
            ax3.bar(years, stats_df['std'], color='green', alpha=0.7)
            ax3.set_xlabel('年份', fontsize=12)
            ax3.set_ylabel('标准差', fontsize=12)
            ax3.set_title('相似度离散程度变化', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            plt.setp(ax3.get_xticklabels(), rotation=45)

            # 4. 极差变化
            ax4 = axes[1, 1]
            ax4.plot(years, stats_df['max'], marker='^', linewidth=2, label='最大值', color='darkred')
            ax4.plot(years, stats_df['min'], marker='v', linewidth=2, label='最小值', color='darkblue')
            ax4.fill_between(years, stats_df['min'], stats_df['max'],
                             alpha=0.2, color='purple', label='极差范围')
            ax4.set_xlabel('年份', fontsize=12)
            ax4.set_ylabel('相似度', fontsize=12)
            ax4.set_title('相似度极差变化', fontsize=14, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            plt.setp(ax4.get_xticklabels(), rotation=45)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'similarity_trend_analysis.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

            print('趋势分析图已保存: %s/similarity_trend_analysis.png' % output_dir)

        except ImportError:
            print('注意：未安装matplotlib和seaborn，跳过可视化')
        except Exception as e:
            print('生成趋势图时出错: %s' % str(e))


# 主程序
def main():
    """
    主程序：加载数据，处理缺失值，计算相似度矩阵
    """
    print('=' * 70)
    print('城市属性相似度计算系统')
    print('=' * 70)

    # 初始化处理器
    processor = RobustCityAttributeProcessor(gamma=1.0, num_years=10)  # 处理10年数据

    # 1. 加载数据
    print('\n[步骤1] 加载数据...')
    data_dir = 'C:/Users/hp/Desktop/14-23attr'  # 修改为目录路径
    yearly_data = processor.load_yearly_data(data_dir)

    # 检查数据质量
    if not yearly_data:
        print('错误: 没有加载到任何数据')
        return None

    # 2. 处理缺失值
    print('\n[步骤2] 处理缺失值...')
    filled_data = processor.handle_missing_values(yearly_data)

    # 3. 计算相似度矩阵
    print('\n[步骤3] 计算属性相似度矩阵...')
    similarity_matrices = processor.compute_yearly_similarity(filled_data)

    # 4. 保存结果
    print('\n[步骤4] 保存结果...')
    output_dir = 'C:/Users/hp/Desktop/14-23_attr_results'
    processor.save_results(similarity_matrices, output_dir)

    print('\n' + '=' * 70)
    print('处理完成!')
    print('=' * 70)

    return similarity_matrices



if __name__ == '__main__':
    # 运行主程序
    similarity_results = main()

    # 示例：访问和显示结果
    if similarity_results and 2020 in similarity_results:
        print('\n2020年相似度矩阵示例（前5个城市）:')
        sample_matrix = similarity_results[2020][:5, :5]
        print(pd.DataFrame(sample_matrix,
                           index=['City_%d' % i for i in range(5)],
                           columns=['City_%d' % i for i in range(5)]).round(4))

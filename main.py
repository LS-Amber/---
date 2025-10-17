# -*- coding: utf-8 -*-
import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置matplotlib使用非GUI后端
import matplotlib
matplotlib.use('Agg')  # 在导入pyplot之前设置

# 设置中文字体
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
import glob
import pandas as pd

from data_processor import DataProcessor
from hierarchy_manager import HierarchyManager
from clustering_calculator import ClusteringCalculator
from temporal_analyzer import TemporalAnalyzer
from visualizer import Visualizer



def main():
    """主函数"""
    # 初始化组件
    data_processor = DataProcessor()
    hierarchy_manager = HierarchyManager()
    clustering_calculator = ClusteringCalculator()
    temporal_analyzer = TemporalAnalyzer()
    visualizer = Visualizer()

    # 1. 加载数据
    print("加载数据...")
    data_dir = "C:/Users/hp/Desktop/Complex_Network_Analyse/2014-2023_py_csv"  # CSV文件所在的目录
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    if not csv_files:
        print("在目录 {} 中未找到CSV文件".format(data_dir))
        return

    # 从文件名提取年份（假设文件名格式为"search_index_2014.csv"）
    panel_data = {}
    for file_path in csv_files:
        try:
            # 从文件名提取年份
            file_name = os.path.basename(file_path)
            year_str = ''.join(filter(str.isdigit, file_name))
            if not year_str or len(year_str) != 4:
                print("无法从文件名 {0} 中提取年份，跳过此文件".format(file_name))
                continue

            year = int(year_str)

            # 读取CSV文件（假设第一列是城市名）
            df = pd.read_csv(file_path, index_col=0)  # 使用第一列作为索引

            # 城市列表
            cities = df.index.tolist()

            # 重新构造矩阵以确保行列一致
            all_cities = sorted(set(cities) | set(df.columns))
            matrix = df.reindex(index=all_cities, columns=all_cities, fill_value=0)

            panel_data[year] = matrix
            print("成功加载 {0} 年数据，包含 {1} 个城市".format(year, len(all_cities)))

        except Exception as e:
            print("加载文件 {0} 时出错: {1}".format(file_path, e))

    if not panel_data:
        print("未能加载任何数据")
        return

    # 2. 权重转换
    print("转换权重...")
    transformed_data = data_processor.transform_weights(panel_data, alpha=1, beta=1)

    # 3. 创建城市层级映射
    print("创建城市层级映射...")
    # 获取所有城市的完整列表
    all_cities = []
    for year, matrix in panel_data.items():
        all_cities.extend(matrix.index.tolist())
    all_cities = sorted(set(all_cities))

    hierarchy_manager.create_city_hierarchy_map(all_cities)

    # 4. 计算聚类系数
    print("计算聚类系数...")
    clustering_results = clustering_calculator.calculate_for_all_years(transformed_data)

    # 5. 计算各层级平均聚类系数
    print("计算各层级平均聚类系数...")
    hierarchy_results = temporal_analyzer.calculate_hierarchy_means(
        clustering_results, hierarchy_manager
    )

    # 6. 计算变化率
    print("计算变化率...")
    growth_rates = temporal_analyzer.calculate_growth_rates()

    # 7. 创建结果目录
    os.makedirs('results', exist_ok=True)

    # 8. 可视化结果
    print("生成可视化结果...")

    # 8.1 绘制各层级时间序列图
    visualizer.plot_hierarchy_timeseries(
        hierarchy_results,
        "各层级城市加权聚类系数时间序列",
        save_path='results/hierarchy_timeseries.png'
    )

    # 8.2 绘制增长率条形图
    visualizer.plot_growth_rates(
        growth_rates,
        "各层级城市加权聚类系数年增长率",
        save_path='results/growth_rates.png'
    )

    # 8.3 绘制最新年份的分布图
    latest_year = max(clustering_results.keys())
    visualizer.plot_hierarchy_distribution(
        clustering_results, hierarchy_manager, latest_year,
        save_path=f'results/distribution_{latest_year}.png'
    )

    # 9. 输出统计结果
    print("\n=== 描述性统计 ===")
    try:
        orig_stats = data_processor.get_descriptive_stats('original')
        trans_stats = data_processor.get_descriptive_stats('transformed')

        print("\n原始搜索指数统计:")
        for year, stats in orig_stats.items():
            print(f"{year}: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}")

        print("\n转换后权重统计:")
        for year, stats in trans_stats.items():
            print(f"{year}: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}")
    except ValueError as e:
        print(f"获取统计信息时出错: {e}")

    print("\n=== 各层级平均加权聚类系数 ===")
    for year, means in hierarchy_results.items():
        print(f"\n{year}年:")
        for hierarchy, value in means.items():
            print(f"  {hierarchy}: {value:.6f}")

    print("\n=== 年度增长率 (%) ===")
    for period, growth in growth_rates.items():
        print(f"\n{period}:")
        for hierarchy, value in growth.items():
            print(f"  {hierarchy}: {value:.2f}%")

    # 10. 保存结果到CSV文件
    print("\n保存结果到CSV文件...")

    # 保存各层级平均聚类系数
    hierarchy_df = pd.DataFrame.from_dict(hierarchy_results, orient='index')
    hierarchy_df.to_csv('results/hierarchy_means.csv')

    # 保存增长率
    growth_df = pd.DataFrame.from_dict(growth_rates, orient='index')
    growth_df.to_csv('results/growth_rates.csv')

    # 保存各城市详细聚类系数
    for year, result in clustering_results.items():
        df = pd.DataFrame({
            'city': result['cities'],
            'clustering_coefficient': result['coefficients'],
            'hierarchy': [hierarchy_manager.city_hierarchy_map.get(city, '未知')
                          for city in result['cities']]
        })
        df.to_csv(f'results/clustering_coefficients_{year}.csv', index=False)

    print("分析完成! 结果已保存到 results/ 目录")


if __name__ == "__main__":
    main()

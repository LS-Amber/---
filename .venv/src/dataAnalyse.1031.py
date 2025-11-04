# -*- coding: utf-8 -*-
import pandas as pd
from collections import defaultdict
import os


def calculate_all_cities_from_csvs(csv_files, output_file=None):
    """
    从指定的CSV文件中统计所有城市的出入度边权重之和，并按总权重降序输出

    参数:
    csv_files: CSV文件路径列表
    output_file: 输出文件路径(可选)
    """

    # 初始化字典来存储城市出入度权重
    city_out_degree = defaultdict(float)  # 出度权重和 (城市作为source)
    city_in_degree = defaultdict(float)  # 入度权重和 (城市作为target)

    # 处理每个CSV文件
    for csv_file in csv_files:
        try:
            print("正在处理文件: {}".format(os.path.basename(csv_file)))

            # 读取CSV文件，尝试多种编码格式
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding)
                    print("使用 {} 编码成功读取文件".format(encoding))
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                print("无法使用任何编码读取文件: {}".format(csv_file))
                continue

            # 计算每个城市的出度权重和 (作为source)
            out_degree = df.groupby('Source')['Weight'].sum()
            for city, weight in out_degree.items():
                city_out_degree[city] += weight

            # 计算每个城市的入度权重和 (作为target)
            in_degree = df.groupby('Target')['Weight'].sum()
            for city, weight in in_degree.items():
                city_in_degree[city] += weight

        except Exception as e:
            print("处理文件 {} 时出错: {}".format(csv_file, e))

    # 获取所有城市的并集
    all_cities = set(city_out_degree.keys()) | set(city_in_degree.keys())
    print("共找到 {} 个城市".format(len(all_cities)))

    # 创建结果DataFrame
    result_data = []
    for city in all_cities:
        out_weight = city_out_degree.get(city, 0)
        in_weight = city_in_degree.get(city, 0)
        total_weight = out_weight + in_weight
        result_data.append({
            'City': city,
            'Out_Degree_Weight': out_weight,
            'In_Degree_Weight': in_weight,
            'Total_Weight': total_weight
        })

    # 创建DataFrame并按总权重倒序排列
    result_df = pd.DataFrame(result_data)
    result_df = result_df.sort_values('Total_Weight', ascending=False)

    # 重置索引
    result_df.reset_index(drop=True, inplace=True)

    # 打印所有城市
    print("\n按集散强度降序排列的所有城市:")
    print(result_df.to_string(index=False))

    # 保存结果
    if output_file:
        result_df.to_csv(output_file, index=False, encoding='utf-8')
        print("\n结果已保存到: {}".format(output_file))

    return result_df


# 使用示例
if __name__ == "__main__":
    # 设置CSV文件路径
    csv_files = [
        "C:/Users/hp/Desktop/省shp/省shp/output_filter_2020.csv",
        "C:/Users/hp/Desktop/省shp/省shp/output_filter_2021.csv",
        "C:/Users/hp/Desktop/省shp/省shp/output_filter_2022.csv",
        "C:/Users/hp/Desktop/省shp/省shp/output_filter_2023.csv"
    ]

    # 输出文件路径
    output_file = "C:/Users/hp/Desktop/信息集散数据/2020-2023_pointTotal_all_cities.csv"

    # 计算并输出所有城市
    all_cities = calculate_all_cities_from_csvs(csv_files, output_file)

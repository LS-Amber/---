# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
import os
import networkx as nx
import numpy as np

# 在绘图之前设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题


def plot_degree_rank(csv_path, output_path="degree_rank.png"):
    """
    绘制城市入度值与出度值的位序分布图。

    参数：
    ----------
    csv_path : str
        输入CSV文件路径，需包含至少 ['City', 'In_Degree', 'Out_Degree'] 三列。
    output_path : str, 可选
        输出图像文件路径，默认 'degree_rank.png'。
    """

    # === 1. 读取数据 ===
    if not os.path.exists(csv_path):
        raise FileNotFoundError("未找到输入文件：{}".format(csv_path))

    df = pd.read_csv(csv_path, encoding="gbk")

    # === 2. 检查列名 ===
    required_cols = {"In_Degree", "Out_Degree"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV文件必须包含列名：{}".format(required_cols))

    # === 3. 排序（按度值从大到小） ===
    df_in = df.sort_values("In_Degree", ascending=False).reset_index(drop=True)
    df_out = df.sort_values("Out_Degree", ascending=False).reset_index(drop=True)

    # === 4. 生成位序 ===
    df_in["Rank"] = df_in.index + 1
    df_out["Rank"] = df_out.index + 1

    # === 5. 绘图 ===
    plt.figure(figsize=(10, 5))
    plt.plot(df_in["Rank"], df_in["In_Degree"], color="dodgerblue", linewidth=1.5, label="入度值")
    plt.plot(df_out["Rank"], df_out["Out_Degree"], color="orangered", linewidth=1.5, label="出度值")

    plt.xlabel("城市位序", fontsize=12)
    plt.ylabel("度值", fontsize=12)
    plt.title("信息流动网络的入度与出度位序分布", fontsize=13, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    # 设置坐标轴范围（自动适配）
    plt.xlim(1, max(len(df_in), len(df_out)))
    plt.ylim(0, max(df_in["In_Degree"].max(), df_out["Out_Degree"].max()) * 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    print("✅ 图像已保存至：{}".format(os.path.abspath(output_path)))


def calculate_closeness_centrality_by_year(edges_csv_path, target_year=None):
    """
    根据年份计算并输出各城市的接近中心性

    参数:
    ----------
    edges_csv_path : str
        包含边信息的CSV文件路径，应包含 Year, Source, Target, Weight 列
    target_year : int, 可选
        指定计算某一年的接近中心性，如果不指定则计算所有年份

    返回:
    -------
    dict or pd.DataFrame
        如果指定了年份，返回该年份的接近中心性DataFrame；
        否则返回包含所有年份接近中心性的字典
    """

    # 读取边数据
    if not os.path.exists(edges_csv_path):
        raise FileNotFoundError("未找到边数据文件：{}".format(edges_csv_path))

    edges_df = pd.read_csv(edges_csv_path, encoding="gbk")

    # 检查必要列
    required_edge_cols = {"Year", "Source", "Target", "Weight"}
    if not required_edge_cols.issubset(edges_df.columns):
        raise ValueError("边数据CSV文件必须包含列名：{}".format(required_edge_cols))

    # 如果指定了年份，则只处理该年份的数据
    if target_year is not None:
        edges_df = edges_df[edges_df["Year"] == target_year]
        if edges_df.empty:
            raise ValueError("未找到 {} 年的数据".format(target_year))

        # 创建有向图
        G = nx.DiGraph()

        # 添加带权重的边到图中
        for _, row in edges_df.iterrows():
            G.add_edge(row["Source"], row["Target"], weight=row["Weight"])

        # 计算接近中心性
        closeness_centrality = nx.closeness_centrality(G)

        # 转换为DataFrame
        centrality_df = pd.DataFrame(list(closeness_centrality.items()),
                                     columns=["City", "Closeness_Centrality"])

        # 按接近中心性排序
        centrality_df = centrality_df.sort_values("Closeness_Centrality", ascending=False).reset_index(drop=True)
        centrality_df["Rank"] = centrality_df.index + 1

        return centrality_df

    else:
        # 处理所有年份
        years = sorted(edges_df["Year"].unique())
        result_dict = {}

        for year in years:
            year_data = edges_df[edges_df["Year"] == year]
            G = nx.DiGraph()

            # 添加边到图中
            for _, row in year_data.iterrows():
                G.add_edge(row["Source"], row["Target"], weight=row["Weight"])

            # 计算接近中心性
            closeness_centrality = nx.closeness_centrality(G)

            # 转换为DataFrame
            centrality_df = pd.DataFrame(list(closeness_centrality.items()),
                                         columns=["City", "Closeness_Centrality"])
            centrality_df = centrality_df.sort_values("Closeness_Centrality", ascending=False).reset_index(drop=True)
            centrality_df["Rank"] = centrality_df.index + 1

            result_dict[year] = centrality_df

        return result_dict


def plot_closeness_centrality(centrality_df, year=None, output_path="closeness_centrality.png"):
    """
    绘制接近中心性位序图

    参数:
    ----------
    centrality_df : pd.DataFrame
        包含城市、接近中心性和排名的数据框
    year : int, 可选
        数据所属年份，用于图表标题
    output_path : str
        输出图像路径
    """

    plt.figure(figsize=(10, 6))
    plt.plot(centrality_df["Rank"], centrality_df["Closeness_Centrality"],
             marker='o', markersize=4, linewidth=1.5, color='purple')

    title = "信息流动网络的接近中心性位序分布"
    if year:
        title += " ({})".format(year)

    plt.xlabel("城市位序", fontsize=12)
    plt.ylabel("接近中心性", fontsize=12)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    # 标注前10个高接近中心性的城市
    top_cities = centrality_df.head(10)
    for idx, row in top_cities.iterrows():
        plt.annotate(row["City"], (row["Rank"], row["Closeness_Centrality"]),
                     xytext=(5, 0), textcoords='offset points',
                     fontsize=9, alpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    print("✅ 接近中心性图像已保存至：{}".format(os.path.abspath(output_path)))


def analyze_closeness_centrality_trend(edges_csv_path, cities=None):
    """
    分析指定城市接近中心性的年度变化趋势

    参数:
    ----------
    edges_csv_path : str
        边数据CSV文件路径
    cities : list, 可选
        要分析的城市列表，如果不指定则分析接近中心性最高的前10个城市

    返回:
    -------
    pd.DataFrame
        包含城市年度接近中心性变化的数据框
    """

    # 获取所有年份的接近中心性数据
    all_years_data = calculate_closeness_centrality_by_year(edges_csv_path)

    # 确定要分析的城市
    if cities is None:
        # 取第一年接近中心性最高的前10个城市
        first_year = sorted(all_years_data.keys())[0]
        cities = all_years_data[first_year].head(10)["City"].tolist()

    # 构建时间序列数据
    trend_data = []
    for year, data in all_years_data.items():
        # 获取该年份的前10个城市
        top_cities = data.head(10)
        for _, row in top_cities.iterrows():
            trend_data.append({
                "Year": year,
                "City": row["City"],
                "Closeness_Centrality": row["Closeness_Centrality"],
                "Rank": row["Rank"]
            })

    trend_df = pd.DataFrame(trend_data)
    return trend_df


def plot_closeness_centrality_trend(trend_df, output_path="closeness_centrality_trend.png"):
    """
    绘制接近中心性年度变化趋势图

    参数:
    ----------
    trend_df : pd.DataFrame
        包含城市年度接近中心性变化的数据框
    output_path : str
        输出图像路径
    """

    plt.figure(figsize=(12, 6))

    # 为不同城市绘制趋势线
    cities = trend_df["City"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(cities)))

    for i, city in enumerate(cities):
        city_data = trend_df[trend_df["City"] == city]
        plt.plot(city_data["Year"], city_data["Closeness_Centrality"],
                 marker='o', linewidth=2, label=city, color=colors[i])

    plt.xlabel("年份", fontsize=12)
    plt.ylabel("接近中心性", fontsize=12)
    plt.title("主要城市接近中心性年度变化趋势", fontsize=13, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("✅ 接近中心性趋势图像已保存至：{}".format(os.path.abspath(output_path)))


# =======================
# 主程序入口
# =======================
if __name__ == "__main__":
    # 原有的度值分析
    input_csv = "C:/Users/hp/Desktop/度值分析（新）/出入度导入数据.csv"
    output_img = "C:/Users/hp/Desktop/度值分析（新）/degree_rank.png"
    # plot_degree_rank(input_csv, output_img)  # 注释掉避免重复运行

    # 新增的接近中心性分析
    edges_csv = "C:/Users/hp/Desktop/度值分析（新）/2014-2023边快速分析.csv"  # 替换为实际的边数据路径

    try:
        # 计算特定年份的接近中心性（例如2023年）
        target_year = 2014
        centrality_result = calculate_closeness_centrality_by_year(edges_csv, target_year=target_year)
        print("\n=== {}年接近中心性计算结果 ===".format(target_year))
        print(centrality_result.head(10))  # 显示前10个

        # 保存结果到CSV
        centrality_output = "C:/Users/hp/Desktop/接近中心性/closeness_centrality_{}.csv".format(target_year)
        centrality_result.to_csv(centrality_output, index=False, encoding="gbk")
        print("✅ {}年接近中心性数据已保存至：{}".format(target_year, os.path.abspath(centrality_output)))

        # 绘制接近中心性图表
        closeness_plot = "C:/Users/hp/Desktop/接近中心性/closeness_centrality_{}.png".format(target_year)
        plot_closeness_centrality(centrality_result, year=target_year, output_path=closeness_plot)

        # 分析接近中心性趋势
        print("\n=== 接近中心性趋势分析 ===")
        trend_df = analyze_closeness_centrality_trend(edges_csv)
        print(trend_df.head(15))  # 显示前15条记录

        # 保存趋势数据
        trend_output = "C:/Users/hp/Desktop/接近中心性/closeness_centrality_trend.csv"
        trend_df.to_csv(trend_output, index=False, encoding="gbk")
        print("✅ 接近中心性趋势数据已保存至：{}".format(os.path.abspath(trend_output)))

        # 绘制趋势图
        trend_plot = "C:/Users/hp/Desktop/接近中心性/closeness_centrality_trend.png"
        plot_closeness_centrality_trend(trend_df, output_path=trend_plot)

    except Exception as e:
        print("❌ 接近中心性计算失败: {}".format(e))

# coding=utf-8
import pandas as pd
import networkx as nx
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def load_network_data(file_path):
    """加载网络数据"""
    df = pd.read_csv(file_path, encoding="gbk")
    return df


def calculate_degree_assortativity(G):
    """计算有向网络的度关联性"""
    # 有向网络的度关联性 (入度-入度, 出度-出度, 入度-出度等)
    in_in_assortativity = nx.degree_assortativity_coefficient(G, x='in', y='in')
    out_out_assortativity = nx.degree_assortativity_coefficient(G, x='out', y='out')
    in_out_assortativity = nx.degree_assortativity_coefficient(G, x='in', y='out')
    out_in_assortativity = nx.degree_assortativity_coefficient(G, x='out', y='in')

    return {
        'in_in': in_in_assortativity,
        'out_out': out_out_assortativity,
        'in_out': in_out_assortativity,
        'out_in': out_in_assortativity
    }


def calculate_average_shortest_path_length(G):
    """计算平均路径长度"""
    try:
        # 如果图是强连通的，直接计算
        if nx.is_strongly_connected(G):
            return nx.average_shortest_path_length(G)
        else:
            # 否则计算强连通分量的平均路径长度
            largest_scc = max(nx.strongly_connected_components(G), key=len)
            subgraph = G.subgraph(largest_scc)
            return nx.average_shortest_path_length(subgraph)
    except:
        # 如果计算失败，返回None
        return None


def calculate_directed_clustering_coefficients(G):
    """计算有向局部聚类系数和全局聚类系数"""

    # 方法1: 使用NetworkX内置函数计算有向聚类系数
    try:
        local_clustering = nx.clustering(G)
        global_clustering = nx.average_clustering(G)
    except:
        local_clustering = {}
        global_clustering = None

    # 方法2: 手动计算有向聚类系数 (基于三元组的定义)
    def directed_triangles_and_triples(node):
        """计算节点的有向三角形和三元组数量"""
        # 获取节点的邻居
        neighbors = set(G.predecessors(node)) | set(G.successors(node))

        # 计算所有可能的三元组
        triples = 0
        triangles = 0

        for u in neighbors:
            for v in neighbors:
                if u != v:
                    # 计算三元组 (node, u, v)
                    if (u in G.predecessors(node) or u in G.successors(node)) and \
                            (v in G.predecessors(node) or v in G.successors(node)) and \
                            (u in G.predecessors(v) or u in G.successors(v)):
                        triples += 1

                        # 检查是否形成有向三角形
                        if (G.has_edge(node, u) or G.has_edge(u, node)) and \
                                (G.has_edge(node, v) or G.has_edge(v, node)) and \
                                (G.has_edge(u, v) or G.has_edge(v, u)):
                            triangles += 1

        # 避免除以零
        if triples == 0:
            return 0, 0, 0

        return triangles, triples, triangles / triples

    # 计算每个节点的局部聚类系数
    manual_local_clustering = {}
    total_triangles = 0
    total_triples = 0

    for node in G.nodes():
        triangles, triples, clustering = directed_triangles_and_triples(node)
        manual_local_clustering[node] = clustering
        total_triangles += triangles
        total_triples += triples

    # 计算全局聚类系数 (传递性)
    if total_triples > 0:
        manual_global_clustering = total_triangles / total_triples
    else:
        manual_global_clustering = 0

    return {
        'local_clustering_nx': local_clustering,
        'global_clustering_nx': global_clustering,
        'local_clustering_manual': manual_local_clustering,
        'global_clustering_manual': manual_global_clustering
    }


def analyze_network(file_path):
    """分析网络的主函数"""

    # 设置输出目录
    output_dir = r"C:\Users\hp\Desktop\度值分析（新）"

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载数据
    df = load_network_data(file_path)
    print("数据加载完成，共{}条边".format(len(df)))

    # 创建有向图
    G = nx.DiGraph()

    # 添加边和权重
    for _, row in df.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'])

    print("网络构建完成，共 {} 个节点，{} 条边".format(G.number_of_nodes(), G.number_of_edges()))

    # 1. 计算度关联性
    print("\n1. 计算度关联性...")
    degree_assortativity = calculate_degree_assortativity(G)
    print("入度-入度关联性: {:.4f}".format(degree_assortativity['in_in']))
    print("出度-出度关联性: {:.4f}".format(degree_assortativity['out_out']))
    print("入度-出度关联性: {:.4f}".format(degree_assortativity['in_out']))
    print("出度-入度关联性: {:.4f}".format(degree_assortativity['out_in']))

    # 2. 计算平均路径长度
    print("\n2. 计算平均路径长度...")
    avg_path_length = calculate_average_shortest_path_length(G)
    if avg_path_length is not None:
        print("平均路径长度: {:.4f}".format(avg_path_length))
    else:
        print("无法计算平均路径长度（图可能不连通）")

    # 3. 计算聚类系数
    print("\n3. 计算聚类系数...")
    clustering_results = calculate_directed_clustering_coefficients(G)

    if clustering_results['global_clustering_nx'] is not None:
        print("全局聚类系数 (NetworkX): {:.4f}".format(clustering_results['global_clustering_nx']))
    print("全局聚类系数 (手动计算): {:.4f}".format(clustering_results['global_clustering_manual']))

    # 输出部分节点的局部聚类系数
    print("\n部分节点的局部聚类系数 (前10个节点):")
    nodes = list(clustering_results['local_clustering_manual'].keys())[:10]
    for node in nodes:
        print("节点 {}: {:.4f}".format(node, clustering_results['local_clustering_manual'][node]))

    # 4. 基本网络统计
    print("\n4. 基本网络统计:")
    print("节点数量: {}".format(G.number_of_nodes()))
    print("边数量: {}".format(G.number_of_edges()))
    print("网络密度: {:.6f}".format(nx.density(G)))

    # 计算度分布
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]

    print("平均入度: {:.2f}".format(np.mean(in_degrees)))
    print("平均出度: {:.2f}".format(np.mean(out_degrees)))
    print("最大入度: {}".format(max(in_degrees)))
    print("最大出度: {}".format(max(out_degrees)))

    # 5. 可视化度分布
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(in_degrees, bins=50, alpha=0.7, color='blue')
    plt.xlabel('入度')
    plt.ylabel('频率')
    plt.title('入度分布')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(out_degrees, bins=50, alpha=0.7, color='red')
    plt.xlabel('出度')
    plt.ylabel('频率')
    plt.title('出度分布')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2023度分布图.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # 6. 保存结果到文件
    results = {
        'degree_assortativity': degree_assortativity,
        'average_path_length': avg_path_length,
        'global_clustering_nx': clustering_results['global_clustering_nx'],
        'global_clustering_manual': clustering_results['global_clustering_manual'],
        'local_clustering': clustering_results['local_clustering_manual'],
        'network_stats': {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_in_degree': np.mean(in_degrees),
            'avg_out_degree': np.mean(out_degrees),
            'max_in_degree': max(in_degrees),
            'max_out_degree': max(out_degrees)
        }
    }

    # 保存局部聚类系数到CSV
    local_clustering_df = pd.DataFrame.from_dict(
        clustering_results['local_clustering_manual'],
        orient='index',
        columns=['局部聚类系数']
    )
    local_clustering_df.to_csv(os.path.join(output_dir, '2023局部聚类系数.csv'), encoding='utf-8-sig')

    # 保存网络统计结果
    with open(os.path.join(output_dir, '2023网络分析结果.txt'), 'w', encoding='utf-8') as f:
        f.write("有向网络分析结果\n")
        f.write("================\n\n")
        f.write("节点数量: {}\n".format(results['network_stats']['nodes']))
        f.write("边数量: {}\n".format(results['network_stats']['edges']))
        f.write("网络密度: {:.6f}\n\n".format(results['network_stats']['density']))

        f.write("度关联性:\n")
        f.write("  入度-入度: {:.4f}\n".format(results['degree_assortativity']['in_in']))
        f.write("  出度-出度: {:.4f}\n".format(results['degree_assortativity']['out_out']))
        f.write("  入度-出度: {:.4f}\n".format(results['degree_assortativity']['in_out']))
        f.write("  出度-入度: {:.4f}\n\n".format(results['degree_assortativity']['out_in']))

        if results['average_path_length'] is not None:
            f.write("平均路径长度: {:.4f}\n".format(results['average_path_length']))
        else:
            f.write("平均路径长度: 无法计算（图可能不连通）\n")

        if results['global_clustering_nx'] is not None:
            f.write("全局聚类系数 (NetworkX): {:.4f}\n".format(results['global_clustering_nx']))
        f.write("全局聚类系数 (手动计算): {:.4f}\n".format(results['global_clustering_manual']))

    print("\n分析完成！结果已保存到文件:")
    print("- " + os.path.join(output_dir, "2023局部聚类系数.csv"))
    print("- " + os.path.join(output_dir, "2023网络分析结果.txt"))
    print("- " + os.path.join(output_dir, "2023度分布图.png"))

    return results


# 主程序
if __name__ == "__main__":
    # 替换为您的CSV文件路径
    file_path = "C:/Users/hp/Desktop/省shp/省shp/output_filter_2023.csv"

    # 执行网络分析
    results = analyze_network(file_path)

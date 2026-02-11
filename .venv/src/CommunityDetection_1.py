# coding=utf-8
import numpy as np
import networkx as nx
from collections import defaultdict
import math
import pandas as pd


class CDDAWN_CDSSA_Revised:
    """
    修正融合型社区检测算法（CD-DAWN-S + CDSSA 原始框架）
    按照新方案实现：简化ρ公式，统一IS计算，复用CDSSA的MD和社区分裂逻辑
    """

    def __init__(self,
                 alpha=0.6,  # 结构-属性权重平衡系数
                 min_size=3  # 最小社区规模
                 ):
        self.alpha = alpha
        self.min_size = min_size

        # 存储中间结果
        self.node_data = {}
        self.S_T = {}  # 结构相似度矩阵
        self.S_A = {}  # 属性相似度矩阵
        self.S = {}  # 融合相似度矩阵
        self.IS = {}  # 节点重要性分数（统一不分入/出）
        self.MD = {}  # 隶属度
        self.G_prime = None  # 简化网络

    def preprocess(self, G, node_attributes):
        """
        Step 1: 数据预处理
        G: 有向加权网络 (networkx DiGraph)
        node_attributes: 节点属性字典 {节点ID: 属性向量}
        """
        self.G = G
        self.node_attrs = node_attributes
        self.nodes = list(G.nodes())
        self.num_nodes = len(self.nodes)

        # 创建节点索引映射
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(self.nodes)}

        # Step 1.1: 计算节点强度、邻居集
        print("正在计算节点强度和邻居集...")
        for u in self.nodes:
            # 出强度 (所有从u出发的边的权重之和)
            out_strength = sum(G[u][v].get('weight', 1.0)
                               for v in G.successors(u) if G.has_edge(u, v))

            # 入强度 (所有指向u的边的权重之和)
            in_strength = sum(G[v][u].get('weight', 1.0)
                              for v in G.predecessors(u) if G.has_edge(v, u))

            # 直接邻居集：所有与u有边关联的节点（不分方向）
            neighbors = set()
            neighbors.update(G.predecessors(u))
            neighbors.update(G.successors(u))

            # 存储节点数据
            self.node_data[u] = {
                'out_strength': out_strength,
                'in_strength': in_strength,
                'neighbors': neighbors
            }

        # Step 1.2: 补全节点属性缺失值
        print("正在处理节点属性...")
        if node_attributes:
            # 获取所有属性维度
            all_attrs = list(node_attributes.values())
            if all_attrs:
                num_dimensions = len(all_attrs[0])

                # 计算每个维度的均值
                attr_means = []
                for dim in range(num_dimensions):
                    values = [attrs[dim] for attrs in all_attrs if dim < len(attrs)]
                    # 假设数值型属性
                    if values and all(isinstance(v, (int, float)) for v in values):
                        attr_means.append(np.mean(values))
                    else:
                        # 对于非数值属性，使用最常见的值
                        from collections import Counter
                        counter = Counter(values)
                        attr_means.append(counter.most_common(1)[0][0])

                # 补全缺失值
                for node in self.nodes:
                    if node in node_attributes:
                        attrs = node_attributes[node]
                        # 如果属性向量长度不足，用均值填充
                        if len(attrs) < num_dimensions:
                            attrs = attrs + [attr_means[i] for i in range(len(attrs), num_dimensions)]
                        node_attributes[node] = attrs
                    else:
                        # 如果节点没有属性，用均值向量
                        node_attributes[node] = attr_means[:]

                self.node_attrs = node_attributes

    def compute_structural_similarity(self):
        """
        Step 2: 计算结构相似度 S_T（简化ρ公式）
        公式：ρ(u,v) = W(u,v) / s⁺_u
        S_T(u,v) = max{ρ(u,v), ρ(v,u)}
        """
        print("正在计算结构相似度...")
        self.S_T = np.zeros((self.num_nodes, self.num_nodes))

        for i, u in enumerate(self.nodes):
            for j, v in enumerate(self.nodes):
                if i == j:
                    self.S_T[i][j] = 1.0  # 自相似度为1
                    continue

                # 计算ρ(u,v) = W(u,v) / s⁺_u
                if self.G.has_edge(u, v):
                    w_uv = self.G[u][v].get('weight', 1.0)
                    s_u_plus = self.node_data[u]['out_strength']
                    rho_uv = w_uv / s_u_plus if s_u_plus > 0 else 0
                else:
                    rho_uv = 0

                # 计算ρ(v,u) = W(v,u) / s⁺_v
                if self.G.has_edge(v, u):
                    w_vu = self.G[v][u].get('weight', 1.0)
                    s_v_plus = self.node_data[v]['out_strength']
                    rho_vu = w_vu / s_v_plus if s_v_plus > 0 else 0
                else:
                    rho_vu = 0

                # 结构相似度为两者的最大值
                self.S_T[i][j] = max(rho_uv, rho_vu)

        print("结构相似度矩阵形状: {0}".format(self.S_T.shape))

    def compute_attribute_similarity(self):
        """
        Step 3: 计算属性相似度 S_A
        """
        print("正在计算属性相似度...")
        self.S_A = np.zeros((self.num_nodes, self.num_nodes))

        if not self.node_attrs:
            print("警告: 没有节点属性，属性相似度设为0")
            return

        for i, u in enumerate(self.nodes):
            for j, v in enumerate(self.nodes):
                if i == j:
                    self.S_A[i][j] = 1.0  # 自相似度为1
                    continue

                # 获取节点属性
                attrs_u = self.node_attrs.get(u, [])
                attrs_v = self.node_attrs.get(v, [])

                if not attrs_u or not attrs_v:
                    self.S_A[i][j] = 0.0
                    continue

                # 统计属性匹配数
                min_len = min(len(attrs_u), len(attrs_v))
                same_attr_count = 0
                for idx in range(min_len):
                    if attrs_u[idx] == attrs_v[idx]:
                        same_attr_count += 1

                # 属性相似度
                total_dimensions = max(len(attrs_u), len(attrs_v))
                if total_dimensions > 0:
                    self.S_A[i][j] = float(same_attr_count) / total_dimensions
                else:
                    self.S_A[i][j] = 0.0

        print("属性相似度矩阵形状: {0}".format(self.S_A.shape))

    def compute_fused_similarity(self):
        """
        Step 4: 计算融合相似度矩阵
        S(u,v) = α * S_T(u,v) + (1-α) * S_A(u,v)
        α = 0.7 (按新方案要求)
        """
        print("正在计算融合相似度...")
        self.S = np.zeros((self.num_nodes, self.num_nodes))

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                # 加权融合结构和属性相似度
                self.S[i][j] = self.alpha * self.S_T[i][j] + (1 - self.alpha) * self.S_A[i][j]

        print("融合相似度矩阵形状: {0}".format(self.S.shape))

    def compute_node_importance(self):
        """
        Step 5: 计算节点重要性分数（IS）
        新公式：IS(u) = α * Σ ρ(v,u) + (1-α) * Σ S_A(u,v)
        其中：Σ ρ(v,u) 是所有终点为u的ρ之和
        """
        print("正在计算节点重要性分数...")

        # 初始化
        self.IS = {}

        for u in self.nodes:
            i = self.node_to_idx[u]

            # 第一部分：α * Σ ρ(v,u) (所有终点为u的ρ之和)
            sum_rho = 0
            for v in self.nodes:
                if self.G.has_edge(v, u):  # 计算v→u的ρ
                    w_vu = self.G[v][u].get('weight', 1.0)
                    s_v_plus = self.node_data[v]['out_strength']
                    if s_v_plus > 0:
                        sum_rho += w_vu / s_v_plus

            # 第二部分：(1-α) * Σ S_A(u,v) (所有与u相关的属性相似度之和)
            sum_sa = 0
            for v in self.nodes:
                j = self.node_to_idx[v]
                sum_sa += self.S_A[i][j]

            # 计算IS
            self.IS[u] = self.alpha * sum_rho + (1 - self.alpha) * sum_sa

        print("节点重要性分数计算完成，共{0}个节点".format(len(self.IS)))

    def simplify_network(self):
        """
        Step 6: 隶属度及网络简化（完全复用CDSSA原始逻辑）
        MD(u,v) = S(u,v) 如果 IS(u) ≤ IS(v)，否则为0
        构建简化网络 G' = (V, E', A')
        """
        print("正在简化网络...")

        # Step 6.1: 计算隶属度 MD
        self.MD = {}

        for u in self.nodes:
            for v in self.nodes:
                if u == v:
                    continue

                i, j = self.node_to_idx[u], self.node_to_idx[v]

                # CDSSA原始逻辑：若IS(u) ≤ IS(v)，则MD(u,v)=S(u,v)
                if self.IS[u] <= self.IS[v]:
                    self.MD[(u, v)] = self.S[i][j]
                else:
                    self.MD[(u, v)] = 0

        # Step 6.2: 构建简化有向网络 G'
        self.G_prime = nx.DiGraph()
        self.G_prime.add_nodes_from(self.nodes)

        for (u, v), weight in self.MD.items():
            if weight > 0:
                self.G_prime.add_edge(u, v, weight=weight)

        print("简化网络构建完成，包含{0}个节点和{1}条边".format(self.G_prime.number_of_nodes(),
                                                               self.G_prime.number_of_edges()))

    def detect_communities(self):
        """
        Step 7: 社区检测（完全复用CDSSA原始分裂思路）
        """
        print("正在检测社区...")

        # Step 7.1: 计算简化网络基础阈值
        edges = []
        for u, v in self.G_prime.edges():
            weight = self.G_prime[u][v]['weight']
            edges.append(((u, v), weight))

        # 平均边权重
        if edges:
            A_avg = sum(w for (_, _), w in edges) / len(edges)
        else:
            A_avg = 0

        # 平均节点重要性
        IS_avg = np.mean(list(self.IS.values())) if self.IS else 0

        # Step 7.2: 初始化社区
        C = [set(self.nodes)]  # 初始社区包含所有节点
        U = set(self.nodes)  # 未处理节点集

        # 存储节点到社区的映射（允许重叠）
        node_communities = {node: {0} for node in self.nodes}  # 初始都在社区0

        # Step 7.3: 迭代分裂社区（CDSSA原始流程）
        iteration = 0
        processed_nodes = set()  # 记录已处理的节点

        while U:
            iteration += 1
            if iteration % 10 == 0:
                print("  第{0}次迭代，剩余未处理节点: {1}".format(iteration, len(U)))

            # a. 从U中选择IS最大的节点r（根节点）
            if not U:
                break

            r = max(U, key=lambda x: self.IS.get(x, 0))

            # b. 定义r的前驱集 T(r) = {u | (u,r) ∈ E' 且 MD(u,r) > 0}
            # 在简化网络G'中，前驱集就是r的入邻居
            T_r = set(self.G_prime.predecessors(r))

            # c. 若 |T(r)| > 1
            if len(T_r) > 1:
                # 从T(r)中选择IS次大的节点r'
                # 按IS值排序
                sorted_nodes = sorted(T_r, key=lambda x: self.IS.get(x, 0), reverse=True)
                if len(sorted_nodes) >= 2:
                    r_prime = sorted_nodes[1]  # IS次大的节点
                else:
                    r_prime = sorted_nodes[0]  # 如果只有一个节点

                # 检查分裂条件：MD(r',r) < A_avg 且 IS(r') > IS_avg
                md_rprime_r = self.MD.get((r_prime, r), 0)

                if md_rprime_r < A_avg and self.IS.get(r_prime, 0) > IS_avg:
                    # 将T(r')分裂为新社区
                    T_r_prime = set(self.G_prime.predecessors(r_prime))
                    new_community = T_r_prime.copy()
                    community_id = len(C)
                    C.append(new_community)

                    # 更新节点到社区的映射
                    for node in new_community:
                        node_communities[node].add(community_id)

                # d. 标记r、r'及T(r)中节点为已处理，从U移除
                nodes_to_remove = {r, r_prime}
                nodes_to_remove.update(T_r)
                processed_nodes.update(nodes_to_remove)
                U.difference_update(nodes_to_remove)
            else:
                # 如果T(r)大小≤1，只标记r为已处理
                processed_nodes.add(r)
                U.difference_update([r])

        # Step 7.4: 合并过小社区（CDSSA补充逻辑）
        print("正在合并过小社区...")
        community_dict = {}
        for comm_id, comm_set in enumerate(C):
            community_dict[comm_id] = comm_set

        small_communities = [comm_id for comm_id, comm_set in community_dict.items()
                             if len(comm_set) < self.min_size]

        for small_comm_id in small_communities:
            if small_comm_id not in community_dict:
                continue

            small_comm = community_dict[small_comm_id]
            if not small_comm:
                continue

            best_match_comm = None
            best_similarity = -1

            for other_comm_id, other_comm in community_dict.items():
                if other_comm_id == small_comm_id:
                    continue

                # 计算两个社区的平均相似度（按S(u,v)均值）
                similarity_sum = 0
                count = 0

                for u in small_comm:
                    for v in other_comm:
                        if u != v:
                            i, j = self.node_to_idx[u], self.node_to_idx[v]
                            similarity_sum += self.S[i][j]
                            count += 1

                if count > 0:
                    avg_similarity = similarity_sum / count
                    if avg_similarity > best_similarity:
                        best_similarity = avg_similarity
                        best_match_comm = other_comm_id

            if best_match_comm is not None:
                # 合并小社区到最相似的社区
                community_dict[best_match_comm].update(small_comm)
                # 更新节点到社区的映射
                for node in small_comm:
                    node_communities[node].discard(small_comm_id)
                    node_communities[node].add(best_match_comm)

                del community_dict[small_comm_id]

        # 转换为最终的社区结构
        final_communities = []
        for comm_id, comm_set in community_dict.items():
            if comm_set:
                final_communities.append(comm_set)

        print("社区检测完成，共发现{0}个社区".format(len(final_communities)))
        return final_communities, node_communities

    def save_fused_similarity_matrix(self):
        """
        将融合相似度矩阵 S 保存为 CSV 文件
        """
        if not hasattr(self, 'S') or self.S is None:
            print("错误: 融合相似度矩阵未计算")
            return

        # 创建DataFrame
        df = pd.DataFrame(self.S, index=self.nodes, columns=self.nodes)

        # 保存为CSV文件
        df.to_csv('fused_symmetry_matrix.csv', index=True, header=True)
        print("融合相似度矩阵已保存至 fused_symmetry_matrix.csv")

    def save_intermediate_results(self):
        """
        保存中间结果用于调试和分析
        """
        # 保存节点重要性分数
        is_df = pd.DataFrame(list(self.IS.items()), columns=['Node', 'IS'])
        is_df.to_csv('node_importance_scores.csv', index=False)
        print("节点重要性分数已保存至 node_importance_scores.csv")

        # 保存结构相似度矩阵
        s_t_df = pd.DataFrame(self.S_T, index=self.nodes, columns=self.nodes)
        s_t_df.to_csv('structural_similarity_matrix.csv', index=True, header=True)
        print("结构相似度矩阵已保存至 structural_similarity_matrix.csv")

        # 保存属性相似度矩阵
        s_a_df = pd.DataFrame(self.S_A, index=self.nodes, columns=self.nodes)
        s_a_df.to_csv('attribute_similarity_matrix.csv', index=True, header=True)
        print("属性相似度矩阵已保存至 attribute_similarity_matrix.csv")

    def run(self, G, node_attributes=None):
        """
        运行完整的社区检测算法
        """
        print("=" * 60)
        print("开始运行修正版CD-DAWN-S + CDSSA算法")
        print("网络节点数: {0}, 边数: {1}".format(G.number_of_nodes(), G.number_of_edges()))
        print("参数: α={0}, min_size={1}".format(self.alpha, self.min_size))
        print("=" * 60)

        # Step 1: 数据预处理
        self.preprocess(G, node_attributes)

        # Step 2: 计算结构相似度（简化ρ公式）
        self.compute_structural_similarity()

        # Step 3: 计算属性相似度
        self.compute_attribute_similarity()

        # Step 4: 计算融合相似度
        self.compute_fused_similarity()

        # Step 5: 计算节点重要性分数（新公式）
        self.compute_node_importance()

        # Step 6: 网络简化（CDSSA原始逻辑）
        self.simplify_network()

        # Step 7-8: 社区检测和输出
        communities, node_communities = self.detect_communities()

        # 保存中间结果
        self.save_fused_similarity_matrix()
        self.save_intermediate_results()

        return communities, node_communities

    def evaluate_communities(self, communities):
        """评估社区质量"""
        print("\n" + "=" * 60)
        print("社区评估结果")
        print("=" * 60)

        # 统计社区规模
        community_sizes = [len(comm) for comm in communities]

        print("社区数量: {0}".format(len(communities)))
        print("社区规模统计:")
        print("  最大规模: {0}".format(max(community_sizes) if community_sizes else 0))
        print("  最小规模: {0}".format(min(community_sizes) if community_sizes else 0))
        print("  平均规模: {0:.2f}".format(np.mean(community_sizes) if community_sizes else 0))
        print("  中位数规模: {0}".format(np.median(community_sizes) if community_sizes else 0))

        # 统计社区大小分布
        print("\n社区大小分布:")
        size_counts = {}
        for size in community_sizes:
            size_counts[size] = size_counts.get(size, 0) + 1
        for size, count in sorted(size_counts.items()):
            print("  大小{0}: {1}个社区".format(size, count))

        # 计算模块度（简化版，针对有向加权网络）
        if hasattr(self, 'G'):
            modularity = self._compute_modularity(communities)
            print("模块度: {0:.4f}".format(modularity))

        # 输出每个社区的前几个节点
        print("\n社区详情 (显示前10个社区):")
        for i, comm in enumerate(communities[:10]):
            comm_nodes = list(comm)
            print("  社区 {0}: {1} 个节点".format(i + 1, len(comm)))
            print("    节点示例: {0}{1}".format(comm_nodes[:min(10, len(comm_nodes))],
                                                '...' if len(comm_nodes) > 10 else ''))

    def _compute_modularity(self, communities):
        """计算有向加权网络的模块度"""
        total_weight = sum(d['weight'] for _, _, d in self.G.edges(data=True))

        if total_weight == 0:
            return 0

        modularity = 0
        for comm in communities:
            comm_nodes = set(comm)
            for u in comm_nodes:
                for v in comm_nodes:
                    if u != v:
                        # 实际权重
                        A_uv = self.G[u][v].get('weight', 0) if self.G.has_edge(u, v) else 0
                        # 期望权重
                        k_u_out = self.node_data[u]['out_strength']
                        k_v_in = self.node_data[v]['in_strength']
                        P_uv = (k_u_out * k_v_in) / total_weight

                        modularity += (A_uv - P_uv)

        return modularity / total_weight


def load_migration_data(migration_filename):
    """
    从CSV文件加载迁移数据并自动生成节点属性
    """
    # 尝试不同的编码方式读取CSV文件
    try:
        df = pd.read_csv(migration_filename, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(migration_filename, encoding='gbk')
        except:
            df = pd.read_csv(migration_filename, encoding='latin1')

    print("CSV文件列名: {0}".format(list(df.columns)))

    # 创建有向加权图
    G = nx.DiGraph()

    # 添加节点（城市）
    cities = set()
    for _, row in df.iterrows():
        cities.add(str(row['Source']))
        cities.add(str(row['Target']))

    # 将城市映射到数字ID
    city_to_id = {city: i for i, city in enumerate(cities)}
    id_to_city = {i: city for city, i in city_to_id.items()}

    # 添加节点
    for city in cities:
        G.add_node(city_to_id[city])

    # 添加边（迁移关系）
    for _, row in df.iterrows():
        source = city_to_id[str(row['Source'])]
        target = city_to_id[str(row['Target'])]

        # 处理权重列，确保是数值类型
        try:
            weight = float(row['Weight'])
        except:
            weight = 1.0  # 默认权重

        # 添加有向边，权重为迁移量
        G.add_edge(source, target, weight=weight)

    # 自动生成节点属性 - 为每个城市生成5维属性向量
    node_attributes = {}
    num_cities = len(cities)
    print("正在为{0}个城市生成节点属性...".format(num_cities))

    for city_id in range(num_cities):
        # 生成5个0-5之间的随机属性值
        attrs = [round(np.random.uniform(0, 5), 2) for _ in range(5)]
        node_attributes[city_id] = attrs

    return G, node_attributes, city_to_id, id_to_city


def main():
    # 从CSV文件加载迁移数据
    print("正在加载迁移数据...")
    migration_file = "C:/Users/hp/Desktop/filter/output_filter_2014.csv"
    G, node_attrs, city_to_id, id_to_city = load_migration_data(migration_file)

    print("\n数据加载完成:")
    print("  节点数: {0}".format(G.number_of_nodes()))
    print("  边数: {0}".format(G.number_of_edges()))
    print("  网络密度: {0:.4f}".format(nx.density(G)))

    # 计算网络的基本统计信息
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    print("  平均入度: {0:.2f}".format(np.mean(in_degrees) if in_degrees else 0))
    print("  平均出度: {0:.2f}".format(np.mean(out_degrees) if out_degrees else 0))

    # 初始化算法（按照新方案设置参数）
    detector = CDDAWN_CDSSA_Revised(
        alpha=0.7,  # 结构-属性权重系数
        min_size=3  # 最小社区规模
    )

    # 运行算法
    communities, node_communities = detector.run(G, node_attrs)

    # 评估结果
    detector.evaluate_communities(communities)

    # 保存结果
    print("\n" + "=" * 60)
    print("结果保存")
    print("=" * 60)

    # 保存社区结构到文件
    with open("migration_communities_results_revised.txt", "w", encoding="utf-8") as f:
        f.write("修正版算法结果\n")
        f.write("=" * 50 + "\n")
        f.write("网络节点数: {}\n".format(G.number_of_nodes()))
        f.write("网络边数: {}\n".format(G.number_of_edges()))
        f.write("检测到的社区数: {}\n\n".format(len(communities)))

        f.write("社区详情:\n")
        for i, comm in enumerate(communities):
            f.write("社区 {} (大小: {}):\n".format(i + 1, len(comm)))
            # 将数字ID转回城市名
            city_names = [id_to_city[node] for node in comm if node in id_to_city]
            f.write("  {}\n\n".format(sorted(city_names)))

        f.write("\n节点所属社区（重叠社区）:\n")
        overlap_count = 0
        for node in sorted(node_communities.keys()):
            comms = sorted(list(node_communities[node]))
            if len(comms) > 1:  # 只显示属于多个社区的节点
                city_name = id_to_city[node] if node in id_to_city else str(node)
                f.write("节点 {}: 社区 {}\n".format(city_name, comms))
                overlap_count += 1

        f.write("\n重叠节点总数: {}\n".format(overlap_count))

    print("结果已保存到 migration_communities_results_revised.txt")

    # 可视化（可选，需要matplotlib）
    try:
        import matplotlib.pyplot as plt

        # 绘制社区规模分布
        community_sizes = [len(comm) for comm in communities]
        plt.figure(figsize=(10, 6))
        plt.hist(community_sizes, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('社区规模')
        plt.ylabel('频数')
        plt.title('社区规模分布（修正版算法）')
        plt.grid(True, alpha=0.3)
        plt.savefig('migration_community_size_distribution_revised.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 绘制节点重要性分布
        plt.figure(figsize=(10, 6))
        is_values = list(detector.IS.values())
        plt.hist(is_values, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('节点重要性分数(IS)')
        plt.ylabel('频数')
        plt.title('节点重要性分布')
        plt.grid(True, alpha=0.3)
        plt.savefig('node_importance_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("可视化图表已保存")

    except ImportError:
        print("注意: 未安装matplotlib，跳过可视化")


if __name__ == "__main__":
    main()
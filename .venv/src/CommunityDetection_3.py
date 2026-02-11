# coding=utf-8
import numpy as np
import networkx as nx
from collections import defaultdict
import pandas as pd
import math
import codecs
import os

# Python 2/3 兼容性处理
import sys

try:
    # Python 2
    reload(sys)
    sys.setdefaultencoding('utf-8')
except NameError:
    # Python 3，无需处理
    pass


class CDDAWN_CDSSA_Fix:
    """
    严格按要求修改：仅保留4个核心优化
    1. 取消初始全节点社区 2. 非重叠模式 3. 优化分裂逻辑 4. 调整合并策略
    支持直接传入预计算的属性相似度矩阵 S_A（替代 node_attributes）
    Python2.7兼容
    """

    def __init__(self,
                 alpha=0.7,
                 min_size=5,  # 最小社区规模
                 min_split_size=1,  # 最小分裂规模
                 non_overlapping=True,  # 非重叠模式
                 precomputed_S_A=None):  # 新增：预计算的属性相似度矩阵 (n x n)
        self.alpha = alpha
        self.min_size = min_size
        self.min_split_size = min_split_size
        self.non_overlapping = non_overlapping
        self.precomputed_S_A = precomputed_S_A  # 保存预计算 S_A
        self.node_data = {}
        self.S_T = None
        self.S_A = None
        self.S = None
        self.IS = {}
        self.MD = {}
        self.G_prime = None

    def preprocess(self, G, node_attributes):
        self.G = G
        self.node_attrs = node_attributes
        self.nodes = list(G.nodes())
        self.num_nodes = len(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(self.nodes)}

        # 计算节点强度、邻居集
        print("正在计算节点强度和邻居集...")
        out_strengths = []
        for u in self.nodes:
            out_strength = sum(G[u][v].get('weight', 1.0)
                               for v in G.successors(u) if G.has_edge(u, v))
            out_strengths.append(out_strength)
            in_strength = sum(G[v][u].get('weight', 1.0)
                              for v in G.predecessors(u) if G.has_edge(v, u))
            neighbors = set()
            neighbors.update(G.predecessors(u))
            neighbors.update(G.successors(u))
            self.node_data[u] = {
                'out_strength': out_strength,
                'in_strength': in_strength,
                'neighbors': neighbors
            }

        # 补全节点属性（仅当未提供 precomputed_S_A 时使用）
        print("正在处理节点属性...")
        if node_attributes and self.precomputed_S_A is None:
            all_attrs = list(node_attributes.values())
            if all_attrs:
                num_dimensions = len(all_attrs[0])
                attr_means = []
                for dim in range(num_dimensions):
                    values = [attrs[dim] for attrs in all_attrs if dim < len(attrs)]
                    if values and all(isinstance(v, (int, float)) for v in values):
                        attr_means.append(np.mean(values))
                    else:
                        from collections import Counter
                        counter = Counter(values)
                        attr_means.append(counter.most_common(1)[0][0])
                for node in self.nodes:
                    if node in node_attributes:
                        attrs = node_attributes[node]
                        if len(attrs) < num_dimensions:
                            attrs += [attr_means[i] for i in range(len(attrs), num_dimensions)]
                        node_attributes[node] = attrs
                    else:
                        node_attributes[node] = attr_means[:]
                self.node_attrs = node_attributes
        elif self.precomputed_S_A is not None:
            print("✓ 预计算 S_A 已提供，跳过节点属性补全")
        else:
            print("⚠️ 无节点属性且未提供 precomputed_S_A，后续属性相似度将为0")

    def compute_structural_similarity(self):
        print("正在计算结构相似度...")
        self.S_T = np.zeros((self.num_nodes, self.num_nodes))
        for i, u in enumerate(self.nodes):
            for j, v in enumerate(self.nodes):
                if i == j:
                    self.S_T[i][j] = 1.0
                    continue
                # 结构相似度计算
                rho_uv = 0
                if self.G.has_edge(u, v):
                    w_uv = self.G[u][v].get('weight', 1.0)
                    s_u_plus = self.node_data[u]['out_strength']
                    if s_u_plus > 0:
                        rho_uv = w_uv / float(s_u_plus)

                rho_vu = 0
                if self.G.has_edge(v, u):
                    w_vu = self.G[v][u].get('weight', 1.0)
                    s_v_plus = self.node_data[v]['out_strength']
                    if s_v_plus > 0:
                        rho_vu = w_vu / float(s_v_plus)

                struct_sim = max(rho_uv, rho_vu)
                self.S_T[i][j] = struct_sim
        print("结构相似度矩阵形状: {0}".format(self.S_T.shape))

    def compute_attribute_similarity(self):
        print("正在计算属性相似度...")
        if self.precomputed_S_A is not None:
            # 直接使用预计算的 S_A
            assert self.precomputed_S_A.shape == (self.num_nodes, self.num_nodes), \
                "预计算 S_A 形状 {} 不匹配节点数 {}".format(self.precomputed_S_A.shape, self.num_nodes)
            self.S_A = self.precomputed_S_A.copy()
            print("✓ 使用预计算的属性相似度矩阵，形状: {0}".format(self.S_A.shape))
            return

        # 否则走原逻辑（仅当有 node_attrs 时）
        self.S_A = np.zeros((self.num_nodes, self.num_nodes))
        if not self.node_attrs:
            print("警告: 没有节点属性，属性相似度设为0")
            return
        for i, u in enumerate(self.nodes):
            for j, v in enumerate(self.nodes):
                if i == j:
                    self.S_A[i][j] = 1.0
                    continue
                attrs_u = self.node_attrs.get(u, [])
                attrs_v = self.node_attrs.get(v, [])
                if not attrs_u or not attrs_v:
                    self.S_A[i][j] = 0.0
                    continue
                min_len = min(len(attrs_u), len(attrs_v))
                same_attr_count = 0
                for idx in range(min_len):
                    if attrs_u[idx] == attrs_v[idx]:
                        same_attr_count += 1
                total_dimensions = max(len(attrs_u), len(attrs_v))
                self.S_A[i][j] = float(same_attr_count) / total_dimensions if total_dimensions > 0 else 0.0
        print("属性相似度矩阵形状: {0}".format(self.S_A.shape))

    def compute_fused_similarity(self):
        print("正在计算融合相似度...")
        self.S = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                self.S[i][j] = self.alpha * self.S_T[i][j] + (1 - self.alpha) * self.S_A[i][j]
        print("融合相似度矩阵形状: {0}".format(self.S.shape))

    def compute_node_importance(self):
        print("正在计算节点重要性分数...")
        self.IS = {}
        for u in self.nodes:
            i = self.node_to_idx[u]
            sum_rho = 0
            for v in self.nodes:
                if self.G.has_edge(v, u):
                    w_vu = self.G[v][u].get('weight', 1.0)
                    s_v_plus = self.node_data[v]['out_strength']
                    if s_v_plus > 0:
                        sum_rho += w_vu / float(s_v_plus)
            sum_sa = np.sum(self.S_A[i])
            self.IS[u] = self.alpha * sum_rho + (1 - self.alpha) * sum_sa
        print("节点重要性分数计算完成，共{0}个节点".format(len(self.IS)))

    def simplify_network(self):
        print("正在简化网络...")
        self.MD = {}
        # 首先计算所有MD值
        for u in self.nodes:
            for v in self.nodes:
                if u == v:
                    continue
                i, j = self.node_to_idx[u], self.node_to_idx[v]
                if self.IS[u] <= self.IS[v]:
                    self.MD[(u, v)] = self.S[i][j]
                else:
                    self.MD[(u, v)] = 0

        # 根据公式E′ = {⟨𝑣𝑖, 𝑣𝑝⟩|𝑀𝐷(𝑣𝑖, 𝑣𝑝) = 𝑚𝑎𝑥𝑣𝑗V𝑀𝐷(𝑣𝑖, 𝑣𝑗 )}重新构建网络
        # 为每个节点vi找到MD(vi, vj)最大的vj
        selected_edges = {}
        for u in self.nodes:
            # 找到节点u与其他节点的MD最大值
            max_md_value = -1
            max_md_target = None
            for v in self.nodes:
                if u == v:
                    continue
                md_value = self.MD.get((u, v), 0)
                if md_value > max_md_value:
                    max_md_value = md_value
                    max_md_target = v

            # 如果找到了最大值，添加到简化网络
            if max_md_target is not None:
                selected_edges[(u, max_md_target)] = max_md_value

        # 清空旧的MD字典，只保留选中的边
        self.MD = selected_edges

        # 构建简化网络
        self.G_prime = nx.DiGraph()
        self.G_prime.add_nodes_from(self.nodes)
        for (u, v), weight in self.MD.items():
            if weight > 0:
                self.G_prime.add_edge(u, v, weight=weight)

        print("简化网络构建完成，包含{0}个节点和{1}条边".format(
            self.G_prime.number_of_nodes(), self.G_prime.number_of_edges()))

    def detect_communities(self):
        """严格落实4个修改点"""
        print("正在检测社区（非重叠+优化分裂+合理合并）...")
        # 使用全局A_avg中位数
        edges = [(u, v, d['weight']) for u, v, d in self.G_prime.edges(data=True)]
        global_A_avg = np.median([w for _, _, w in edges]) if edges else 0
        # 使用IS中位数作为分裂门槛
        IS_values = list(self.IS.values())  # 转换为列表
        IS_avg = np.median(IS_values) if IS_values else 0
        print("全局A_avg中位数: {0:.4f}, IS中位数: {1:.4f}".format(global_A_avg, IS_avg))

        # 1. 取消初始全节点社区：C初始为空
        C = []
        U = set(self.nodes)
        node_communities = {node: set() for node in self.nodes}

        iteration = 0
        while U and len(C) < 30:  # 限制最大社区数为30个，避免碎片化
            iteration += 1
            if iteration % 3 == 0:
                print("  第{0}次迭代，剩余未处理节点: {1}，已检测社区数: {2}".format(
                    iteration, len(U), len(C)))

            # 选择根节点
            r = max(U, key=lambda x: self.IS.get(x, 0))
            # 优化分裂逻辑：获取r的前驱集（根据简化网络的结构）
            T_r = set(self.G_prime.predecessors(r)) & U
            print("  根节点{0}，前驱集规模: {1}".format(self.idx_to_node.get(r, r), len(T_r)))

            # 分裂条件：门槛收紧
            if len(T_r) >= self.min_split_size:
                r_prime = max(T_r, key=lambda x: self.IS.get(x, 0))

                # 使用全局A_avg中位数
                md_rprime_r = self.MD.get((r_prime, r), 0)

                # 分裂条件：MD < 全局A_avg中位数且IS > IS中位数
                if md_rprime_r < global_A_avg and self.IS.get(r_prime, 0) > IS_avg:
                    new_community = T_r.copy()
                    new_community.add(r_prime)
                    new_community.add(r)
                    if len(new_community) >= self.min_size:
                        community_id = len(C)
                        C.append(new_community)
                        # 2. 非重叠模式：清空原有归属
                        for node in new_community:
                            if self.non_overlapping:
                                node_communities[node].clear()
                            node_communities[node].add(community_id)
                        # 彻底从待处理集移除
                        U.difference_update(new_community)
                        print("  分裂新社区{0}，规模: {1}".format(community_id + 1, len(new_community)))
                else:
                    # 如果不满足分裂条件，但有前驱节点，尝试将r分配到与r_prime相同的社区
                    if r_prime in node_communities and len(node_communities[r_prime]) > 0:
                        # 将r分配到r_prime所在的社区
                        r_prime_comm = next(iter(node_communities[r_prime]))
                        C[r_prime_comm].add(r)
                        if self.non_overlapping:
                            node_communities[r].clear()
                        node_communities[r].add(r_prime_comm)
                        U.discard(r)
                    else:
                        # 将r作为单独社区
                        single_community = {r}
                        community_id = len(C)
                        C.append(single_community)
                        if self.non_overlapping:
                            node_communities[r].clear()
                        node_communities[r].add(community_id)
                        U.discard(r)
                        print("  创建单节点社区{0}，节点: {1}".format(community_id + 1, self.idx_to_node.get(r, r)))
            else:
                # 如果没有足够的前驱节点，将r分配到最相似的现有社区
                assigned = False
                if len(C) > 0:
                    best_comm = None
                    best_sim = -1

                    # 寻找与该节点相似度最高的现有社区
                    for comm_id, comm in enumerate(C):
                        sim_sum = 0
                        count = 0
                        for u in comm:
                            sim_sum += self.S[self.node_to_idx[r]][self.node_to_idx[u]]
                            count += 1
                        if count > 0:
                            avg_sim = sim_sum / float(count)
                            if avg_sim > best_sim:
                                best_sim = avg_sim
                                best_comm = comm_id

                    # 如果找到了相似社区，将节点加入
                    if best_comm is not None:
                        C[best_comm].add(r)
                        if self.non_overlapping:
                            node_communities[r].clear()
                        node_communities[r].add(best_comm)
                        assigned = True

                if not assigned:
                    # 创建单节点社区
                    single_community = {r}
                    community_id = len(C)
                    C.append(single_community)
                    if self.non_overlapping:
                        node_communities[r].clear()
                    node_communities[r].add(community_id)

                U.discard(r)

        # 处理剩余未分裂节点
        print("正在分配剩余未分裂节点...")
        for node in U.copy():  # 使用copy避免在迭代中修改集合
            best_comm = None
            best_sim = -1

            # 寻找与该节点相似度最高的现有社区
            for comm_id, comm in enumerate(C):
                sim_sum = 0
                count = 0
                for u in comm:
                    sim_sum += self.S[self.node_to_idx[node]][self.node_to_idx[u]]
                    count += 1
                if count > 0:
                    avg_sim = sim_sum / float(count)
                    if avg_sim > best_sim:
                        best_sim = avg_sim
                        best_comm = comm_id

            # 如果找到了相似社区，将节点加入
            if best_comm is not None:
                C[best_comm].add(node)
                if self.non_overlapping:
                    node_communities[node].clear()
                node_communities[node].add(best_comm)
            else:
                # 创建单节点社区
                new_community = {node}
                community_id = len(C)
                C.append(new_community)
                node_communities[node].add(community_id)

            U.discard(node)  # 确保节点从未处理集合中移除

        # 调整合并策略：过小社区合并
        print("正在合并过小社区...")
        # 分离大小社区
        large_comms = []  # 符合规模的社区
        small_comms = []  # 过小社区（<min_size）
        for comm in C:
            if len(comm) >= self.min_size:
                large_comms.append(comm)
            else:
                small_comms.append(comm)

        # 合并过小社区
        for small_comm in small_comms:
            best_match = None
            best_sim = -1
            # 找最相似的大社区
            for comm_id, large_comm in enumerate(large_comms):
                sim_sum = 0
                count = 0
                for u in small_comm:
                    for v in large_comm:
                        sim_sum += self.S[self.node_to_idx[u]][self.node_to_idx[v]]
                        count += 1
                avg_sim = sim_sum / float(count) if count > 0 else 0
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_match = comm_id
            # 执行合并
            if best_match is not None:
                large_comms[best_match].update(small_comm)
                for node in small_comm:
                    if self.non_overlapping:
                        node_communities[node].clear()
                    node_communities[node].add(best_match)

        # 最终社区列表（去重+过滤）
        final_communities = []
        seen = set()
        for comm in large_comms:
            comm_tuple = tuple(sorted(comm))
            if comm_tuple not in seen and len(comm) >= self.min_size:
                seen.add(comm_tuple)
                final_communities.append(comm)

        # 统计重叠节点（非重叠模式下应为0）
        overlap_count = sum(1 for node in node_communities if len(node_communities[node]) > 1)
        print("社区检测完成，共发现{0}个有效社区，重叠节点数: {1}".format(
            len(final_communities), overlap_count))
        return final_communities, node_communities

    def save_results(self, communities, node_communities, id_to_city=None, year=None):
        print("\n正在保存结果...")
        # 计算重叠节点数
        overlap_count = sum(1 for node in node_communities if len(node_communities[node]) > 1)

        # 输出目录
        output_dir = r"C:\Users\hp\Desktop\社区结果"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 年份前缀
        prefix = "{}_".format(year) if year else ""

        # 1. fixed_communities.txt
        filename = os.path.join(output_dir, "{}fixed_communities.txt".format(prefix))
        with codecs.open(filename, "w", encoding="utf-8") as f:
            f.write("CDSSA优化版算法结果（仅保留4个核心修改）\n")
            f.write("=" * 50 + "\n")
            f.write("网络节点数: {0}\n".format(self.num_nodes))
            f.write("网络边数: {0}\n".format(self.G.number_of_edges()))
            f.write("检测到的社区数: {0}\n".format(len(communities)))
            f.write("重叠节点数: {0}\n\n".format(overlap_count))
            for i, comm in enumerate(communities):
                f.write("社区 {0} (大小: {1}):\n".format(i + 1, len(comm)))
                if id_to_city:
                    comm_nodes = [id_to_city[node] for node in comm if node in id_to_city]
                else:
                    comm_nodes = list(comm)
                f.write("  {0}\n\n".format(sorted(comm_nodes)))
            # 输出重叠节点（非重叠模式下应为空）
            overlap_nodes = [node for node in node_communities if len(node_communities[node]) > 1]
            if overlap_nodes:
                f.write("\n重叠节点:\n")
                for node in sorted(overlap_nodes):
                    comms = sorted(list(node_communities[node]))
                    city_name = id_to_city[node] if (id_to_city and node in id_to_city) else str(node)
                    f.write("节点 {0}: 社区 {1}\n".format(city_name, comms))
        print("结果已保存到 {}".format(filename))

        # 2. fixed_fused_similarity.csv
        csv_filename = os.path.join(output_dir, "{}fixed_fused_similarity.csv".format(prefix))
        df = pd.DataFrame(self.S, index=self.nodes, columns=self.nodes)
        df.to_csv(csv_filename, index=True, header=True)
        print("融合相似度矩阵已保存至 {}".format(csv_filename))

        # 3. fixed_node_importance.csv
        is_csv_filename = os.path.join(output_dir, "{}fixed_node_importance.csv".format(prefix))
        is_df = pd.DataFrame(self.IS.items(), columns=['Node', 'IS'])
        is_df['City'] = is_df['Node'].map(self.idx_to_node)
        is_df.to_csv(is_csv_filename, index=False)
        print("节点重要性分数已保存至 {}".format(is_csv_filename))

    def evaluate_communities(self, communities):
        print("\n" + "=" * 60)
        print("社区评估结果")
        print("=" * 60)
        community_sizes = [len(comm) for comm in communities]
        print("社区数量: {0}".format(len(communities)))
        print("社区规模统计:")
        print("  最大规模: {0}".format(max(community_sizes) if community_sizes else 0))
        print("  最小规模: {0}".format(min(community_sizes) if community_sizes else 0))
        print("  平均规模: {0:.2f}".format(np.mean(community_sizes) if community_sizes else 0))
        print("  中位数规模: {0}".format(np.median(community_sizes) if community_sizes else 0))
        # 模块度
        total_weight = sum(d['weight'] for _, _, d in self.G.edges(data=True))
        modularity = 0
        for comm in communities:
            comm_nodes = set(comm)
            for u in comm_nodes:
                for v in comm_nodes:
                    if u != v:
                        A_uv = self.G[u][v].get('weight', 0) if self.G.has_edge(u, v) else 0
                        k_u_out = self.node_data[u]['out_strength']
                        k_v_in = self.node_data[v]['in_strength']
                        P_uv = (k_u_out * k_v_in) / float(total_weight) if total_weight > 0 else 0
                        modularity += (A_uv - P_uv)
        modularity = modularity / float(total_weight) if total_weight > 0 else 0
        print("\n模块度: {0:.4f}".format(modularity))

    def run(self, G, node_attributes=None, id_to_city=None, precomputed_S_A=None, year=None):
        print("=" * 60)
        print("开始运行仅保留4个核心修改的CDSSA算法")
        print("网络节点数: {0}, 边数: {1}".format(G.number_of_nodes(), G.number_of_edges()))
        print("参数: α={0}, min_size={1}, min_split_size={2}".format(
            self.alpha, self.min_size, self.min_split_size))
        print("=" * 60)

        # 优先使用传入的 precomputed_S_A（覆盖 __init__ 中的）
        if precomputed_S_A is not None:
            self.precomputed_S_A = precomputed_S_A

        self.preprocess(G, node_attributes)
        self.compute_structural_similarity()
        self.compute_attribute_similarity()
        self.compute_fused_similarity()
        self.compute_node_importance()
        self.simplify_network()
        communities, node_communities = self.detect_communities()
        self.save_results(communities, node_communities, id_to_city, year=year)
        self.evaluate_communities(communities)
        self.visualize_results(communities, year=year)
        return communities, node_communities

    def visualize_results(self, communities, year=None):
        try:
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False

            plt.figure(figsize=(10, 6))
            community_sizes = [len(comm) for comm in communities]
            plt.hist(community_sizes, bins=12, edgecolor='black', alpha=0.7, color='#1f77b4')
            plt.xlabel('社区规模', fontsize=12)
            plt.ylabel('频数', fontsize=12)
            plt.title('社区规模分布（非重叠+优化分裂）', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)

            # 构建带年份的文件名
            output_dir = r"C:\Users\hp\Desktop\社区结果"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            prefix = "{}_".format(year) if year else ""
            png_filename = os.path.join(output_dir, "{}fixed_community_size.png".format(prefix))
            plt.savefig(png_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print("可视化图表已保存至 {}".format(png_filename))
        except ImportError:
            print("注意: 未安装matplotlib，跳过可视化")


def load_migration_data(migration_filename, attribute_files):
    print("正在加载迁移数据: {0}".format(migration_filename))
    try:
        df = pd.read_csv(migration_filename, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(migration_filename, encoding='gbk')
        except:
            df = pd.read_csv(migration_filename, encoding='latin1')
    print("数据列名: {0}".format(list(df.columns)))
    print("数据总行数: {0}".format(len(df)))

    # 提取城市列表并映射为数字ID
    cities = set()
    cities.update(df['Source'].astype(str).unique())
    cities.update(df['Target'].astype(str).unique())
    cities = sorted(list(cities))
    city_to_id = {city: i for i, city in enumerate(cities)}
    id_to_city = {i: city for city, i in city_to_id.items()}
    print("共包含{0}个城市".format(len(cities)))

    # 构建有向加权图
    G = nx.DiGraph()
    G.add_nodes_from(range(len(cities)))
    for _, row in df.iterrows():
        source_city = str(row['Source'])
        target_city = str(row['Target'])
        source = city_to_id[source_city]
        target = city_to_id[target_city]
        # 处理权重（确保为数值型）
        try:
            weight = float(row['Weight'])
        except:
            weight = 1.0
        # 添加边（避免重复边，保留最大权重）
        if G.has_edge(source, target):
            if weight > G[source][target]['weight']:
                G[source][target]['weight'] = weight
        else:
            G.add_edge(source, target, weight=weight)
    print("图构建完成：{0}个节点，{1}条边".format(G.number_of_nodes(), G.number_of_edges()))

    # 加载属性数据（仅用于兼容，若用 precomputed_S_A 则忽略）
    node_attributes = {}
    for file_path in attribute_files:
        print("正在加载属性文件: {0}".format(file_path))
        try:
            if file_path.endswith('.csv'):
                attr_df = pd.read_csv(file_path, encoding='utf-8')
            elif file_path.endswith('.xlsx'):
                attr_df = pd.read_excel(file_path)
            else:
                raise ValueError("不支持的文件格式: {0}".format(file_path))
        except Exception as e:
            print("加载文件失败: {0}".format(e))
            continue

        # 假设第一列为城市名
        for _, row in attr_df.iterrows():
            city_name = str(row.iloc[0])
            if city_name in city_to_id:
                city_id = city_to_id[city_name]
                attributes = row.iloc[1:].tolist()
                if city_id not in node_attributes:
                    node_attributes[city_id] = []
                node_attributes[city_id].extend(attributes)

    print("节点属性加载完成，共{0}个节点".format(len(node_attributes)))
    return G, node_attributes, city_to_id, id_to_city


def main():
    # 配置
    year = 2023  # 👈 请按实际年份修改（如 2015, 2023）
    migration_file = "C:/Users/hp/Desktop/filter/output_filter_2023.csv"
    attribute_files = []  # 不再需要属性文件

    # 加载图
    G, node_attrs, city_to_id, id_to_city = load_migration_data(migration_file, attribute_files)

    # ✅ 加载你自己的 296×296 属性相似度矩阵
    try:
        attr_sim_df = pd.read_csv(
            "C:/Users/hp/Desktop/14-23_attr_results/attribute_similarity_2023.csv",
            index_col=0
        )
        # 校验维度
        if attr_sim_df.shape != (len(G.nodes()), len(G.nodes())):
            raise ValueError("S_A 形状 {} 不匹配图节点数 {}".format(
                attr_sim_df.shape, len(G.nodes())))
        S_A_matrix = attr_sim_df.values.astype(float)
        print("✅ 成功加载预计算 S_A，形状: {}".format(S_A_matrix.shape))
    except Exception as e:
        print("❌ 加载属性相似度矩阵失败: {}".format(e))
        return

    # 初始化算法
    detector = CDDAWN_CDSSA_Fix(
        alpha=0.7,
        min_size=5,
        min_split_size=5,
        non_overlapping=True,
        precomputed_S_A=S_A_matrix
    )

    # 运行（传入 year）
    communities, node_communities = detector.run(
        G=G,
        node_attributes={},
        id_to_city=id_to_city,
        precomputed_S_A=S_A_matrix,
        year=year
    )

    print("\n" + "=" * 60)
    print("运行完成！")
    print("=" * 60)
    print("输出文件已保存至：C:\\Users\\hp\\Desktop\\社区结果")
    print("  - {}fixed_communities.txt".format(year))
    print("  - {}fixed_fused_similarity.csv".format(year))
    print("  - {}fixed_node_importance.csv".format(year))
    print("  - fixed_community_size.png")


if __name__ == "__main__":
    main()

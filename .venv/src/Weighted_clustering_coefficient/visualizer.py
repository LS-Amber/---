import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Visualizer:
    """可视化工具类，负责结果可视化"""

    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.fig_size = (12, 8)
        # 为不同层级定义颜色
        self.hierarchy_colors = {
            '直辖市': '#E41A1C',  # 红色
            '副省级城市': '#377EB8',  # 蓝色
            '省会城市': '#4DAF4A',  # 绿色
            '普通地级市': '#984EA3'  # 紫色
        }

    def plot_hierarchy_timeseries(self, hierarchy_results, title, save_path=None):
        """
        绘制各层级时间序列图

        参数:
        hierarchy_results: 各层级结果字典
        title: 图表标题
        save_path: 保存路径（可选）

        返回:
        matplotlib图对象
        """
        fig, ax = plt.subplots(figsize=self.fig_size)

        years = sorted(hierarchy_results.keys())

        for hierarchy in next(iter(hierarchy_results.values())).keys():
            values = [hierarchy_results[year][hierarchy] for year in years]
            color = self.hierarchy_colors.get(hierarchy, '#000000')

            ax.plot(years, values, marker='o', linewidth=2,
                    markersize=8, label=hierarchy, color=color)

            # 添加趋势线
            if len(years) > 1:
                mask = ~np.isnan(values)
                if np.sum(mask) > 1:
                    # 修复索引问题
                    x_data = np.array(range(len(years)))[mask]
                    y_data = np.array(values)[mask]
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    ax.plot(years, p(range(len(years))), "--",
                            alpha=0.7, color=color)

        ax.set_xlabel('年份')
        ax.set_ylabel('加权聚类系数')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close(fig)  # 关闭图形以释放内存
        return fig, ax

    def plot_growth_rates(self, growth_rates, title, save_path=None):
        """
        绘制增长率条形图

        参数:
        growth_rates: 增长率字典
        title: 图表标题
        save_path: 保存路径（可选）

        返回:
        matplotlib图对象
        """
        fig, ax = plt.subplots(figsize=self.fig_size)

        periods = list(growth_rates.keys())
        hierarchies = list(growth_rates[periods[0]].keys())

        # 准备数据
        data = {h: [] for h in hierarchies}
        for period in periods:
            for h in hierarchies:
                data[h].append(growth_rates[period][h])

        # 绘制分组条形图
        x = np.arange(len(periods))
        width = 0.2

        for i, (hierarchy, values) in enumerate(data.items()):
            color = self.hierarchy_colors.get(hierarchy, '#000000')
            offset = width * (i - len(hierarchies) / 2 + 0.5)
            ax.bar(x + offset, values, width, label=hierarchy, color=color)

        ax.set_xlabel('时期')
        ax.set_ylabel('增长率 (%)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(periods, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close(fig)  # 关闭图形以释放内存
        return fig, ax

    def plot_hierarchy_distribution(self, clustering_results, hierarchy_manager, year, save_path=None):
        """
        绘制特定年份各层级聚类系数分布

        参数:
        clustering_results: 聚类系数结果字典
        hierarchy_manager: 层级管理器实例
        year: 年份
        save_path: 保存路径（可选）

        返回:
        matplotlib图对象
        """
        if year not in clustering_results:
            raise ValueError(f"没有 {year} 年的数据")

        fig, ax = plt.subplots(figsize=self.fig_size)

        cities = clustering_results[year]['cities']
        coefficients = clustering_results[year]['coefficients']

        # 按层级分组数据
        data = []
        labels = []
        colors = []

        for hierarchy in hierarchy_manager.hierarchy_names:
            hierarchy_cities = [
                city for city in cities
                if hierarchy_manager.city_hierarchy_map.get(city) == hierarchy
            ]
            hierarchy_indices = [i for i, city in enumerate(cities) if city in hierarchy_cities]

            if hierarchy_indices:
                hierarchy_values = [coefficients[i] for i in hierarchy_indices]
                data.append(hierarchy_values)
                labels.append(hierarchy)
                colors.append(self.hierarchy_colors.get(hierarchy, '#000000'))

        # 绘制箱线图
        boxplot = ax.boxplot(data, labels=labels, patch_artist=True)

        # 设置颜色
        for patch, color in zip(boxplot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('加权聚类系数')
        ax.set_title(f'{year}年各层级城市加权聚类系数分布')
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close(fig)  # 关闭图形以释放内存
        return fig, ax

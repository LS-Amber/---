# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')  # 设置非GUI后端
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class Visualizer:
    """可视化工具类，负责结果可视化"""

    def __init__(self):
        # 使用兼容的样式
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('ggplot')
        self.fig_size = (10, 8)

    def plot_heatmap(self, data, title, ax=None):
        """
        绘制热力图

        参数:
        data: 要绘制的数据矩阵
        title: 图表标题
        ax: matplotlib轴对象（可选）

        返回:
        matplotlib轴对象
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.fig_size)

        mask = np.isnan(data) | (data >= 1e6)  # 掩码NaN和不可达值
        im = ax.imshow(data, cmap='viridis_r', interpolation='nearest')

        # 设置刻度
        ax.set_xticks(range(len(data.columns)))
        ax.set_yticks(range(len(data.index)))
        ax.set_xticklabels(data.columns, rotation=45, ha='right')
        ax.set_yticklabels(data.index)

        # 添加颜色条
        plt.colorbar(im, ax=ax)
        ax.set_title(title)

        return ax

    def plot_timeseries(self, years, values, title, ax=None):
        """
        绘制时间序列图

        参数:
        years: 年份列表
        values: 值列表
        title: 图表标题
        ax: matplotlib轴对象（可选）

        返回:
        matplotlib轴对象
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.fig_size)

        ax.plot(years, values, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('年份')
        ax.set_ylabel('路径长度')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # 添加趋势线
        if len(years) > 1 and not all(np.isnan(values)):
            values_array = np.array(values)
            mask = ~np.isnan(values_array)
            if np.sum(mask) > 1:
                # 修复索引问题
                x_data = np.array(range(len(years)))[mask]
                y_data = values_array[mask]
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                ax.plot(years, p(range(len(years))), "r--", alpha=0.7)

        return ax

    def plot_growth_rates(self, growth_rates, title, ax=None):
        """
        绘制增长率热力图

        参数:
        growth_rates: 增长率矩阵字典
        title: 图表标题
        ax: matplotlib轴对象（可选）

        返回:
        matplotlib轴对象
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.fig_size)

        # 提取第一个时期的增长率作为示例
        period = list(growth_rates.keys())[0]
        data = growth_rates[period]

        im = ax.imshow(data, cmap='RdYlGn', interpolation='nearest')

        # 设置刻度
        ax.set_xticks(range(len(data.columns)))
        ax.set_yticks(range(len(data.index)))
        ax.set_xticklabels(data.columns, rotation=45, ha='right')
        ax.set_yticklabels(data.index)

        # 添加颜色条
        plt.colorbar(im, ax=ax)
        ax.set_title("{} ({})".format(title, period))

        return ax


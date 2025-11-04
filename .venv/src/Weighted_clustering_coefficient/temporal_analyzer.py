# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

class TemporalAnalyzer:
    """时间序列分析类，负责多年度比较和变化率计算"""

    def __init__(self):
        self.hierarchy_results = {}
        self.growth_rates = {}

    def calculate_hierarchy_means(self, clustering_results, hierarchy_manager):
        """
        计算各层级的平均聚类系数

        参数:
        clustering_results: 聚类系数结果字典
        hierarchy_manager: 层级管理器实例

        返回:
        各层级平均聚类系数字典 {year: {hierarchy: mean_value}}
        """
        self.hierarchy_results = {}

        for year, result in clustering_results.items():
            cities = result['cities']
            coefficients = result['coefficients']

            # 获取各层级的统计信息
            stats = hierarchy_manager.get_hierarchy_stats(coefficients, cities)

            # 提取均值
            self.hierarchy_results[year] = {
                hierarchy: stats[hierarchy]['mean']
                for hierarchy in hierarchy_manager.hierarchy_names
            }

        return self.hierarchy_results

    def calculate_growth_rates(self):
        """
        计算年度变化率

        返回:
        年度变化率字典
        """
        if not self.hierarchy_results:
            raise ValueError("请先计算层级均值")

        years = sorted(self.hierarchy_results.keys())
        self.growth_rates = {}

        for i in range(1, len(years)):
            prev_year = years[i-1]
            curr_year = years[i]

            # 计算年度变化率
            growth = {}
            for hierarchy in self.hierarchy_results[prev_year].keys():
                prev_val = self.hierarchy_results[prev_year][hierarchy]
                curr_val = self.hierarchy_results[curr_year][hierarchy]

                if prev_val != 0:
                    growth[hierarchy] = (curr_val - prev_val) / prev_val * 100
                else:
                    growth[hierarchy] = 0

            self.growth_rates[f"{prev_year}-{curr_year}"] = growth

        return self.growth_rates

    def get_trend_analysis(self, hierarchy):
        """
        获取特定层级的趋势分析

        参数:
        hierarchy: 层级名称

        返回:
        趋势分析字典
        """
        if not self.hierarchy_results:
            raise ValueError("请先计算层级均值")

        years = sorted(self.hierarchy_results.keys())
        values = [self.hierarchy_results[year][hierarchy] for year in years]

        # 计算线性趋势
        x = np.arange(len(years))
        mask = ~np.isnan(values)

        if np.sum(mask) > 1:  # 至少需要两个点才能拟合
            slope, intercept = np.polyfit(x[mask], np.array(values)[mask], 1)
            trend = slope * len(years)  # 整个期间的总变化
        else:
            slope, intercept, trend = np.nan, np.nan, np.nan

        return {
            'years': years,
            'values': values,
            'trend_slope': slope,
            'trend_intercept': intercept,
            'overall_trend': trend
        }

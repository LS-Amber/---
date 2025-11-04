# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


class TemporalAnalyzer:
    """时间序列分析类，负责多年度比较和变化率计算"""

    def __init__(self):
        self.results = {}
        self.growth_rates = {}

    def add_year_result(self, year, region_path_matrix):
        """
        添加单年度的区域路径矩阵结果

        参数:
        year: 年份
        region_path_matrix: 区域间路径矩阵
        """
        self.results[year] = region_path_matrix

    def calculate_growth_rates(self):
        """
        计算年度变化率

        返回:
        年度变化率字典
        """
        years = sorted(self.results.keys())
        self.growth_rates = {}

        for i in range(1, len(years)):
            prev_year = years[i - 1]
            curr_year = years[i]

            # 计算年度变化率
            with np.errstate(divide='ignore', invalid='ignore'):
                growth = (self.results[curr_year] - self.results[prev_year]) / self.results[prev_year] * 100

            # 处理无穷大和NaN值
            growth = growth.replace([np.inf, -np.inf], np.nan)
            self.growth_rates["{}-{}".format(prev_year, curr_year)] = growth

        return self.growth_rates

    def get_trend_analysis(self, region_pair):
        """
        获取特定区域对的趋势分析

        参数:
        region_pair: 区域对元组 (source_region, target_region)

        返回:
        趋势分析字典
        """
        source, target = region_pair
        years = sorted(self.results.keys())
        values = [self.results[year].loc[source, target] for year in years]

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
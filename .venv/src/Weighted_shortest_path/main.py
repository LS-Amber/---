# -*- coding: utf-8 -*-
import os

# 设置matplotlib使用非GUI后端
import matplotlib

matplotlib.use('Agg')  # 必须在导入pyplot之前设置
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import numpy as np
import glob
import pandas as pd
from data_processor import DataProcessor
from network_analyzer import NetworkAnalyzer
from region_manager import RegionManager
from temporal_analyzer import TemporalAnalyzer
from visualizer import Visualizer

# 英文区域名称映射
REGION_ENGLISH_NAMES = {
    '东部沿海': 'Eastern Coastal',
    '华中地区': 'Central China',
    '西部地区': 'Western China',
    '东北地区': 'Northeast China',
    '其他': 'Other'
}


def translate_region_names(name):
    """翻译区域名称为英文"""
    return REGION_ENGLISH_NAMES.get(name, name)


def plot_heatmap_en(data, title, ax=None):
    """
    绘制英文版热力图

    参数:
    data: 要绘制的数据矩阵
    title: 图表标题
    ax: matplotlib轴对象（可选）

    返回:
    matplotlib轴对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    mask = np.isnan(data) | (data >= 1e6)  # 掩码NaN和不可达值
    im = ax.imshow(data, cmap='viridis_r', interpolation='nearest')

    # 设置刻度（使用英文区域名）
    ax.set_xticks(range(len(data.columns)))
    ax.set_yticks(range(len(data.index)))
    ax.set_xticklabels([translate_region_names(name) for name in data.columns], rotation=45, ha='right')
    ax.set_yticklabels([translate_region_names(name) for name in data.index])

    # 添加颜色条
    plt.colorbar(im, ax=ax)
    ax.set_title(title)

    return ax


def plot_timeseries_en(years, values, title, ax=None):
    """
    绘制英文版时间序列图

    参数:
    years: 年份列表
    values: 值列表
    title: 图表标题
    ax: matplotlib轴对象（可选）

    返回:
    matplotlib轴对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(years, values, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('Year')
    ax.set_ylabel('Path Length')
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


def plot_growth_rates_en(growth_rates, title, ax=None):
    """
    绘制英文版增长率热力图

    参数:
    growth_rates: 增长率矩阵字典
    title: 图表标题
    ax: matplotlib轴对象（可选）

    返回:
    matplotlib轴对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # 提取第一个时期的增长率作为示例
    period = list(growth_rates.keys())[0]
    data = growth_rates[period]

    im = ax.imshow(data, cmap='RdYlGn', interpolation='nearest')

    # 设置刻度（使用英文区域名）
    ax.set_xticks(range(len(data.columns)))
    ax.set_yticks(range(len(data.index)))
    ax.set_xticklabels([translate_region_names(name) for name in data.columns], rotation=45, ha='right')
    ax.set_yticklabels([translate_region_names(name) for name in data.index])

    # 添加颜色条
    plt.colorbar(im, ax=ax)
    ax.set_title("{} ({})".format(title, period))

    return ax


def main():
    """主函数"""
    # 初始化组件
    data_processor = DataProcessor()
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
    transformed_data = data_processor.transform_weights(panel_data, alpha=1, c=1)

    # 3. 定义区域
    print("定义区域...")
    # 获取所有城市的完整列表
    all_cities = []
    for year, matrix in panel_data.items():
        all_cities.extend(matrix.index.tolist())
    all_cities = sorted(set(all_cities))

    # 这里应根据实际情况定义区域
    region_definitions = {
        '东部沿海': [
            '北京市', '天津市', '石家庄市', '唐山市', '秦皇岛市',
            '邯郸市', '邢台市', '保定市', '张家口市', '承德市',
            '沧州市', '廊坊市', '衡水市', '上海市', '南京市',
            '无锡市', '徐州市', '常州市', '苏州市', '南通市',
            '连云港市', '淮北市', '盐城市', '扬州市', '镇江市',
            '泰州市', '宿迁市', '杭州市', '宁波市', '温州市',
            '嘉兴市', '湖州市', '绍兴市', '金华市', '衢州市',
            '舟山市', '台州市', '丽水市', '福州市', '厦门市',
            '莆田市', '三明市', '泉州市', '漳州市', '南平市',
            '龙岩市', '宁德市', '济南市', '青岛市', '淄博市',
            '枣庄市', '东营市', '烟台市', '潍坊市', '济宁市',
            '泰安市', '威海市', '日照市', '临沂市', '德州市',
            '聊城市', '滨州市', '菏泽市', '广州市', '韶关市',
            '深圳市', '珠海市', '汕头市', '佛山市', '江门市',
            '湛江市', '茂名市', '肇庆市', '惠州市', '梅州市',
            '汕尾市', '河源市', '阳江市', '清远市', '东莞市',
            '中山市', '潮州市', '揭阳市', '云浮市', '海口市',
            '三亚市', '儋州市'
        ],

        '华中地区': [
            '太原市', '大同市', '阳泉市', '长治市', '晋城市',
            '朔州市', '晋中市', '运城市', '忻州市', '临汾市',
            '吕梁市', '合肥市', '芜湖市', '蚌埠市', '淮南市',
            '马鞍山市', '淮安市', '铜陵市', '安庆市', '黄山市',
            '滁州市', '阜阳市', '宿州市', '六安市', '亳州市',
            '池州市', '宣城市', '南昌市', '景德镇市', '萍乡市',
            '九江市', '新余市', '鹰潭市', '赣州市', '吉安市',
            '宜春市', '抚州市', '上饶市', '郑州市', '开封市',
            '洛阳市', '平顶山市', '安阳市', '鹤壁市', '新乡市',
            '焦作市', '濮阳市', '许昌市', '漯河市', '三门峡市',
            '南阳市', '商丘市', '信阳市', '周口市', '驻马店市',
            '武汉市', '黄石市', '十堰市', '宜昌市', '襄阳市',
            '鄂州市', '荆门市', '孝感市', '荆州市', '黄冈市',
            '咸宁市', '随州市', '长沙市', '株洲市', '湘潭市',
            '衡阳市', '邵阳市', '岳阳市', '常德市', '张家界市',
            '益阳市', '郴州市', '永州市', '怀化市', '娄底市'
        ],

        '西部地区': [
            '呼和浩特市', '包头市', '乌海市', '赤峰市', '通辽市',
            '鄂尔多斯市', '呼伦贝尔市', '巴彦淖尔市', '乌兰察布市',
            '南宁市', '柳州市', '桂林市', '梧州市', '北海市',
            '防城港市', '钦州市', '贵港市', '玉林市', '百色市',
            '贺州市', '河池市', '来宾市', '崇左市', '重庆市',
            '成都市', '自贡市', '攀枝花市', '泸州市', '德阳市',
            '绵阳市', '广元市', '遂宁市', '内江市', '乐山市',
            '南充市', '眉山市', '宜宾市', '广安市', '达州市',
            '雅安市', '巴中市', '资阳市', '贵阳市', '六盘水市',
            '遵义市', '安顺市', '毕节市', '铜仁市', '昆明市',
            '曲靖市', '玉溪市', '保山市', '昭通市', '丽江市',
            '普洱市', '临沧市', '拉萨市', '日喀则市', '昌都市',
            '林芝市', '山南市', '那曲市', '西安市', '铜川市',
            '宝鸡市', '咸阳市', '渭南市', '延安市', '汉中市',
            '榆林市', '安康市', '商洛市', '兰州市', '嘉峪关市',
            '金昌市', '白银市', '天水市', '武威市', '张掖市',
            '平凉市', '酒泉市', '庆阳市', '定西市', '陇南市',
            '西宁市', '海东市', '银川市', '石嘴山市', '吴忠市',
            '固原市', '中卫市', '乌鲁木齐市', '克拉玛依市',
            '吐鲁番市', '哈密市'
        ],

        '东北地区': ['沈阳市', '大连市', '鞍山市', '抚顺市', '本溪市',
                     '丹东市', '锦州市', '营口市', '阜新市', '辽阳市',
                     '盘锦市', '铁岭市', '朝阳市', '葫芦岛市', '长春市',
                     '吉林市', '四平市', '辽源市', '通化市', '白山市',
                     '松原市', '白城市', '哈尔滨市', '齐齐哈尔市',
                     '鸡西市', '鹤岗市', '双鸭山市', '大庆市', '伊春市',
                     '佳木斯市', '七台河市', '牡丹江市', '黑河市', '绥化市']
    }

    # 确保所有城市都被分配到某个区域
    unassigned_cities = set(all_cities) - set(city for cities in region_definitions.values() for city in cities)
    if unassigned_cities:
        print("警告: 以下城市未被分配到任何区域: {0}".format(sorted(unassigned_cities)))
        # 可以将未分配的城市添加到"其他"区域
        region_definitions['其他'] = list(unassigned_cities)

    region_manager = RegionManager(region_definitions)

    # 存储每年的最短路径前5名结果
    top_paths_results = {}

    # 4. 逐年分析
    print("进行逐年分析...")
    for year, weight_matrix in transformed_data.items():
        print("处理 {0} 年数据...".format(year))

        # 获取城市列表
        cities = weight_matrix.index.tolist()

        # 构建网络并计算最短路径
        network_analyzer = NetworkAnalyzer(weight_matrix, cities)
        distance_matrix = network_analyzer.compute_shortest_paths()

        # 计算区域间平均路径长度
        region_paths = region_manager.compute_region_path_length(distance_matrix, cities)

        # 添加到时间序列分析器
        temporal_analyzer.add_year_result(year, region_paths)

        # 计算区域间城市节点最短路径的前5名
        top_paths_by_region = {}

        # 创建城市到区域的映射
        city_to_region = region_manager.city_region_map

        # 遍历所有城市对
        path_list = []
        for i, source in enumerate(cities):
            source_region = city_to_region.get(source, 'Unknown')
            for j, target in enumerate(cities):
                if i != j and target in city_to_region:  # 排除自身和未知区域城市
                    target_region = city_to_region.get(target, 'Unknown')
                    distance = distance_matrix.iloc[i, j]

                    # 只考虑可达路径
                    if distance < 1e6:
                        path_list.append({
                            'source': source,
                            'source_region': source_region,
                            'target': target,
                            'target_region': target_region,
                            'distance': distance
                        })

        # 转换为DataFrame便于处理
        path_df = pd.DataFrame(path_list)

        # 按区域对分组并获取前5名
        if not path_df.empty:
            grouped = path_df.groupby(['source_region', 'target_region'])
            for (src_region, tgt_region), group in grouped:
                # 按距离排序并取前5名
                top_paths = group.nsmallest(5, 'distance')

                if (src_region, tgt_region) not in top_paths_by_region:
                    top_paths_by_region[(src_region, tgt_region)] = []

                for _, row in top_paths.iterrows():
                    top_paths_by_region[(src_region, tgt_region)].append({
                        'source': row['source'],
                        'target': row['target'],
                        'distance': row['distance']
                    })

        # 保存结果
        top_paths_results[year] = top_paths_by_region

    # 5. 计算变化率
    print("计算变化率...")
    growth_rates = temporal_analyzer.calculate_growth_rates()

    # 6. 可视化结果
    print("生成可视化结果...")

    # 创建结果目录
    if not os.path.exists('results'):
        os.makedirs('results')

    # 6.1 绘制区域间平均路径长度热力图（2014, 2017, 2020和最新年份）
    years_to_plot = [2014, 2017, 2020]
    latest_year = max(temporal_analyzer.results.keys())
    if latest_year not in years_to_plot:
        years_to_plot.append(latest_year)

    # 确保只选择存在的年份
    available_years = [year for year in years_to_plot if year in temporal_analyzer.results]

    # 中文版
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    for i, year in enumerate(available_years):
        if i < 4:  # 最多绘制4张图
            visualizer.plot_heatmap(
                temporal_analyzer.results[year],
                "区域间平均路径长度 ({0}年)".format(year),
                ax=axes[i]
            )

    # 隐藏多余的子图
    for i in range(len(available_years), 4):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig('results/region_path_length_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 英文版
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    for i, year in enumerate(available_years):
        if i < 4:  # 最多绘制4张图
            plot_heatmap_en(
                temporal_analyzer.results[year],
                "Inter-Regional Average Path Length ({0})".format(year),
                ax=axes[i]
            )

    # 隐藏多余的子图
    for i in range(len(available_years), 4):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig('results/region_path_length_heatmaps_en.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6.2 绘制各区域对之间的路径长度变化趋势图
    region_names = list(region_definitions.keys())

    # 为每个区域绘制与其他区域的路径长度变化图
    for src_region in region_names:
        # 中文版
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        plot_idx = 0

        for tgt_region in region_names:
            if plot_idx >= 4:
                break

            try:
                trend_data = temporal_analyzer.get_trend_analysis((src_region, tgt_region))
                visualizer.plot_timeseries(
                    trend_data['years'],
                    trend_data['values'],
                    "{0}→{1}路径长度变化".format(src_region, tgt_region),
                    ax=axes[plot_idx]
                )
                plot_idx += 1
            except KeyError:
                # 如果区域对不存在数据，则跳过
                continue

        # 隐藏多余的子图
        for i in range(plot_idx, 4):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig('results/trend_{0}_to_all.png'.format(src_region), dpi=300, bbox_inches='tight')
        plt.close()

        # 英文版
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        plot_idx = 0

        for tgt_region in region_names:
            if plot_idx >= 4:
                break

            try:
                trend_data = temporal_analyzer.get_trend_analysis((src_region, tgt_region))
                plot_timeseries_en(
                    trend_data['years'],
                    trend_data['values'],
                    "{0}→{1} Path Length Change".format(
                        translate_region_names(src_region),
                        translate_region_names(tgt_region)
                    ),
                    ax=axes[plot_idx]
                )
                plot_idx += 1
            except KeyError:
                # 如果区域对不存在数据，则跳过
                continue

        # 隐藏多余的子图
        for i in range(plot_idx, 4):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig('results/trend_{0}_to_all_en.png'.format(translate_region_names(src_region).replace(' ', '_')),
                    dpi=300, bbox_inches='tight')
        plt.close()

    # 6.3 绘制增长率热力图（选择特定年份对）
    growth_periods = ['2014-2015', '2017-2018', '2019-2020', '2022-2023']
    available_periods = [period for period in growth_periods if period in growth_rates]

    if available_periods:
        # 中文版
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()

        for i, period in enumerate(available_periods):
            if i < 4:
                # 创建中文版的增长率图表
                im = axes[i].imshow(growth_rates[period], cmap='RdYlGn', interpolation='nearest')

                # 设置刻度
                axes[i].set_xticks(range(len(growth_rates[period].columns)))
                axes[i].set_yticks(range(len(growth_rates[period].index)))
                axes[i].set_xticklabels(growth_rates[period].columns, rotation=45, ha='right')
                axes[i].set_yticklabels(growth_rates[period].index)

                # 添加颜色条
                plt.colorbar(im, ax=axes[i])
                axes[i].set_title("区域间路径长度增长率 ({})".format(period))

        # 隐藏多余的子图
        for i in range(len(available_periods), 4):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig('results/growth_rates_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 英文版
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()

        for i, period in enumerate(available_periods):
            if i < 4:
                plot_growth_rates_en(
                    {period: growth_rates[period]},
                    "Inter-Regional Path Length Growth Rate",
                    ax=axes[i]
                )

        # 隐藏多余的子图
        for i in range(len(available_periods), 4):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig('results/growth_rates_heatmaps_en.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 6.4 绘制原有的综合分析图
    # 中文版
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 最新年份的热力图
    latest_year = max(temporal_analyzer.results.keys())
    visualizer.plot_heatmap(
        temporal_analyzer.results[latest_year],
        "区域间平均路径长度 ({0}年)".format(latest_year),
        ax=axes[0, 0]
    )

    # 东部到西部的时间序列
    trend_data = temporal_analyzer.get_trend_analysis(('东部沿海', '华中地区'))
    visualizer.plot_timeseries(
        trend_data['years'],
        trend_data['values'],
        "东部沿海→华中地区路径长度变化",
        ax=axes[0, 1]
    )

    # 增长率热力图（使用第一个可用时期）
    if growth_rates:
        visualizer.plot_growth_rates(
            growth_rates,
            "区域间路径长度年增长率",
            ax=axes[1, 0]
        )
    else:
        axes[1, 0].text(0.5, 0.5, "无增长率数据", ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title("区域间路径长度年增长率")

    # 区域统计信息
    region_stats = region_manager.get_region_stats(
        network_analyzer.compute_shortest_paths(),
        cities
    )

    # 创建区域统计条形图
    regions = list(region_stats.keys())
    intra_means = [region_stats[r]['intra_region_mean'] for r in regions]
    inter_means = [region_stats[r]['inter_region_mean'] for r in regions]

    x = np.arange(len(regions))
    width = 0.35

    axes[1, 1].bar(x - width / 2, intra_means, width, label='区域内平均路径')
    axes[1, 1].bar(x + width / 2, inter_means, width, label='区域间平均路径')
    axes[1, 1].set_xlabel('区域')
    axes[1, 1].set_ylabel('平均路径长度')
    axes[1, 1].set_title('区域内与区域间平均路径长度比较')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(regions)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('results/regional_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()  # 使用close而不是show，因为我们使用的是Agg后端

    # 英文版
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 最新年份的热力图
    latest_year = max(temporal_analyzer.results.keys())
    plot_heatmap_en(
        temporal_analyzer.results[latest_year],
        "Inter-Regional Average Path Length ({0})".format(latest_year),
        ax=axes[0, 0]
    )

    # 东部到西部的时间序列
    trend_data = temporal_analyzer.get_trend_analysis(('东部沿海', '华中地区'))
    plot_timeseries_en(
        trend_data['years'],
        trend_data['values'],
        "{0}→{1} Path Length Change".format(
            translate_region_names('东部沿海'),
            translate_region_names('华中地区')
        ),
        ax=axes[0, 1]
    )

    # 增长率热力图（使用第一个可用时期）
    if growth_rates:
        plot_growth_rates_en(
            growth_rates,
            "Inter-Regional Path Length Annual Growth Rate",
            ax=axes[1, 0]
        )
    else:
        axes[1, 0].text(0.5, 0.5, "No Growth Rate Data", ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title("Inter-Regional Path Length Annual Growth Rate")

    # 区域统计信息
    region_stats = region_manager.get_region_stats(
        network_analyzer.compute_shortest_paths(),
        cities
    )

    # 创建区域统计条形图
    regions = list(region_stats.keys())
    intra_means = [region_stats[r]['intra_region_mean'] for r in regions]
    inter_means = [region_stats[r]['inter_region_mean'] for r in regions]
    regions_en = [translate_region_names(r) for r in regions]

    x = np.arange(len(regions))
    width = 0.35

    axes[1, 1].bar(x - width / 2, intra_means, width, label='Intra-Regional Average Path')
    axes[1, 1].bar(x + width / 2, inter_means, width, label='Inter-Regional Average Path')
    axes[1, 1].set_xlabel('Region')
    axes[1, 1].set_ylabel('Average Path Length')
    axes[1, 1].set_title('Comparison of Intra- and Inter-Regional Average Path Length')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(regions_en)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('results/regional_analysis_en.png', dpi=300, bbox_inches='tight')
    plt.close()  # 使用close而不是show，因为我们使用的是Agg后端

    # 7. 输出统计结果
    print("\n=== 描述性统计 ===")
    orig_stats = data_processor.get_descriptive_stats('original')
    trans_stats = data_processor.get_descriptive_stats('transformed')

    print("\n原始搜索指数统计:")
    for year, stats in orig_stats.items():
        print("{0}: 均值={1:.2f}, 标准差={2:.2f}".format(year, stats['mean'], stats['std']))

    print("\n转换后权重统计:")
    for year, stats in trans_stats.items():
        print("{0}: 均值={1:.2f}, 标准差={2:.2f}".format(year, stats['mean'], stats['std']))

    print("\n=== 区域间平均路径长度 ===")
    for year, matrix in temporal_analyzer.results.items():
        print("\n{0}年:".format(year))
        print(matrix.round(2))

    print("\n=== 年度增长率 (%) ===")
    for period, growth in growth_rates.items():
        print("\n{0}:".format(period))
        print(growth.round(2))

    # 8. 保存结果到CSV文件
    print("\n保存结果到CSV文件...")
    # 保存区域间路径长度
    for year, matrix in temporal_analyzer.results.items():
        matrix.to_csv('results/regional_paths_{0}.csv'.format(year))

    # 保存增长率
    for period, growth in growth_rates.items():
        # 创建英文版的列名和行名
        en_growth = growth.copy()
        en_growth.columns = [translate_region_names(name) for name in growth.columns]
        en_growth.index = [translate_region_names(name) for name in growth.index]
        en_growth.to_csv('results/growth_rates_{0}_en.csv'.format(period))

        # 保存原始中文版
        growth.to_csv('results/growth_rates_{0}.csv'.format(period))

    # 保存每年的区域间城市节点最短路径前5名结果
    for year, top_paths in top_paths_results.items():
        # 创建一个列表来存储所有数据
        all_top_paths = []
        all_top_paths_en = []

        for (src_region, tgt_region), paths in top_paths.items():
            for i, path in enumerate(paths, 1):
                # 中文版
                all_top_paths.append({
                    'source_region': src_region,
                    'target_region': tgt_region,
                    'rank': i,
                    'source_city': path['source'],
                    'target_city': path['target'],
                    'distance': path['distance']
                })

                # 英文版
                all_top_paths_en.append({
                    'source_region': translate_region_names(src_region),
                    'target_region': translate_region_names(tgt_region),
                    'rank': i,
                    'source_city': path['source'],
                    'target_city': path['target'],
                    'distance': path['distance']
                })

        # 转换为DataFrame并保存
        if all_top_paths:
            df = pd.DataFrame(all_top_paths)
            df.to_csv('results/top_paths_{}.csv'.format(year), index=False, encoding='utf-8-sig')

            # 保存英文版
            df_en = pd.DataFrame(all_top_paths_en)
            df_en.to_csv('results/top_paths_{}_en.csv'.format(year), index=False, encoding='utf-8-sig')

    print("分析完成! 结果已保存到 results/ 目录")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import pandas as pd
import os


def add_coordinates_to_csv(csv_file, coordinates_file, output_file=None):
    """
    为CSV文件中的Source和Target城市添加经纬度坐标

    参数:
    csv_file: 输入的CSV文件路径
    coordinates_file: 包含城市经纬度的CSV文件路径
    output_file: 输出文件路径(可选)
    """

    # 读取经纬度数据
    try:
        try:
            coords_df = pd.read_csv(coordinates_file, encoding='utf-8')
            print(u"使用UTF-8编码读取经纬度文件成功")
        except UnicodeDecodeError:
            print(u"UTF-8编码读取失败，尝试使用GBK编码...")
            coords_df = pd.read_csv(coordinates_file, encoding='gbk')
            print(u"使用GBK编码读取经纬度文件成功")
        # 构建完整的城市经纬度字典
        coords_dict = dict(zip(coords_df['Label'], zip(coords_df['Longtitude'], coords_df['Latitude'])))
        print(u"成功加载 {0} 个城市的经纬度数据".format(len(coords_dict)))
        print(u"城市示例: {0}...".format(list(coords_dict.keys())[:10]))  # 显示前10个城市作为示例
    except Exception as e:
        print(u"读取经纬度文件失败: {0}".format(str(e)))
        return

    try:
        print(u"\n正在处理文件: {0}".format(os.path.basename(csv_file)))

        # 读取CSV文件（先尝试UTF-8，失败则使用GBK）
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            print(u"使用UTF-8编码读取主文件成功")
        except UnicodeDecodeError:
            print(u"UTF-8编码读取失败，尝试使用GBK编码...")
            df = pd.read_csv(csv_file, encoding='gbk')
            print(u"使用GBK编码读取主文件成功")

        print(u"原始数据行数: {0}".format(len(df)))

        # 添加经纬度列
        df = add_coordinates_to_dataframe(df, coords_dict)

        # 生成输出文件名
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(csv_file))[0]
            output_file = u"{0}_with_coordinates.csv".format(base_name)

        # 保存结果
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(u"已保存: {0}".format(os.path.basename(output_file)))
        print(u"处理后数据行数: {0}".format(len(df)))

    except Exception as e:
        print(u"处理文件 {0} 时出错: {1}".format(csv_file, str(e)))
        import traceback
        traceback.print_exc()


def add_coordinates_to_dataframe(df, coords_dict):
    """
    为DataFrame添加经纬度列
    """
    # 检查数据框结构
    print(u"数据框列: {0}".format(list(df.columns)))

    # 为Source城市添加经纬度
    source_coords = df['Source'].map(lambda city: coords_dict.get(city, (None, None)))
    df['Source_Longitude'] = source_coords.map(lambda x: x[0] if x else None)
    df['Source_Latitude'] = source_coords.map(lambda x: x[1] if x else None)

    # 为Target城市添加经纬度
    target_coords = df['Target'].map(lambda city: coords_dict.get(city, (None, None)))
    df['Target_Longitude'] = target_coords.map(lambda x: x[0] if x else None)
    df['Target_Latitude'] = target_coords.map(lambda x: x[1] if x else None)

    # 检查缺失的坐标
    missing_source = df[df['Source_Longitude'].isna()]['Source'].unique()
    missing_target = df[df['Target_Longitude'].isna()]['Target'].unique()

    if len(missing_source) > 0:
        print(u"警告: 以下 {0} 个Source城市缺少坐标: {1}".format(len(missing_source), list(missing_source)))
    else:
        print(u"所有Source城市都有对应的坐标")

    if len(missing_target) > 0:
        print(u"警告: 以下 {0} 个Target城市缺少坐标: {1}".format(len(missing_target), list(missing_target)))
    else:
        print(u"所有Target城市都有对应的坐标")

    # 重新排列列的顺序，使经纬度列紧跟在城市列后面
    columns_order = []
    for col in df.columns:
        columns_order.append(col)
        if col == 'Source':
            columns_order.extend(['Source_Longitude', 'Source_Latitude'])
        elif col == 'Target':
            columns_order.extend(['Target_Longitude', 'Target_Latitude'])

    # 移除重复的列
    seen = set()
    unique_columns = []
    for col in columns_order:
        if col not in seen:
            unique_columns.append(col)
            seen.add(col)

    df = df[unique_columns]

    return df


# 使用示例
if __name__ == "__main__":
    # 处理单个CSV文件
    csv_file = "C:/Users/hp/Desktop/filter/filter_5%_2023.csv"
    coordinates_file = "C:/Users/hp/Desktop/省shp/省shp/经纬度备用.csv"
    output_file = "C:/Users/hp/Desktop/filter/output_filter_2023.csv"

    add_coordinates_to_csv(csv_file, coordinates_file, output_file)

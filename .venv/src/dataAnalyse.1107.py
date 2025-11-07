# coding=utf-8
import pandas as pd
import os
from collections import defaultdict

# 城市分类定义
MUNICIPALITIES = ["北京市", "天津市", "上海市", "重庆市"]  # 直辖市

SUB_PROVINCIAL_CITIES = [  # 副省级城市
    "沈阳市", "大连市", "长春市", "哈尔滨市", "南京市", "杭州市", "宁波市", "厦门市",
    "济南市", "青岛市", "武汉市", "广州市", "深圳市", "成都市", "西安市"
]

PROVINCIAL_CAPITALS = [  # 省会级城市
    "石家庄市", "太原市", "呼和浩特市", "合肥市", "福州市", "南昌市", "郑州市",
    "长沙市", "南宁市", "海口市", "贵阳市", "昆明市", "拉萨市", "兰州市",
    "西宁市", "银川市", "乌鲁木齐市"
]


def load_city_coordinates(city_dict_file):
    """
    从城市字典CSV文件加载城市经纬度信息

    参数:
    city_dict_file: 城市字典CSV文件路径

    返回:
    city_coords: 城市经纬度字典 {城市名: (经度, 纬度)}
    """
    try:
        city_df = pd.read_csv(city_dict_file, encoding="gbk")
        # 假设CSV文件有列: Id, Label, Latitude, Longtitude
        city_coords = dict(zip(city_df['Label'], zip(city_df['Longtitude'], city_df['Latitude'])))
        print("成功加载 {} 个城市的经纬度数据".format(len(city_coords)))
        return city_coords
    except Exception as e:
        print("读取城市字典文件失败: {}".format(e))
        return {}


def classify_city(city_name):
    """
    根据城市名称分类

    参数:
    city_name: 城市名称

    返回:
    category: 城市类别
    """
    if city_name in MUNICIPALITIES:
        return "直辖市"
    elif city_name in SUB_PROVINCIAL_CITIES:
        return "副省级城市"
    elif city_name in PROVINCIAL_CAPITALS:
        return "省会级城市"
    else:
        return "地级市"


def calculate_city_degrees_from_csv(city_coords, csv_files, output_file="city_degrees_total.csv"):
    """
    从多个CSV文件计算城市的出入度权重之和

    参数:
    city_coords: 城市经纬度字典
    csv_files: CSV文件路径列表
    output_file: 输出文件路径
    """

    # 初始化字典来存储城市出入度权重
    city_out_degree = defaultdict(float)  # 出度权重和 (城市作为source)
    city_in_degree = defaultdict(float)  # 入度权重和 (城市作为target)

    # 处理每个CSV文件
    for i, csv_file in enumerate(csv_files, 1):
        try:
            print("正在处理文件 {}/{}: {}".format(i, len(csv_files), os.path.basename(csv_file)))

            # 读取CSV文件
            df = pd.read_csv(csv_file,encoding="gbk")

            # 检查必要的列是否存在
            required_columns = ['Source', 'Target', 'Weight']
            if not all(col in df.columns for col in required_columns):
                print("警告: 文件 {} 缺少必要的列，跳过处理".format(csv_file))
                continue

            # 计算每个城市的出度权重和 (作为source)
            out_degree = df.groupby('Source')['Weight'].sum()
            for city, weight in out_degree.items():
                if city in city_coords:  # 只统计城市字典中的城市
                    city_out_degree[city] += weight

            # 计算每个城市的入度权重和 (作为target)
            in_degree = df.groupby('Target')['Weight'].sum()
            for city, weight in in_degree.items():
                if city in city_coords:  # 只统计城市字典中的城市
                    city_in_degree[city] += weight

        except Exception as e:
            print("处理文件 {} 时出错: {}".format(csv_file, e))

    # 创建结果DataFrame
    result_data = []
    for city in city_coords.keys():
        out_weight = city_out_degree.get(city, 0)
        in_weight = city_in_degree.get(city, 0)
        total_weight = out_weight + in_weight
        category = classify_city(city)

        result_data.append({
            'City': city,
            'Category': category,
            'Out_Degree_Weight': out_weight,
            'In_Degree_Weight': in_weight,
            'Total_Weight': total_weight
        })

    # 创建DataFrame并按总权重倒序排列
    result_df = pd.DataFrame(result_data)

    # 按照城市类别和总权重排序
    # 首先定义类别的排序顺序
    category_order = {"直辖市": 0, "副省级城市": 1, "省会级城市": 2, "地级市": 3}
    result_df['Category_Order'] = result_df['Category'].map(category_order)
    result_df = result_df.sort_values(['Category_Order', 'Total_Weight'], ascending=[True, False])
    result_df = result_df.drop('Category_Order', axis=1)

    # 重置索引
    result_df.reset_index(drop=True, inplace=True)

    # 打印各类别的前5个城市
    print("\n各类别城市权重统计 (前5名):")
    for category in ["直辖市", "副省级城市", "省会级城市", "地级市"]:
        category_df = result_df[result_df['Category'] == category].head(5)
        print("\n{}:".format(category))
        print(category_df[['City', 'Total_Weight']].to_string(index=False))

    # 保存结果
    result_df.to_csv(output_file, index=False)
    print("\n结果已保存到: {}".format(output_file))

    # 打印一些统计信息
    print("\n统计信息:")
    print("总城市数量: {}".format(len(result_df)))
    for category in ["直辖市", "副省级城市", "省会级城市", "地级市"]:
        count = len(result_df[result_df['Category'] == category])
        print("{}数量: {}".format(category, count))

    if len(result_df) > 0:
        print("总权重最高的城市: {} (总权重: {:.2f})".format(result_df.iloc[0]['City'], result_df.iloc[0]['Total_Weight']))
        print("总权重最低的城市: {} (总权重: {:.2f})".format(result_df.iloc[-1]['City'], result_df.iloc[-1]['Total_Weight']))
        print("平均总权重: {:.2f}".format(result_df['Total_Weight'].mean()))

    return result_df


def main():
    """
    主函数 - 处理城市字典和10个CSV文件
    """
    # 加载城市字典
    city_dict_file = "C:/Users/hp/Desktop/省shp/省shp/经纬度备用.csv"
    city_coords = load_city_coordinates(city_dict_file)

    if not city_coords:
        print("无法加载城市字典，程序终止")
        return

    # 10个CSV文件的路径
    csv_files = [
        "C:/Users/hp/Desktop/filter/output_filter_2014.csv",
        "C:/Users/hp/Desktop/filter/output_filter_2015.csv",
        "C:/Users/hp/Desktop/filter/output_filter_2016.csv",
        "C:/Users/hp/Desktop/filter/output_filter_2017.csv",
        "C:/Users/hp/Desktop/filter/output_filter_2018.csv",
        "C:/Users/hp/Desktop/filter/output_filter_2019.csv",
        "C:/Users/hp/Desktop/filter/output_filter_2020.csv",
        "C:/Users/hp/Desktop/filter/output_filter_2021.csv",
        "C:/Users/hp/Desktop/filter/output_filter_2022.csv",
        "C:/Users/hp/Desktop/filter/output_filter_2023.csv"
    ]

    # 检查文件是否存在
    existing_files = []
    for file in csv_files:
        if os.path.exists(file):
            existing_files.append(file)
        else:
            print("警告: 文件不存在: {}".format(file))

    if not existing_files:
        print("没有找到任何CSV文件，请检查文件路径")
        return

    print("找到 {} 个CSV文件".format(len(existing_files)))

    # 计算城市度值
    result_df = calculate_city_degrees_from_csv(city_coords, existing_files, "city_degrees_total.csv")

    # 可选：添加经纬度信息到结果
    result_with_coords = result_df.copy()
    result_with_coords['Longitude'] = result_with_coords['City'].map(
        lambda city: city_coords.get(city, (None, None))[0])
    result_with_coords['Latitude'] = result_with_coords['City'].map(lambda city: city_coords.get(city, (None, None))[1])

    # 保存包含经纬度的结果
    result_with_coords.to_csv("city_degrees_with_coordinates.csv", index=False)
    print("包含经纬度的结果已保存到: city_degrees_with_coordinates.csv")


if __name__ == "__main__":
    main()

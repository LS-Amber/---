# -*- coding: utf-8 -*-
import pandas as pd
import os

# 定义直辖市和省会城市列表
municipalities_and_capitals = [
    # 直辖市
    '北京市', '天津市', '上海市', '重庆市',
    # 省会城市
    '石家庄市', '太原市', '呼和浩特市', '沈阳市', '长春市', '哈尔滨市',
    '南京市', '杭州市', '合肥市', '福州市', '南昌市', '济南市',
    '郑州市', '武汉市', '长沙市', '广州市', '南宁市', '海口市',
    '成都市', '贵阳市', '昆明市', '拉萨市', '西安市', '兰州市',
    '西宁市', '银川市', '乌鲁木齐市'
]


def process_data(input_file, output_file, description):
    """
    处理数据：筛选直辖市和省会城市并保存为CSV
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_file):
            print(u"错误: 输入文件 {} 不存在".format(input_file))
            return None
            
        # 读取Excel文件
        df = pd.read_excel(input_file, sheet_name='Sheet1')
        print(u"成功读取 {}".format(input_file))

        # 筛选直辖市和省会城市
        filtered_df = df[df['城市/年份'].isin(municipalities_and_capitals)]

        # 保存为CSV文件
        filtered_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(u"成功保存 {} 数据到 {}".format(description, output_file))
        print(u"筛选出的城市数量: {}".format(len(filtered_df)))

        # 显示筛选出的城市
        if len(filtered_df) > 0:
            cities = u", ".join(filtered_df['城市/年份'].tolist())
            print(u"筛选出的城市: {}".format(cities))
        else:
            print(u"未找到匹配的城市")

        return filtered_df

    except FileNotFoundError:
        print(u"错误: 无法找到文件 {}".format(input_file))
        return None
    except Exception as e:
        print(u"处理 {} 时出错: {}".format(input_file, str(e)))
        return None


def main():
    # 文件路径
    in_degree_file = 'C:/Users/hp/Desktop/10年入度.xlsx'  # 入度数据文件
    out_degree_file = 'C:/Users/hp/Desktop/10年出度.xlsx'  # 出度数据文件

    # 输出文件
    in_degree_output = '直辖市省会入度数据.csv'
    out_degree_output = '直辖市省会出度数据.csv'

    print(u"开始处理数据...")
    print("=" * 50)

    # 处理入度数据
    print(u"处理入度数据:")
    in_degree_data = process_data(in_degree_file, in_degree_output, u"入度")

    print("\n" + "=" * 50)

    # 处理出度数据
    print(u"处理出度数据:")
    out_degree_data = process_data(out_degree_file, out_degree_output, u"出度")

    print("\n" + "=" * 50)

    # 汇总信息
    if in_degree_data is not None and out_degree_data is not None:
        print(u"数据处理完成!")
        print(u"入度数据文件: {}".format(in_degree_output))
        print(u"出度数据文件: {}".format(out_degree_output))
        print(u"总共处理了 {} 个直辖市和省会城市".format(len(municipalities_and_capitals)))
    else:
        print(u"数据处理过程中出现错误，请检查文件路径和数据格式")


if __name__ == "__main__":
    main()
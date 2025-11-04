# -*- coding: utf-8 -*-
from __future__ import print_function
import pandas as pd
import os
import sys


def extract_top_edges(input_path, output_path="top15_edges_per_city.csv", top_n=15):
    """
    从城市边权表中提取每个城市（source或target）联系最强的前 top_n 条边。

    参数：
    ----------
    input_path : str
        输入 CSV 文件路径，必须包含列名 ['Source', 'Target', 'Weight']。
    output_path : str, 可选
        输出文件路径（默认：top15_edges_per_city.csv）。
    top_n : int, 可选
        每个城市保留的最大边数（默认 15）。
    """

    try:
        # === 1. 检查输入文件是否存在 ===
        if not os.path.exists(input_path):
            raise FileNotFoundError("未找到输入文件：{}".format(input_path))

        # === 2. 读取数据 ===
        try:
            df = pd.read_csv(input_path, encoding="utf-8")
            print(" 使用UTF-8编码读取文件成功")
        except UnicodeDecodeError:
            print(" UTF-8编码读取失败，尝试使用GBK编码...")
            df = pd.read_csv(input_path, encoding="gbk")
            print(" 使用GBK编码读取文件成功")

        expected_cols = {"Source", "Target", "Weight"}
        if not expected_cols.issubset(df.columns):
            raise ValueError("CSV 文件必须包含列名：{}，实际为：{}".format(expected_cols, set(df.columns)))

        print(" 成功读取数据，共 {} 行。".format(len(df)))

        # === 3. 构建无向边表（因为 source/target 都要算） ===
        df_rev = df.rename(columns={"Source": "Target", "Target": "Source"})
        df_all = pd.concat([df, df_rev], ignore_index=True)

        print(" 已扩展为无向边表，共 {} 行。".format(len(df_all)))

        # === 4. 提取每个城市最强的 top_n 条边 ===
        top_edges = (
            df_all.sort_values("Weight", ascending=False)
            .groupby("Source")
            .head(top_n)
            .reset_index(drop=True)
        )

        # === 5. 去重（避免双向重复边） ===
        top_edges["EdgeKey"] = top_edges.apply(
            lambda x: tuple(sorted([x["Source"], x["Target"]])), axis=1
        )
        top_edges = top_edges.drop_duplicates(subset="EdgeKey").drop(columns="EdgeKey")

        # === 6. 保存结果 ===
        top_edges.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(" 处理完成！共保留 {} 条边。".format(len(top_edges)))
        print(" 输出文件路径：{}".format(os.path.abspath(output_path)))

    except Exception as e:
        print(" 运行出错：", e)
        sys.exit(1)


# === 主程序入口 ===
if __name__ == "__main__":
    # 改变io路径遍历10年数据进行清洗
    input_file = "C:/Users/hp/Desktop/Complex_Network_Analyse/2014-2023_db_csv/2023.csv"
    output_file = "C:/Users/hp/Desktop/filter/filter_5%_2023.csv"
    extract_top_edges(input_file, output_file, top_n=15)

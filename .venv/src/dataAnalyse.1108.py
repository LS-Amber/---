# coding=utf-8
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 在绘图之前设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'FangSong', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题


def plot_degree_rank(csv_path, output_path="degree_rank.png"):
    """
    绘制城市入度值与出度值的位序分布图。

    参数：
    ----------
    csv_path : str
        输入CSV文件路径，需包含至少 ['City', 'In_Degree', 'Out_Degree'] 三列。
    output_path : str, 可选
        输出图像文件路径，默认 'degree_rank.png'。
    """

    # === 1. 读取数据 ===
    if not os.path.exists(csv_path):
        raise FileNotFoundError("未找到输入文件：{}".format(csv_path))

    df = pd.read_csv(csv_path, encoding="gbk")

    # === 2. 检查列名 ===
    required_cols = {"In_Degree", "Out_Degree"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV文件必须包含列名：{}".format(required_cols))

    # === 3. 排序（按度值从大到小） ===
    df_in = df.sort_values("In_Degree", ascending=False).reset_index(drop=True)
    df_out = df.sort_values("Out_Degree", ascending=False).reset_index(drop=True)

    # === 4. 生成位序 ===
    df_in["Rank"] = df_in.index + 1
    df_out["Rank"] = df_out.index + 1

    # === 5. 绘图 ===
    plt.figure(figsize=(10, 5))
    plt.plot(df_in["Rank"], df_in["In_Degree"], color="dodgerblue", linewidth=1.5, label="入度值")
    plt.plot(df_out["Rank"], df_out["Out_Degree"], color="orangered", linewidth=1.5, label="出度值")

    plt.xlabel("城市位序", fontsize=12)
    plt.ylabel("度值", fontsize=12)
    plt.title("信息流动网络的入度与出度位序分布", fontsize=13, fontweight="bold")
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    # 设置坐标轴范围（自动适配）
    plt.xlim(1, max(len(df_in), len(df_out)))
    plt.ylim(0, max(df_in["In_Degree"].max(), df_out["Out_Degree"].max()) * 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

    print("✅ 图像已保存至：{}".format(os.path.abspath(output_path)))


# =======================
# 主程序入口
# =======================
if __name__ == "__main__":
    input_csv = "C:/Users/hp/Desktop/度值分析（新）/出入度导入数据.csv"
    output_img = "C:/Users/hp/Desktop/度值分析（新）/degree_rank.png"
    plot_degree_rank(input_csv, output_img)

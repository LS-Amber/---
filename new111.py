import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
import math

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False

# 创建有向图
G = nx.DiGraph()
G.add_edge("节点1", "节点2", weight=7)
G.add_edge("节点2", "节点1", weight=4)
G.add_edge("节点2", "节点3", weight=3)
G.add_edge("节点3", "节点1", weight=2)

# 节点布局
pos = nx.spring_layout(G, seed=42)

fig, ax = plt.subplots(figsize=(6,6))

# 画节点
nx.draw_networkx_nodes(G, pos, node_size=2000, node_color="lightblue", ax=ax)
nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)

# 边偏移量，正负值用于双向平行显示
offset_map = {
    ("节点1","节点2"):  0.08,
    ("节点2","节点1"): -0.08,
    ("节点2","节点3"):  0.08,
    ("节点3","节点2"): -0.08,
    ("节点3","节点1"):  0.06,
    ("节点1","节点3"): -0.06
}

# 边绘制函数
def offset_endpoints(p1, p2, offset):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        return p1, p2
    ux = -dy / dist
    uy = dx / dist
    return (x1 + ux*offset, y1 + uy*offset), (x2 + ux*offset, y2 + uy*offset)

# 画每条边（带箭头和偏移）
for u, v, w in G.edges(data='weight'):
    p1, p2 = offset_endpoints(pos[u], pos[v], offset_map.get((u,v), 0))
    arrow = FancyArrowPatch(posA=p1, posB=p2,
                            arrowstyle='-|>', mutation_scale=18,
                            linewidth=1.4, color='black',
                            shrinkA=12, shrinkB=12)
    ax.add_patch(arrow)
    # 权重文字
    mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
    ax.text(mx, my, str(w), fontsize=12, ha='center', va='center',
            bbox=dict(facecolor='white', edgecolor='none', pad=0.2, alpha=0.8))

ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
plt.show()

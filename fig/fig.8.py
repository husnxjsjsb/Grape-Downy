import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 
from matplotlib.ticker  import MultipleLocator

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 加载数据
csv_file = "./miou_out/model.csv"
df = pd.read_csv(csv_file)
df.set_index("Model", inplace=True)

# 颜色方案
palette = sns.color_palette("Set2")
colors = palette[:4]

# 创建图形并调整间距
fig = plt.figure(figsize=(6.5, 7))
gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1], hspace=0.05)  # 减小子图间距
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2], sharex=ax1)

# 调整刻度位置
tick_padding = 0.2

# 第一个子图：柱状图
bar_width = 0.15
index = np.arange(len(df))
metrics = ["mPA", "IoU (Background)", "IoU (Leaf)", "mIoU"]

# 只绘制水平网格线
ax1.grid(True, axis='y', linestyle='-', linewidth=0.5, zorder=0.5)
for i, metric in enumerate(metrics):
    ax1.bar(index + i * bar_width, df[metric], width=bar_width, 
            label=metric, color=colors[i], zorder=2)

# 添加垂直线
for x in index:
    ax1.axvline(x + tick_padding, color='gray', linestyle='-', linewidth=0.5, zorder=1)

# 设置第一个子图
ax1.set_ylabel("Percentage (%)")
ax1.set_xticks([])
ax1.set_xticklabels([])
ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1), 
           ncol=len(metrics), fontsize='small', 
           prop={'family':'Times New Roman'})
ax1.set_ylim(90, 100)
ax1.yaxis.set_tick_params(labelsize=9)

# 第二个子图：参数量
ax2.grid(True,  axis='y', linestyle='-', linewidth=0.5, zorder=0.5)
ax2.plot(index  + tick_padding, df["Params (M)"], marker="o", 
         label="Params (M)", color="red", zorder=2)  

# 添加垂直线
for x in index:
    ax2.axvline(x + tick_padding, color='gray', linestyle='-', linewidth=0.5, zorder=1)

ax2.set_ylabel("Params  (M)")
ax2.set_xticks([]) 
ax2.set_xticklabels([]) 
ax2.tick_params(axis='y',  labelsize=9)
ax2.yaxis.set_major_locator(MultipleLocator(10)) 
 
# 第三个子图：推理时间 
ax3.grid(True,  axis='y', linestyle='-', linewidth=0.5, zorder=0.5)
ax3.plot(index  + tick_padding, df["Inference Time (ms)"], marker="o", 
         label="Inference Time (ms)", color="#1f77b4", zorder=2)  

# 添加垂直线
for x in index:
    ax3.axvline(x + tick_padding, color='gray', linestyle='-', linewidth=0.5, zorder=1)

ax3.set_ylabel("Inference  Time (ms)")
ax3.set_xticks(index  + tick_padding)
ax3.set_xticklabels(df.index,  ha='center', fontstyle='normal')
ax3.tick_params(axis='y',  labelsize=9)

# 移除共享x轴的设置
ax1.xaxis.set_visible(False)
ax2.xaxis.set_visible(False)

# 最终调整
plt.tight_layout() 
plt.subplots_adjust(top=0.92,  hspace=0.05)  # 进一步减小子图间距 
 
# 保存为 300 DPI 的 PNG 文件 
plt.savefig("leaf_model.png",  dpi=300, bbox_inches='tight')  # 保存为高分辨率 PNG 
plt.show()

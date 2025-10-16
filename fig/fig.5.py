import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def plot_model_comparison_bar():
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})

    file = r'C:\model\deeplabv3-plus-pytorch-main2\miou_out\result_back.csv'
    df = pd.read_csv(file)
    models = df['Model'].tolist()
    x = np.arange(len(models))

    # 指标数据
    metrics = {
        'mPA': df['mPA'].tolist(),
        'IoU-Background': df['IoU-Background'].tolist(),
        'IoU-Leaf': df['IoU-Leaf'].tolist(),
        'mIoU': df['mIoU'].tolist()
    }

    params = df['Params (M)'].tolist()
    inference_time = df['Inference Time (ms)'].tolist()

    # ✅ 使用 seaborn Set2 调色板（取前4 + 第7、8）
    palette = sns.color_palette("Set2", 8)
    color_dict = {
        'mPA': palette[0],
        'IoU-Background': palette[1],
        'IoU-Leaf': palette[2],
        'mIoU': palette[3],
        'Params': palette[6],
        'Inference Time': palette[7]
    }
    width = 0.12
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # 左侧主指标柱状图
    for i, (label, values) in enumerate(metrics.items()):
        offset = (i - 1.5) * width
        # 先绘制柱子
        bars = ax1.bar(x + offset, values, width=width, label=label, color=color_dict[label])
        
        # 在柱子顶部添加文本标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            # 将文本放在柱子顶部上方0.1处
            ax1.text(bar.get_x() + bar.get_width()/2., 
                     height + 0.1, 
                     f'{value:.2f}', 
                     ha='center', 
                     va='bottom', 
                     fontsize=8, 
                     color='black')

    ax1.set_ylabel('mPA / IoU (Background) / IoU (Leaf) / mIoU (%)')
    ax1.set_ylim(90, 100)  # 增加一点空间给文本标签
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Params 柱状图（右轴1）
    ax2 = ax1.twinx()
    bars_params = []
    for j in range(len(x)):
        offset = 2.5 * width
        # 先绘制柱子
        bar = ax2.bar(x[j] + offset, params[j], width=width,
                      color=color_dict['Params'], label='Params (M)' if j == 0 else "")
        bars_params.append(bar)
    
    # 在柱子顶部添加文本标签
    for bar, param_val in zip(bars_params, params):
        height = bar[0].get_height()
        ax2.text(bar[0].get_x() + bar[0].get_width()/2., 
                 height + 0.1, 
                 f'{param_val:.2f}', 
                 ha='center', 
                 va='bottom', 
                 fontsize=8, 
                 color='black')
    
    ax2.set_ylabel('Params (M)', color=color_dict['Params'])
    ax2.tick_params(axis='y', labelcolor=color_dict['Params'])
    ax2.set_ylim(0, 15)  # 增加一点空间给文本标签
    ax2.spines["right"].set_edgecolor(color_dict['Params'])

    # Inference Time 柱状图（右轴2）
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))
    bars_time = []
    for j in range(len(x)):
        offset = 3.5 * width
        # 先绘制柱子
        bar = ax3.bar(x[j] + offset, inference_time[j], width=width,
                      color=color_dict['Inference Time'], label='Inference Time (ms)' if j == 0 else "")
        bars_time.append(bar)
    
    # 在柱子顶部添加文本标签
    for bar, time_val in zip(bars_time, inference_time):
        height = bar[0].get_height()
        ax3.text(bar[0].get_x() + bar[0].get_width()/2., 
                 height + 0.1, 
                 f'{time_val:.2f}', 
                 ha='center', 
                 va='bottom', 
                 fontsize=8, 
                 color='black')
    
    ax3.set_ylabel('Inference Time (ms)', color=color_dict['Inference Time'])
    ax3.tick_params(axis='y', labelcolor=color_dict['Inference Time'])
    ax3.set_ylim(5, 90.5)  # 增加一点空间给文本标签
    ax3.spines["right"].set_edgecolor(color_dict['Inference Time'])

    # 合并图例
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles3, labels3 = ax3.get_legend_handles_labels()
    fig.legend(handles1 + handles2 + handles3,
               labels1 + labels2 + labels3,
               loc='upper center', bbox_to_anchor=(0.5, 1.02),
               fontsize=9, ncol=6, frameon=False)

    plt.tight_layout(pad=0.4, rect=[0, 0, 1, 0.95])
    plt.savefig(r"C:\model\deeplabv3-plus-pytorch-main2\miou_out\Fig_5_bar_all_blacktext.png",
                dpi=300, bbox_inches='tight')
    plt.show()

plot_model_comparison_bar()
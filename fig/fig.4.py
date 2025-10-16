import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def plot_model_comparison_bar():
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})

    file = r'C:\model\deeplabv3-plus-pytorch-main2\miou_out\model.csv'
    df = pd.read_csv(file)
    models = df['Model'].tolist()
    x = np.arange(len(models))

    metrics = {
        
        'IoU-Background': df['IoU-Background'].tolist(),
        'IoU-Leaf': df['IoU-Leaf'].tolist(),
        'mIoU': df['mIoU'].tolist(),
        'mPA': df['mPA'].tolist(),
        
    }

    params = df['Params (M)'].tolist()
    inference_time = df['Inference Time (ms)'].tolist()

    # ✅ 使用 seaborn Set2 调色板（取前4 + 第7、8）
    palette = sns.color_palette("Set2", 8)
    color_dict = {
        
        'IoU-Background': palette[0],
        'IoU-Leaf': palette[1],
        'mPA': palette[2],
        'mIoU': palette[3],
        'Params': palette[6],
        'Inference Time': palette[7]
    }

    width = 0.12
    fig, ax1 = plt.subplots(figsize=(6.5, 5))

    # 左侧主指标柱状图
    metric_offsets = []
    for i, (label, values) in enumerate(metrics.items()):
        offset = (i - 1.5) * width
        metric_offsets.append(offset)
        ax1.bar(x + offset, values, width=width, label=label,
                color=color_dict[label], alpha=0.85, zorder=1)

    ax1.set_ylabel('mPA / IoU (Background) / IoU (Leaf) / mIoU (%)')
    ax1.set_ylim(90, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    # Params 柱状图（右轴1）
    ax2 = ax1.twinx()
    offset_p = 2.5 * width
    bars2 = ax2.bar(x + offset_p, params, width=width,
                    color=color_dict['Params'], label='Params (M)',
                    alpha=0.85, zorder=1)
    ax2.set_ylabel('Params (M)', color=color_dict['Params'])
    ax2.tick_params(axis='y', labelcolor=color_dict['Params'])
    ax2.set_ylim(0, 30)
    ax2.spines["right"].set_edgecolor(color_dict['Params'])

    # Inference Time 柱状图（右轴2）
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))
    offset_t = 3.5 * width
    bars3 = ax3.bar(x + offset_t, inference_time, width=width,
                    color=color_dict['Inference Time'], label='Inference Time (ms)',
                    alpha=0.85, zorder=1)
    ax3.set_ylabel('Inference Time (ms)', color=color_dict['Inference Time'])
    ax3.tick_params(axis='y', labelcolor=color_dict['Inference Time'])
    ax3.set_ylim(15, 100)
    ax3.spines["right"].set_edgecolor(color_dict['Inference Time'])

    # ✅ 延后绘制所有文字，确保不被柱状图覆盖
    # 主指标文字
    for i, (label, values) in enumerate(metrics.items()):
        offset = metric_offsets[i]
        for j, v in enumerate(values):
            ax1.text(x[j] + offset, v + 0.1, f'{v:.2f}', ha='center', va='bottom',
                     fontsize=8, color='black', zorder=10, clip_on=False)

    # Params 数值
    for j in range(len(x)):
        ax2.text(x[j] + offset_p, params[j] + 0.1, f'{params[j]:.2f}',
                 ha='center', va='bottom', fontsize=8, color='black',
                 zorder=10, clip_on=False)

    # Inference Time 数值
    for j in range(len(x)):
        ax3.text(x[j] + offset_t, inference_time[j] + 0.1, f'{inference_time[j]:.2f}',
                 ha='center', va='bottom', fontsize=8, color='black',
                 zorder=10, clip_on=False)

    # 合并图例
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles3, labels3 = ax3.get_legend_handles_labels()
    fig.legend(handles1 + handles2 + handles3,
               labels1 + labels2 + labels3,
               loc='upper center', bbox_to_anchor=(0.5, 1.02),
               fontsize=9, ncol=6, frameon=False)

    plt.tight_layout(pad=0.4, rect=[0, 0, 1, 0.95])
    plt.savefig(r"C:\model\deeplabv3-plus-pytorch-main2\miou_out\Fig_4_bar_set2.png",
                dpi=300, bbox_inches='tight')
    plt.show()

plot_model_comparison_bar()

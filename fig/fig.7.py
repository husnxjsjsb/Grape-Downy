import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

def plot_all_metrics_as_bar_chart():
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})

    file = r'data\leison\RESULT\backbone.csv'
    df = pd.read_csv(file)
    models = df['Model'].tolist()
    x = np.arange(len(models))

    metrics = {
    'mPA': df['mPA(Binary)'].tolist(),
    'IoU-Background': df['IoU(Non-Disease: BG+Leaf)'].tolist(),
    'IoU-Disease': df['IoU(Disease)'].tolist(),
    'mIoU': df['mIoU(Binary)'].tolist(),
}

    params = df['Params (M)'].tolist()
    inference_time = df['Inference Time (ms)'].tolist()

    # ✅ 使用 seaborn Set2 调色板
    palette = sns.color_palette("Set2", 8)
    color_dict = {
        'IoU-Background': palette[1],
        'IoU-Disease': palette[2],
        'mIoU': palette[3],
        'mPA': palette[0],
        'Params': palette[6],
        'Inference Time': palette[7]
    }


    width = 0.12
    fig, ax1 = plt.subplots(figsize=(7, 5))

    # 左边指标
    for i, (label, values) in enumerate(metrics.items()):
        offset = (i - 1.5) * width
        ax1.bar(x + offset, values, width=width, label=label, color=color_dict[label])
        for j, v in enumerate(values):
            ax1.text(x[j] + offset, v + 0.1, f'{v:.2f}',
                     ha='center', va='bottom', fontsize=8, color='black')

    ax1.set_ylabel('IoU / mIoU / mPA (%)')
    ax1.set_ylim(60, 101)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Params
    ax2 = ax1.twinx()
    for j in range(len(x)):
        offset = 2.5 * width
        ax2.bar(x[j] + offset, params[j], width=width,
                color=color_dict['Params'], label='Params' if j == 0 else "")
        ax2.text(x[j] + offset, params[j] + 0.1, f'{params[j]:.2f}',
                 ha='center', va='bottom', fontsize=8, color='black')
    ax2.set_ylabel('Params (M)', color=color_dict['Params'])
    ax2.tick_params(axis='y', labelcolor=color_dict['Params'])
    ax2.set_ylim(0, 60)
    ax2.spines["right"].set_edgecolor(color_dict['Params'])

    # Inference Time
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('axes', 1.12))
    for j in range(len(x)):
        offset = 3.5 * width
        ax3.bar(x[j] + offset, inference_time[j], width=width,
                color=color_dict['Inference Time'], label='Inference Time' if j == 0 else "")
        ax3.text(x[j] + offset, inference_time[j] + 0.1, f'{inference_time[j]:.2f}',
                 ha='center', va='bottom', fontsize=8, color='black')
    ax3.set_ylabel('Inference Time (ms)', color=color_dict['Inference Time'])
    ax3.tick_params(axis='y', labelcolor=color_dict['Inference Time'])
    ax3.set_ylim(15, 65)
    ax3.spines["right"].set_edgecolor(color_dict['Inference Time'])

    # 图例
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles3, labels3 = ax3.get_legend_handles_labels()
    fig.legend(handles1 + handles2 + handles3,
               labels1 + labels2 + labels3,
               loc='upper center', bbox_to_anchor=(0.5, 1.02),
               fontsize=9, ncol=6, frameon=False)

    plt.tight_layout(pad=0.4, rect=[0, 0, 1, 0.95])
    plt.savefig(r"fig\Fig_7_bar_all_blacktext.png",
                dpi=300, bbox_inches='tight')
    plt.show()

plot_all_metrics_as_bar_chart()

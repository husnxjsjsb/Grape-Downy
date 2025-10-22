import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# --- 配置本地文件路径 ---
file_path = r'data\comparsion.csv' 

# *** 导入表格数据 ***
try:
    df = pd.read_csv(file_path)
    
    print(f"原始导入的列名: {df.columns.tolist()}")
    df.columns = df.columns.str.strip()
    print(f"清理空格后的列名: {df.columns.tolist()}")
    
    expected_cols = {
        'IoU-Leaf(%)': 'IoU-Leaf(%)',
        'IoU-Disease(%)': 'IoU-Disease(%)',
        'Params(M)': 'Params(M)',
        'TPI(ms)': 'TPI(ms)',
    }
    
    current_cols = df.columns.tolist()
    rename_dict = {}
    
    def simplify_col(col):
        return col.replace(' ', '').replace('-', '').lower()

    for code_key in expected_cols.keys():
        if code_key in current_cols:
            continue
        simplified_code_key = simplify_col(code_key)
        for current_col in current_cols:
            if simplify_col(current_col) == simplified_code_key:
                rename_dict[current_col] = code_key
                break
        
        if 'IoU-Disease (%)' in current_cols and code_key == 'IoU-Disease(%)':
            rename_dict['IoU-Disease (%)'] = code_key
             
    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)
        print(f"进行了列名重命名: {rename_dict}")

    numeric_cols = ['IoU-leaf(%)', 'IoU-disease(%)', 'Params(M)', 'TPI(ms)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            
except Exception as e:
    print(f"导入数据或预处理时发生错误: {e}")
    raise


def plot_model_comparison_bar(df, shift_index=6, label_shift=0.2):
    """
    绘制模型对比图，并将第 shift_index 个模型名向右移动 label_shift 距离
    shift_index: 从 0 开始计数，例如 6 表示第七个模型
    label_shift: 移动距离（正数向右，负数向左），单位与x轴刻度一致
    """
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})

    models_raw = df['Model'].tolist()
    models_wrapped = [m.replace('+', '+\n') for m in models_raw]
    x = np.arange(len(models_wrapped))

    metrics = {
        'IoU-Leaf (%)': df['IoU-Leaf(%)'].tolist(),
        'IoU-Disease (%)': df['IoU-Disease(%)'].tolist(),
    }

    params = df['Params(M)'].tolist()
    inference_time = df['TPI(ms)'].tolist()

    # --- 配色 ---
    palette = sns.color_palette("Set2", 8)
    color_dict = {
        'IoU-Leaf (%)': palette[0],
        'IoU-Disease (%)': palette[1],
        'Params (M)': palette[2],
        'TPI (ms)': palette[3]
    }

    width = 0.15
    fig, ax1 = plt.subplots(figsize=(7, 6), constrained_layout=True)

    offset_leaf = -0.5 * width
    offset_disease = 0.5 * width

    # --- IoU 指标 ---
    ax1.bar(x + offset_leaf, metrics['IoU-Leaf (%)'], width=width,
            label='IoU-Leaf (%)', color=color_dict['IoU-Leaf (%)'], alpha=0.85, zorder=1)
    ax1.bar(x + offset_disease, metrics['IoU-Disease (%)'], width=width,
            label='IoU-Disease (%)', color=color_dict['IoU-Disease (%)'], alpha=0.85, zorder=1)

    ax1.set_ylabel('IoU-Leaf / IoU-Disease (%)', fontsize=12)
    ax1.set_ylim(45, 100)
    ax1.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    # --- Params (M) ---
    ax2 = ax1.twinx()
    offset_p = 1.5 * width
    ax2.bar(x + offset_p, params, width=width,
            color=color_dict['Params (M)'], label='Params (M)',
            alpha=0.85, zorder=1)
    ax2.set_ylabel('Params (M)', fontsize=12)
    ax2.set_ylim(0, 40)
    ax2.spines["right"].set_edgecolor(color_dict['Params (M)'])

    # --- TPI (ms) ---
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    offset_t = 2.5 * width
    ax3.bar(x + offset_t, inference_time, width=width,
            color=color_dict['TPI (ms)'], label='TPI (ms)',
            alpha=0.85, zorder=1)
    ax3.set_ylabel('TPI (ms)', fontsize=12)
    ax3.set_ylim(300, 2500)
    ax3.spines["right"].set_edgecolor(color_dict['TPI (ms)'])

    # --- 数据标签 ---
    for offset, values in [(-0.5 * width, metrics['IoU-Leaf (%)']), (0.5 * width, metrics['IoU-Disease (%)'])]:
        for j, v in enumerate(values):
            ax1.text(x[j] + offset, v + 0.5, f'{v:.2f}', ha='center', va='bottom',
                     fontsize=7, color='black', zorder=10, clip_on=False)

    for j in range(len(x)):
        ax2.text(x[j] + offset_p, params[j] + 1, f'{params[j]:.2f}',
                 ha='center', va='bottom', fontsize=7, color='black', zorder=10, clip_on=False)
        ax3.text(x[j] + offset_t, inference_time[j] + 50, f'{inference_time[j]:.0f}',
                 ha='center', va='bottom', fontsize=7, color='black', zorder=10, clip_on=False)

    y_offset = 44  # 原来是 43，略微上移
    for i, label in enumerate(models_wrapped):
        if i == shift_index:
            ax1.text(x[i] + label_shift, y_offset, label, ha='center', va='top', fontsize=10)
        else:
            ax1.text(x[i], y_offset, label, ha='center', va='top', fontsize=10)

    # ✅ 去掉 x 轴刻度数字
    ax1.set_xticks([])


    # ✅ 图例
    handles, labels = [], []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    unique = dict(zip(labels, handles))

    fig.legend(
        unique.values(), unique.keys(),
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        fontsize=10,
        ncol=4,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.2
    )

    plt.savefig("model_comparison_bar_chart.png", dpi=300, bbox_inches='tight')
    plt.show()


# --- 执行绘图 ---
if 'df' in locals() and not df.empty:
    plot_model_comparison_bar(df, shift_index=6, label_shift=0.25)

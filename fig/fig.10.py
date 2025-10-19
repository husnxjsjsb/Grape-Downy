import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import io

# --- 配置本地文件路径 ---
# 请确保这个路径是正确的，并且文件存在
file_path = r'C:\model\DeepLap_UNet\data\comparsion.csv' 

# *** 导入表格数据 (仅保留 try 块，移除 except 块) ***
try:
    df = pd.read_csv(file_path)
    
    # 1. 打印原始列名，以便调试
    print(f"原始导入的列名: {df.columns.tolist()}")

    # 2. 清理列名：去除列名首尾可能存在的空格
    df.columns = df.columns.str.strip()

    # 3. 再次打印清理后的列名
    print(f"清理空格后的列名: {df.columns.tolist()}")
    
    # 4. 解决KeyError：将所有列名标准化为代码期望的格式 (如 'IoU-leaf(%)')
    expected_cols = {
        'IoU-leaf(%)': 'IoU-leaf(%)',
        'IoU-disease(%)': 'IoU-disease (%)', 
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
        
        found = False
        for current_col in current_cols:
            if simplify_col(current_col) == simplified_code_key:
                rename_dict[current_col] = code_key
                found = True
                break
        
        if not found and 'IoU-disease (%)' in current_cols and code_key == 'IoU-disease(%)':
             rename_dict['IoU-disease (%)'] = code_key
             
    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)
        print(f"进行了列名重命名: {rename_dict}")
        print(f"重命名后的列名: {df.columns.tolist()}")

    # *** 强制类型转换以解决 TypeError ***
    numeric_cols = ['IoU-leaf(%)', 'IoU-disease(%)', 'Params(M)', 'TPI(ms)']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            
except Exception as e:
    # 捕获所有异常
    print(f"导入数据或预处理时发生错误: {e}")
    raise 
# end of try block


def plot_model_comparison_bar(df):
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})

    # --- 模型名称处理：在 '+' 后面换行 ---
    models_raw = df['Model'].tolist()
    models_wrapped = [m.replace('+', '+\n') for m in models_raw] 
    
    x = np.arange(len(models_wrapped))

    # --- 左轴主指标 ---
    metrics = {
        'IoU-leaf (%)': df['IoU-leaf(%)'].tolist(),
        'IoU-disease (%)': df['IoU-disease(%)'].tolist(), 
    }

    # --- 右轴指标 ---
    params = df['Params(M)'].tolist()
    inference_time = df['TPI(ms)'].tolist()

    # 颜色分配
    palette = sns.color_palette("Set2", 8)
    color_dict = {
        'IoU-leaf (%)': palette[0],
        'IoU-disease (%)': palette[1],
        'Params (M)': palette[2],
        'TPI (ms)': palette[3]
    }

    # *** 修改柱子宽度 ***
    width = 0.15 # 柱子宽度
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # --- 左侧主指标柱状图 (ax1) ---
    # 定义两个IoU指标的偏移量
    offset_leaf = -0.5 * width  # 左边的柱子
    offset_disease = 0.5 * width # 右边的柱子

    # 绘制 IoU-leaf
    ax1.bar(x + offset_leaf, metrics['IoU-leaf (%)'], width=width, 
            label='IoU-leaf (%)', color=color_dict['IoU-leaf (%)'], 
            alpha=0.85, zorder=1)

    # 绘制 IoU-disease
    ax1.bar(x + offset_disease, metrics['IoU-disease (%)'], width=width, 
            label='IoU-disease (%)', color=color_dict['IoU-disease (%)'], 
            alpha=0.85, zorder=1)
    
    # 存储偏移量和值，用于后续的文本标签
    metric_offsets_data = [
        (offset_leaf, metrics['IoU-leaf (%)']),
        (offset_disease, metrics['IoU-disease (%)'])
    ]

    ax1.set_ylabel('IoU-leaf / IoU-disease (%)', fontsize=12)
    # **这里是根据你的要求修改后的代码**
    ax1.set_ylim(45, 100) # 根据你的要求更改了下限
    
    ax1.set_xticks(x)
    
    # *** 应用换行后的模型名称列表 ***
    ax1.set_xticklabels(models_wrapped, rotation=0, ha="center") 
    
    ax1.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    # --- Params (M) 柱状图（右轴 1: ax2） ---
    ax2 = ax1.twinx()
    offset_p = 1.5 * width
    ax2.bar(x + offset_p, params, width=width,
            color=color_dict['Params (M)'], label='Params (M)',
            alpha=0.85, zorder=1)
    ax2.set_ylabel('Params (M)', color='black', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, 40)
    ax2.spines["right"].set_edgecolor(color_dict['Params (M)'])

    # --- TPI (ms) 柱状图（右轴 2: ax3） ---
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.1))
    offset_t = 2.5 * width
    ax3.bar(x + offset_t, inference_time, width=width,
            color=color_dict['TPI (ms)'], label='TPI (ms)',
            alpha=0.85, zorder=1)
    ax3.set_ylabel('TPI (ms)', color='black', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='black')
    ax3.set_ylim(300, 2500)
    ax3.spines["right"].set_edgecolor(color_dict['TPI (ms)'])

    # --- 添加数据标签 ---
    # 使用 metric_offsets_data 来循环
    for offset, values in metric_offsets_data:
        for j, v in enumerate(values):
            ax1.text(x[j] + offset, v + 0.5, f'{v:.2f}', ha='center', va='bottom',
                     fontsize=7, color='black', zorder=10, clip_on=False)

    for j in range(len(x)):
        ax2.text(x[j] + offset_p, params[j] + 1, f'{params[j]:.2f}',
                 ha='center', va='bottom', fontsize=7, color='black',
                 zorder=10, clip_on=False)

    for j in range(len(x)):
        ax3.text(x[j] + offset_t, inference_time[j] + 50, f'{inference_time[j]:.0f}',
                 ha='center', va='bottom', fontsize=7, color='black',
                 zorder=10, clip_on=False)

    # --- 合并图例 ---
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles3, labels3 = ax3.get_legend_handles_labels()
    fig.legend(handles1 + handles2 + handles3,
               labels1 + labels2 + labels3,
               loc='upper center', bbox_to_anchor=(0.5, 1.05),
               fontsize=9, ncol=4, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("model_comparison_bar_chart.png", dpi=300)
    plt.show() # 显示图表的命令

# 确保 df 存在且非空
if 'df' in locals() and not df.empty:
    # 执行绘图函数
    plot_model_comparison_bar(df)
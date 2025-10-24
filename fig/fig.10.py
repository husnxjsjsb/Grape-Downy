import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from matplotlib.lines import Line2D

# ===================== 1️⃣ 全局字体与样式 =====================
plt.rcParams['font.family'] = 'Times New Roman'  # 全局字体
plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示问题
plt.rcParams['mathtext.fontset'] = 'custom'       # 数学字体统一为 Times New Roman
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
sns.set_style("white")

# ===================== 2️⃣ 文件路径 =====================
predicted_csv = r"data\predict.csv"
true_csv = r"data\true.csv"
output_fig = r"fig10.png"

# ===================== 3️⃣ 读取并合并数据 =====================
try:
    df_pred = pd.read_csv(predicted_csv)
    df_true = pd.read_csv(true_csv)
except FileNotFoundError as e:
    print(f"Error: Required file not found. Please check the path: {e}")
    print("Using mock data for demonstration.")
    np.random.seed(42)
    y_true = np.linspace(0, 1, 100)
    y_pred = y_true * 0.9 + 0.05 + np.random.normal(0, 0.05, 100)
    y_pred[y_pred < 0] = 0
    y_pred[y_pred > 1] = 1
    df_merged = pd.DataFrame({'severity_pred': y_pred, 'severity_true': y_true})
    y_pred = df_merged['severity_pred'].values
    y_true = df_merged['severity_true'].values

if 'df_pred' in locals() and 'df_true' in locals():
    df_pred.columns = df_pred.columns.str.replace('!NL', '', regex=False)
    df_true.columns = df_true.columns.str.replace('!NL', '', regex=False)

    if 'image_name' in df_pred.columns and 'image_name' in df_true.columns:
        df_merged = pd.merge(df_pred, df_true, on='image_name', suffixes=('_pred', '_true'))
    elif 'severity' in df_pred.columns and 'severity' in df_true.columns:
        df_merged = pd.concat([df_pred['severity'], df_true['severity']], axis=1)
        df_merged.columns = ['severity_pred', 'severity_true']

    y_pred = df_merged['severity_pred'].values
    y_true = df_merged['severity_true'].values

# ===================== 4️⃣ 计算指标 =====================
r2 = r2_score(y_true, y_pred)
mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
slope, intercept = np.polyfit(y_true, y_pred, 1)

# ===================== 5️⃣ 绘图 =====================
fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
ax.grid(False)

# --- 散点 ---
ax.scatter(y_true, y_pred,
           alpha=0.6,
           edgecolors='k',
           facecolors='#1f77b4',
           linewidth=0.4,
           s=25,
           label='Predicted Points')

# --- 理想线 ---
ax.plot([0, 1.1], [0, 1.1], 'r--', linewidth=1.2, label='Ideal Line (y=x)')

# --- 拟合线 ---
x_fit = np.array([0, 1.1])
y_fit = slope * x_fit + intercept
if intercept >= 0:
    reg_label = rf'Predicted Fit ($y = {slope:.4f}x + {intercept:.4f}$)'
else:
    reg_label = rf'Predicted Fit ($y = {slope:.4f}x - {-intercept:.4f}$)'
ax.plot(x_fit, y_fit, '--', color='#FFD700', linewidth=1.5, label=reg_label)

# --- 坐标轴设置 ---
ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1.1)
ax.set_xlabel('Ground Truth', fontsize=14, fontname='Times New Roman')
ax.set_ylabel('Predicted', fontsize=14, fontname='Times New Roman')
ax.tick_params(axis='both', labelsize=10)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Times New Roman')

# ===================== 统一信息框（图例 + 指标） =====================
# 自定义图例句柄
custom_lines = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markeredgecolor='k',
           markersize=6, label='Predicted Points'),
    Line2D([0], [0], color='r', linestyle='--', label='Ideal Line (y=x)'),
    Line2D([0], [0], color='#FFD700', linestyle='--', label=reg_label)
]

# 指标文字（Times New Roman）
metrics_text = rf'$R^2$ = {r2:.3f},  MAE = {mae:.3f},  RMSE = {rmse:.3f}'
custom_lines.append(Line2D([], [], color='none', label=metrics_text))

# 创建统一图例框
legend = ax.legend(
    handles=custom_lines,
    loc='upper left',
    fontsize=10,
    frameon=True,
    fancybox=True,
    framealpha=0.9,
    borderpad=0.8
)

# 强制图例中的字体为 Times New Roman
for text in legend.get_texts():
    text.set_fontname('Times New Roman')

plt.tight_layout()
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
plt.close()

# ===================== 输出 =====================
print(f"Enhanced unified plot saved to {output_fig}")
print("\nEvaluation Metrics:")
print(f"R² = {r2:.4f}")
print(f"MAE = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"Regression Line: y = {slope:.4f}x + {intercept:.4f}")

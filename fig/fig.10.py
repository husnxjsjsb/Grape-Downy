import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# ===================== 1️⃣ 全局字体与样式 =====================
plt.rcParams['font.family'] = 'Times New Roman'
# 🚀 修改：将 "whitegrid" 更改为 "white" 来移除默认网格
sns.set_style("white")

# ===================== 2️⃣ 文件路径 =====================
predicted_csv = r"data\predict.csv"
true_csv = r"data\true.csv"
output_fig = r"fig11.png"

# ===================== 3️⃣ 读取并合并数据 =====================
df_pred = pd.read_csv(predicted_csv)
df_true = pd.read_csv(true_csv)

# 清理可能的“!NL”异常符号
df_pred.columns = df_pred.columns.str.replace('!NL', '', regex=False)
df_true.columns = df_true.columns.str.replace('!NL', '', regex=False)

if 'image_name' in df_pred.columns and 'image_name' in df_true.columns:
    df_merged = pd.merge(df_pred, df_true, on='image_name', suffixes=('_pred', '_true'))
else:
    df_merged = pd.concat([df_pred['severity'], df_true['severity']], axis=1)
    df_merged.columns = ['severity_pred', 'severity_true']

y_pred = df_merged['severity_pred'].values
y_true = df_merged['severity_true'].values

# ===================== 4️⃣ 计算评价指标 =====================
r2 = r2_score(y_true, y_pred)
mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

# 🚀 新增：计算回归线（拟合线）系数
# 计算一元线性回归系数 (y_pred = slope * y_true + intercept)
slope, intercept = np.polyfit(y_true, y_pred, 1)

# ===================== 5️⃣ 绘图 =====================
fig, ax = plt.subplots(figsize=(5, 4), dpi=300)

# 🚀 新增：显式关闭网格线
ax.grid(False)

# --- 散点 ---
ax.scatter(y_true, y_pred,
           alpha=0.6,
           edgecolors='k',
           facecolors='#1f77b4',
           linewidth=0.4,
           s=25,
           label='Predicted Points')

# --- 理想线 y = x ---
ax.plot([0, 1.1], [0, 1.1], 'r--', linewidth=1.2, label='True')

# 🚀 修改：计算并绘制回归线（拟合线） - 黄色虚线
# 创建用于绘制回归线的 x 值范围
x_fit = np.array([0, 1.1])
# 计算对应的 y 值
y_fit = slope * x_fit + intercept

# 绘制回归线
ax.plot(x_fit, y_fit,
        '--',                     # 虚线
        color='#FFD700',          # 黄色/金色
        linewidth=1.5,
        label='Predicted Fit')    # 图例文字为 'Predicted Fit'

# --- 坐标范围 ---
ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1.1)

# --- 坐标轴标签 & 标题 ---
ax.set_xlabel('True Severity', fontsize=11)
ax.set_ylabel('Predicted Severity', fontsize=11)
ax.set_title('Severity Prediction Comparison', fontsize=12, pad=10)

# --- 坐标刻度字体 ---
ax.tick_params(axis='both', labelsize=9)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Times New Roman')

# --- 指标文本框 (已移除回归方程) ---
textstr = '\n'.join((
    rf'$R^2$ = {r2:.3f}',
    f'MAE = {mae:.3f}',
    f'RMSE = {rmse:.3f}')) # 🚀 已移除回归方程
ax.text(0.97, 0.03, textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        family='Times New Roman')

# --- 图例 ---
ax.legend(loc='upper left', fontsize=8, frameon=False)

# --- 保存图片 ---
plt.tight_layout()
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
plt.close()

# ===================== 6️⃣ 打印指标 =====================
print(f"Enhanced comparison plot saved to {output_fig}")
print("\nEvaluation Metrics:")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Regression Line: y = {slope:.4f}x + {intercept:.4f}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# ===================== 1️⃣ 全局字体与样式 =====================
plt.rcParams['font.family'] = 'Times New Roman'
sns.set_style("whitegrid")

# ===================== 2️⃣ 文件路径 =====================
predicted_csv = r"data\predict.csv"
true_csv = r"data\true.csv"
output_fig = r"data\fig11.png"

# ===================== 3️⃣ 读取并合并数据 =====================
df_pred = pd.read_csv(predicted_csv)
df_true = pd.read_csv(true_csv)

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

# ===================== 5️⃣ 绘图 =====================
fig, ax = plt.subplots(figsize=(5, 4), dpi=300)

# 散点
ax.scatter(y_true, y_pred,
           alpha=0.6,
           edgecolors='k',
           facecolors='#1f77b4',
           linewidth=0.4,
           s=50,
           label='Predicted Points')

# 理想线 y = x
ax.plot([0, 1.1], [0, 1.1], 'r--', linewidth=1.2, label='Ideal (y = x)')

# 回归线
sns.regplot(x=y_true, y=y_pred,
            scatter=False,
            color='orange',
            line_kws={'linestyle': '--',
                      'linewidth': 1.5,
                      'label': 'Fit Line'},
            ax=ax)

# 坐标范围固定 0~1.1（如需更宽可改成 1.2）
ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1.1)

# 坐标轴标签 & 标题
ax.set_xlabel('True Severity', fontsize=11, family='Times New Roman')
ax.set_ylabel('Predicted Severity', fontsize=11, family='Times New Roman')
ax.set_title('Severity Prediction Comparison',
             fontsize=12, family='Times New Roman', pad=10)

# 坐标刻度字体
ax.tick_params(axis='both', labelsize=9)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Times New Roman')

# 指标框
textstr = '\n'.join((
    rf'$R^2$ = {r2:.3f}',
    f'MAE = {mae:.3f}',
    f'RMSE = {rmse:.3f}'))
props = dict(boxstyle='round,pad=0.4', facecolor='lightgrey', alpha=0.7)
ax.text(0.97, 0.03, textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=props,
        family='Times New Roman')

# 图例
ax.legend(loc='upper left', fontsize=8, frameon=True)

# 保存图片
plt.tight_layout()
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
plt.close()

# ===================== 6️⃣ 打印指标 =====================
print(f"Enhanced comparison plot saved to {output_fig}")
print("\nEvaluation Metrics:")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

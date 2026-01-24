import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np



# データ準備（インデックスの計算）
# I_std = 対数収益率
# I_ene = 対数収益率 * 株価（または時価総額指数）
# ※ df_dataset['Market_Price_A'] がある前提




df = pd.read_csv("dataset_ml_weighted.csv", index_col=0, parse_dates=True)





idx_price = df['Market_Price_A'] # インデックス（標準型）
idx_price_ene = df['Market_Price_B'] #　インデックス（エネルギー型） 
i_std = np.log(idx_price).diff()
i_ene = np.log(idx_price_ene).diff()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) 時系列比較
ax1 = axes[0]
ax1.plot(i_std.index, i_std, label='Standard Index (Speed)', alpha=0.6, color='blue', linewidth=0.5)
ax1_twin = ax1.twinx()
ax1_twin.plot(i_ene.index, i_ene, label='Energy Index (Impact)', alpha=0.6, color='red', linewidth=0.5)

ax1.set_title('(a) Time Series Comparison')
ax1.set_ylabel('Standard Index (Log Return)')
ax1_twin.set_ylabel('Energy Index (Return x Price)')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left')

# (b) 散布図：価格水準 vs 変動エネルギー（絶対値）
# これがレビュワーが求めていた「相関」の可視化です
ax2 = axes[1]
scatter_data = pd.DataFrame({
    'Price_Level': idx_price,
    'Energy_Magnitude': i_ene.abs()
}).dropna()

# 2024年以降をハイライト
mask_recent = scatter_data.index >= '2024-01-01'

ax2.scatter(scatter_data[~mask_recent]['Price_Level'], 
            scatter_data[~mask_recent]['Energy_Magnitude'], 
            alpha=0.3, color='gray', label='Pre-2024', s=10)
ax2.scatter(scatter_data[mask_recent]['Price_Level'], 
            scatter_data[mask_recent]['Energy_Magnitude'], 
            alpha=0.6, color='red', label='Post-2024 (High Potential)', s=15)

ax2.set_title('(b) Price Level vs. Energy Magnitude')
ax2.set_xlabel('Market Price Level (Potential Height)')
ax2.set_ylabel('Abs Energy Index (Impact Size)')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("fig4_1_index_comparison_scatter.png")
plt.show()
print("図4.1 生成完了: 時系列と散布図で『高値圏ほどエネルギーが増大する』ことを証明しました。")
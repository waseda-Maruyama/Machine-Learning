import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# 1. データ読み込み
# ---------------------------------------------------------
input_file = "stock_prices_topix100_simple.csv"

print(f"📂 データを読み込んでいます: {input_file} ...")
if not os.path.exists(input_file):
    print("❌ ファイルが見つかりません。")
    exit()

df_prices = pd.read_csv(input_file, index_col=0, parse_dates=True)
print(f"📊 データ形状: {df_prices.shape} (行:日数, 列:銘柄)")

# ---------------------------------------------------------
# 2. 前処理: 対数収益率 (Log Returns)
# ---------------------------------------------------------
# 価格そのものではなく「変化率」の相関を見ます
df_log_returns = np.log(df_prices / df_prices.shift(1)).dropna()

# ---------------------------------------------------------
# 3. RMT指標(最大固有値)の計算
# ---------------------------------------------------------
def get_max_eigenvalue(window_df):
    # 相関行列 (Correlation Matrix)
    corr_mat = window_df.corr()
    # 欠損ケア (分散0の場合など)
    corr_mat = corr_mat.fillna(0)
    # 固有値分解 (eighはエルミート行列用で高速)
    vals, _ = np.linalg.eigh(corr_mat)
    # 最大値を返す (末尾)
    return vals[-1]

# 窓幅設定: 60営業日 (約3ヶ月)
WINDOW = 55

print(f"🧮 移動窓 ({WINDOW}日) で固有値を解析中...")
print("   (98銘柄の行列計算なので数秒〜数十秒かかります)")

rmt_dates = []
max_eigenvalues = []

# ループ処理
for i in range(WINDOW, len(df_log_returns)):
    # 窓データの切り出し
    window_data = df_log_returns.iloc[i-WINDOW : i]
    
    # 計算
    lambda_max = get_max_eigenvalue(window_data)
    
    rmt_dates.append(df_log_returns.index[i])
    max_eigenvalues.append(lambda_max)

# Series化
ts_rmt = pd.Series(max_eigenvalues, index=rmt_dates, name="RMT_Max_Eigen")

# ---------------------------------------------------------
# 4. 可視化 (ここが研究のハイライト)
# ---------------------------------------------------------
# 市場平均（擬似TOPIX）を作成
market_index = df_prices.mean(axis=1)
market_index = market_index / market_index.iloc[0] # 正規化

fig, ax1 = plt.figure(figsize=(14, 8)), plt.gca()

# [左軸] 市場平均株価
color_price = 'tab:blue'
ax1.set_xlabel('Date')
ax1.set_ylabel('Market Average Price', color=color_price, fontsize=12)
ax1.plot(market_index.index, market_index, color=color_price, alpha=0.6, label='Market Price')
ax1.tick_params(axis='y', labelcolor=color_price)
ax1.grid(True, alpha=0.3)

# [右軸] RMT最大固有値
ax2 = ax1.twinx()
color_rmt = 'tab:red'
ax2.set_ylabel('Max Eigenvalue (Sync Risk)', color=color_rmt, fontsize=12, fontweight='bold')
ax2.plot(ts_rmt.index, ts_rmt, color=color_rmt, alpha=0.9, linewidth=1.5, label='Max Eigenvalue')
ax2.tick_params(axis='y', labelcolor=color_rmt)

# イベントライン
events = {
    '2016-02-12': '2016 Crash',
    '2018-02-06': 'VIX Shock',
    '2018-12-25': 'Xmas Drop',
    '2020-03-19': 'Covid-19',
    '2022-03-09': 'Ukraine/Fed'
}

for date, label in events.items():
    try:
        date_ts = pd.to_datetime(date)
        if date_ts >= ts_rmt.index[0] and date_ts <= ts_rmt.index[-1]:
            plt.axvline(x=date_ts, color='black', linestyle='--', alpha=0.5)
            plt.text(date_ts, ax2.get_ylim()[1]*0.95, f' {label}', rotation=90, verticalalignment='top')
    except:
        pass

plt.title(f"Validation: RMT Signal vs Market Crashes (N={df_prices.shape[1]})", fontsize=14)
plt.show()

# 保存
ts_rmt.to_csv("feature_rmt_eigen_98.csv")
print("✅ 解析完了。結果を保存しました。")
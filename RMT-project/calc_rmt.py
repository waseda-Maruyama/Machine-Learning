import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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


WINDOW = 110

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

# DataFrame化
df_features = pd.DataFrame(max_eigenvalues, index=rmt_dates, columns=['RMT_Raw'])

print("⚙️ 速度、加速度、Zスコアを計算中...")

# 1. 平滑化 (Smoothing)
# 微分（速度計算）の前にノイズを低減させます。
# 窓5日 = 1週間のトレンドを見るイメージです。
rmt_smooth = df_features['RMT_Raw'].rolling(window=5).mean()

# 2. 速度 (Velocity)
# 平滑化したデータの「前日比」を取ります。
df_features['RMT_Vel'] = rmt_smooth.diff()

# 3. 加速度 (Acceleration)
# 速度の変化量です。「変化の激しさ」を表します。
df_features['RMT_Accel'] = df_features['RMT_Vel'].diff()

# 4. 緊張度 (Z-Score)
# 生データ(Raw)が、過去250日(約1年)の平均から「標準偏差いくつ分」離れているか。
# これが +2.0 や +3.0 を超えると「異常事態」です。
window_z = 250
rmt_mean = df_features['RMT_Raw'].rolling(window_z).mean()
rmt_std = df_features['RMT_Raw'].rolling(window_z).std()

# ゼロ除算回避のため、stdが極端に小さい場合は考慮が必要ですが、
# RMT固有値でstd=0になることは稀なので、今回はそのまま計算します。
df_features['RMT_Zscore'] = (df_features['RMT_Raw'] - rmt_mean) / rmt_std

# ※計算初期は窓分のデータがないため NaN (欠損) が発生します。
# 後の工程で dropna() するか、ここで埋めるかは戦略次第ですが、
# ここでは「データの実態」を優先して NaN のまま保存します。


ts_rmt = df_features['RMT_Raw']

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

from config import scenarios

for name, period_tuple in scenarios.items():
# タプルから開始日を取り出す
    start_date_str = period_tuple[0] 
    end_date_str = period_tuple[1]
    
    # 変換（ここでエラーが出たら、データが間違っていると気づける）
    date_ts = pd.to_datetime(start_date_str)
    end_date_ts = pd.to_datetime(end_date_str)

    ax1.axvspan(date_ts, end_date_ts, color='gray', alpha=0.2, label='_nolegend_')

    plt.text(date_ts, ax1.get_ylim()[1]*0.95, name, rotation=90, verticalalignment='top', fontsize=10, color='black',fontweight='bold')
    
    ax1.axvline(x=date_ts, color='gray', linestyle=':', alpha=0.6)

plt.title(f"Validation: RMT Signal vs Market Crashes (N={df_prices.shape[1]})", fontsize=14)
plt.tight_layout()
plt.show()

# 保存
output_file = "feature_rmt_eigen_98.csv"
df_features.to_csv(output_file)
print("✅ 解析完了。結果を保存しました。")
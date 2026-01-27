import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from config import scenarios

# =========================================================
# ⚙️ 設定エリア
# =========================================================
WINDOW_S = 65   # 短期窓 (急変検知用)
WINDOW = 135  # 中期窓 (中間スケール)
WINDOW_L = 170  # 長期窓 (構造変化検知用)

print(f"🧪 マルチスケールRMT解析: 短期={WINDOW_S}日, 中期={WINDOW}日, 長期={WINDOW_L}日")

# ---------------------------------------------------------
# 1. データ読み込み
# ---------------------------------------------------------
input_file = "stock_adj_close.csv" 
if not os.path.exists(input_file):
    print(f"❌ ファイルが見つかりません: {input_file}")
    exit()

df_prices = pd.read_csv(input_file, index_col=0, parse_dates=True)
print(f"📊 データ形状: {df_prices.shape}")

# 対数収益率
df_log_returns = np.log(df_prices / df_prices.shift(1)).dropna()

# ---------------------------------------------------------
# 2. RMT計算関数 (理論値 lambda_plus も計算するように改良)
# ---------------------------------------------------------
def get_rmt_metrics(returns_df, window_size):
    """
    指定した窓幅で以下を計算する
    1. 最大固有値 (lambda_max): 市場の同期シグナル
    2. 理論的上限値 (lambda_plus): これ以下はノイズとみなされる境界線
    """
    data_values = returns_df.values
    dates = returns_df.index
    n_samples, n_assets = data_values.shape
    
    # Q値 (T/N): 時系列長 / 銘柄数
    # ※ ウィンドウ内で銘柄数が変動しない前提
    q = window_size / n_assets
    
    # マルチェンコ・パスツール分布の理論上限値 (固定値)
    # λ+ = (1 + sqrt(1/Q))^2
    lambda_plus_val = (1 + np.sqrt(1/q))**2
    
    vals_max = np.full(n_samples, np.nan)
    vals_threshold = np.full(n_samples, lambda_plus_val) # 時系列として持たせる

    # 高速化のためループ処理
    print(f"   ...Processing Window T={window_size}, N={n_assets}, Q={q:.2f}, Threshold={lambda_plus_val:.2f}")
    
    for i in range(window_size, n_samples):
        # 窓切り出し
        sub_data = data_values[i-window_size : i]

        # 欠損ケア
        sub_data = np.nan_to_num(sub_data)
        if np.all(sub_data == 0):
            vals_max[i] = 0
            continue

        # 相関行列 & 固有値分解
        # rowvar=False なので (N, N) の行列ができる
        corr = np.corrcoef(sub_data, rowvar=False)
        corr = np.nan_to_num(corr)
        
        # 最大固有値のみ取得 (eigvalshは昇順ソートされるので最後尾)
        eigvals = np.linalg.eigvalsh(corr)
        vals_max[i] = eigvals[-1] 

    return pd.Series(vals_max, index=dates), pd.Series(vals_threshold, index=dates)

# ---------------------------------------------------------
# 3. 計算実行 & 物理指標の生成
# ---------------------------------------------------------
df_features = pd.DataFrame(index=df_log_returns.index)

def add_physics_indicators(df_target, suffix):
    """
    suffix が空文字の場合は添え字なしの列名を使い、
    S/L には従来通りサフィックス付き列名を使う。
    """
    def col(base):
        return base if suffix == '' else f"{base}_{suffix}"

    raw_col = col("RMT_Raw")
    threshold_col = col("RMT_Threshold")

    # ノイズに対するシグナルの強度比 (Signal-to-Noise Ratio的な意味合い)
    # これが1.0を超えている部分が「有意な同期」
    df_target[col("RMT_Ratio")] = df_target[raw_col] / df_target[threshold_col]
    
    # 速度・加速度 (平滑化してから計算)
    smooth = df_target[raw_col].rolling(window=5).mean()
    df_target[col("RMT_Vel")] = smooth.diff()
    df_target[col("RMT_Accel")] = df_target[col("RMT_Vel")].diff()

# --- A. 短期窓 (Short) ---
s_max, s_th = get_rmt_metrics(df_log_returns, WINDOW_S)
df_features['RMT_Raw_S'] = s_max
df_features['RMT_Threshold_S'] = s_th
add_physics_indicators(df_features, 'S')

# --- B. 中期窓 (Mid) ---
m_max, m_th = get_rmt_metrics(df_log_returns, WINDOW)
df_features['RMT_Raw'] = m_max
df_features['RMT_Threshold'] = m_th
add_physics_indicators(df_features, '')

# --- C. 長期窓 (Long) ---
l_max, l_th = get_rmt_metrics(df_log_returns, WINDOW_L)
df_features['RMT_Raw_L'] = l_max
df_features['RMT_Threshold_L'] = l_th
add_physics_indicators(df_features, 'L')

# =========================================================
# 4. 可視化 (理論境界線を追加)
# =========================================================

# --- 市場インデックス準備 (既存コードと同じ) ---
market_cap_file = "market_caps.csv"
if os.path.exists(market_cap_file):
    df_caps = pd.read_csv(market_cap_file, index_col=0, parse_dates=True)
    df_caps = df_caps.reindex(df_prices.index).ffill()
    market_index = df_caps.sum(axis=1)
else:
    market_index = df_prices.mean(axis=1)
market_index = market_index / market_index.iloc[0]

# --- プロット ---
fig, ax1 = plt.subplots(figsize=(14, 8))

# 1. 市場インデックス
color_market = 'black'
ax1.set_xlabel('Date', fontsize=14)
ax1.set_ylabel('Market Index', color=color_market, fontsize=16)
ax1.plot(market_index.index, market_index, color=color_market, alpha=0.9, label='Market Index')
ax1.tick_params(axis='y', labelcolor=color_market)

# 2. RMT 最大固有値 & 理論境界
ax2 = ax1.twinx()

# Short
ax2.plot(df_features.index, df_features['RMT_Raw_S'], 
         color='tab:blue', linewidth=0.8, alpha=0.9, label=rf'W={WINDOW_S} $\lambda_{{max}}$')
# Short Threshold (点線)
ax2.plot(df_features.index, df_features['RMT_Threshold_S'], 
         color='tab:blue', linestyle='--', linewidth=0.8, alpha=0.5, label=rf'W={WINDOW_S} Noise Limit ($\lambda_+$)')

# Mid
ax2.plot(df_features.index, df_features['RMT_Raw'], 
         color='tab:green', linewidth=0.9, alpha=0.9, label=rf'W={WINDOW} $\lambda_{{max}}$')
# Mid Threshold (点線)
ax2.plot(df_features.index, df_features['RMT_Threshold'], 
         color='tab:green', linestyle='--', linewidth=0.8, alpha=0.5, label=rf'W={WINDOW} Noise Limit ($\lambda_+$)')

# Long
ax2.plot(df_features.index, df_features['RMT_Raw_L'], 
         color='tab:red', linewidth=1.0, alpha=0.9, label=rf'W={WINDOW_L} $\lambda_{{max}}$')
# Long Threshold (点線)
ax2.plot(df_features.index, df_features['RMT_Threshold_L'], 
         color='tab:red', linestyle='--', linewidth=0.8, alpha=0.5, label=rf'W={WINDOW_L} Noise Limit ($\lambda_+$)')

ax2.set_ylabel(rf'Max Eigenvalue ($\lambda_{{max}}$) vs Noise Limit ($\lambda_+$)', fontsize=16)
ax2.tick_params(axis='y', labelsize=14)

# 凡例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

# イベント帯
data_start = df_features.index[0]
data_end = df_features.index[-1]
ymin, ymax = ax2.get_ylim()

for name, period in scenarios.items():
    start, end = pd.to_datetime(period[0]), pd.to_datetime(period[1])
    # 表示範囲とシナリオ期間の重なりだけを可視化
    span_start = max(start, data_start)
    span_end = min(end, data_end)
    if span_start <= span_end:
        ax1.axvspan(span_start, span_end, color='gray', alpha=0.1)  # 塗りつぶしだけにしてダッシュ指定を外す
        mid_point = span_start + (span_end - span_start) / 2
        plt.text(mid_point, ymax * 0.98, f" {name}", rotation=90, va='top', fontsize=10, color='gray')

# タイトルは表示しない
plt.tight_layout()
plt.savefig("rmt_signal_detection.pdf")
plt.show()

# 保存
output_file = "feature_rmt_dual.csv"
df_features.to_csv(output_file)
print(f"✅ 保存完了: {output_file}")
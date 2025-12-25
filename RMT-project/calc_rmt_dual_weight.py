import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from config import scenarios

# =========================================================
# ⚙️ 設定エリア: 2つの窓幅を定義
# =========================================================
# 感度分析の結果を参考に設定してください
WINDOW_S = 70  # 短期窓 (急変検知用)
WINDOW_L = 135  # 長期窓 (構造変化検知用)

print(f"🧪 マルチスケールRMT解析: 短期={WINDOW_S}日, 長期={WINDOW_L}日")

# ---------------------------------------------------------
# 1. データ読み込み
# ---------------------------------------------------------
input_file = "stock_prices.csv" 
if not os.path.exists(input_file):
    print(f"❌ ファイルが見つかりません: {input_file}")
    exit()

df_prices = pd.read_csv(input_file, index_col=0, parse_dates=True)
print(f"📊 データ形状: {df_prices.shape}")

# 対数収益率
df_log_returns = np.log(df_prices / df_prices.shift(1)).dropna()

# ---------------------------------------------------------
# 2. RMT計算関数 (共通化)
# ---------------------------------------------------------
def get_max_eigenvalue_series(returns_df, window_size):
    """指定した窓幅で最大固有値の時系列を計算する"""
    data_values = returns_df.values
    dates = returns_df.index
    n_samples = len(dates)
    
    vals = np.full(n_samples, np.nan)
    
    # 高速化のためループ処理
    for i in range(window_size, n_samples):
        # 窓切り出し
        sub_data = data_values[i-window_size : i]
        
        # 相関行列 & 固有値分解
        # 欠損(NaN)ケア: fillna(0)相当
        sub_data = np.nan_to_num(sub_data)
        if np.all(sub_data == 0):
            vals[i] = 0
            continue
            
        corr = np.corrcoef(sub_data, rowvar=False)
        corr = np.nan_to_num(corr)
        eigvals = np.linalg.eigvalsh(corr)
        vals[i] = eigvals[-1] # 最大固有値
        
    return pd.Series(vals, index=dates)

# ---------------------------------------------------------
# 3. 計算実行 & 物理指標の生成
# ---------------------------------------------------------
# 結果をまとめるDataFrame
df_features = pd.DataFrame(index=df_log_returns.index)

# --- 関数: 物理指標を追加する ---
def add_physics_indicators(df_target, base_col, suffix):
    # 1. Raw (生データ)
    raw_col = f"RMT_Raw_{suffix}"
    
    # 2. Smooth (平滑化) -> Velocity計算用
    smooth = df_target[raw_col].rolling(window=5).mean()
    
    # 3. Velocity (速度)
    df_target[f"RMT_Vel_{suffix}"] = smooth.diff()
    
    # 4. Acceleration (加速度)
    df_target[f"RMT_Accel_{suffix}"] = df_target[f"RMT_Vel_{suffix}"].diff()
    



# --- A. 短期窓 (Short) ---
print(f"🧮 短期窓 ({WINDOW_S}日) を計算中...")
df_features[f'RMT_Raw_S'] = get_max_eigenvalue_series(df_log_returns, WINDOW_S)
add_physics_indicators(df_features, f'RMT_Raw_S', 'S')

# --- B. 長期窓 (Long) ---
print(f"🧮 長期窓 ({WINDOW_L}日) を計算中...")
df_features[f'RMT_Raw_L'] = get_max_eigenvalue_series(df_log_returns, WINDOW_L)
add_physics_indicators(df_features, f'RMT_Raw_L', 'L')

# ---------------------------------------------------------
# 4. 可視化 (2つのRMTを表示)
# ---------------------------------------------------------
market_cap_file = "market_caps.csv"
if os.path.exists(market_cap_file):
    print("📈 時価総額加重インデックスを使用")
    df_caps = pd.read_csv(market_cap_file, index_col=0, parse_dates=True)
    df_caps = df_caps.reindex(df_prices.index).ffill()
    market_index = df_caps.sum(axis=1)
else:
    print("📉 単純平均インデックスを使用")
    market_index = df_prices.mean(axis=1)

market_index = market_index / market_index.iloc[0]

# market_indexの最大値の値とその日付を表示
max_value = market_index.max()
max_date = market_index.idxmax()
print(f"   -> 市場インデックス最大値: {max_value:.2f} (日付: {max_date.date()})")

fig, ax1 = plt.subplots(figsize=(14, 8))

# 市場インデックス
ax1.plot(market_index.index, market_index, color='tab:blue', alpha=0.5, label='Market Index')
ax1.set_ylabel('Market Index', fontsize=16)  # フォントサイズ変更
ax1.tick_params(axis='both', labelsize=16)  # 軸目盛りのフォントサイズ変更
ax1.grid(True, alpha=0.3)

# RMT指標 (2本)
ax2 = ax1.twinx()
ax2.plot(df_features.index, df_features['RMT_Raw_S'], color='tab:orange', linewidth=1, alpha=0.8, label=f'Short ({WINDOW_S}d)')
ax2.plot(df_features.index, df_features['RMT_Raw_L'], color='tab:red', linewidth=1.5, alpha=0.9, label=f'Long ({WINDOW_L}d)')
ax2.set_ylabel('RMT', fontsize=16)  # フォントサイズ変更
ax2.tick_params(axis='both', labelsize=18)  # 軸目盛りのフォントサイズ変更

# 凡例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=16)  # フォントサイズ変更

# イベント帯
for name, period in scenarios.items():
    start_dt = pd.to_datetime(period[0])
    if start_dt >= market_index.index[0] and start_dt <= market_index.index[-1]:
        ax1.axvline(x=start_dt, color='gray', linestyle=':', alpha=0.6)
        plt.text(start_dt, ax2.get_ylim()[1]*0.98, f" {name}", rotation=90, va='top', fontsize=9)

plt.title("RMT Analysis", fontsize=22)  # タイトルのフォントサイズ変更
plt.tight_layout()
plt.show()

# 保存
output_file = "feature_rmt_dual.csv" # ファイル名変更
df_features.to_csv(output_file)
print(f"✅ 保存完了: {output_file} (列数: {df_features.shape[1]})")

# (前略... データフレーム作成まで同じ)

# --- 修正版: Z-Score を可視化する ---
fig, ax1 = plt.subplots(figsize=(14, 8))

# 1. 市場インデックス (対数表示の方が見やすいかも)
ax1.plot(market_index.index, market_index, color='black', alpha=0.6, label='Market Index')
ax1.set_ylabel('Market Index', fontsize=14)  # フォントサイズ変更
ax1.tick_params(axis='both', labelsize=12)  # 軸目盛りのフォントサイズ変更
ax1.set_yscale('log') # 対数スケール推奨

# 2. RMT Z-Score (AIが見ている異常度)
ax2 = ax1.twinx()
# 危険ライン (Z=2.0)
ax2.axhline(2.0, color='gray', linestyle='--', alpha=0.5, label='Warning Threshold (2σ)')

# Short (60) -> 青
ax2.plot(df_features.index, df_features['RMT_Zscore_S'], 
         color='tab:blue', linewidth=1, alpha=0.8, label=f'Short Z ({WINDOW_S}d)')

# Long (110) -> 赤
ax2.plot(df_features.index, df_features['RMT_Zscore_L'], 
         color='tab:red', linewidth=1.5, alpha=0.8, label=f'Long Z ({WINDOW_L}d)')

ax2.set_ylabel('RMT Z-Score (Anomaly Degree)', fontsize=14)  # フォントサイズ変更
ax2.tick_params(axis='both', labelsize=12)  # 軸目盛りのフォントサイズ変更
ax2.set_ylim(-2, 10) # 異常値を際立たせるため範囲固定

# 凡例など
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)  # フォントサイズ変更
plt.title(f"RMT Anomaly Detection (Z-Window=400)", fontsize=16)  # タイトルのフォントサイズ変更

plt.tight_layout()
plt.show()
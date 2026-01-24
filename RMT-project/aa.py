
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
WINDOW_S = 68  # 短期窓 (急変検知用)
WINDOW_L = 135 # 長期窓 (構造変化検知用)

print(f"🧪 マルチスケールRMT解析: 短期={WINDOW_S}日, 長期={WINDOW_L}日")

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
def add_physics_indicators(df_target, suffix):
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
add_physics_indicators(df_features, 'S')

# --- B. 長期窓 (Long) ---
print(f"🧮 長期窓 ({WINDOW_L}日) を計算中...")
df_features[f'RMT_Raw_L'] = get_max_eigenvalue_series(df_log_returns, WINDOW_L)
add_physics_indicators(df_features, 'L')

# =========================================================
# 4. 可視化 (修正版: Zスコア廃止 -> 生データの推移を表示)
# =========================================================

# --- 市場インデックスの準備 ---
market_cap_file = "market_caps.csv"
if os.path.exists(market_cap_file) and os.path.exists("close_shares.csv") and os.path.exists("stock_close.csv"):
    print("📈 時価総額加重インデックスを使用")
    df_close = pd.read_csv("stock_close.csv", index_col=0, parse_dates=True)
    df_shares = pd.read_csv("close_shares.csv", index_col=0, parse_dates=True)
    df_caps = pd.read_csv(market_cap_file, index_col=0, parse_dates=True)
    df_caps = df_caps.reindex(df_prices.index).ffill()

    market_index = ((df_caps*0)+(df_close)*df_shares).sum(axis=1)
else:
    print("📉 単純平均インデックスを使用")
    market_index = df_prices.mean(axis=1)

# 指数化 (最初を1.0とする)
market_index = market_index / market_index.iloc[0]

# --- プロット作成 ---




fig, ax1 = plt.subplots(figsize=(14, 8))

# 1. 市場インデックス (左軸: 対数表示)
# 株価は指数関数的に動くため、対数スケールの方急落が見やすいです
color_market = 'black'
ax1.set_xlabel('Date', fontsize=14)
ax1.set_ylabel('Market Index (Log Scale)', color=color_market, fontsize=16)
ax1.plot(market_index.index, market_index, color=color_market, alpha=0.6, label='Market Index')
ax1.tick_params(axis='y', labelcolor=color_market, labelsize=14)
ax1.set_yscale('log') # 対数スケール

# 2. RMT 最大固有値 (右軸: 生データ)
ax2 = ax1.twinx()
color_short = 'tab:blue'
color_long = 'tab:red'





ax2.set_ylabel(rf'Max Eigenvalue ($\lambda_{max}$)', color='tab:purple', fontsize=16) # LaTeX表記




# Short (短期窓)
ax2.plot(df_features.index, df_features['RMT_Raw_S'], 
         color=color_short, linewidth=1, alpha=0.8, label=rf'Short $\lambda$ ({WINDOW_S}d)')

# Long (長期窓)
ax2.plot(df_features.index, df_features['RMT_Raw_L'], 
         color=color_long, linewidth=1.5, alpha=0.8, label=rf'Long $\lambda$ ({WINDOW_L}d)')
ax2.tick_params(axis='y', labelsize=14)





# --- 凡例の整理 ---
# 2つの軸の凡例を1箇所にまとめます
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)




# --- イベント帯の表示 ---
# データの範囲内にあるイベントだけを描画
data_start = df_features.index[0]
data_end = df_features.index[-1]

ymin, ymax = ax2.get_ylim() # テキスト位置調整用



for name, period in scenarios.items():
    event_date = pd.to_datetime(period[0])
    if data_start <= event_date <= data_end:
        ax1.axvline(x=event_date, color='gray', linestyle=':', alpha=0.6)
        # グラフの上端より少し下に文字を表示
        plt.text(event_date, ymax * 0.98, f" {name}", rotation=90, va='top', fontsize=10, color='gray')

plt.title("Market Index vs RMT Eigenvalue Analysis", fontsize=20)
plt.tight_layout()
plt.show()

# --- 保存 ---
output_file = "feature_rmt_dual.csv"
df_features.to_csv(output_file)
print(f"✅ 保存完了: {output_file} (列数: {df_features.shape[1]})")
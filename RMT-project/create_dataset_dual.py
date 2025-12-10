import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# 1. 設定
# ---------------------------------------------------------
# 最適化で見つけたターゲット定義
TERM_SHORT = 5
DROP_SHORT = -0.02
TERM_LONG = 10
DROP_LONG = -0.06
ONSET_FILTER = 5

print(f"🎯 データセット構築 (Dual Window RMT)...")

# ---------------------------------------------------------
# 2. 市場データ読み込み
# ---------------------------------------------------------
if os.path.exists("market_caps.csv") and os.path.exists("stock_prices.csv"):
    df_prices = pd.read_csv("stock_prices.csv", index_col=0, parse_dates=True)
    df_caps = pd.read_csv("market_caps.csv", index_col=0, parse_dates=True)
    df_caps = df_caps.reindex(df_prices.index).ffill()
    market_index = df_caps.sum(axis=1)
    market_index = market_index / market_index.iloc[0]
else:
    print("❌ データ不足: get_data.py を実行してください")
    exit()

df_dataset = pd.DataFrame(index=market_index.index)
df_dataset.index.name = "Date"
df_dataset['Market_Price'] = market_index

# ---------------------------------------------------------
# 3. RMT特徴量の結合 (ここが変わった)
# ---------------------------------------------------------
feature_file = "feature_rmt_dual.csv" # calc.pyで保存した新ファイル名

if not os.path.exists(feature_file):
    print("❌ RMTファイルなし: calc.py を実行してください")
    exit()

print("🔗 Dual-Window RMT特徴量を結合中...")
df_features = pd.read_csv(feature_file, index_col=0, parse_dates=True)

# 結合 (Raw_S, Vel_S, ..., Raw_L, Vel_L ... 全部入ります)
df_dataset = df_dataset.join(df_features, how='left')

# ---------------------------------------------------------
# 4. テクニカル指標 & ターゲット (変更なし)
# ---------------------------------------------------------
print("⚙️ テクニカル指標 & ターゲット生成中...")
price = df_dataset['Market_Price']

# Tech
df_dataset['Return'] = price.pct_change()
df_dataset['Vol_20'] = df_dataset['Return'].rolling(20).std()
df_dataset['Momentum_10'] = price / price.shift(10) - 1.0

delta = price.diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df_dataset['RSI_14'] = 100 - (100 / (1 + gain/loss))

# Target
ret_short = price.shift(-TERM_SHORT) / price - 1.0
ret_long = price.shift(-TERM_LONG) / price - 1.0
raw_target = (ret_short <= DROP_SHORT) & (ret_long <= DROP_LONG)
raw_target = raw_target.astype(int)

event_id = (raw_target.diff() != 0).cumsum()
days_since = raw_target.groupby(event_id).cumcount()
final_target = raw_target.copy()
mask_late = (raw_target == 1) & (days_since >= ONSET_FILTER)
final_target[mask_late] = 0

df_dataset['Target'] = final_target

# ---------------------------------------------------------
# 5. 保存
# ---------------------------------------------------------
# 長期窓(L)の計算開始前はNaNになるので削除
df_dataset = df_dataset.dropna()

print(f"📊 最終データ形状: {df_dataset.shape}")
print(f"   ターゲット数: {df_dataset['Target'].sum()}")

output_file = "dataset_ml_dual.csv"
df_dataset.to_csv(output_file)
print(f"💾 保存完了: {output_file}")
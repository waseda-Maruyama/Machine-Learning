import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# 1. 設定
# ---------------------------------------------------------
TERM_SHORT = 5      # 5日後
DROP_SHORT = -0.03  # -3%
TERM_LONG = 10      # 10日後
DROP_LONG = -0.05   # -5%
ONSET_FILTER = 5    # 暴落開始から5日間だけ正解とする

print(f"🎯 データセット構築を開始...")

# ---------------------------------------------------------
# 2. データ読み込み & 結合
# ---------------------------------------------------------
# (A) 株価
df_prices = pd.read_csv("stock_prices_topix100_simple.csv", index_col=0, parse_dates=True)
market_index = df_prices.mean(axis=1)

# (B) RMT特徴量
feature_file = "feature_rmt_eigen_98.csv"
if not os.path.exists(feature_file):
    print("❌ RMTファイルがありません。calc.py を実行してください。")
    exit()
df_features = pd.read_csv(feature_file, index_col=0, parse_dates=True)

# ベースの箱を作成
df_dataset = pd.DataFrame(index=market_index.index)
df_dataset.index.name = "Date"
df_dataset['Market_Price'] = market_index

# RMT結合
df_dataset = df_dataset.join(df_features, how='left')

# ---------------------------------------------------------
# 3. テクニカル指標の計算 (ここに移動！)
# ---------------------------------------------------------
print("⚙️ テクニカル指標を生成中...")
price = df_dataset['Market_Price']

# (1) リターン
df_dataset['Return'] = price.pct_change()

# (2) ボラティリティ (20日)
df_dataset['Vol_20'] = df_dataset['Return'].rolling(20).std()

# (3) モメンタム (10日)
df_dataset['Momentum_10'] = price / price.shift(10) - 1.0

# (4) RSI (14日)
delta = price.diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df_dataset['RSI_14'] = 100 - (100 / (1 + gain/loss))

# ---------------------------------------------------------
# 4. 正解ラベル (Target) の作成
# ---------------------------------------------------------
print("🔨 正解ラベル(Target)を作成中...")

ret_short = price.shift(-TERM_SHORT) / price - 1.0
ret_long = price.shift(-TERM_LONG) / price - 1.0

raw_target = (ret_short <= DROP_SHORT) & (ret_long <= DROP_LONG)
raw_target = raw_target.astype(int)

# Onsetフィルタリング
event_id = (raw_target.diff() != 0).cumsum()
days_since = raw_target.groupby(event_id).cumcount()
final_target = raw_target.copy()
mask_late = (raw_target == 1) & (days_since >= ONSET_FILTER)
final_target[mask_late] = 0

df_dataset['Target'] = final_target

# ---------------------------------------------------------
# 5. クリーニングと保存
# ---------------------------------------------------------
# テクニカル計算(RSI等)やRMT計算で生じたNaNを一括削除
df_dataset = df_dataset.dropna()

print(f"📊 最終データ形状: {df_dataset.shape}")
print(f"   ターゲット数: {df_dataset['Target'].sum()}")

output_file = "dataset_ml.csv"
df_dataset.to_csv(output_file)
print(f"💾 保存完了: {output_file}")
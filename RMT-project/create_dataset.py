import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# 1. 設定 (ターゲット定義)
# ---------------------------------------------------------
TERM_SHORT = 5      # 5日後
DROP_SHORT = -0.03  # -3%
TERM_LONG = 10      # 10日後
DROP_LONG = -0.05   # -5%
ONSET_FILTER = 5    # 暴落開始から5日間だけ正解とする

print(f"🎯 データセット構築を開始...")

# ---------------------------------------------------------
# 2. 市場インデックス (TOPIX型) の作成
# ---------------------------------------------------------
# get_data.py で作成した「時価総額データ」を使います
mcap_file = "market_caps.csv"
price_file = "stock_prices.csv" # 日付合わせ用

if not os.path.exists(mcap_file):
    print(f"⚠️ {mcap_file} が見つかりません。")
    print("   -> 代わりに単純平均(stock_prices.csv)を使います。")
    
    if not os.path.exists(price_file):
        print("❌ ファイルが何もありません。get_data.py を実行してください。")
        exit()
    
    df_prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
    market_index = df_prices.mean(axis=1)

else:
    print(f"⚖️ {mcap_file} を読み込み中 (正確なTOPIX型指数を作成)...")
    df_caps = pd.read_csv(mcap_file, index_col=0, parse_dates=True)
    
    # 全銘柄の時価総額を合計 = 市場全体の規模
    market_cap_total = df_caps.sum(axis=1)
    
    # 指数化 (初期値を1.0とする)
    market_index = market_cap_total / market_cap_total.iloc[0]

# ベースの箱を作成
df_dataset = pd.DataFrame(index=market_index.index)
df_dataset.index.name = "Date"
df_dataset['Market_Price'] = market_index

# ---------------------------------------------------------
# 3. RMT特徴量の結合
# ---------------------------------------------------------
feature_file = "feature_rmt_eigen_98.csv" # calc.pyの出力
if not os.path.exists(feature_file):
    print("❌ RMTファイルがありません。calc.py を実行してください。")
    exit()

print("🔗 RMT特徴量を結合中...")
df_features = pd.read_csv(feature_file, index_col=0, parse_dates=True)

# 結合 (Left Join)
df_dataset = df_dataset.join(df_features, how='left')

# ---------------------------------------------------------
# 4. テクニカル指標の計算 (Market_Priceに対して)
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
# 5. 正解ラベル (Target) の作成
# ---------------------------------------------------------
print("🔨 正解ラベル(Target)を作成中...")

# 未来のリターン (Shiftで未来を覗く)
ret_short = price.shift(-TERM_SHORT) / price - 1.0
ret_long = price.shift(-TERM_LONG) / price - 1.0

# 複合条件 (AND)
raw_target = (ret_short <= DROP_SHORT) & (ret_long <= DROP_LONG)
raw_target = raw_target.astype(int)

# Onsetフィルタリング (暴落の入り口だけを1にする)
event_id = (raw_target.diff() != 0).cumsum()
days_since = raw_target.groupby(event_id).cumcount()

final_target = raw_target.copy()
# 条件を満たしていても、発生からN日以上経過したら0に戻す
mask_late = (raw_target == 1) & (days_since >= ONSET_FILTER)
final_target[mask_late] = 0

df_dataset['Target'] = final_target

# ---------------------------------------------------------
# 6. クリーニングと保存
# ---------------------------------------------------------
# RMT計算前の期間や、テクニカル計算の初期期間(NaN)を削除
df_dataset = df_dataset.dropna()

print(f"📊 最終データ形状: {df_dataset.shape}")
print(f"   ターゲット(暴落)数: {df_dataset['Target'].sum()} 日")

output_file = "dataset_ml.csv"
df_dataset.to_csv(output_file)
print(f"💾 保存完了: {output_file}")
print("   -> 次は train.py で学習を開始してください！")
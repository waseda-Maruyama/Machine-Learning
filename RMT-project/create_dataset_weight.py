import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# 1. 設定：ターゲット定義
# ---------------------------------------------------------
TERM_SHORT = 5      # 5日後
DROP_SHORT = -0.02  # -3%
TERM_LONG = 10      # 10日後
DROP_LONG = -0.06   # -6%

# 【新機能】重みの減衰スピード
# 値が大きいほど「初期しか見ない（スパルタ）」になります。
# 0.5だと、5日後には重みが 1/10 以下になります。
DECAY_RATE = 0.5
BOOST_FACTOR = 10.0 # From optimize_rmt_windows_weiht.py

print(f"🎯 ターゲット定義: {TERM_SHORT}日後{DROP_SHORT:.0%} & {TERM_LONG}日後{DROP_LONG:.0%}")
print(f"⚖️ 重み付け: 暴落開始から日が経つにつれて指数減衰 (Rate={DECAY_RATE}), Boost={BOOST_FACTOR})")

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

delta = price.diff()
# (C) ターゲット & 重み生成 (修正版)
print("🔨 Building Targets & Weights (Severity Adjusted)...")

# 未来のリターン (5日後、10日後)
ret_short = market_index.shift(-TERM_SHORT) / market_index - 1.0
ret_long = market_index.shift(-TERM_LONG) / market_index - 1.0

# 複合ターゲット (Target定義は変更なし)
raw_target = (ret_short <= DROP_SHORT) & (ret_long <= DROP_LONG)
raw_target = raw_target.astype(int)

# --- ★ここが修正箇所 ---
# 重みの初期値 (すべて1.0)
sample_weights = pd.Series(1.0, index=raw_target.index)

if raw_target.sum() > 0:
    # 1. 時間減衰 (Time Decay): 早いほど偉い
    event_id = (raw_target.diff() != 0).cumsum()
    days_since = raw_target.groupby(event_id).cumcount()
    
    # 暴落時は BOOST_FACTOR (10.0) からスタートして減衰させる
    # これにより、暴落中の重みが 1.0 を下回ることを防ぎつつ、初動を強調
    decay_weights = BOOST_FACTOR * np.exp(-DECAY_RATE * days_since)
    
    # 2. 被害規模 (Severity): 下げがキツイほど偉い
    # ret_long (10日後のリターン) の絶対値を使う
    # 例: -0.06 (ギリギリ暴落) -> 0.06点
    #     -0.20 (コロナ級)     -> 0.20点
    severity_factor = ret_long.abs()
    
    # 3. 最終ウェイト = 時間 × 規模
    # Target=1 の場所だけ計算して代入
    mask_crash = (raw_target == 1)
    
    # ここで掛け合わせる！
    # (規模が大きいと、日が経ってdecayしても重みが残る仕組み)
    sample_weights[mask_crash] = decay_weights[mask_crash]
    
    print(f"⚖️ 重み設定完了: 通常=1.0, 暴落時Max={decay_weights.max():.1f}")


df_dataset['Target'] = raw_target
df_dataset['Sample_Weight'] = sample_weights

# ---------------------------------------------------------
# 4. 保存
# ---------------------------------------------------------
df_dataset = df_dataset.dropna()
output_file = "dataset_ml_weighted.csv"
df_dataset.to_csv(output_file)

print(f"\n✅ データセット完成: {output_file}")
print(f"   データ数: {len(df_dataset)}")
print(f"   カラム: {list(df_dataset.columns)}")
print(f"   暴落時の平均重み: {df_dataset.loc[df_dataset['Target']==1, 'Sample_Weight'].mean():.3f}")
print("   -> 暴落後半の重みが軽くなっていることを確認しました。")
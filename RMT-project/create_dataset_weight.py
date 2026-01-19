import pandas as pd
import numpy as np
import os

# =========================================================
# 1. 設定：ターゲット定義 & 重みパラメータ
# =========================================================
# 前回の分析結果(Row 3)に基づき、ここを修正するとより強力になります
TERM_SHORT = 3      # 推奨: 3日
DROP_SHORT = -0.02  # 推奨: -2%
TERM_LONG = 10      # 推奨: 10日
DROP_LONG = -0.08  # 推奨: -8% (より深い暴落を狙う場合)

# 重みの設定
DECAY_RATE = 0.5    # 減衰スピード
BOOST_FACTOR = 10.0 # 初動の基本ブースト値

print(f"🎯 ターゲット定義: {TERM_SHORT}日後{DROP_SHORT:.0%} & {TERM_LONG}日後{DROP_LONG:.0%}")
print(f"⚖️ 重み付け: 暴落開始から指数減衰 (Rate={DECAY_RATE}) × 深刻度倍率")

# =========================================================
# 2. 市場データ読み込み
# =========================================================
if os.path.exists("market_caps.csv") and os.path.exists("stock_prices.csv"):
    df_prices = pd.read_csv("stock_prices.csv", index_col=0, parse_dates=True)
    
    # 時価総額ファイルがある場合
    df_caps = pd.read_csv("market_caps.csv", index_col=0, parse_dates=True)
    df_caps = df_caps.reindex(df_prices.index).ffill()
    
    # 共通のカラムのみで計算（エラー回避）
    common_cols = df_prices.columns.intersection(df_caps.columns)
    # 時価総額の合計（既に株価×発行株式数が含まれている）
    total_market_cap = df_caps[common_cols].sum(axis=1)
    
    # 正規化
    market_index = total_market_cap / total_market_cap.iloc[0]
else:
    print("❌ データ不足: stock_prices.csv または market_caps.csv が見つかりません")
    exit()

df_dataset = pd.DataFrame(index=market_index.index)
df_dataset.index.name = "Date"
df_dataset['Market_Price'] = market_index

# =========================================================
# 3. RMT特徴量の結合
# =========================================================
feature_file = "feature_rmt_dual.csv"

if not os.path.exists(feature_file):
    print("❌ RMTファイルなし: calc.py を実行してください")
    exit()

print("🔗 Dual-Window RMT特徴量を結合中...")
df_features = pd.read_csv(feature_file, index_col=0, parse_dates=True)
df_dataset = df_dataset.join(df_features, how='left')

# =========================================================
# 4. テクニカル指標作成
# =========================================================
print("⚙️ テクニカル指標を生成中...")
price = df_dataset['Market_Price']

df_dataset['Return'] = price.pct_change()
df_dataset['Vol_20'] = df_dataset['Return'].rolling(20).std()
df_dataset['Momentum_10'] = price / price.shift(10) - 1.0

delta = price.diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df_dataset['RSI_14'] = 100 - (100 / (1 + gain/loss))

# =========================================================
# 5. ターゲット & 重み生成 (★ここを修正済み)
# =========================================================
print("🔨 Building Targets & Weights (Severity Adjusted)...")

# 未来のリターン計算
ret_short = market_index.shift(-TERM_SHORT) / market_index - 1.0
ret_long = market_index.shift(-TERM_LONG) / market_index - 1.0

# 複合ターゲット判定
raw_target = (ret_short <= DROP_SHORT) & (ret_long <= DROP_LONG)
raw_target = raw_target.astype(int)

# 重みの初期化 (デフォルトは 1.0)
sample_weights = pd.Series(1.0, index=raw_target.index)

if raw_target.sum() > 0:
    # 1. 時間減衰 (Time Decay)
    # 暴落イベントごとにグループ化し、経過日数をカウント
    event_id = (raw_target.diff() != 0).cumsum()
    days_since = raw_target.groupby(event_id).cumcount()
    
    # 減衰成分: 日が経つにつれて小さくなる (例: 10.0 -> 6.0 -> 3.6 ...)
    decay_comp = BOOST_FACTOR * np.exp(-DECAY_RATE * days_since)
    
    # 2. 被害規模 (Severity Ratio)
    # 基準(-6%)に対して、実際どれくらい酷いか？
    # 例: 実績が -12% なら Ratio = 2.0 (倍の重みを与える)
    severity_ratio = ret_long.abs() / abs(DROP_LONG)
    
    # 3. 結合 (Decay × Severity)
    # マスク作成 (Target=1 の場所のみ計算)
    mask_crash = (raw_target == 1)
    
    # 掛け合わせる
    final_weights = decay_comp * severity_ratio
    
    # 代入
    sample_weights[mask_crash] = final_weights[mask_crash]
    
    # 統計情報の表示
    max_w = sample_weights.max()
    mean_w = sample_weights[mask_crash].mean()
    print(f"⚖️ 重み計算完了:")
    print(f"   - Max Weight : {max_w:.2f} (大暴落の初動)")
    print(f"   - Mean Weight: {mean_w:.2f} (暴落期間の平均)")
    print(f"   - Normal     : 1.00")

# データフレームに格納
df_dataset['Target'] = raw_target
df_dataset['Sample_Weight'] = sample_weights

# =========================================================
# 6. 保存
# =========================================================
# NaNを含む行（計算できない初期データや未来データ）を削除
df_dataset = df_dataset.dropna()

output_file = "dataset_ml_weighted.csv"
df_dataset.to_csv(output_file)

print(f"\n✅ データセット完成: {output_file}")
print(f"   データ数: {len(df_dataset)}")
print(f"   使用カラム: {list(df_dataset.columns)}")
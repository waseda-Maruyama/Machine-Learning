import pandas as pd
import numpy as np
import os

# =========================================================
# 1. 設定：ターゲット定義 & 重みパラメータ
# =========================================================
# 前回の分析結果(Row 3)に基づき、ここを修正するとより強力になります
TERM_SHORT = 3      # 推奨: 3日
DROP_SHORT = -0.02  # 推奨: -2%
TERM_LONG = 10     # 推奨: 10日
DROP_LONG = -0.08  # 推奨: -8% (より深い暴落を狙う場合)

# 重みの設定
DECAY_RATE = 0.5    # 減衰スピード
BOOST_FACTOR = 5.0 # 初動の基本ブースト値

print(f"🎯 ターゲット定義: {TERM_SHORT}日後{DROP_SHORT:.0%} & {TERM_LONG}日後{DROP_LONG:.0%}")
print(f"⚖️ 重み付け: 暴落開始から指数減衰 (Rate={DECAY_RATE}) × 深刻度倍率")

# =========================================================
# 2. 市場データ読み込み
# =========================================================
if os.path.exists("market_caps.csv") and os.path.exists("stock_adj_close.csv") and os.path.exists("stock_close.csv"):
    df_adj_close = pd.read_csv("stock_adj_close.csv", index_col=0, parse_dates=True)
    df_close = pd.read_csv("stock_close.csv", index_col=0, parse_dates=True)

    # 時価総額ファイルがある場合
    df_caps = pd.read_csv("market_caps.csv", index_col=0, parse_dates=True)
    df_caps = df_caps.reindex(df_close.index).ffill()
    
    # 三つのファイルで共通のカラムのみで計算（エラー回避）
    common_cols = df_adj_close.columns.intersection(df_caps.columns)

    # 時価総額の合計（既に株価×発行株式数が含まれている）
    total_market_cap_a = df_caps[common_cols].sum(axis=1)
    total_market_cap_b = (df_caps[common_cols] * df_adj_close[common_cols]).sum(axis=1)

    # 正規化
    market_index_a = total_market_cap_a / total_market_cap_a.iloc[0]
    market_index_b = total_market_cap_b / total_market_cap_b.iloc[0]

else:
    print("❌ データ不足: stock_adj_close.csv または market_caps.csv が見つかりません")
    exit()

df_dataset = pd.DataFrame({
    'Market_Price_A': market_index_a,
    'Market_Price_B': market_index_b
})
df_dataset.index.name = 'Date'

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
price_a = df_dataset['Market_Price_A']
price_b = df_dataset['Market_Price_B']

df_dataset['Return'] = price_a.pct_change()
df_dataset['Vol_20'] = df_dataset['Return'].rolling(20).std()
df_dataset['Momentum_10'] = price_a / price_a.shift(10) - 1.0

delta_a = price_a.diff()
gain_a = (delta_a.where(delta_a > 0, 0)).rolling(14).mean()
loss_a = (-delta_a.where(delta_a < 0, 0)).rolling(14).mean()
df_dataset['RSI_14'] = 100 - (100 / (1 + gain_a/loss_a))




# =========================================================
# ★ 4.5. ポテンシャル・エネルギー特徴量（株式分割ノイズ除去版）
# =========================================================
print("🔋 ポテンシャル・エネルギー特徴量を生成中（ノイズ除去モード）...")

# データのマッピング（前提）
# df_raw: 生株価 (ユーザー環境の df_close)
# df_adj: 調整株価 (ユーザー環境の df_close_adj)
# df_caps: 時価総額

# 1. 共通カラムの抽出
common_cols = df_close.columns.intersection(df_caps.columns)
df_raw_common = df_close[common_cols]
df_adj_common = df_adj_close[common_cols]
df_caps_common = df_caps[common_cols]

# 2. 歪み項（Distortion）の計算（調整株価ベース）
window = 60
P_bar = df_adj_common.rolling(window).mean()
sigma = df_adj_common.rolling(window).std()

# z^2 (無次元の歪みエネルギー)
# これは連続量なので、微分してもスパイクしない
distortion = ((df_adj_common - P_bar) / (sigma + 1e-8)) ** 2

# 3. 物理スケール項（有効質量）
# Scale = 時価総額 * 生株価 (値がさ株ほど重い)
# ※ここは株式分割で「段差」ができるが、微分はしないのでOK
physical_mass = df_caps_common * df_adj_common 

# 正規化（市場全体での相対的な重み）
# これをしないと、市場全体の株価水準上昇で値がインフレし続ける
# 特徴量として安定させるなら正規化推奨、絶対額を見たいなら不要
# ここでは「全体の中での局所的なエネルギー集中」を見るため、総和で割ります
total_mass = physical_mass.sum(axis=1) + 1e-8
w_physical = physical_mass.div(total_mass, axis=0)

# =========================================================
# ★重要: スパイクを防ぐための計算順序の変更
# =========================================================

# [A] エネルギー準位 (Level)
# これは「状態」を表すので、分割による段差があっても良い（ツリーモデルが処理する）
df_dataset['E_pot'] = (w_physical * distortion).sum(axis=1)

# [B] エネルギー速度 (Velocity) & 加速度 (Accel)
# ここで段差を微分しないように、「歪みの変化」に「質量」を掛ける

# 1. 個別銘柄ごとの歪み変化量 (Δz^2)
delta_distortion = distortion.diff()

# 2. 物理的パワー (Power) = 質量 * 歪み変化速度
# Sum( m_i * Δz^2_i )
power_series = (w_physical * delta_distortion).sum(axis=1)

# 滑らかにして特徴量化
smooth_window = 5
df_dataset['E_pot_Vel'] = power_series.rolling(smooth_window).mean()

# 加速度は、速度の微分ではなく、再度「変化量」として計算するのが安全だが、
# 既にVelが滑らかなので、単純なdiffでもスパイクは抑制されているはず
df_dataset['E_pot_Accel'] = df_dataset['E_pot_Vel'].diff()

print("✅ 生成完了: 分割によるスパイクを除去しました。")



# =========================================================
# 5. ターゲット & 重み生成 (★ここを修正済み)
# =========================================================
print("🔨 Building Targets & Weights (Severity Adjusted)...")

#未来のリターン計算 a
#ret_short = market_index_a.shift(-TERM_SHORT) / market_index_a - 1.0
#ret_long = market_index_a.shift(-TERM_LONG) / market_index_a - 1.0


# 未来のリターン計算 b
ret_short = market_index_b.shift(-TERM_SHORT) / market_index_b - 1.0
ret_long = market_index_b.shift(-TERM_LONG) / market_index_b - 1.0

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
    
# 2. イベント全体の被害規模 (Event Severity)
    # イベントごとに、期間中の「最大の下落幅」を特定する
    # ※ ret_long は負の値なので、min() で最大の下落（最も負に大きい値）を取得
    event_max_drop = ret_long.groupby(event_id).transform('min')
    
    # イベント全体の深刻度スコア (基準値 DROP_LONG に対する比率の二乗など)
    # これにより「一度でも-20%まで行ったイベント」は、その期間全体の重みが底上げされる
    event_severity_score = (event_max_drop.abs() / abs(DROP_LONG))
    # 3. 結合 (Decay × Severity)
    # マスク作成 (Target=1 の場所のみ計算)
    mask_crash = (raw_target == 1)
    
    # 掛け合わせる
    final_weights = decay_comp * event_severity_score
    
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

output_file = "dataset_ml_weighted_2.csv"
df_dataset.to_csv(output_file)

print(f"\n✅ データセット完成: {output_file}")
print(f"   データ数: {len(df_dataset)}")
print(f"   使用カラム: {list(df_dataset.columns)}")
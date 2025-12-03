import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. データ読み込み & 結合 (Alignment)
# ---------------------------------------------------------
print("🛠️ データセット構築を開始...")

# (1) マスターデータの読み込み (株価)
try:
    df_prices = pd.read_csv("stock_prices_topix100_simple.csv", index_col=0, parse_dates=True)
    market_index = df_prices.mean(axis=1)
except FileNotFoundError:
    print("❌ 'stock_prices_topix100_simple.csv' が見つかりません。")
    exit()

# (2) RMT特徴量の読み込み
try:
    ts_rmt = pd.read_csv("feature_rmt_eigen_98.csv", index_col=0, parse_dates=True)
    # Seriesとして整形
    if isinstance(ts_rmt, pd.DataFrame):
        ts_rmt = ts_rmt.iloc[:, 0]
    ts_rmt.name = "RMT_Raw"
except FileNotFoundError:
    print("❌ 'feature_rmt_eigen_98.csv' が見つかりません。")
    exit()

# (3) 厳密な結合 (株価の日付を正とする)
df_ml = pd.DataFrame(index=market_index.index)
df_ml['Market_Price'] = market_index
df_ml['RMT_Raw'] = ts_rmt # 日付マッチングで結合（窓枠分はNaNになる）

# ---------------------------------------------------------
# 2. 特徴量エンジニアリング
# ---------------------------------------------------------
print("⚙️ 特徴量生成中 (物理指標の平滑化を含む)...")

# --- (A) ベースライン特徴量 (Tech) ---
# リターン & ボラティリティ
df_ml['Return'] = df_ml['Market_Price'].pct_change()
df_ml['Vol_20'] = df_ml['Return'].rolling(20).std()
df_ml['Momentum_10'] = df_ml['Market_Price'] / df_ml['Market_Price'].shift(10) - 1.0

# RSI (14)
delta = df_ml['Market_Price'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df_ml['RSI_14'] = 100 - (100 / (1 + gain/loss))

# --- (B) RMT特徴量の「物理的進化」 (ここが修正の核心) ---
# 1. 位置 (Raw) -> そのまま使う
# (既に df_ml['RMT_Raw'] にある)

# 【重要】ノイズ対策: 5日移動平均で平滑化
# 微分する前に必ず平滑化を行い、高周波ノイズを除去する
rmt_smooth = df_ml['RMT_Raw'].rolling(window=5).mean()

# 2. 速度 (Velocity) ※平滑化したものを使う
df_ml['RMT_Vel'] = rmt_smooth.diff()

# 3. 加速度 (Acceleration) ※平滑化したものを使う
df_ml['RMT_Accel'] = df_ml['RMT_Vel'].diff()

# 4. 緊張度 (Z-Score)
# ※ Z-Scoreは「位置の異常度」なので、生データ(Raw)を使って分布を見る
window_z = 250
rmt_mean = df_ml['RMT_Raw'].rolling(window_z).mean()
rmt_std = df_ml['RMT_Raw'].rolling(window_z).std()
df_ml['RMT_Zscore'] = (df_ml['RMT_Raw'] - rmt_mean) / rmt_std

# ---------------------------------------------------------
# 3. ターゲット生成 (10日 / -7% 固定)
# ---------------------------------------------------------
LOOKAHEAD = 10
THRESHOLD = -0.07

future_min = df_ml['Market_Price'].rolling(LOOKAHEAD).min().shift(-LOOKAHEAD)
drawdown = (future_min - df_ml['Market_Price']) / df_ml['Market_Price']
df_ml['Target'] = (drawdown <= THRESHOLD).astype(int)

# NaN削除 (先頭の窓枠、Z-score計算期間、ターゲットの未来分を一括削除)
df_ml = df_ml.dropna()

print(f"📊 学習データセット完了: {len(df_ml)} 行")
print(f"   暴落発生率: {df_ml['Target'].mean():.2%}")

# ---------------------------------------------------------
# 4. イベント駆動型検証 (Train & Test)
# ---------------------------------------------------------
# 特徴量セット定義
features_A = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14']
# 進化版RMTセット (Raw, Velocity, Accel, Zscore)
features_B = features_A + ['RMT_Raw', 'RMT_Vel', 'RMT_Accel', 'RMT_Zscore']

scenarios = {
    "1. 2018 VIX": ("2018-01-01", "2018-06-30"),
    "2. 2020 Covid": ("2020-01-01", "2020-06-30"),
    "3. 2024 Ueda": ("2024-06-01", "2024-10-31"),
    "4. 2025 Tariff": ("2025-01-01", "2025-08-30")
}

print("\n🤖 学習開始 (Noise-Reduced RMT Model)...")

results = []
last_model = None

for name, (start, end) in scenarios.items():
    test_start = pd.to_datetime(start)
    test_end = pd.to_datetime(end)
    
    # マスク作成 (Walk-Forward)
    train_mask = df_ml.index < test_start
    test_mask = (df_ml.index >= test_start) & (df_ml.index <= test_end)
    
    y_test = df_ml.loc[test_mask, 'Target']
    
    if y_test.sum() == 0:
        print(f"⚠️ {name}: 期間中に暴落なし。スキップ。")
        continue

    # データ分割
    X_train = df_ml.loc[train_mask]
    y_train = df_ml.loc[train_mask, 'Target']
    X_test = df_ml.loc[test_mask]
    
    # 不均衡対策
    if y_train.sum() == 0: continue
    w = len(X_train) / (2 * y_train.sum())
    
    # --- Model A (Tech Only) ---
    clf_a = lgb.LGBMClassifier(random_state=42, scale_pos_weight=w, verbose=-1, n_jobs=1)
    clf_a.fit(X_train[features_A], y_train)
    pred_a = clf_a.predict(X_test[features_A])
    recall_a = recall_score(y_test, pred_a, zero_division=0)
    
    # --- Model B (RMT Enhanced) ---
    clf_b = lgb.LGBMClassifier(random_state=42, scale_pos_weight=w, verbose=-1, n_jobs=1)
    clf_b.fit(X_train[features_B], y_train)
    pred_b = clf_b.predict(X_test[features_B])
    recall_b = recall_score(y_test, pred_b, zero_division=0)
    
    last_model = clf_b
    results.append({'Scenario': name, 'A': recall_a, 'B': recall_b})
    
    print(f"{name}: Recall A={recall_a:.3f} vs B={recall_b:.3f}")

# ---------------------------------------------------------
# 5. 最終結果の集計
# ---------------------------------------------------------
if results:
    df_res = pd.DataFrame(results)
    mean_a = df_res['A'].mean()
    mean_b = df_res['B'].mean()
    
    print("\n🏆 最終結果 (平均Recall):")
    print(f"Model A (Tech Only)  : {mean_a:.3f}")
    print(f"Model B (RMT Enhanced): {mean_b:.3f}")
    
    diff = mean_b - mean_a
    if diff > 0:
        print(f"\n✅ RMT導入により精度向上 (+{diff:.3f})")
    else:
        print(f"\n⚠️ RMT導入効果なし ({diff:.3f})")

    # 特徴量重要度 (最後のモデル)
    print("\n🔍 重要度ランキング (Gain):")
    imp = pd.DataFrame({'Feature': features_B, 'Gain': last_model.feature_importances_})
    print(imp.sort_values('Gain', ascending=False))
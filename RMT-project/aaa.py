import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. データ読み込み (株価データのみ外部ファイル)
# ---------------------------------------------------------
print("🛠️ データセット構築を開始...")
try:
    df_prices = pd.read_csv("stock_prices_topix100_simple.csv", index_col=0, parse_dates=True)
except FileNotFoundError:
    print("❌ 'stock_prices_topix100_simple.csv' が見つかりません。")
    exit()

# ---------------------------------------------------------
# 2. RMT計算 (ここで計算します！)
# ---------------------------------------------------------
# 窓幅設定: 最適化された 80日
WINDOW_RMT = 80

print(f"🧮 RMT(Window={WINDOW_RMT}) をリアルタイム計算中...")

# 対数収益率
df_log_returns = np.log(df_prices / df_prices.shift(1)).dropna()

# 高速計算用関数
def calculate_rmt_fast(returns, window):
    vals = [np.nan] * window
    # numpy配列化して高速化
    data_values = returns.values
    for i in range(window, len(returns)):
        # スライス
        window_data = data_values[i-window : i]
        # 相関行列 (ここだけpandasの挙動を再現するためDataFrameに戻す手もあるが、
        # 速度優先でnumpy完結させる)
        # ※欠損がない前提(dropna済み)なのでnumpy.corrcoefでOK
        # 行=変数にするため転置
        corr_mat = np.corrcoef(window_data.T)
        # NaNケア (万が一分散0の場合)
        np.nan_to_num(corr_mat, copy=False)
        
        # 固有値分解 (エルミート)
        eigvals = np.linalg.eigvalsh(corr_mat)
        vals.append(eigvals[-1])
        
    return pd.Series(vals, index=returns.index, name="RMT_Raw")

ts_rmt = calculate_rmt_fast(df_log_returns, window=WINDOW_RMT)

# ---------------------------------------------------------
# 3. 特徴量エンジニアリング & 結合
# ---------------------------------------------------------
print("⚙️ 特徴量生成中...")

# ベースライン (市場平均)
market_index = df_prices.mean(axis=1)
df_ml = pd.DataFrame(index=market_index.index)
df_ml['Market_Price'] = market_index

# (A) Tech Features
df_ml['Return'] = df_ml['Market_Price'].pct_change()
df_ml['Vol_20'] = df_ml['Return'].rolling(20).std()
df_ml['Momentum_10'] = df_ml['Market_Price'] / df_ml['Market_Price'].shift(10) - 1.0
delta = df_ml['Market_Price'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df_ml['RSI_14'] = 100 - (100 / (1 + gain/loss))

# (B) RMT Features (結合)
df_ml['RMT_Raw'] = ts_rmt

# 【重要】ノイズ対策: 5日移動平均で平滑化してから微分
rmt_smooth = df_ml['RMT_Raw'].rolling(window=5).mean()

# 速度 (Velocity)
df_ml['RMT_Vel'] = rmt_smooth.diff()
# 加速度 (Acceleration)
df_ml['RMT_Accel'] = df_ml['RMT_Vel'].diff()
# 緊張度 (Z-Score)
window_z = 250
rmt_mean = df_ml['RMT_Raw'].rolling(window_z).mean()
rmt_std = df_ml['RMT_Raw'].rolling(window_z).std()
df_ml['RMT_Zscore'] = (df_ml['RMT_Raw'] - rmt_mean) / rmt_std

# (C) ターゲット (10日 / -7% 固定)
LOOKAHEAD = 10
THRESHOLD = -0.07
future_min = df_ml['Market_Price'].rolling(LOOKAHEAD).min().shift(-LOOKAHEAD)
drawdown = (future_min - df_ml['Market_Price']) / df_ml['Market_Price']
df_ml['Target'] = (drawdown <= THRESHOLD).astype(int)

# NaN削除
df_ml = df_ml.dropna()

print(f"📊 データセット完了: {len(df_ml)} 行")
print(f"   暴落発生率: {df_ml['Target'].mean():.2%}")

# ---------------------------------------------------------
# 4. イベント駆動型検証
# ---------------------------------------------------------
features_A = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14']
features_B = features_A + ['RMT_Raw', 'RMT_Vel', 'RMT_Accel', 'RMT_Zscore']

scenarios = {
    "1. 2018 VIX": ("2018-01-01", "2018-06-30"),
    "2. 2020 Covid": ("2020-01-01", "2020-06-30"),
    "3. 2024 Ueda": ("2024-06-01", "2024-10-31"),
    "4. 2025 Tariff": ("2025-01-01", "2025-08-30")
}

print("\n🤖 学習開始...")

results = []
last_model = None

for name, (start, end) in scenarios.items():
    test_start = pd.to_datetime(start)
    test_end = pd.to_datetime(end)
    
    train_mask = df_ml.index < test_start
    test_mask = (df_ml.index >= test_start) & (df_ml.index <= test_end)
    
    y_test = df_ml.loc[test_mask, 'Target']
    
    if y_test.sum() == 0:
        print(f"⚠️ {name}: 期間中に暴落なし。スキップ。")
        continue

    X_train = df_ml.loc[train_mask]
    y_train = df_ml.loc[train_mask, 'Target']
    X_test = df_ml.loc[test_mask]
    
    w = len(X_train) / (2 * y_train.sum())
    
    # Model A
    clf_a = lgb.LGBMClassifier(random_state=42, scale_pos_weight=w, verbose=-1, n_jobs=1)
    clf_a.fit(X_train[features_A], y_train)
    pred_a = clf_a.predict(X_test[features_A])
    recall_a = recall_score(y_test, pred_a, zero_division=0)
    
    # Model B
    clf_b = lgb.LGBMClassifier(random_state=42, scale_pos_weight=w, verbose=-1, n_jobs=1)
    clf_b.fit(X_train[features_B], y_train)
    pred_b = clf_b.predict(X_test[features_B])
    recall_b = recall_score(y_test, pred_b, zero_division=0)
    
    last_model = clf_b
    results.append({'Scenario': name, 'A': recall_a, 'B': recall_b})
    
    print(f"{name}: Recall A={recall_a:.3f} vs B={recall_b:.3f}")

if results:
    df_res = pd.DataFrame(results)
    mean_a = df_res['A'].mean()
    mean_b = df_res['B'].mean()
    
    print("\n🏆 最終結果 (平均Recall):")
    print(f"Model A: {mean_a:.3f}")
    print(f"Model B: {mean_b:.3f}")
    
    if mean_b > mean_a:
        print(f"\n✅ RMT導入により精度向上 (+{mean_b - mean_a:.3f})")
    else:
        print(f"\n⚠️ RMT導入効果なし ({mean_b - mean_a:.3f})")

    print("\n🔍 重要度ランキング:")
    imp = pd.DataFrame({'Feature': features_B, 'Gain': last_model.feature_importances_})
    print(imp.sort_values('Gain', ascending=False))
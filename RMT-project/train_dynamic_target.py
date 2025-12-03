import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import TimeSeriesSplit

# 1. データ準備
print("📊 データを準備中...")
try:
    df_prices = pd.read_csv("stock_prices_topix100_simple.csv", index_col=0, parse_dates=True)
except FileNotFoundError:
    print("❌ ファイルが見つかりません。")
    exit()

# RMT計算 (Window=20) ※ここが変わりました
df_log_returns = np.log(df_prices / df_prices.shift(1)).dropna()
def calculate_rmt(returns, window):
    vals = [np.nan] * window
    for i in range(window, len(returns)):
        window_data = returns.iloc[i-window : i]
        corr_mat = window_data.corr().fillna(0)
        vals.append(np.linalg.eigh(corr_mat)[0][-1])
    return pd.Series(vals, index=returns.index)

print("🧮 RMTを計算中...")
ts_rmt = calculate_rmt(df_log_returns, window=120)

# 特徴量作成
market_index = df_prices.mean(axis=1)
df_ml = pd.DataFrame(index=market_index.index)

# A: Tech
df_ml['Return'] = market_index.pct_change()
df_ml['Vol_20'] = df_ml['Return'].rolling(20).std()
df_ml['Momentum_10'] = market_index / market_index.shift(10) - 1.0
delta = market_index.diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df_ml['RSI_14'] = 100 - (100 / (1 + gain/loss))

# B: RMT (進化版)
df_ml['RMT_Raw'] = ts_rmt
df_ml['RMT_Diff'] = df_ml['RMT_Raw'].diff()
df_ml['RMT_Accel'] = df_ml['RMT_Diff'].diff()
# Z-Score (Window=20の場合は、Zスコアの基準期間も短くすべきか検討余地ありだが、一旦250で固定)
df_ml['RMT_Z'] = (df_ml['RMT_Raw'] - df_ml['RMT_Raw'].rolling(250).mean()) / df_ml['RMT_Raw'].rolling(250).std()

# C: ターゲット (動的: -2.0 * Sigma)
LOOKAHEAD = 5 # 期間も5日に短縮して感度を合わせる
indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=LOOKAHEAD)
future_min = df_ml['Return'].rolling(window=indexer).min().shift(-1)
threshold = -2.5 * df_ml['Vol_20']
df_ml['Target'] = (future_min < threshold).astype(int)

df_ml = df_ml.dropna()

print(f"暴落発生率(Y=1): {df_ml['Target'].mean():.2%}")

# 2. 全シナリオ A/Bテスト
features_A = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14']
features_B = features_A + ['RMT_Raw', 'RMT_Diff', 'RMT_Accel', 'RMT_Z']

scenarios = {
    "1. 2018 VIX": ("2018-01-01", "2018-06-30"),
    "2. 2020 Covid": ("2020-01-01", "2020-06-30"),
    "3. 2024 Ueda": ("2024-06-01", "2024-10-31"),
    "4. 2025 Tariff": ("2025-01-01", "2025-08-30")
}

print("\n🤖 動的ターゲット × 最適窓(20日) 検証開始...")

results = []
for name, (start, end) in scenarios.items():
    test_start = pd.to_datetime(start)
    test_end = pd.to_datetime(end)
    train_mask = df_ml.index < test_start
    test_mask = (df_ml.index >= test_start) & (df_ml.index <= test_end)
    
    y_test = df_ml.loc[test_mask, 'Target']
    if y_test.sum() == 0: continue
        
    X_train = df_ml.loc[train_mask]
    y_train = df_ml.loc[train_mask, 'Target']
    X_test = df_ml.loc[test_mask]
    
    w = len(X_train) / (2 * y_train.sum())
    
    # Model A
    clf_a = lgb.LGBMClassifier(random_state=42, scale_pos_weight=w, verbose=-1)
    clf_a.fit(X_train[features_A], y_train)
    
    # Model B
    clf_b = lgb.LGBMClassifier(random_state=42, scale_pos_weight=w, verbose=-1)
    clf_b.fit(X_train[features_B], y_train)
    
    rec_a = recall_score(y_test, clf_a.predict(X_test[features_A]), zero_division=0)
    rec_b = recall_score(y_test, clf_b.predict(X_test[features_B]), zero_division=0)
    
    results.append({'Scenario': name, 'Recall_A': rec_a, 'Recall_B': rec_b})
    print(f"{name}: Recall A={rec_a:.3f} vs B={rec_b:.3f}")

if results:
    df_res = pd.DataFrame(results)
    avg_a = df_res['Recall_A'].mean()
    avg_b = df_res['Recall_B'].mean()
    print(f"\n🏆 平均Recall: Model A {avg_a:.3f} vs Model B {avg_b:.3f}")
    if avg_b > avg_a:
        print(f"✅ RMT勝利 (+{avg_b - avg_a:.3f})")
    else:
        print("⚠️ RMT敗北")
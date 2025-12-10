import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------
# 1. データ準備
# ---------------------------------------------------------
print("📊 データを準備中...")
try:
    df_prices = pd.read_csv("stock_prices_topix100_simple.csv", index_col=0, parse_dates=True)
except FileNotFoundError:
    print("❌ ファイルが見つかりません。")
    exit()

# 対数収益率
df_log_returns = np.log(df_prices / df_prices.shift(1)).dropna()

# RMT計算関数
def calculate_rmt(returns, window):
    vals = [np.nan] * window
    # 高速化のため単純ループ
    for i in range(window, len(returns)):
        window_data = returns.iloc[i-window : i]
        # 欠損ケア
        corr_mat = window_data.corr().fillna(0)
        vals.append(np.linalg.eigh(corr_mat)[0][-1])
    return pd.Series(vals, index=returns.index)

# ★ マルチスケール計算 ★
print("🧮 RMT(Window=20: 急変検知)を計算中...")
rmt_short = calculate_rmt(df_log_returns, window=20)

print("🧮 RMT(Window=105: 構造変化検知)を計算中...")
rmt_long = calculate_rmt(df_log_returns, window=105)

# ---------------------------------------------------------
# 2. 特徴量エンジニアリング (全部入り)
# ---------------------------------------------------------
market_index = df_prices.mean(axis=1)
df_ml = pd.DataFrame(index=market_index.index)

# (A) Tech Baseline
df_ml['Return'] = market_index.pct_change()
df_ml['Vol_20'] = df_ml['Return'].rolling(20).std()
df_ml['Momentum_10'] = market_index / market_index.shift(10) - 1.0
delta = market_index.diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df_ml['RSI_14'] = 100 - (100 / (1 + gain/loss))

# (B) Multi-Scale RMT Features
# Short (20)
df_ml['RMT_S_Raw'] = rmt_short
df_ml['RMT_S_Diff'] = df_ml['RMT_S_Raw'].diff()
# ShortのZ-scoreは「直近の異常」を見るため短め(60日)で
df_ml['RMT_S_Z'] = (df_ml['RMT_S_Raw'] - df_ml['RMT_S_Raw'].rolling(60).mean()) / df_ml['RMT_S_Raw'].rolling(60).std()

# Long (105)
df_ml['RMT_L_Raw'] = rmt_long
df_ml['RMT_L_Diff'] = df_ml['RMT_L_Raw'].diff()
# LongのZ-scoreは「歴史的な異常」を見るため長め(250日)で
df_ml['RMT_L_Z'] = (df_ml['RMT_L_Raw'] - df_ml['RMT_L_Raw'].rolling(250).mean()) / df_ml['RMT_L_Raw'].rolling(250).std()

# (C) ターゲット (動的: -2.5 * Sigma)
LOOKAHEAD = 5
indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=LOOKAHEAD)
future_min = df_ml['Return'].rolling(window=indexer).min().shift(-1)
threshold = -2.5 * df_ml['Vol_20']
df_ml['Target'] = (future_min < threshold).astype(int)

df_ml = df_ml.dropna()

print(f"学習データ数: {len(df_ml)}")
print(f"暴落発生率: {df_ml['Target'].mean():.2%}")

# ---------------------------------------------------------
# 3. 最終A/Bテスト
# ---------------------------------------------------------
features_A = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14']
# ShortとLongの両方を入れる
features_B = features_A + ['RMT_S_Raw', 'RMT_S_Diff', 'RMT_S_Z', 'RMT_L_Raw', 'RMT_L_Diff', 'RMT_L_Z']

scenarios = {
    "1. 2018 VIX": ("2018-01-01", "2018-06-30"),
    "2. 2020 Covid": ("2020-01-01", "2020-06-30"),
    "3. 2024 Ueda": ("2024-06-01", "2024-10-31"),
    "4. 2025 Tariff": ("2025-01-01", "2025-08-30")
}

print("\n🤖 最終決戦: Multi-Scale RMT 検証開始...")

results = []
# 重要度確認用のモデル保存
last_model_b = None

for name, (start, end) in scenarios.items():
    test_mask = (df_ml.index >= start) & (df_ml.index <= end)
    train_mask = (df_ml.index < start)
    
    y_test = df_ml.loc[test_mask, 'Target']
    if y_test.sum() == 0:
        continue
        
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
    last_model_b = clf_b
    
    rec_a = recall_score(y_test, clf_a.predict(X_test[features_A]), zero_division=0)
    rec_b = recall_score(y_test, clf_b.predict(X_test[features_B]), zero_division=0)
    
    print(f"{name}: Recall A={rec_a:.3f} vs B={rec_b:.3f}")
    results.append({'Recall_A': rec_a, 'Recall_B': rec_b})

# ---------------------------------------------------------
# 4. 結果総括
# ---------------------------------------------------------
if results:
    df_res = pd.DataFrame(results)
    avg_a = df_res['Recall_A'].mean()
    avg_b = df_res['Recall_B'].mean()
    
    print("\n🏆 平均Recall:")
    print(f"Model A: {avg_a:.3f}")
    print(f"Model B: {avg_b:.3f}")
    
    if avg_b > avg_a:
        print(f"✅ RMT勝利 (+{avg_b - avg_a:.3f})")
    
    # 重要度 (最新モデル)
    print("\n🔍 AIの特徴量評価 (Feature Importance):")
    imp = pd.DataFrame({'Feat': features_B, 'Gain': last_model_b.feature_importances_})
    print(imp.sort_values('Gain', ascending=False))
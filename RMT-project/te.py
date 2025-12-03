import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import recall_score
from sklearn.model_selection import TimeSeriesSplit

# ---------------------------------------------------------
# 1. データ準備 (固定ターゲット: 10日 / -7%)
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
    for i in range(window, len(returns)):
        window_data = returns.iloc[i-window : i]
        corr_mat = window_data.corr().fillna(0)
        vals.append(np.linalg.eigh(corr_mat)[0][-1])
    return pd.Series(vals, index=returns.index)

# ★ マルチスケール計算 ★
print("🧮 RMT(Window=20)を計算中...")
rmt_80 = calculate_rmt(df_log_returns, window=80)

print("🧮 RMT(Window=80)を計算中...")
rmt_135 = calculate_rmt(df_log_returns, window=135)

# 特徴量作成
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

# (B) RMT Features (80 & 135)
# Short (80)
df_ml['RMT_80_Raw'] = rmt_80
df_ml['RMT_80_Diff'] = df_ml['RMT_80_Raw'].diff()
df_ml['RMT_80_Z'] = (df_ml['RMT_80_Raw'] - df_ml['RMT_80_Raw'].rolling(250).mean()) / df_ml['RMT_80_Raw'].rolling(250).std()
# Long (135)
df_ml['RMT_135_Raw'] = rmt_135
df_ml['RMT_135_Diff'] = df_ml['RMT_135_Raw'].diff()
df_ml['RMT_135_Z'] = (df_ml['RMT_135_Raw'] - df_ml['RMT_135_Raw'].rolling(250).mean()) / df_ml['RMT_135_Raw'].rolling(250).std()
# (C) ターゲット (固定: -7%)
LOOKAHEAD = 10
THRESHOLD = -0.07 
future_min = market_index.rolling(LOOKAHEAD).min().shift(-LOOKAHEAD)
target = ((future_min - market_index) / market_index <= THRESHOLD).astype(int)
df_ml['Target'] = target

df_ml = df_ml.dropna()

print(f"学習データ数: {len(df_ml)}")
print(f"暴落発生率: {df_ml['Target'].mean():.2%}")

# ---------------------------------------------------------
# 2. 3つ巴の戦い (Model A vs B vs C)
# ---------------------------------------------------------
# 特徴量セット
feat_A = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14']
feat_B = feat_A + ['RMT_135_Raw', 'RMT_135_Diff', 'RMT_135_Z'] # Single (Best)
feat_C = feat_B + ['RMT_80_Raw', 'RMT_80_Diff', 'RMT_80_Z'] # Multi-Scale

scenarios = {
    "1. 2018 VIX": ("2018-01-01", "2018-06-30"),
    "2. 2020 Covid": ("2020-01-01", "2020-06-30"),
    "3. 2024 Ueda": ("2024-06-01", "2024-10-31"),
    "4. 2025 Tariff": ("2025-01-01", "2025-08-30")
}

print("\n🤖 比較検証開始: Tech vs Single(80) vs Multi(20+80)")

results = []
model_c_last = None

for name, (start, end) in scenarios.items():
    test_mask = (df_ml.index >= start) & (df_ml.index <= end)
    train_mask = (df_ml.index < start)
    
    y_test = df_ml.loc[test_mask, 'Target']
    if y_test.sum() == 0: continue
        
    X_train = df_ml.loc[train_mask]
    y_train = df_ml.loc[train_mask, 'Target']
    X_test = df_ml.loc[test_mask]
    
    w = len(X_train) / (2 * y_train.sum())
    
    # 学習 & 評価
    # Model A
    clf_a = lgb.LGBMClassifier(random_state=42, scale_pos_weight=w, verbose=-1).fit(X_train[feat_A], y_train)
    rec_a = recall_score(y_test, clf_a.predict(X_test[feat_A]), zero_division=0)
    
    # Model B
    clf_b = lgb.LGBMClassifier(random_state=42, scale_pos_weight=w, verbose=-1).fit(X_train[feat_B], y_train)
    rec_b = recall_score(y_test, clf_b.predict(X_test[feat_B]), zero_division=0)
    
    # Model C
    clf_c = lgb.LGBMClassifier(random_state=42, scale_pos_weight=w, verbose=-1).fit(X_train[feat_C], y_train)
    rec_c = recall_score(y_test, clf_c.predict(X_test[feat_C]), zero_division=0)
    
    model_c_last = clf_c
    
    print(f"\n{name} (Crash: {y_test.sum()}日)")
    print(f"   A (Tech): {rec_a:.3f}")
    print(f"   B (RMT80): {rec_b:.3f}")
    print(f"   C (Multi): {rec_c:.3f}")
    
    results.append({'Scenario': name, 'A': rec_a, 'B': rec_b, 'C': rec_c})

# ---------------------------------------------------------
# 3. 結論
# ---------------------------------------------------------
if results:
    df_res = pd.DataFrame(results)
    avg_a = df_res['A'].mean()
    avg_b = df_res['B'].mean()
    avg_c = df_res['C'].mean()
    
    print("\n🏆 平均Recall:")
    print(f"Model A: {avg_a:.3f}")
    print(f"Model B: {avg_b:.3f}")
    print(f"Model C: {avg_c:.3f}")
    
    if avg_c > avg_b:
        print("✅ Model C (Multi) が最強でした。短期窓の追加は有効です。")
    elif avg_c == avg_b:
        print("➖ 変わりません。短期窓は無視されました。")
    else:
        print("⚠️ Model B (Single) の方が良いです。短期窓はノイズになりました。")

    # 重要度 (Model C)
    print("\n🔍 Model C の特徴量重要度:")
    imp = pd.DataFrame({'Feat': feat_C, 'Gain': model_c_last.feature_importances_})
    print(imp.sort_values('Gain', ascending=False))
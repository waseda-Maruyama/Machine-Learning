import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------
# 1. データ準備 (動的ターゲット版)
# ---------------------------------------------------------
print("📊 データを準備中...")
try:
    df_prices = pd.read_csv("stock_prices_topix100_simple.csv", index_col=0, parse_dates=True)
except FileNotFoundError:
    print("❌ ファイルが見つかりません。")
    exit()

# 対数収益率 & ベースライン
df_log_returns = np.log(df_prices / df_prices.shift(1)).dropna()
market_index = df_prices.mean(axis=1)
df_base = pd.DataFrame(index=market_index.index)
df_base['Return'] = market_index.pct_change()
df_base['Vol_20'] = df_base['Return'].rolling(20).std()
df_base['Momentum_10'] = market_index / market_index.shift(10) - 1.0
delta = market_index.diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df_base['RSI_14'] = 100 - (100 / (1 + gain/loss))

# ターゲットY (動的: -2.5 * Sigma)
LOOKAHEAD = 5
indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=LOOKAHEAD)
future_min = df_base['Return'].rolling(window=indexer).min().shift(-1)
threshold_dynamic = -2.5 * df_base['Vol_20']
target = (future_min < threshold_dynamic).astype(int)

# 検証シナリオ
scenarios = {
    "2018 VIX": ("2018-01-01", "2018-06-30"),
    "2020 Covid": ("2020-01-01", "2020-06-30"),
    "2024 Ueda": ("2024-06-01", "2024-10-31"),
    "2025 Tariff": ("2025-01-01", "2025-08-30")
}

# ---------------------------------------------------------
# 2. RMT計算関数
# ---------------------------------------------------------
def calculate_rmt_fast(returns, window):
    data_values = returns.values
    n_samples, n_features = data_values.shape
    rmt_vals = np.full(n_samples, np.nan)
    for i in range(window, n_samples):
        sub_df = returns.iloc[i-window : i]
        corr = sub_df.corr().fillna(0).values
        eigvals = np.linalg.eigvalsh(corr)
        rmt_vals[i] = eigvals[-1]
    return pd.Series(rmt_vals, index=returns.index)

# ---------------------------------------------------------
# 3. シナリオ別 感度分析
# ---------------------------------------------------------
# 10日から150日まで (長めにとってクロスを見る)
windows_to_scan = list(range(10, 155, 5))

print(f"\n🧪 シナリオ別感度分析: {min(windows_to_scan)}日 〜 {max(windows_to_scan)}日")

# 結果格納用辞書: {シナリオ名: [スコアリスト]}
scenario_scores = {name: [] for name in scenarios.keys()}
avg_scores = []

for w in tqdm(windows_to_scan):
    # RMT計算
    ts_rmt = calculate_rmt_fast(df_log_returns, w)
    
    # 特徴量セット
    df_ml = df_base.copy()
    df_ml['RMT_Raw'] = ts_rmt
    df_ml['RMT_Diff'] = df_ml['RMT_Raw'].diff()
    df_ml['RMT_Accel'] = df_ml['RMT_Diff'].diff()
    df_ml['RMT_Z'] = (df_ml['RMT_Raw'] - df_ml['RMT_Raw'].rolling(250).mean()) / df_ml['RMT_Raw'].rolling(250).std()
    
    df_ml['Target'] = target
    df_ml = df_ml.dropna()
    
    # 各シナリオでテスト
    current_recalls = []
    feats = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14', 'RMT_Raw', 'RMT_Diff', 'RMT_Accel', 'RMT_Z']
    
    for s_name, (s_start, s_end) in scenarios.items():
        test_mask = (df_ml.index >= s_start) & (df_ml.index <= s_end)
        train_mask = (df_ml.index < s_start)
        
        y_test = df_ml.loc[test_mask, 'Target']
        
        # 暴落がない場合は0扱いではなくNaN(無効)扱いにしないと平均が歪むが、
        # 今回は「検知ゼロ」として0を記録する
        if y_test.sum() == 0: 
            rec = 0.0
        else:
            X_train = df_ml.loc[train_mask, feats]
            y_train = df_ml.loc[train_mask, 'Target']
            X_test = df_ml.loc[test_mask, feats]
            
            w_train = len(X_train) / (2 * y_train.sum())
            model = lgb.LGBMClassifier(random_state=42, scale_pos_weight=w_train, verbose=-1, n_jobs=1)
            model.fit(X_train, y_train)
            rec = recall_score(y_test, model.predict(X_test), zero_division=0)
            
        scenario_scores[s_name].append(rec)
        current_recalls.append(rec)
    
    avg_scores.append(np.mean(current_recalls))

# ---------------------------------------------------------
# 4. グラフ描画 (マルチライン)
# ---------------------------------------------------------
plt.figure(figsize=(12, 7))

# 各シナリオの線
colors = ['cyan', 'orange', 'green', 'purple']
for (name, scores), color in zip(scenario_scores.items(), colors):
    plt.plot(windows_to_scan, scores, marker='.', linestyle='--', linewidth=1.5, alpha=0.7, label=name, color=color)

# 平均線 (太く)
plt.plot(windows_to_scan, avg_scores, marker='o', linestyle='-', linewidth=3, color='blue', label='Average')

# ピーク注釈
max_avg = max(avg_scores)
best_w = windows_to_scan[avg_scores.index(max_avg)]
plt.annotate(f'Best Avg: {best_w}d ({max_avg:.2f})', xy=(best_w, max_avg), xytext=(best_w, max_avg+0.05),
             arrowprops=dict(facecolor='black', shrink=0.05), ha='center', fontweight='bold')

plt.title('RMT Sensitivity by Crisis Scenario (Dynamic Target)', fontsize=16)
plt.xlabel('Window Size (Days)', fontsize=14)
plt.ylabel('Recall', fontsize=14)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('sensitivity_by_scenario.png')
plt.show()

print(f"\n✅ 分析完了。'sensitivity_by_scenario.png' を確認してください。")
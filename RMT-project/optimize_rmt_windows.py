import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm

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

# ベースライン
market_index = df_prices.mean(axis=1)
df_base = pd.DataFrame(index=market_index.index)
df_base['Return'] = market_index.pct_change()
df_base['Vol_20'] = df_base['Return'].rolling(20).std()
df_base['Momentum_10'] = market_index / market_index.shift(10) - 1.0
delta = market_index.diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df_base['RSI_14'] = 100 - (100 / (1 + gain/loss))

# --- ターゲットY (固定: 10日 / -7%) ---
LOOKAHEAD = 5
THRESHOLD = -0.05 
future_min = df_base['Return'].rolling(window=LOOKAHEAD).min().shift(-LOOKAHEAD) # 近似
# 正確には価格ベースで計算
future_min_price = market_index.rolling(window=LOOKAHEAD).min().shift(-LOOKAHEAD)
drawdown = (future_min_price - market_index) / market_index
target = (drawdown <= THRESHOLD).astype(int)

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
        # 簡易計算 (速度優先)
        sub_df = returns.iloc[i-window : i]
        corr = sub_df.corr().fillna(0).values
        eigvals = np.linalg.eigvalsh(corr)
        rmt_vals[i] = eigvals[-1]
        
    return pd.Series(rmt_vals, index=returns.index)

# ---------------------------------------------------------
# 3. シナリオ別 感度分析
# ---------------------------------------------------------
# 探索範囲: 10〜160 (10刻み) + 重要ポイント
windows_to_scan = list(range(10, 165, 10))
windows_to_scan.extend([20, 80]) # 注目ポイント
windows_to_scan = sorted(list(set(windows_to_scan)))

print(f"\n🧪 固定ターゲット(-7%) 感度分析: {windows_to_scan} 日")

scenario_scores = {name: [] for name in scenarios.keys()}

for w in tqdm(windows_to_scan):
    # RMT計算
    ts_rmt = calculate_rmt_fast(df_log_returns, w)
    
    # 特徴量セット
    df_ml = df_base.copy()
    df_ml['RMT_Raw'] = ts_rmt
    smooth_window = 5
    rmt_smooth = df_ml['RMT_Raw'].rolling(window=smooth_window).mean()
    
    df_ml['Vel_Smooth'] = rmt_smooth.diff()
    df_ml['Accel_Smooth'] = df_ml['Vel_Smooth'].diff()
    df_ml['RMT_Z'] = (df_ml['RMT_Raw'] - df_ml['RMT_Raw'].rolling(250).mean()) / df_ml['RMT_Raw'].rolling(250).std()
    
    
    df_ml['Target'] = target
    df_ml = df_ml.dropna()
    
    feats = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14', 'RMT_Raw', 'Vel_Smooth', 'Accel_Smooth', 'RMT_Z']
    
    for s_name, (s_start, s_end) in scenarios.items():
        test_mask = (df_ml.index >= s_start) & (df_ml.index <= s_end)
        train_mask = (df_ml.index < s_start)
        
        y_test = df_ml.loc[test_mask, 'Target']
        
        if y_test.sum() == 0:
            rec = 0.0 # 暴落なし期間はスコア0扱い
        else:
            X_train = df_ml.loc[train_mask, feats]
            y_train = df_ml.loc[train_mask, 'Target']
            X_test = df_ml.loc[test_mask, feats]
            
            w_train = len(X_train) / (2 * y_train.sum())
            model = lgb.LGBMClassifier(random_state=42, scale_pos_weight=w_train, verbose=-1, n_estimators=100)
            model.fit(X_train, y_train)
            
            rec = recall_score(y_test, model.predict(X_test), zero_division=0)
            
        scenario_scores[s_name].append(rec)

# ---------------------------------------------------------
# 4. グラフ描画
# ---------------------------------------------------------
plt.figure(figsize=(12, 7))

colors = ['gray', 'tab:blue', 'tab:orange', 'tab:green']
styles = [':', '-', '-', '-']

for (name, scores), color, style in zip(scenario_scores.items(), colors, styles):
    plt.plot(windows_to_scan, scores, marker='o', linestyle=style, linewidth=2, label=name, color=color)

# 注目ポイント縦線
plt.axvline(x=20, color='gray', linestyle='--', alpha=0.3)
plt.text(20, -0.02, '20', color='gray', ha='center')

plt.axvline(x=80, color='red', linestyle='--', alpha=0.3)
plt.text(80, -0.02, 'Best(80)', color='red', ha='center')

plt.title('RMT Sensitivity: Fixed Target (10d/-7%)', fontsize=16)
plt.xlabel('Window Size (Days)', fontsize=14)
plt.ylabel('Recall', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.5)

plt.tight_layout()
plt.savefig('fixed_sensitivity_by_scenario.png')
plt.show()

print(f"\n✅ 分析完了。'fixed_sensitivity_by_scenario.png' を確認してください。")
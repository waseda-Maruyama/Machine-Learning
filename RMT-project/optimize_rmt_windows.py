import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from config import scenarios # 設定ファイル

# =========================================================
# 1. 設定 & データ読み込み
# =========================================================
# ターゲット定義 (create_dataset.py に合わせる)
TERM_SHORT = 5      # 5日後
DROP_SHORT = -0.02  # -2%
TERM_LONG = 10      # 10日後
DROP_LONG = -0.06   # -6%
ONSET_FILTER = 5    # Onsetフィルタ

# モデル判定閾値 (train.py で決めた値)
DECISION_THRESHOLD = 0.20

print("📊 データを読み込み、市場インデックスを作成中...")

if not os.path.exists("stock_prices.csv"):
    print("❌ stock_prices.csv がありません。")
    exit()
df_prices = pd.read_csv("stock_prices.csv", index_col=0, parse_dates=True)

if os.path.exists("market_caps.csv"):
    df_caps = pd.read_csv("market_caps.csv", index_col=0, parse_dates=True)
    df_caps = df_caps.reindex(df_prices.index).ffill()
    market_index = df_caps.sum(axis=1)
    print("   -> 時価総額加重平均インデックスを使用します")
else:
    market_index = df_prices.mean(axis=1)
    print("   -> 単純平均インデックスを使用します")

market_index = market_index / market_index.iloc[0]

# ---------------------------------------------------------
# 2. 共通特徴量 & ターゲット生成
# ---------------------------------------------------------
print("⚙️ ベース特徴量とターゲットを作成中...")

df_log_returns = np.log(df_prices / df_prices.shift(1)).dropna()

df_base = pd.DataFrame(index=market_index.index)
price = market_index
df_base['Return'] = price.pct_change()
df_base['Vol_20'] = df_base['Return'].rolling(20).std()
df_base['Momentum_10'] = price / price.shift(10) - 1.0
delta = price.diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df_base['RSI_14'] = 100 - (100 / (1 + gain/loss))

ret_short = price.shift(-TERM_SHORT) / price - 1.0
ret_long = price.shift(-TERM_LONG) / price - 1.0
raw_target = (ret_short <= DROP_SHORT) & (ret_long <= DROP_LONG)
raw_target = raw_target.astype(int)

event_id = (raw_target.diff() != 0).cumsum()
days_since = raw_target.groupby(event_id).cumcount()
final_target = raw_target.copy()
mask_late = (raw_target == 1) & (days_since >= ONSET_FILTER)
final_target[mask_late] = 0
df_base['Target'] = final_target

# ---------------------------------------------------------
# 3. RMT計算関数
# ---------------------------------------------------------
def calculate_rmt_fast(returns_df, window_size):
    data_values = returns_df.values
    dates = returns_df.index
    n_samples = len(dates)
    rmt_vals = np.full(n_samples, np.nan)
    
    for i in range(window_size, n_samples):
        sub_data = data_values[i-window_size : i]
        sub_data = np.nan_to_num(sub_data)
        if np.all(sub_data == 0):
            rmt_vals[i] = 0
            continue
        corr = np.corrcoef(sub_data, rowvar=False)
        corr = np.nan_to_num(corr)
        eigvals = np.linalg.eigvalsh(corr)
        rmt_vals[i] = eigvals[-1]
        
    return pd.Series(rmt_vals, index=dates, name='RMT_Raw')

# ---------------------------------------------------------
# 4. 感度分析ループ
# ---------------------------------------------------------
windows_to_scan = list(range(100, 300, 5))
print(f"\n🧪 RMT窓幅の感度分析を開始: {windows_to_scan} 日")

scenario_scores = {name: [] for name in scenarios.keys()}

for w in tqdm(windows_to_scan):
    ts_rmt = calculate_rmt_fast(df_log_returns, w)
    
    df_ml = df_base.copy()
    df_ml['RMT_Raw'] = ts_rmt
    
    rmt_smooth = df_ml['RMT_Raw'].rolling(window=5).mean()
    df_ml['RMT_Vel'] = rmt_smooth.diff()
    df_ml['RMT_Accel'] = df_ml['RMT_Vel'].diff()
    
    window_z = 250
    rmt_mean = df_ml['RMT_Raw'].rolling(window_z).mean()
    rmt_std = df_ml['RMT_Raw'].rolling(window_z).std()
    df_ml['RMT_Zscore'] = (df_ml['RMT_Raw'] - rmt_mean) / rmt_std
    
    df_ml = df_ml.dropna()
    features = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14', 
                'RMT_Raw', 'RMT_Vel', 'RMT_Accel', 'RMT_Zscore']
    
    for s_name, (s_start, s_end) in scenarios.items():
        test_start = pd.to_datetime(s_start)
        test_end = pd.to_datetime(s_end)
        
        train_mask = df_ml.index < test_start
        test_mask = (df_ml.index >= test_start) & (df_ml.index <= test_end)
        
        y_test = df_ml.loc[test_mask, 'Target']
        y_train = df_ml.loc[train_mask, 'Target']
        
        if y_test.sum() == 0 or y_train.sum() == 0:
            scenario_scores[s_name].append(0.0)
            continue
            
        X_train = df_ml.loc[train_mask, features]
        X_test = df_ml.loc[test_mask, features]
        pos_weight = len(y_train) / (2 * y_train.sum())
        
        model = lgb.LGBMClassifier(random_state=42, scale_pos_weight=pos_weight, verbose=-1, n_jobs=1)
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= DECISION_THRESHOLD).astype(int)
        rec = recall_score(y_test, preds, zero_division=0)
        scenario_scores[s_name].append(rec)

# =========================================================
# 5. 結果の集計と保存 (CSV出力)
# =========================================================
print("\n💾 結果を集計中...")

# 辞書をDataFrameに変換
df_sens = pd.DataFrame(scenario_scores)
df_sens['Window_Size'] = windows_to_scan

# 平均スコアを計算 (シナリオ列のみの平均をとる)
scenario_cols = list(scenarios.keys())
df_sens['Average_Recall'] = df_sens[scenario_cols].mean(axis=1)

# 列の順番を整える (Window, Average, その他の順)
cols = ['Window_Size', 'Average_Recall'] + scenario_cols
df_sens = df_sens[cols]

# CSV保存
output_csv = 'rmt_window_sensitivity.csv'
df_sens.to_csv(output_csv, index=False)

print(f"📊 最適化結果 (上位5件):")
print(df_sens.sort_values('Average_Recall', ascending=False).head(5))
print(f"✅ CSV保存完了: {output_csv}")

# ---------------------------------------------------------
# 6. 可視化
# ---------------------------------------------------------
plt.figure(figsize=(12, 7))
colors = ['gray', 'tab:blue', 'tab:orange', 'tab:green', 'tab:purple']
markers = ['o', 's', '^', 'D']

for (name, scores), color, marker in zip(scenario_scores.items(), colors, markers):
    alpha = 1.0 if max(scores) > 0 else 0.3
    plt.plot(windows_to_scan, scores, marker=marker, linewidth=2, label=name, color=color, alpha=alpha)

best_idx = df_sens['Average_Recall'].idxmax()
best_window = df_sens.loc[best_idx, 'Window_Size']
best_score = df_sens.loc[best_idx, 'Average_Recall']

plt.axvline(x=best_window, color='red', linestyle='--', alpha=0.5)
plt.text(best_window, -0.05, f'Best: {best_window}d\n(Avg: {best_score:.1%})', 
         color='red', ha='center', fontweight='bold')

plt.title(f'RMT Sensitivity Analysis (Threshold: {DECISION_THRESHOLD:.0%})', fontsize=14)
plt.xlabel('Window Size (Days)', fontsize=12)
plt.ylabel('Recall (Sensitivity)', fontsize=12)
plt.ylim(-0.1, 1.1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rmt_sensitivity_result.png')
# plt.show() # 必要ならコメントアウト解除
print(f"✅ グラフ保存完了: rmt_sensitivity_result.png")

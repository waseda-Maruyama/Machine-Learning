import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import warnings

# 警告を抑制
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# ⚙️ 実験設定
# ---------------------------------------------------------
# 検証するRMTウィンドウサイズの範囲
RMT_WINDOWS = list(range(10, 305, 5))

OUTPUT_DIR = "output_optimization_dual"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 暴落定義
TERM_SHORT = 3
DROP_SHORT = -0.02
TERM_LONG = 10
DROP_LONG = -0.08
DECISION_THRESHOLD = 0.3

# 重み設定
DECAY_RATE = 0.5
BOOST_FACTOR = 10.0

# シナリオ定義
try:
    from config import scenarios
except ImportError:
    print("⚠️ config.pyが見つかりません。ダミーシナリオを使用します。")
    scenarios = {
        "2020 Covid": ("2020-02-01", "2020-04-01"),
        "2024 Ueda":  ("2024-07-20", "2024-08-15")
    }

# =========================================================
# 1. データ読み込み & ベース作成
# =========================================================
print("📊 データを読み込み中...")

has_prices = os.path.exists("stock_prices.csv")
has_caps = os.path.exists("market_caps.csv")

if has_prices:
    df_prices = pd.read_csv("stock_prices.csv", index_col=0, parse_dates=True)
    if has_caps:
        print("   -> 加重平均インデックスを作成")
        df_caps = pd.read_csv("market_caps.csv", index_col=0, parse_dates=True)
        df_caps = df_caps.reindex(df_prices.index).ffill()
        common_cols = df_prices.columns.intersection(df_caps.columns)
        market_index = (df_prices[common_cols] * df_caps[common_cols]).sum(axis=1)
        df_multi_log = np.log(df_prices[common_cols] / df_prices[common_cols].shift(1)).dropna()
    else:
        print("   -> 単純平均インデックスを作成")
        market_index = df_prices.mean(axis=1)
        df_multi_log = np.log(df_prices / df_prices.shift(1)).dropna()
else:
    print("❌ データファイルがありません。ダミーデータを生成します。")
    dates = pd.date_range("2010-01-01", "2025-12-31", freq="B")
    dummy_data = np.random.normal(0, 0.01, (len(dates), 5))
    df_prices = pd.DataFrame(np.cumprod(1 + dummy_data, axis=0), index=dates, columns=[f"S_{i}" for i in range(5)])
    market_index = df_prices.mean(axis=1)
    df_multi_log = np.log(df_prices / df_prices.shift(1)).dropna()

# 正規化
market_index = market_index / market_index.iloc[0]

# ベース特徴量
df_base = pd.DataFrame(index=market_index.index)
price = market_index
df_base['Return'] = price.pct_change()
df_base['Vol_20'] = df_base['Return'].rolling(20).std()
df_base['Momentum_10'] = price / price.shift(10) - 1.0

# RSI
delta = price.diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df_base['RSI_14'] = 100 - (100 / (1 + gain/loss))

# ターゲット生成
ret_short = price.shift(-TERM_SHORT) / price - 1.0
ret_long = price.shift(-TERM_LONG) / price - 1.0
target = ((ret_short <= DROP_SHORT) & (ret_long <= DROP_LONG)).astype(int)
df_base['Target'] = target

# 重み生成
sample_weights = pd.Series(1.0, index=target.index)
if target.sum() > 0:
    event_id = (target.diff() != 0).cumsum()
    days_since = target.groupby(event_id).cumcount()
    decay_comp = BOOST_FACTOR * np.exp(-DECAY_RATE * days_since)
    severity_ratio = ret_long.abs() / abs(DROP_LONG)
    mask_crash = (target == 1)
    sample_weights[mask_crash] = decay_comp[mask_crash] * severity_ratio[mask_crash]

df_base['Sample_Weight'] = sample_weights

# =========================================================
# 2. RMT計算関数
# =========================================================
def calculate_rmt_eigen(df_returns_multi, window_size):
    n_rows = len(df_returns_multi)
    res = np.full(n_rows, np.nan)
    values = df_returns_multi.values
    
    if n_rows <= window_size:
        return pd.Series(res, index=df_returns_multi.index)

    for i in range(window_size, n_rows):
        sub = values[i-window_size : i, :]
        try:
            if np.isnan(sub).any(): sub = np.nan_to_num(sub)
            std = np.std(sub, axis=0)
            valid_cols = std > 1e-9
            if valid_cols.sum() < 2:
                res[i] = 0.0
                continue
            
            sub_valid = sub[:, valid_cols]
            corr = np.corrcoef(sub_valid, rowvar=False)
            corr = np.nan_to_num(corr)
            eigvals = np.linalg.eigvalsh(corr)
            res[i] = eigvals[-1]
        except:
            res[i] = 0.0
    return pd.Series(res, index=df_returns_multi.index)

# =========================================================
# 3. 最適化ループ (Dual Evaluation)
# =========================================================
print(f"\n🧪 RMTウィンドウ最適化 (Standard vs Weighted)...")

results = []
smoothing_window = 5  # 速度計算用の平滑化

for rmt_w in tqdm(RMT_WINDOWS, desc="Scanning"):
    
    # --- A. 特徴量生成 ---
    rmt_raw = calculate_rmt_eigen(df_multi_log, rmt_w)
    
    df_ml = df_base.copy()
    df_ml['RMT_Raw'] = rmt_raw
    
    # ★重要: 平滑化してから微分 (物理的意味の保持)
    df_ml['RMT_Level'] = df_ml['RMT_Raw'].rolling(window=smoothing_window).mean()
    df_ml['RMT_Vel'] = df_ml['RMT_Level'].diff()     # 速度
    df_ml['RMT_Accel'] = df_ml['RMT_Vel'].diff()     # 加速度
    
    df_ml = df_ml.dropna()
    
    # 使用する特徴量
    feats = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14', 
             'RMT_Level', 'RMT_Vel', 'RMT_Accel'] # RawではなくLevelを使うのが物理的に綺麗
    
    # --- B. シナリオ別評価 ---
    total_std_recall = 0
    total_wgt_recall = 0
    valid_count = 0
    
    for s_name, (s_start, s_end) in scenarios.items():
        t_start = pd.to_datetime(s_start)
        t_end = pd.to_datetime(s_end)
        
        if t_start < df_ml.index.min() or t_end > df_ml.index.max(): continue
            
        mask_train = df_ml.index < t_start
        mask_test = (df_ml.index >= t_start) & (df_ml.index <= t_end)
        
        X_tr = df_ml.loc[mask_train, feats]
        y_tr = df_ml.loc[mask_train, 'Target']
        w_tr = df_ml.loc[mask_train, 'Sample_Weight']
        
        X_te = df_ml.loc[mask_test, feats]
        y_te = df_ml.loc[mask_test, 'Target']
        w_te = df_ml.loc[mask_test, 'Sample_Weight']
        
        if len(y_tr) < 50 or y_tr.sum() == 0 or y_te.sum() == 0: continue
            
        # LightGBM学習
        pos_ratio = y_tr.sum() / len(y_tr)
        model = lgb.LGBMClassifier(random_state=42, n_jobs=1, verbose=-1, 
                                   scale_pos_weight=1.0/pos_ratio if pos_ratio > 0 else 1.0)
        model.fit(X_tr, y_tr, sample_weight=w_tr) # 学習は常にWeightedで全力
        
        probs = model.predict_proba(X_te)[:, 1]
        preds = (probs >= DECISION_THRESHOLD).astype(int)
        
        # --- ★ここが変更点: 2つのスコアを計算 ---
        # 1. Standard Score (件数ベース: 普通の正解率)
        score_std = recall_score(y_te, preds, zero_division=0)
        
        # 2. Weighted Score (重要度ベース: 致命傷回避率)
        score_wgt = recall_score(y_te, preds, sample_weight=w_te, zero_division=0)
        
        total_std_recall += score_std
        total_wgt_recall += score_wgt
        valid_count += 1
        
    if valid_count > 0:
        res = {
            "Window_Size": rmt_w,
            "Avg_Recall_Std": total_std_recall / valid_count,
            "Avg_Recall_Wgt": total_wgt_recall / valid_count
        }
        results.append(res)

# =========================================================
# 4. 結果保存 & 可視化
# =========================================================
if not results:
    print("❌ 結果なし")
    exit()

df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(OUTPUT_DIR, "rmt_optimization_dual.csv"), index=False)

# --- グラフ描画 ---
plt.figure(figsize=(12, 7))

# 1. Standard (件数ベース)
plt.plot(df_res["Window_Size"], df_res["Avg_Recall_Std"], 
         color='gray', linestyle='--', linewidth=2, alpha=0.7, label="Standard Recall (Count-based)")

# 2. Weighted (インパクトベース)
plt.plot(df_res["Window_Size"], df_res["Avg_Recall_Wgt"], 
         color='crimson', linestyle='-', linewidth=3, label="Weighted Recall (Impact-based)")

# 最大値の強調
best_idx_wgt = df_res["Avg_Recall_Wgt"].idxmax()
best_win_wgt = df_res.loc[best_idx_wgt, "Window_Size"]
best_score_wgt = df_res.loc[best_idx_wgt, "Avg_Recall_Wgt"]

plt.axvline(best_win_wgt, color='red', linestyle=':', alpha=0.5)
plt.scatter(best_win_wgt, best_score_wgt, color='red', s=100, zorder=10)
plt.text(best_win_wgt + 5, best_score_wgt, f"Best Weighted: {best_win_wgt}d\n({best_score_wgt:.2f})", 
         color='darkred', va='center')

plt.title("RMT Window Optimization: Standard vs Weighted Evaluation")
plt.xlabel("Window Size (Days)")
plt.ylabel("Recall Score")
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

png_path = os.path.join(OUTPUT_DIR, "optimization_chart_dual.png")
plt.savefig(png_path)
print(f"📊 保存完了: {png_path}")
print(f"🏆 Best Weighted Window: {best_win_wgt} days")
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
# RMTウィンドウサイズの範囲
RMT_WINDOWS = list(range(5, 300, 1))

OUTPUT_DIR = "output_optimization_split"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 暴落定義
TERM_SHORT = 3
DROP_SHORT = -0.02
TERM_LONG = 10
DROP_LONG = -0.08
DECISION_THRESHOLD = 0.5

# 重み設定
DECAY_RATE = 0.5
BOOST_FACTOR = 10.0

# ★ PR曲線を出すウィンドウサイズを固定したい場合はここで指定
# None にすると、最適化ループの中で一番成績が良かったものを自動採用します
MANUAL_WINDOW_SIZE = 135  # 例: 135日で固定

# シナリオ定義
try:
    from config import scenarios
except ImportError:
    print("⚠️ config.pyが見つかりません。ダミーシナリオを使用します。")
    scenarios = {
        "2020 Covid": ("2020-02-01", "2020-04-01"),
        "2024 Ueda":  ("2024-07-20", "2024-08-15"),
        "2025 Tariff": ("2025-01-01", "2025-10-31") 
    }

# =========================================================
# 1. データ読み込み & ベース作成
# =========================================================
print("📊 データを読み込み中...")

has_close = os.path.exists("stock_close.csv")
has_caps = os.path.exists("market_caps.csv")
has_adj = os.path.exists("stock_adj_close.csv")

if has_close:
    df_close = pd.read_csv("stock_close.csv", index_col=0, parse_dates=True)
    df_adj = pd.read_csv("stock_adj_close.csv", index_col=0, parse_dates=True)
    if has_caps:
        print("   -> 加重平均インデックスを作成")
        df_caps = pd.read_csv("market_caps.csv", index_col=0, parse_dates=True)
        df_caps = df_caps.reindex(df_close.index).ffill()
        common_cols = df_close.columns.intersection(df_caps.columns)
        market_index_a = (df_caps[common_cols]).sum(axis=1)
        market_index_b = (df_close[common_cols] * df_caps[common_cols]).sum(axis=1)

        df_multi_log = np.log(df_adj[common_cols] / df_adj[common_cols].shift(1)).dropna()
    else:
        print("   -> 単純平均インデックスを作成")
else:
    print("❌ データファイルがありません。ダミーデータを生成します。")

# 正規化
market_index_a = market_index_a / market_index_a.iloc[0]
market_index_b = market_index_b / market_index_b.iloc[0]

# ベース特徴量
df_base = pd.DataFrame(index=market_index_a.index)
price = market_index_a
df_base['Return'] = price.pct_change()
df_base['Vol_20'] = df_base['Return'].rolling(20).std()
df_base['Momentum_10'] = price / price.shift(10) - 1.0

# RSI
delta = price.diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
df_base['RSI_14'] = 100 - (100 / (1 + gain/loss))


# ターゲット生成
price_b = market_index_b
ret_short = price_b.shift(-TERM_SHORT) / price_b - 1.0
ret_long = price_b.shift(-TERM_LONG) / price_b - 1.0
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
    max_w = sample_weights.max()
    mean_w = sample_weights[mask_crash].mean()
    print(f"⚖️ 重み計算完了:")
    print(f"   - Max Weight : {max_w:.2f}")
    print(f"   - Mean Weight: {mean_w:.2f}")

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
# 3. 最適化ループ
# =========================================================
print(f"\n🧪 RMTウィンドウ最適化 (シナリオ別)...")

results = []
smoothing_window = 5

for rmt_w in tqdm(RMT_WINDOWS, desc="Scanning"):
    
    # --- A. 特徴量生成 ---
    rmt_raw = calculate_rmt_eigen(df_multi_log, rmt_w)
    
    df_ml = df_base.copy()
    df_ml['RMT_Raw'] = rmt_raw
    df_ml['RMT_Level'] = df_ml['RMT_Raw'].rolling(window=smoothing_window).mean()
    df_ml['RMT_Vel'] = df_ml['RMT_Level'].diff()
    df_ml['RMT_Accel'] = df_ml['RMT_Vel'].diff()
    df_ml = df_ml.dropna()
    
    feats = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14', 
             'RMT_Level', 'RMT_Vel', 'RMT_Accel']
    
    
    






    row_result = {"Window_Size": rmt_w}
    
    total_std = 0
    total_wgt = 0
    valid_count = 0
    
    # --- B. シナリオ別ループ ---
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
            
        pos_ratio = y_tr.sum() / len(y_tr)
        model = lgb.LGBMClassifier(random_state=42, n_jobs=1, verbose=-1, 
                                   scale_pos_weight=1.0/pos_ratio if pos_ratio > 0 else 1.0)
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        
        probs = model.predict_proba(X_te)[:, 1]
        preds = (probs >= DECISION_THRESHOLD).astype(int)
        
        s_std = recall_score(y_te, preds, zero_division=0)
        s_wgt = recall_score(y_te, preds, sample_weight=w_te, zero_division=0)
        
        row_result[f"{s_name}_Std"] = s_std
        row_result[f"{s_name}_Wgt"] = s_wgt
        
        total_std += s_std
        total_wgt += s_wgt
        valid_count += 1
        
    if valid_count > 0:
        row_result["Average_Std"] = total_std / valid_count
        row_result["Average_Wgt"] = total_wgt / valid_count
        results.append(row_result)

if not results:
    print("❌ 結果なし")
    exit()

df_res = pd.DataFrame(results)
df_res.to_csv(os.path.join(OUTPUT_DIR, "rmt_optimization_split.csv"), index=False)

# =========================================================
# 4. グラフ描画関数 (修正: 極細の実線に変更)
# =========================================================
def plot_optimization_chart(df, suffix, title, filename):
    plt.figure(figsize=(12, 7))
    
    # カラーパレット
    colors = plt.cm.tab10(np.linspace(0, 1, len(scenarios)))
    
    # 各シナリオの線を描画
    scenario_cols = [c for c in df.columns if c.endswith(f"_{suffix}") and "Average" not in c]
    
    for i, col in enumerate(scenario_cols):
        clean_name = col.replace(f"_{suffix}", "")
        
        # Tariffを目立たせる（オプション）
        if "Tariff" in clean_name:
            lw = 1.5  # 少しだけ太く
            alpha = 1.0
            zorder = 10
        else:
            lw = 0.8  # 他は極細
            alpha = 0.6
            zorder = 1
            
        # marker=None, linestyle='-' (実線) に変更
        plt.plot(df["Window_Size"], df[col], 
                 marker=None, linestyle='-', linewidth=lw, alpha=alpha, 
                 label=clean_name, color=colors[i % len(colors)], zorder=zorder)
    
    plt.title(title, fontsize=16)
    plt.xlabel("Window Size (Days)", fontsize=14)
    plt.ylabel("Recall Score", fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300)
    print(f"📊 保存完了: {path}")

# --- 実行 ---
plot_optimization_chart(
    df_res, "Std", 
    "Figure A: Optimization of Standard Recall", 
    "optimization_standard.png"
)

plot_optimization_chart(
    df_res, "Wgt", 
    "Figure B: Optimization of Weighted Recall", 
    "optimization_weighted.png"
)

print("\n✅ 全処理完了。output_optimization_split フォルダを確認してください。")


# =========================================================
# 5. 指定ウィンドウでのPR曲線
# =========================================================
from sklearn.metrics import precision_recall_curve, average_precision_score

print("\n📉 PR曲線を描画します...")

# ★ 修正: マニュアル設定がある場合はそれを優先、なければTariffベスト
if MANUAL_WINDOW_SIZE is not None:
    best_window = MANUAL_WINDOW_SIZE
    print(f"🔧 Manual Window Selected: {best_window}")
else:
    # 自動選択ロジック
    target_metric_col = "2025 Tariff_Wgt"
    if target_metric_col in df_res.columns:
        best_idx = df_res[target_metric_col].idxmax()
    else:
        best_idx = df_res["Average_Wgt"].idxmax()
    best_window = df_res.loc[best_idx, "Window_Size"]
    print(f"🏆 Auto Best Window (Tariff): {best_window}")

# 2. そのウィンドウサイズで特徴量を再計算
rmt_raw_best = calculate_rmt_eigen(df_multi_log, int(best_window))

df_best = df_base.copy()
df_best['RMT_Raw'] = rmt_raw_best
df_best['RMT_Level'] = df_best['RMT_Raw'].rolling(window=smoothing_window).mean()
df_best['RMT_Vel'] = df_best['RMT_Level'].diff()
df_best['RMT_Accel'] = df_best['RMT_Vel'].diff()
df_best = df_best.dropna()

feats = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14', 
         'RMT_Level', 'RMT_Vel', 'RMT_Accel']


# 3. シナリオごとに推論してプロット
plt.figure(figsize=(10, 8))

for s_name, (s_start, s_end) in scenarios.items():
    t_start = pd.to_datetime(s_start)
    t_end = pd.to_datetime(s_end)
    
    if t_start < df_best.index.min() or t_end > df_best.index.max(): continue

    # データ分割
    mask_train = df_best.index < t_start
    mask_test = (df_best.index >= t_start) & (df_best.index <= t_end)
    
    X_tr = df_best.loc[mask_train, feats]
    y_tr = df_best.loc[mask_train, 'Target']
    w_tr = df_best.loc[mask_train, 'Sample_Weight']
    
    X_te = df_best.loc[mask_test, feats]
    y_te = df_best.loc[mask_test, 'Target']
    
    if len(y_tr) < 50 or y_tr.sum() == 0 or y_te.sum() == 0: continue

    # 再学習
    pos_ratio = y_tr.sum() / len(y_tr)
    model = lgb.LGBMClassifier(random_state=42, n_jobs=1, verbose=-1, 
                               scale_pos_weight=1.0/pos_ratio if pos_ratio > 0 else 1.0)
    model.fit(X_tr, y_tr, sample_weight=w_tr)
    
    # 確率を出力 (0〜1)
    probs = model.predict_proba(X_te)[:, 1]
    
    # PR曲線の計算
    precision, recall, _ = precision_recall_curve(y_te, probs)
    ap_score = average_precision_score(y_te, probs)
    
    # プロット
    plt.plot(recall, precision, lw=2, label=f'{s_name} (AP={ap_score:.3f})')
    
    # 現在の閾値(0.5)の位置をプロット
    preds_default = (probs >= DECISION_THRESHOLD).astype(int)
    curr_rec = recall_score(y_te, preds_default)
    
    tp = ((preds_default == 1) & (y_te == 1)).sum()
    fp = ((preds_default == 1) & (y_te == 0)).sum()
    curr_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    plt.scatter(curr_rec, curr_prec, marker='o', s=100, zorder=10, 
                edgecolor='black', label=f'{s_name} Threshold={DECISION_THRESHOLD}')

# 装飾
plt.xlabel('Recall (Sensitivity)', fontsize=14)
plt.ylabel('Precision (Positive Predictive Value)', fontsize=14)
plt.title(f'Precision-Recall Curve (Window: {best_window})', fontsize=16)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

baseline = df_best['Target'].mean()
plt.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline (Random): {baseline:.3f}')

path_pr = os.path.join(OUTPUT_DIR, "best_pr_curve.png")
plt.savefig(path_pr, dpi=300)
print(f"📊 PR曲線を保存しました: {path_pr}")
plt.show()

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import warnings

# 警告を抑制（クリーンな出力のため）
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# ⚙️ 実験設定
# ---------------------------------------------------------
# 検証するRMTウィンドウサイズの範囲
RMT_WINDOWS = list(range(10, 305, 5))

OUTPUT_DIR = "output_optimization_weighted"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 暴落定義 (Code Bの設定に準拠)
TERM_SHORT = 3
DROP_SHORT = -0.02
TERM_LONG = 10
DROP_LONG = -0.08  # -8%
DECISION_THRESHOLD = 0.3

# 重み設定
DECAY_RATE = 0.5    # 時間減衰
BOOST_FACTOR = 10.0 # 暴落初日の重み倍率

# シナリオ定義
try:
    from config import scenarios
except ImportError:
    # ダミーシナリオ
    scenarios = {
        "2020 Covid": ("2020-02-01", "2020-04-01"),
        "2024 Ueda":  ("2024-07-20", "2024-08-15")
    }

# =========================================================
# 1. データ読み込み & ベース作成 (加重平均対応)
# =========================================================
print("📊 データを読み込み中...")

# stock_prices.csv と market_caps.csv の存在確認
has_prices = os.path.exists("stock_prices.csv")
has_caps = os.path.exists("market_caps.csv")

if has_prices:
    df_prices = pd.read_csv("stock_prices.csv", index_col=0, parse_dates=True)
    
    if has_caps:
        # Code B: 時価総額加重平均
        print("   -> 時価総額データを検出: 加重平均インデックスを作成します")
        df_caps = pd.read_csv("market_caps.csv", index_col=0, parse_dates=True)
        df_caps = df_caps.reindex(df_prices.index).ffill()
        common_cols = df_prices.columns.intersection(df_caps.columns)
        market_index = (df_prices[common_cols] * df_caps[common_cols]).sum(axis=1)
        
        # RMT計算用にログリターン(全銘柄)も準備
        df_multi_log = np.log(df_prices[common_cols] / df_prices[common_cols].shift(1)).dropna()
    else:
        # 単純平均
        print("   -> 時価総額データなし: 単純平均インデックスを作成します")
        market_index = df_prices.mean(axis=1)
        df_multi_log = np.log(df_prices / df_prices.shift(1)).dropna()

else:
    print("❌ データファイルがありません。ダミーデータを生成します。")
    dates = pd.date_range("2010-01-01", "2025-12-31", freq="B")
    # 5銘柄分のダミーデータ
    dummy_data = np.random.normal(0, 0.01, (len(dates), 5))
    df_prices = pd.DataFrame(np.cumprod(1 + dummy_data, axis=0), index=dates, columns=[f"Stock_{i}" for i in range(5)])
    
    market_index = df_prices.mean(axis=1)
    df_multi_log = np.log(df_prices / df_prices.shift(1)).dropna()

# 正規化
market_index = market_index / market_index.iloc[0]
df_log_returns = np.log(market_index / market_index.shift(1)).dropna().to_frame()

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

# ---------------------------------------------------------
# 🎯 ターゲット作成
# ---------------------------------------------------------
print("🔨 ターゲットを生成中...")

ret_short = price.shift(-TERM_SHORT) / price - 1.0
ret_long = price.shift(-TERM_LONG) / price - 1.0

raw_target = (ret_short <= DROP_SHORT) & (ret_long <= DROP_LONG)
target = raw_target.astype(int)

df_base['Target'] = target

# ---------------------------------------------------------
# ⚖️ 重み付け計算
# ---------------------------------------------------------
print("⚖️ 重みを計算中...")

sample_weights = pd.Series(1.0, index=target.index)

if target.sum() > 0:
    # 1. 時間減衰
    event_id = (target.diff() != 0).cumsum()
    days_since = target.groupby(event_id).cumcount()
    decay_comp = BOOST_FACTOR * np.exp(-DECAY_RATE * days_since)
    
    # 2. 被害規模
    severity_ratio = ret_long.abs() / abs(DROP_LONG)
    
    # 3. 結合
    mask_crash = (target == 1)
    final_calculated_weights = decay_comp * severity_ratio
    
    sample_weights[mask_crash] = final_calculated_weights[mask_crash]
    
    print(f"   - Max Weight : {sample_weights.max():.2f}")
    print(f"   - Mean Weight: {sample_weights[mask_crash].mean():.2f}")

df_base['Sample_Weight'] = sample_weights

# =========================================================
# 2. RMT計算関数 (修正版)
# =========================================================
def calculate_rmt_eigen(df_returns_multi, window_size):
    """
    複数銘柄のリターンDFを受け取り、スライディングウィンドウで最大固有値を計算
    """
    n_rows = len(df_returns_multi)
    res = np.full(n_rows, np.nan)
    values = df_returns_multi.values
    
    # ループ
    # エラー回避: データがウィンドウより短い場合は計算不可
    if n_rows <= window_size:
        return pd.Series(res, index=df_returns_multi.index)

    for i in range(window_size, n_rows):
        # ウィンドウ切り出し
        sub = values[i-window_size : i, :]
        
        try:
            # 欠損除去
            if np.isnan(sub).any():
                sub = np.nan_to_num(sub)
            
            # 標準偏差が0の列があると相関が計算できないためチェック
            std = np.std(sub, axis=0)
            valid_cols = std > 1e-9
            
            if valid_cols.sum() < 2:
                # 有効な列が2つ未満なら相関行列作れない -> 0
                res[i] = 0.0
                continue

            sub_valid = sub[:, valid_cols]
            corr = np.corrcoef(sub_valid, rowvar=False)
            corr = np.nan_to_num(corr)
            
            # 最大固有値
            eigvals = np.linalg.eigvalsh(corr)
            res[i] = eigvals[-1]
        except Exception:
            res[i] = 0.0
            
    return pd.Series(res, index=df_returns_multi.index)

# =========================================================
# 3. ウィンドウサイズごとの検証ループ
# =========================================================
print(f"\n🧪 RMTウィンドウサイズの最適化を開始します...")

results = []

# 検証ループ
for rmt_w in tqdm(RMT_WINDOWS, desc="Scanning"):
    
    # --- A. 特徴量生成 ---
    # ここで複数銘柄データを使って固有値を計算
    rmt_raw = calculate_rmt_eigen(df_multi_log, rmt_w)
    
    # ★重要: 必ずコピーして使う (元のdf_baseを汚さない)
    df_ml = df_base.copy()
    df_ml['RMT_Raw'] = rmt_raw

    # 2. Smooth (平滑化) -> Velocity計算用
    smooth = df_ml['RMT_Raw'].rolling(window=5).mean()
    
    # 3. Velocity (速度)
    df_ml['RMT_Vel'] = smooth.diff()
    
    # 4. Acceleration (加速度)
    df_ml['RMT_Accel'] = df_ml['RMT_Vel'].diff()

    # 変化率
    df_ml['RMT_Vel'] = df_ml['RMT_Raw'].diff()
    df_ml['RMT_Accel'] = df_ml['RMT_Vel'].diff()
    
    # NaN除去
    df_ml = df_ml.dropna()
    
    feats = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14', 
             'RMT_Raw', 'RMT_Vel', 'RMT_Accel']
    
    # --- B. シナリオ別評価 ---
    scenario_scores = {}
    total_recall = 0
    valid_count = 0
    
    for s_name, (s_start, s_end) in scenarios.items():
        t_start = pd.to_datetime(s_start)
        t_end = pd.to_datetime(s_end)
        
        # 期間チェック
        if t_start < df_ml.index.min() or t_end > df_ml.index.max():
            continue
            
        # Walk-Forward分割
        mask_train = df_ml.index < t_start
        mask_test = (df_ml.index >= t_start) & (df_ml.index <= t_end)
        
        X_tr = df_ml.loc[mask_train, feats]
        y_tr = df_ml.loc[mask_train, 'Target']
        w_tr = df_ml.loc[mask_train, 'Sample_Weight']
        
        X_te = df_ml.loc[mask_test, feats]
        y_te = df_ml.loc[mask_test, 'Target']
        w_te = df_ml.loc[mask_test, 'Sample_Weight']
        
        # 学習可能なデータがあるか確認
        if len(y_tr) < 50 or y_tr.sum() == 0 or y_te.sum() == 0:
            continue
            
        # LightGBM
        pos_ratio = y_tr.sum() / len(y_tr)
        # scale_pos_weightの簡易計算
        scale_weight = 1.0 / pos_ratio if pos_ratio > 0 else 1.0
        
        model = lgb.LGBMClassifier(
            random_state=42, n_jobs=1, verbose=-1,
            scale_pos_weight=scale_weight
        )
        
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        
        probs = model.predict_proba(X_te)[:, 1]
        preds = (probs >= DECISION_THRESHOLD).astype(int)
        
        # 重み付きRecall
        score = recall_score(y_te, preds, sample_weight=w_te, zero_division=0)
        
        scenario_scores[s_name] = score
        total_recall += score
        valid_count += 1
        
    avg_score = total_recall / valid_count if valid_count > 0 else 0
    
    res = {"Window_Size": rmt_w, "Average_Recall": avg_score}
    res.update(scenario_scores)
    results.append(res)

# =========================================================
# 4. 結果保存 & 可視化
# =========================================================
if not results:
    print("❌ 有効な結果が得られませんでした。データ期間やシナリオ設定を確認してください。")
    exit()

df_res = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "rmt_window_optimization.csv")
df_res.to_csv(csv_path, index=False)

# グラフ描画
plt.figure(figsize=(12, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
plot_cols = [c for c in df_res.columns if c not in ["Window_Size", "Average_Recall"]]

for i, col in enumerate(plot_cols):
    plt.plot(df_res["Window_Size"], df_res[col], 
             marker='.', linestyle='None', alpha=0.4, label=col, color=colors[i % len(colors)])

plt.plot(df_res["Window_Size"], df_res["Average_Recall"], 
         color='black', linewidth=2, label="Average Score")

best_idx = df_res["Average_Recall"].idxmax()
best_win = df_res.loc[best_idx, "Window_Size"]
best_score = df_res.loc[best_idx, "Average_Recall"]

plt.axvline(best_win, color='red', linestyle='--', alpha=0.8)
plt.title(f"RMT Window Size Optimization\nBest Window: {best_win} days (Score: {best_score:.2f})")
plt.xlabel("Window Size (Days)")
plt.ylabel("Weighted Recall Score")
plt.legend()
plt.grid(True, alpha=0.3)

png_path = os.path.join(OUTPUT_DIR, "optimization_chart.png")
plt.savefig(png_path)
print(f"📊 グラフを保存しました: {png_path}")
print(f"🏆 最適ウィンドウサイズ: {best_win}")
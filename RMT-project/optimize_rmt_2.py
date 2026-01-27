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
RMT_WINDOWS = list(range(5,250, 1))

OUTPUT_DIR = "output_optimization_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 暴落定義
TERM_SHORTA = 3
DROP_SHORTA = -0.01
TERM_LONGA = 10
DROP_LONGA = -0.04
TERM_SHORTB = 3
DROP_SHORTB = -0.02
TERM_LONGB = 10
DROP_LONGB = -0.08
DECISION_THRESHOLD = 0.4

# 重み設定
DECAY_RATE = 0.5
BOOST_FACTOR = 5.0

# ★ PR曲線を出すウィンドウサイズを固定したい場合はここで指定
# None にすると、最適化ループの中で一番成績が良かったものを自動採用します
MANUAL_WINDOW_SIZE = 135  # 例: 135日で固定

# 可視化スタイル: "line" or "scatter"
PLOT_STYLE = "scatter"

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
        market_index_b = (df_caps[common_cols]).sum(axis=1)
        market_index_b = market_index_b ** 2

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

# =========================================================
# ★ 4.5. ポテンシャル・エネルギー特徴量
# =========================================================
# 1. 共通カラムの抽出
common_cols = df_close.columns.intersection(df_caps.columns)
df_raw_common = df_close[common_cols]
df_adj_common = df_adj[common_cols]
df_caps_common = df_caps[common_cols]

# 2. 歪み項（Distortion）の計算（調整株価ベース）
window = 60
P_bar = df_adj_common.rolling(window).mean()
sigma = df_adj_common.rolling(window).std()

# z^2 (無次元の歪みエネルギー)
# これは連続量なので、微分してもスパイクしない
distortion = ((df_adj_common - P_bar) / (sigma + 1e-8)) ** 2

# 3. 物理スケール項（有効質量）
# Scale = 時価総額 * 生株価 (値がさ株ほど重い)
# ※ここは株式分割で「段差」ができるが、微分はしないのでOK
physical_mass = df_caps_common * df_adj_common 

# 正規化（市場全体での相対的な重み）
# これをしないと、市場全体の株価水準上昇で値がインフレし続ける
# 特徴量として安定させるなら正規化推奨、絶対額を見たいなら不要
# ここでは「全体の中での局所的なエネルギー集中」を見るため、総和で割ります
total_mass = physical_mass.sum(axis=1) + 1e-8
w_physical = physical_mass.div(total_mass, axis=0)

# =========================================================
# ★重要: スパイクを防ぐための計算順序の変更
# =========================================================

# [A] エネルギー準位 (Level)
# これは「状態」を表すので、分割による段差があっても良い（ツリーモデルが処理する）
df_base['E_pot'] = (w_physical * distortion).sum(axis=1)

# [B] エネルギー速度 (Velocity) & 加速度 (Accel)
# ここで段差を微分しないように、「歪みの変化」に「質量」を掛ける

# 1. 個別銘柄ごとの歪み変化量 (Δz^2)
delta_distortion = distortion.diff()

# 2. 物理的パワー (Power) = 質量 * 歪み変化速度
# Sum( m_i * Δz^2_i )
power_series = (w_physical * delta_distortion).sum(axis=1)

# 滑らかにして特徴量化
smooth_window = 5
df_base['E_pot_Vel'] = power_series.rolling(smooth_window).mean()

# 加速度は、速度の微分ではなく、再度「変化量」として計算するのが安全だが、
# 既にVelが滑らかなので、単純なdiffでもスパイクは抑制されているはず
df_base['E_pot_Accel'] = df_base['E_pot_Vel'].diff()
print("🔋 ポテンシャル・エネルギー特徴量を生成中...")

# 学習用ターゲット生成
price_b = market_index_b
ret_short_b = price_b.shift(-TERM_SHORTB) / price_b - 1.0
ret_long_b = price_b.shift(-TERM_LONGB) / price_b - 1.0
target_b = ((ret_short_b <= DROP_SHORTB) & (ret_long_b <= DROP_LONGB)).astype(int)
df_base['Target'] = target_b

#テスト用ターゲット生成
price_a= market_index_a
ret_short_a = price_a.shift(-TERM_SHORTA) / price_a - 1.0
ret_long_a = price_a.shift(-TERM_LONGA) / price_a - 1.0
target_a = ((ret_short_a <= DROP_SHORTA) & (ret_long_a <= DROP_LONGA)).astype(int)
df_base['Target_Test'] = target_a


# 学習用重み生成
sample_weights_train = pd.Series(1.0, index=target_b.index)
if target_b.sum() > 0:
    # 1. 時間減衰 (Time Decay)
    event_id = (target_b.diff() != 0).cumsum()
    days_since = target_b.groupby(event_id).cumcount()
    decay_comp = BOOST_FACTOR * np.exp(-DECAY_RATE * days_since)
    
    # 2. イベント全体の被害規模 (Event Severity)
    # イベントごとに、期間中の「最大の下落幅」を特定する
    event_max_drop = ret_long_b.groupby(event_id).transform('min')
    
    # イベント全体の深刻度スコア（基準値に対する比率）
    event_severity_score = (event_max_drop.abs() / abs(DROP_LONGB))
    
    # 3. 結合 (Decay × Severity)
    mask_crash = (target_b == 1)
    final_weights = decay_comp * event_severity_score
    sample_weights_train[mask_crash] = final_weights[mask_crash]
    
    max_w = sample_weights_train.max()
    mean_w = sample_weights_train[mask_crash].mean()
    print(f"⚖️ 重み計算完了:")
    print(f"   - Max Weight : {max_w:.2f}")
    print(f"   - Mean Weight: {mean_w:.2f}")

df_base['Sample_Weight_train'] = sample_weights_train   


#テスト用重み生成
sample_weights_test = pd.Series(1.0, index=target_a.index)
if target_a.sum() > 0:
    # 1. 時間減衰 (Time Decay)
    event_id = (target_a.diff() != 0).cumsum()
    days_since = target_a.groupby(event_id).cumcount()
    decay_comp = BOOST_FACTOR * np.exp(-DECAY_RATE * days_since)
    
    # 2. イベント全体の被害規模 (Event Severity)
    # イベントごとに、期間中の「最大の下落幅」を特定する
    event_max_drop = ret_long_a.groupby(event_id).transform('min')
    
    # イベント全体の深刻度スコア（基準値に対する比率）
    event_severity_score = (event_max_drop.abs() / abs(DROP_LONGA))
    
    # 3. 結合 (Decay × Severity)
    mask_crash = (target_a == 1)
    final_weights = decay_comp * event_severity_score
    sample_weights_test[mask_crash] = final_weights[mask_crash]
    
    max_w = sample_weights_test.max()
    mean_w = sample_weights_test[mask_crash].mean()
    print(f"⚖️ 重み計算完了:")
    print(f"   - Max Weight : {max_w:.2f}")
    print(f"   - Mean Weight: {mean_w:.2f}")
df_base['Sample_Weight_test'] = sample_weights_test

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
        w_tr = df_ml.loc[mask_train, 'Sample_Weight_train']
        
        X_te = df_ml.loc[mask_test, feats]
        y_te = df_ml.loc[mask_test, 'Target_Test']
        w_te = df_ml.loc[mask_test, 'Sample_Weight_test']
        
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
            lw = 1.2
            alpha = 1.0
            zorder = 10
            msize = 10
        else:
            lw = 0.6
            alpha = 0.6
            zorder = 1
            msize = 8

        color = colors[i % len(colors)]

        if PLOT_STYLE == "scatter":
            y_vals = df[col]
            zero_mask = (y_vals == 0)

            sizes = np.where(zero_mask, msize * 0.6, msize)
            alphas = np.where(zero_mask, 0.2, alpha)

            plt.scatter(df["Window_Size"], y_vals, s=sizes, alpha=1.0,
                        label=clean_name, color=color, zorder=zorder)
            if zero_mask.any():
                plt.scatter(df["Window_Size"][zero_mask], y_vals[zero_mask],
                            s=sizes[zero_mask], alpha=0.2,
                            color=color, zorder=zorder)
        else:
            plt.plot(df["Window_Size"], df[col],
                     marker=None, linestyle='-', linewidth=lw, alpha=alpha,
                     label=clean_name, color=color, zorder=zorder)
    
    # タイトルは表示しない
    plt.xlabel("Window Size (Days)", fontsize=14)
    plt.ylabel("Recall Score", fontsize=14)
    plt.legend(loc='upper right')
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
         'RMT_Level', 'RMT_Vel', 'RMT_Accel', 'E_pot']

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
    w_tr = df_best.loc[mask_train, 'Sample_Weight_train']
    
    X_te = df_best.loc[mask_test, feats]
    y_te = df_best.loc[mask_test, 'Target_Test']
    
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

baseline = df_best['Target_Test'].mean()
plt.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline (Random): {baseline:.3f}')

path_pr = os.path.join(OUTPUT_DIR, "best_pr_curve.png")
plt.savefig(path_pr, dpi=300)
print(f"📊 PR曲線を保存しました: {path_pr}")
plt.show()
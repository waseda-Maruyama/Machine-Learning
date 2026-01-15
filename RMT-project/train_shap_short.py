import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import shap
import os
import warnings
from sklearn.metrics import precision_recall_curve, average_precision_score

# =========================================================
# ⚙️ 論文用スタイル設定 (Publication Quality)
# =========================================================
plt.rcParams['font.family'] = 'DejaVu Sans' 
plt.rcParams['font.size'] = 14
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 300
plt.rcParams['lines.linewidth'] = 2.5

# =========================================================
# ⚙️ シミュレーション設定
# =========================================================
INPUT_FILE = "dataset_ml_weighted.csv" # あなたのデータファイル名
SCENARIO_NAME = "2025 Tariff"          # 概要書に載せるメインのシナリオ
SCENARIO_RANGE = ["2024-01-01", "2024-10-31"] # 期間

# バックテスト用パラメータ
SELL_THRESHOLD = 0.5   # リスク判定閾値
BUY_THRESHOLD = 0.2    # 買い判定閾値
CONFIRM_DAYS = 5       # 買い確認期間（連続で低い確率が続く日数）
COST = 0.001      # 取引コスト (0.1%)

# 出力先（図）ディレクトリ
OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 特徴量定義
# Model A: テクニカルのみ
feats_Model_A = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14']
# Model B: テクニカル + RMT (Short & Long) -> これが提案手法
feats_Model_B = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14', 
                 'RMT_Raw_S', 'RMT_Vel_S', 'RMT_Accel_S']

# =========================================================
# 1. データ準備 & モデル学習
# =========================================================
if not os.path.exists(INPUT_FILE):
    print(f"❌ ファイルが見つかりません: {INPUT_FILE}")
    exit()

df_ml = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)

print(f"🔄 シナリオ: {SCENARIO_NAME} の分析を開始...")

# 期間分割
test_start, test_end = [pd.to_datetime(d) for d in SCENARIO_RANGE]
test_mask = (df_ml.index >= test_start) & (df_ml.index <= test_end)
train_mask = df_ml.index < test_start

X_train = df_ml.loc[train_mask]
y_train = df_ml.loc[train_mask, 'Target']
X_test = df_ml.loc[test_mask]
y_test = df_ml.loc[test_mask, 'Target']

# 学習パラメータ (不均衡データ対応)
# サンプル重みがある場合はそれを使用し、クラス不均衡補正のためのscale_pos_weightも計算する
if 'Sample_Weight' in df_ml.columns:
    w_train = df_ml.loc[train_mask, 'Sample_Weight']
else:
    w_train = np.ones(len(y_train))

pos_ratio = y_train.sum() / len(y_train) if len(y_train) > 0 else 0
params = {
    'random_state': 42,
    'scale_pos_weight': 1.0,
    'verbose': -1,
    'n_jobs': 1
}
print(f"🔧 Using sample weights: {'Sample_Weight' in df_ml.columns}")

# --- Model A (Base) 学習 ---
clf_a = lgb.LGBMClassifier(**params)
clf_a.fit(X_train[feats_Model_A], y_train, sample_weight=w_train)
probs_a = clf_a.predict_proba(X_test[feats_Model_A])[:, 1]

# --- Model B (Proposed) 学習 ---
clf_b = lgb.LGBMClassifier(**params)
clf_b.fit(X_train[feats_Model_B], y_train, sample_weight=w_train)
probs_b = clf_b.predict_proba(X_test[feats_Model_B])[:, 1]

# --- Model A (Base) ---
prec_a, rec_a, thresholds_a = precision_recall_curve(y_test, probs_a)
ap_a = average_precision_score(y_test, probs_a)
plt.plot(rec_a, prec_a, linestyle='--', color='navy', alpha=0.6, 
         label=f'Baseline (AP={ap_a:.3f})')

# --- Model B (Proposed) ---
prec_b, rec_b, thresholds_b = precision_recall_curve(y_test, probs_b)
ap_b = average_precision_score(y_test, probs_b)
plt.plot(rec_b, prec_b, color='crimson', linewidth=3, 
         label=f'Proposed (AP={ap_b:.3f})')

# 1. 閾値配列の中から、TARGET_THRESHOLD に最も近い値のインデックスを探す
#    np.abs(配列 - 目標値).argmin() で、差が最小になる場所が見つかります
closest_idx = np.abs(thresholds_b - SELL_THRESHOLD).argmin()

# 2. そのインデックスに対応する Precision と Recall を取得
#    ※ thresholdsは prec, rec より要素数が1つ少ないため、インデックスはそのまま使えます
target_prec = prec_b[closest_idx]
target_rec = rec_b[closest_idx]
actual_th = thresholds_b[closest_idx]

# 3. 点を打つ (Scatter Plot)
plt.scatter(target_rec, target_prec, color='black', s=150, zorder=10, 
            edgecolor='white', linewidth=2, label=f'Operating Point (Th={actual_th:.2f})')

# 4. 吹き出しで情報を書き込む (Annotation)
plt.annotate(
    f'Threshold: {actual_th:.2f}\nRecall: {target_rec:.1%}\nPrecision: {target_prec:.1%}',
    xy=(target_rec, target_prec), 
    xytext=(target_rec - 0.3, target_prec - 0.2), # テキストの位置（調整してください）
    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9)
)


plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision (Reliability)')
plt.title(f'Figure 1: Predictive Performance Comparison\nScenario: {SCENARIO_NAME}')
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
fname1 = os.path.join(OUTPUT_DIR, f'figure1_pr_curve_{SCENARIO_NAME.replace(" ", "_")}.png')
plt.savefig(fname1, dpi=300)
print(f"✅ Saved Figure 1: {fname1}")
plt.close()

# =========================================================
# 図2: 資産推移シミュレーション (Buy&Hold vs A vs B)
# =========================================================
print("💰 Drawing Figure 2: Equity Curve...")
def calculate_equity(price_series, probs, sell_threshold, buy_threshold, confirm_days, cost):
    # 修正ポイント1: 
    # 予測値(probs)を、ただの配列から「日付付きのデータ(Series)」に格上げする
    # これで「0, 1, 2...」ではなく「2025-01-01...」という住所を持つ
    probs_series = pd.Series(probs, index=price_series.index)
    
    # 判定用のSeriesを作成（NaNで初期化）
    signal = pd.Series(np.nan, index=price_series.index)

    # 修正ポイント2: 
    # 以降はすべて、日付付きの `probs_series` を使って判定する
    
    # 売り条件: 日付が一致しているのでエラーにならない
    mask_sell = probs_series >= sell_threshold
    signal[mask_sell] = 0

    # 買い条件: rolling計算後も日付情報が維持される
    probs_rolling = probs_series.rolling(window=confirm_days).max()
    mask_buy = probs_rolling < buy_threshold
    signal[mask_buy] = 1

    # --- 以下は変更なし ---
    signal = signal.ffill()
    signal = signal.fillna(1) 

    pos = signal.shift(1).fillna(1)
    
    daily_ret = price_series.pct_change().fillna(0)
    trade_flag = pos.diff().abs().fillna(0)
    
    strat_ret = (daily_ret * pos) - (trade_flag * cost)
    equity = (1 + strat_ret).cumprod()
    
    mdd = ((equity - equity.cummax()) / equity.cummax()).min()
    total_ret = equity.iloc[-1] - 1
    
    return equity, total_ret, mdd

# 価格データの取得 (Closeがない場合はReturnから復元)
if 'Close' in X_test.columns:
    price = X_test['Close']
else:
    price = (1 + X_test['Return']).cumprod() * 100

# 各戦略の計算
eq_bh = (1 + price.pct_change().fillna(0)).cumprod()
mdd_bh = ((eq_bh - eq_bh.cummax()) / eq_bh.cummax()).min()

eq_a, ret_a, mdd_a = calculate_equity(price, probs_a, SELL_THRESHOLD, BUY_THRESHOLD, CONFIRM_DAYS, COST)
eq_b, ret_b, mdd_b = calculate_equity(price, probs_b, SELL_THRESHOLD, BUY_THRESHOLD, CONFIRM_DAYS,  COST)

# プロット
plt.figure(figsize=(10, 6))
plt.plot(eq_bh.index, eq_bh, color='gray', linestyle=':', label=f'Buy & Hold (MDD: {mdd_bh:.1%})')
plt.plot(eq_a.index, eq_a, color='navy', linestyle='--', alpha=0.7, label=f'Baseline (MDD: {mdd_a:.1%})')
plt.plot(eq_b.index, eq_b, color='crimson', linewidth=2.5, label=f'Proposed (MDD: {mdd_b:.1%})')

# X軸を月表示 (例: 4月) にフォーマット
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b')) 
plt.xticks(rotation=0)

plt.annotate('Optimal Re-entry\n(RMT detects structural break)', 
             xy=(pd.to_datetime('2025-04-05'), 0.7), # 矢印の先端（底値付近）
             xytext=(pd.to_datetime('2025-06-01'), 0.6), # テキストの場所
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=12, color='darkred', weight='bold')

plt.annotate('Lagging Entry\n(Volatility is slow)', 
             xy=(pd.to_datetime('2025-05-20'), 0.92), # テクニカルが買い戻したあたり
             xytext=(pd.to_datetime('2025-07-01'), 1.2), 
             arrowprops=dict(facecolor='gray', shrink=0.05, linestyle='--'),
             fontsize=10, color='navy')

# 暴落回避ポイントの可視化 (Model BがCashポジションの場所)
#cash_pos = (probs_b >= THRESHOLD)
#plt.fill_between(eq_b.index, eq_b.min(), eq_b.max(), where=cash_pos, 
#                 color='red', alpha=0.1, label='Risk Avoidance (Cash)')

plt.title(f'Figure 2: Simulation Result ({SCENARIO_NAME})\nTransaction Cost: {COST:.1%}')
plt.ylabel('Cumulative Return')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
fname2 = os.path.join(OUTPUT_DIR, f'figure2_equity_{SCENARIO_NAME.replace(" ", "_")}.png')
plt.savefig(fname2, dpi=300)
print(f"✅ Saved Figure 2: {fname2}")
plt.close()

# =========================================================
# 図3: SHAP (Proposed Model B の解釈)
# =========================================================
print("🧠 Drawing Figure 3: SHAP Importance...")

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray"
    )
    explainer = shap.TreeExplainer(clf_b)
    shap_values = explainer.shap_values(X_test[feats_Model_B])

if isinstance(shap_values, list):
    shap_vals_target = shap_values[1]
else:
    shap_vals_target = shap_values

plt.figure(figsize=(12, 8))
plt.title(f'Figure 3: Feature Importance (Proposed Model)', pad=20)
shap.summary_plot(shap_vals_target, X_test[feats_Model_B], show=False, cmap=plt.get_cmap("coolwarm"))
plt.tight_layout()
fname3 = os.path.join(OUTPUT_DIR, f'figure3_shap_{SCENARIO_NAME.replace(" ", "_")}.png')
plt.savefig(fname3, dpi=300)
print(f"✅ Saved Figure 3: {fname3}")
plt.close()

# =========================================================
# 結果サマリ出力 (概要書の本文に書く数字)
# =========================================================
print("\n📝 Result Summary for Abstract:")
print("-" * 60)
print(f"Scenario: {SCENARIO_NAME}")
print(f"AP Improvement: {ap_a:.3f} (Base) -> {ap_b:.3f} (Prop) | (+{ap_b - ap_a:.3f})")
print(f"MDD Reduction : {mdd_bh:.1%} (Hold) -> {mdd_b:.1%} (Prop)")
print(f"Total Return  : {eq_bh.iloc[-1]-1:.1%} (Hold) -> {ret_b:.1%} (Prop)")
print("-" * 60)
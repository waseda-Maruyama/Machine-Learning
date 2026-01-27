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
INPUT_FILE = "dataset_ml_weighted_1.csv" # あなたのデータファイル名
SCENARIO_NAME = "2025 Tariff"          # 概要書に載せるメインのシナリオ
SCENARIO_RANGE = ["2024-01-01", "2025-8-30"] # 期間

# バックテスト用パラメータ
SELL_THRESHOLD = 0.5   # リスク判定閾値
CONFIRM_DAYS = 3      # 買い確認期間（確率合計の計算期間）
CONFIRM_SUM = 0.01      # 買い判定閾値（期間合計）
COST = 0.001      # 取引コスト (0.1%)

# 出力先（図）ディレクトリ
OUTPUT_DIR = "figures_index1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 特徴量定義
# Model A: テクニカルのみ
feats_Model_A = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14']
# Model B: テクニカル + RMT (Long) -> これが提案手法
feats_Model_B = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14', 
                 'RMT_Raw', 'RMT_Vel', 'RMT_Accel']
feats_Model_C = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14', 
                 'RMT_Raw', 'RMT_Vel',  'RMT_Accel', 'E_pot', 'E_pot_Vel']
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
    'scale_pos_weight': 1.0/pos_ratio if pos_ratio > 0 else 1.0,
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

# ===Model C (proposed b) 学習===
clf_c = lgb.LGBMClassifier(**params)
clf_c.fit(X_train[feats_Model_C], y_train, sample_weight=w_train)
probs_c = clf_c.predict_proba(X_test[feats_Model_C])[:, 1]

# --- Model A (Base) ---
prec_a, rec_a, thresholds_a = precision_recall_curve(y_test, probs_a)
ap_a = average_precision_score(y_test, probs_a)
plt.figure(figsize=(10, 7))
plt.plot(rec_a, prec_a, linestyle='--', color='navy', alpha=0.6, 
         label=f'Baseline (AP={ap_a:.3f})')

# --- Model B (Proposed) ---
prec_b, rec_b, thresholds_b = precision_recall_curve(y_test, probs_b)
ap_b = average_precision_score(y_test, probs_b)
plt.plot(rec_b, prec_b, color='crimson', linewidth=3, 
         label=f'Proposed (AP={ap_b:.3f})')
# --- Model C (Proposed b) ---
prec_c, rec_c, thresholds_c = precision_recall_curve(y_test, probs_c)
ap_c = average_precision_score(y_test, probs_c)
plt.plot(rec_c, prec_c, color='darkgreen', linewidth=2,
            label=f'Proposed b (AP={ap_c:.3f})')

# 全モデルの閾値プロット
# Model A
if len(thresholds_a) > 0:
    closest_idx_a = np.abs(thresholds_a - SELL_THRESHOLD).argmin()
    target_prec_a = prec_a[closest_idx_a]
    target_rec_a = rec_a[closest_idx_a]
    actual_th_a = thresholds_a[closest_idx_a]
    plt.scatter(target_rec_a, target_prec_a, color='navy', s=120, zorder=10, 
                edgecolor='white', linewidth=2, marker='o', label=f'A (Th={actual_th_a:.2f})')

# Model B
if len(thresholds_b) > 0:
    closest_idx_b = np.abs(thresholds_b - SELL_THRESHOLD).argmin()
    target_prec_b = prec_b[closest_idx_b]
    target_rec_b = rec_b[closest_idx_b]
    actual_th_b = thresholds_b[closest_idx_b]
    plt.scatter(target_rec_b, target_prec_b, color='crimson', s=120, zorder=10, 
                edgecolor='white', linewidth=2, marker='s', label=f'B (Th={actual_th_b:.2f})')

# Model C
if len(thresholds_c) > 0:
    closest_idx_c = np.abs(thresholds_c - SELL_THRESHOLD).argmin()
    target_prec_c = prec_c[closest_idx_c]
    target_rec_c = rec_c[closest_idx_c]
    actual_th_c = thresholds_c[closest_idx_c]
    plt.scatter(target_rec_c, target_prec_c, color='darkgreen', s=120, zorder=10, 
                edgecolor='white', linewidth=2, marker='^', label=f'C (Th={actual_th_c:.2f})')




plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision (Reliability)')
plt.legend(loc='upper right', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
fname1 = os.path.join(OUTPUT_DIR, f'fig_5_4a.png')
plt.savefig(fname1, dpi=300)
print(f"✅ Saved Figure 1: {fname1}")
plt.close()

# =========================================================
# 図2: 資産推移シミュレーション (Buy&Hold vs A vs B vs C)
# =========================================================
print("💰 Drawing Figure 2: Equity Curve...")
def calculate_equity(price_series, probs, sell_threshold, confirm_days, cost):
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

    # 買い条件: confirm_days期間の確率合計がCONFIRM_SUMより小さい
    probs_rolling = probs_series.rolling(window=confirm_days).sum()
    mask_buy = probs_rolling < CONFIRM_SUM
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
if 'Market_Price_A' in X_test.columns:
    price = X_test['Market_Price_A']
else:
    print("❌ データエラー: Market_Price_A 列が見つかりません")

# 各戦略の計算
eq_bh = (1 + price.pct_change().fillna(0)).cumprod()
mdd_bh = ((eq_bh - eq_bh.cummax()) / eq_bh.cummax()).min()

eq_a, ret_a, mdd_a = calculate_equity(price, probs_a, SELL_THRESHOLD, CONFIRM_DAYS, COST)
eq_b, ret_b, mdd_b = calculate_equity(price, probs_b, SELL_THRESHOLD, CONFIRM_DAYS, COST)
eq_c, ret_c, mdd_c = calculate_equity(price, probs_c, SELL_THRESHOLD, CONFIRM_DAYS, COST)

# プロット
plt.figure(figsize=(10, 6))
plt.plot(eq_bh.index, eq_bh, color='gray', linestyle=':', label=f'Buy & Hold (MDD: {mdd_bh:.1%})')
plt.plot(eq_a.index, eq_a, color='navy', linestyle='--', alpha=0.7, label=f'Baseline (MDD: {mdd_a:.1%})')
plt.plot(eq_b.index, eq_b, color='crimson', linewidth=2.5, label=f'Proposed (MDD: {mdd_b:.1%})')
plt.plot(eq_c.index, eq_c, color='darkgreen', linewidth=2.5, label=f'Proposed b (MDD: {mdd_c:.1%})')


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
shap.summary_plot(shap_vals_target, X_test[feats_Model_B], show=False, cmap=plt.get_cmap("coolwarm"))
plt.tight_layout()
fname3 = os.path.join(OUTPUT_DIR, f'fig_5_10a.png')
plt.savefig(fname3, dpi=300)
print(f"✅ Saved Figure 3: {fname3}")
plt.close()


# =========================================================
# 図5: SHAP (Proposed Model B の解釈)
# =========================================================
print("🧠 Drawing Figure 5: SHAP Importance...")

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray"
    )
    explainer = shap.TreeExplainer(clf_c)
    shap_values = explainer.shap_values(X_test[feats_Model_C])

if isinstance(shap_values, list):
    shap_vals_target = shap_values[1]
else:
    shap_vals_target = shap_values

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_vals_target, X_test[feats_Model_C], show=False, cmap=plt.get_cmap("coolwarm"))
plt.tight_layout()
fname3 = os.path.join(OUTPUT_DIR, f'fig_5_12a.png')
plt.savefig(fname3, dpi=300)
print(f"✅ Saved Figure 5: {fname3}")
plt.close()


# =========================================================
# 結果サマリ出力 (概要書の本文に書く数字)
# =========================================================
print("\n📝 Result Summary for Abstract:")
print("-" * 60)
print(f"Scenario: {SCENARIO_NAME}")
print(f"AP Improvement: {ap_a:.3f} (Base) -> {ap_b:.3f} (Prop) | (+{ap_b - ap_a:.3f}) ->  {ap_c:.3f}(prob b)")
print(f"MDD Reduction : {mdd_bh:.1%} (Hold) -> {mdd_b:.1%} (Prop) -> {mdd_c:.1%}(prob b)")
print(f"Total Return  : {eq_bh.iloc[-1]-1:.1%} (Hold) -> {ret_b:.1%} (Prop) -> {ret_c:.1%}(prob b)")
print("-" * 60)

# =========================================================
# 図4: PR曲線 (Weighted 比較)
# =========================================================
print("📊 Drawing Figure 4: PR Curve ( Weighted)...")

# 1. テストデータ用の重みを取得
if 'Sample_Weight' in df_ml.columns:
    w_test = df_ml.loc[test_mask, 'Sample_Weight']
else:
    w_test = np.ones(len(y_test))

# --- A. ベースライン (Model A) - weighted ---
prec_a_w, rec_a_w, _ = precision_recall_curve(y_test, probs_a, sample_weight=w_test)
ap_a_w = average_precision_score(y_test, probs_a, sample_weight=w_test)


# --- B. 提案モデル (Model B) - Weighted (金額インパクトベース) ---
# sample_weight を渡して計算
prec_b_w, rec_b_w, _ = precision_recall_curve(y_test, probs_b, sample_weight=w_test)
ap_b_w = average_precision_score(y_test, probs_b, sample_weight=w_test)

# ----C. 提案モデル (Model C) - Weighted (金額インパクトベース) ---
prec_c_w, rec_c_w, _ = precision_recall_curve(y_test, probs_c, sample_weight=w_test)
ap_c_w = average_precision_score(y_test, probs_c, sample_weight=w_test)


# --- プロット作成 ---
plt.figure(figsize=(10, 7))

# 1. Baseline (点線)
plt.plot(rec_a_w, prec_a_w, linestyle=':', color='navy', alpha=0.8, linewidth=2,
         label=f'Baseline (Weighted AP={ap_a_w:.3f})')

# 2. Proposed Standard (破線)
plt.plot(rec_b_w, prec_b_w, linestyle='--', color='crimson', alpha=0.8, linewidth=2,
         label=f'Proposed (Weighted AP={ap_b_w:.3f})')

# 3. Proposed Weighted (実線・太線) -> これが主役！
plt.plot(rec_c_w, prec_c_w, color='darkgreen', linewidth=3, zorder=10,
         label=f'Proposed (Weighted AP={ap_c_w:.3f})')

# --- 動作点 (Operating Point) の描画：全モデル ---
# Model A (Weighted)
if len(thresholds_a) > 0:
    closest_idx_aw = np.abs(thresholds_a - SELL_THRESHOLD).argmin()
    target_prec_aw = prec_a_w[closest_idx_aw]
    target_rec_aw = rec_a_w[closest_idx_aw]
    actual_th_aw = thresholds_a[closest_idx_aw]
    plt.scatter(target_rec_aw, target_prec_aw, color='gray', s=120, zorder=11, 
                edgecolor='white', linewidth=2, marker='o', label=f'A (Th={actual_th_aw:.2f})')

# Model B (Weighted)
if len(thresholds_b) > 0:
    closest_idx_bw = np.abs(thresholds_b - SELL_THRESHOLD).argmin()
    target_prec_bw = prec_b_w[closest_idx_bw]
    target_rec_bw = rec_b_w[closest_idx_bw]
    actual_th_bw = thresholds_b[closest_idx_bw]
    plt.scatter(target_rec_bw, target_prec_bw, color='crimson', s=120, zorder=11, 
                edgecolor='white', linewidth=2, marker='s', label=f'B (Th={actual_th_bw:.2f})')

# Model C (Weighted)
if len(thresholds_c) > 0:
    closest_idx_cw = np.abs(thresholds_c - SELL_THRESHOLD).argmin()
    target_prec_cw = prec_c_w[closest_idx_cw]
    target_rec_cw = rec_c_w[closest_idx_cw]
    actual_th_cw = thresholds_c[closest_idx_cw]
    plt.scatter(target_rec_cw, target_prec_cw, color='darkgreen', s=120, zorder=11, 
                edgecolor='white', linewidth=2, marker='^', label=f'C (Th={actual_th_cw:.2f})')



plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision (Reliability)')
plt.legend(loc='lower right') # 左下に配置変更（曲線とかぶらないように）
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

fname1 = os.path.join(OUTPUT_DIR, f'fig_5_12b.png')
plt.savefig(fname1, dpi=300)
print(f"✅ Saved Weighted PR Curve: {fname1}")
plt.close()

# --- コンソール出力用 ---
print("\n⚖️ Weighted Evaluation Results:")
print(f"Baseline Standard AP: {ap_a_w:.4f}")
print(f"Proposed Standard AP: {ap_b_w:.4f}")
print(f"Proposed Weighted AP: {ap_c_w:.4f}")
if ap_c_w > ap_b:
    print("🚀 Result: Weighted score is HIGHER. The model effectively captures large crashes!")
else:
    print("⚠️ Result: Weighted score is lower. The model might be missing some large crashes.")

# === サマリCSV保存 ===
summary_data = {
    'Scenario': [SCENARIO_NAME],
    'AP_Baseline': [ap_a],
    'AP_Proposed': [ap_b],
    'AP_Proposed_b': [ap_c],
    'MDD_BuyHold': [mdd_bh],
    'MDD_Proposed': [mdd_b],
    'MDD_Proposed_b': [mdd_c],
    'Return_BuyHold': [eq_bh.iloc[-1] - 1],
    'Return_Proposed': [ret_b],
    'Return_Proposed_b': [ret_c],
    'Weighted_AP_Baseline': [ap_a_w],
    'Weighted_AP_Proposed': [ap_b_w],
    'Weighted_AP_Proposed_b': [ap_c_w]
}
df_summary = pd.DataFrame(summary_data)
summary_output_file = os.path.join(OUTPUT_DIR, f"summary_results_{SCENARIO_NAME.replace(' ', '_')}.csv")
df_summary.to_csv(summary_output_file, index=False)
print(f"✅ Summary results saved to: {summary_output_file}")

# === 学習結果のCSV保存（Train AP等） ===
train_probs_a = clf_a.predict_proba(X_train[feats_Model_A])[:, 1]
train_probs_b = clf_b.predict_proba(X_train[feats_Model_B])[:, 1]
train_probs_c = clf_c.predict_proba(X_train[feats_Model_C])[:, 1]

ap_a_train = average_precision_score(y_train, train_probs_a)
ap_b_train = average_precision_score(y_train, train_probs_b)
ap_c_train = average_precision_score(y_train, train_probs_c)

if 'Sample_Weight' in df_ml.columns:
    ap_a_train_w = average_precision_score(y_train, train_probs_a, sample_weight=w_train)
    ap_b_train_w = average_precision_score(y_train, train_probs_b, sample_weight=w_train)
    ap_c_train_w = average_precision_score(y_train, train_probs_c, sample_weight=w_train)
else:
    ap_a_train_w = ap_a_train
    ap_b_train_w = ap_b_train
    ap_c_train_w = ap_c_train

train_summary = {
    'Scenario': [SCENARIO_NAME],
    'Train_AP_Baseline': [ap_a_train],
    'Train_AP_Proposed': [ap_b_train],
    'Train_AP_Proposed_b': [ap_c_train],
    'Train_Weighted_AP_Baseline': [ap_a_train_w],
    'Train_Weighted_AP_Proposed': [ap_b_train_w],
    'Train_Weighted_AP_Proposed_b': [ap_c_train_w]
}
df_train_summary = pd.DataFrame(train_summary)
train_output_file = os.path.join(OUTPUT_DIR, f"training_results_{SCENARIO_NAME.replace(' ', '_')}.csv")
df_train_summary.to_csv(train_output_file, index=False)
print(f"✅ Training results saved to: {train_output_file}")

# === テスト予測のCSV保存（確率と正解ラベル） ===
# テスト期間の重み（存在しなければ1で埋める）
if 'Sample_Weight' in df_ml.columns:
    weight_test = df_ml.loc[test_mask, 'Sample_Weight']
else:
    weight_test = pd.Series(np.ones(len(X_test)), index=X_test.index)

df_predictions = pd.DataFrame({
    'Probs_A': probs_a,
    'Probs_B': probs_b,
    'Probs_C': probs_c,
    'Target': y_test,
    'Weight': weight_test
}, index=X_test.index)
preds_output_file = os.path.join(OUTPUT_DIR, f"predictions_{SCENARIO_NAME.replace(' ', '_')}.csv")
df_predictions.to_csv(preds_output_file)
print(f"✅ Predictions saved to: {preds_output_file}")

# === 特徴量重要度のCSV保存（LightGBM） ===
imp_b = pd.DataFrame({
    'feature': feats_Model_B,
    'importance': clf_b.feature_importances_
})
imp_c = pd.DataFrame({
    'feature': feats_Model_C,
    'importance': clf_c.feature_importances_
})
imp_b_file = os.path.join(OUTPUT_DIR, f"importance_model_B_{SCENARIO_NAME.replace(' ', '_')}.csv")
imp_c_file = os.path.join(OUTPUT_DIR, f"importance_model_C_{SCENARIO_NAME.replace(' ', '_')}.csv")
imp_b.to_csv(imp_b_file, index=False)
imp_c.to_csv(imp_c_file, index=False)
print(f"✅ Feature importances saved to: {imp_b_file}, {imp_c_file}")


   # =========================================================
# 📊 図6: ミクロ分析（特定の暴落イベントの深掘り）
# =========================================================
print("🔍 Drawing Figure 6: Micro-analysis of Indicator Surges...")

# 表示する期間を暴落前後に絞る（細かい時間範囲）
FOCUS_RANGE = ["2024-07-01", "2024-09-30"]
focus_start, focus_end = [pd.to_datetime(d) for d in FOCUS_RANGE]
df_focus = df_ml.loc[focus_start:focus_end]
probs_a_series = pd.Series(probs_a, index=X_test.index).loc[focus_start:focus_end]
probs_b_series = pd.Series(probs_b, index=X_test.index).loc[focus_start:focus_end]
probs_c_series = pd.Series(probs_c, index=X_test.index).loc[focus_start:focus_end]

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

# (1) Model A 確率 + 価格ライン
ax1.fill_between(probs_a_series.index, 0, probs_a_series, color='navy', alpha=0.25, label='Prob A')
ax1.plot(df_focus.index, df_focus['Market_Price_A'], color='black', linewidth=1.5, label='Index Price')
ax1.set_ylim(0, 1)
ax1.set_ylabel('Prob A')
ax1.text(0.5, -0.25, '(a) Model A (prob) + Price', transform=ax1.transAxes, ha='center', fontsize=12)
ax1.legend(loc='upper left')
ax1_tw = ax1.twinx()
for i in range(len(df_focus) - 1):
    current_date = df_focus.index[i]
    next_date = df_focus.index[i+1]
    current_price = df_focus['Market_Price_A'].iloc[i]
    next_price = df_focus['Market_Price_A'].iloc[i+1]
    if current_date in y_test.index:
        target_val = y_test[current_date]
    else:
        target_val = 0
    linewidth = 2.0 if target_val == 1 else 1.0
    color = 'red' if target_val == 1 else 'black'
    ax1_tw.plot([current_date, next_date], [current_price, next_price], color=color, linewidth=linewidth, alpha=0.7)
ax1_tw.set_ylabel('Price')
ax1_tw.legend(loc='upper right')

# (2) Model B 確率
ax2.fill_between(probs_b_series.index, 0, probs_b_series, color='crimson', alpha=0.25, label='Prob B')
ax2.set_ylim(0, 1)
ax2.set_ylabel('Prob B')
ax2.text(0.5, -0.25, '(b) Model B (prob)', transform=ax2.transAxes, ha='center', fontsize=12)
ax2.legend(loc='upper left')
ax2_tw = ax2.twinx()
for i in range(len(df_focus) - 1):
    current_date = df_focus.index[i]
    next_date = df_focus.index[i+1]
    current_price = df_focus['Market_Price_A'].iloc[i]
    next_price = df_focus['Market_Price_A'].iloc[i+1]
    if current_date in y_test.index:
        target_val = y_test[current_date]
    else:
        target_val = 0
    linewidth = 2.0 if target_val == 1 else 1.0
    color = 'red' if target_val == 1 else 'black'
    ax2_tw.plot([current_date, next_date], [current_price, next_price], color=color, linewidth=linewidth, alpha=0.7)
ax2_tw.set_ylabel('Price')
ax2_tw.legend(loc='upper right')

# (3) Model C 確率
ax3.fill_between(probs_c_series.index, 0, probs_c_series, color='darkgreen', alpha=0.25, label='Prob C')
ax3.set_ylim(0, 1)
ax3.set_ylabel('Prob C')
ax3.text(0.5, -0.25, '(c) Model C (prob)', transform=ax3.transAxes, ha='center', fontsize=12)
ax3.legend(loc='upper left')
ax3_tw = ax3.twinx()
for i in range(len(df_focus) - 1):
    current_date = df_focus.index[i]
    next_date = df_focus.index[i+1]
    current_price = df_focus['Market_Price_A'].iloc[i]
    next_price = df_focus['Market_Price_A'].iloc[i+1]
    if current_date in y_test.index:
        target_val = y_test[current_date]
    else:
        target_val = 0
    linewidth = 2.0 if target_val == 1 else 1.0
    color = 'red' if target_val == 1 else 'black'
    ax3_tw.plot([current_date, next_date], [current_price, next_price], color=color, linewidth=linewidth, alpha=0.7)
ax3_tw.set_ylabel('Price')
ax3_tw.legend(loc='upper right')

# (4) RMT 最大固有値のサージ (Timing)
ax4.plot(df_focus.index, df_focus['RMT_Raw_L'], color='blue', label='RMT Max Eigenvalue (λ_max)')
ax4.set_ylabel('Synchronization (λ)')
ax4.text(0.5, -0.25, '(d) RMT λ_max', transform=ax4.transAxes, ha='center', fontsize=12)
ax4.legend(loc='upper left')
ax4_tw = ax4.twinx()
for i in range(len(df_focus) - 1):
    current_date = df_focus.index[i]
    next_date = df_focus.index[i+1]
    current_price = df_focus['Market_Price_A'].iloc[i]
    next_price = df_focus['Market_Price_A'].iloc[i+1]
    if current_date in y_test.index:
        target_val = y_test[current_date]
    else:
        target_val = 0
    linewidth = 2.0 if target_val == 1 else 1.0
    color = 'red' if target_val == 1 else 'black'
    ax4_tw.plot([current_date, next_date], [current_price, next_price], color=color, linewidth=linewidth, alpha=0.7)
ax4_tw.set_ylabel('Price')
ax4_tw.legend(loc='upper right')

# (5) ポテンシャルエネルギーの蓄積 (Scale)
ax5.plot(df_focus.index, df_focus['E_pot'], color='darkgreen', label='Market Potential Energy (P^2)')
ax5.set_ylabel('Potential Energy')
ax5.text(0.5, -0.25, '(e) Potential Energy', transform=ax5.transAxes, ha='center', fontsize=12)
ax5.legend(loc='upper left')
ax5_tw = ax5.twinx()
for i in range(len(df_focus) - 1):
    current_date = df_focus.index[i]
    next_date = df_focus.index[i+1]
    current_price = df_focus['Market_Price_A'].iloc[i]
    next_price = df_focus['Market_Price_A'].iloc[i+1]
    if current_date in y_test.index:
        target_val = y_test[current_date]
    else:
        target_val = 0
    linewidth = 2.0 if target_val == 1 else 1.0
    color = 'red' if target_val == 1 else 'black'
    ax5_tw.plot([current_date, next_date], [current_price, next_price], color=color, linewidth=linewidth, alpha=0.7)
ax5_tw.set_ylabel('Price')
ax5_tw.legend(loc='upper right')

# X軸設定
ax5.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

plt.tight_layout()
fname6 = os.path.join(OUTPUT_DIR, f'fig5_13a_micro_data_index1_scenario2_ueda.png')
plt.savefig(fname6, dpi=300)
print(f"✅ Saved Figure 6: {fname6}")
plt.close()

# === 図6データのCSV保存 ===
df_figure6 = pd.DataFrame({
    'Date': df_focus.index,
    'Prob_A': probs_a_series.values,
    'Prob_B': probs_b_series.values,
    'Prob_C': probs_c_series.values,
    'RMT_Raw_L': df_focus['RMT_Raw_L'].values,
    'E_pot': df_focus['E_pot'].values,
    'Market_Price_A': df_focus['Market_Price_A'].values,
    'Target': [y_test.get(d, 0) for d in df_focus.index],
    'Weight': [df_ml.loc[d, 'Sample_Weight'] if 'Sample_Weight' in df_ml.columns and d in df_ml.index else 1.0 for d in df_focus.index]
})
figure6_csv_path = os.path.join(OUTPUT_DIR, f'fig5_13a_micro_data_index1_scenario2_ueda.csv')
df_figure6.to_csv(figure6_csv_path, index=False)
print(f"📊 Saved Figure 6 data to: {figure6_csv_path}")

    # =========================================================
# 📊 図6: ミクロ分析（特定の暴落イベントの深掘り）
# =========================================================
print("🔍 Drawing Figure 6: Micro-analysis of Indicator Surges...")

# 表示する期間を暴落前後に絞る（細かい時間範囲）
FOCUS_RANGE = ["2025-03-01", "2025-04-30"]
focus_start, focus_end = [pd.to_datetime(d) for d in FOCUS_RANGE]
df_focus = df_ml.loc[focus_start:focus_end]
probs_a_series = pd.Series(probs_a, index=X_test.index).loc[focus_start:focus_end]
probs_b_series = pd.Series(probs_b, index=X_test.index).loc[focus_start:focus_end]
probs_c_series = pd.Series(probs_c, index=X_test.index).loc[focus_start:focus_end]

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

# (1) Model A 確率 + 価格ライン
ax1.fill_between(probs_a_series.index, 0, probs_a_series, color='navy', alpha=0.25, label='Prob A')
ax1.plot(df_focus.index, df_focus['Market_Price_A'], color='black', linewidth=1.5, label='Index Price')
ax1.set_ylim(0, 1)
ax1.set_ylabel('Prob A')
ax1.text(0.5, -0.25, '(a) Model A (prob) + Price', transform=ax1.transAxes, ha='center', fontsize=12)
ax1.legend(loc='upper left')
ax1_tw = ax1.twinx()
for i in range(len(df_focus) - 1):
    current_date = df_focus.index[i]
    next_date = df_focus.index[i+1]
    current_price = df_focus['Market_Price_A'].iloc[i]
    next_price = df_focus['Market_Price_A'].iloc[i+1]
    if current_date in y_test.index:
        target_val = y_test[current_date]
    else:
        target_val = 0
    linewidth = 2.0 if target_val == 1 else 1.0
    color = 'red' if target_val == 1 else 'black'
    ax1_tw.plot([current_date, next_date], [current_price, next_price], color=color, linewidth=linewidth, alpha=0.7)
ax1_tw.set_ylabel('Price')
ax1_tw.legend(loc='upper right')

# (2) Model B 確率
ax2.fill_between(probs_b_series.index, 0, probs_b_series, color='crimson', alpha=0.25, label='Prob B')
ax2.set_ylim(0, 1)
ax2.set_ylabel('Prob B')
ax2.text(0.5, -0.25, '(b) Model B (prob)', transform=ax2.transAxes, ha='center', fontsize=12)
ax2.legend(loc='upper left')
ax2_tw = ax2.twinx()
for i in range(len(df_focus) - 1):
    current_date = df_focus.index[i]
    next_date = df_focus.index[i+1]
    current_price = df_focus['Market_Price_A'].iloc[i]
    next_price = df_focus['Market_Price_A'].iloc[i+1]
    if current_date in y_test.index:
        target_val = y_test[current_date]
    else:
        target_val = 0
    linewidth = 2.0 if target_val == 1 else 1.0
    color = 'red' if target_val == 1 else 'black'
    ax2_tw.plot([current_date, next_date], [current_price, next_price], color=color, linewidth=linewidth, alpha=0.7)
ax2_tw.set_ylabel('Price')
ax2_tw.legend(loc='upper right')

# (3) Model C 確率
ax3.fill_between(probs_c_series.index, 0, probs_c_series, color='darkgreen', alpha=0.25, label='Prob C')
ax3.set_ylim(0, 1)
ax3.set_ylabel('Prob C')
ax3.text(0.5, -0.25, '(c) Model C (prob)', transform=ax3.transAxes, ha='center', fontsize=12)
ax3.legend(loc='upper left')
ax3_tw = ax3.twinx()
for i in range(len(df_focus) - 1):
    current_date = df_focus.index[i]
    next_date = df_focus.index[i+1]
    current_price = df_focus['Market_Price_A'].iloc[i]
    next_price = df_focus['Market_Price_A'].iloc[i+1]
    if current_date in y_test.index:
        target_val = y_test[current_date]
    else:
        target_val = 0
    linewidth = 2.0 if target_val == 1 else 1.0
    color = 'red' if target_val == 1 else 'black'
    ax3_tw.plot([current_date, next_date], [current_price, next_price], color=color, linewidth=linewidth, alpha=0.7)
ax3_tw.set_ylabel('Price')
ax3_tw.legend(loc='upper right')

# (4) RMT 最大固有値のサージ (Timing)
ax4.plot(df_focus.index, df_focus['RMT_Raw_L'], color='blue', label='RMT Max Eigenvalue (λ_max)')
ax4.set_ylabel('Synchronization (λ)')
ax4.text(0.5, -0.25, '(d) RMT λ_max', transform=ax4.transAxes, ha='center', fontsize=12)
ax4.legend(loc='upper left')
ax4_tw = ax4.twinx()
for i in range(len(df_focus) - 1):
    current_date = df_focus.index[i]
    next_date = df_focus.index[i+1]
    current_price = df_focus['Market_Price_A'].iloc[i]
    next_price = df_focus['Market_Price_A'].iloc[i+1]
    if current_date in y_test.index:
        target_val = y_test[current_date]
    else:
        target_val = 0
    linewidth = 2.0 if target_val == 1 else 1.0
    color = 'red' if target_val == 1 else 'black'
    ax4_tw.plot([current_date, next_date], [current_price, next_price], color=color, linewidth=linewidth, alpha=0.7)
ax4_tw.set_ylabel('Price')
ax4_tw.legend(loc='upper right')

# (5) ポテンシャルエネルギーの蓄積 (Scale)
ax5.plot(df_focus.index, df_focus['E_pot'], color='darkgreen', label='Market Potential Energy (P^2)')
ax5.set_ylabel('Potential Energy')
ax5.text(0.5, -0.25, '(e) Potential Energy', transform=ax5.transAxes, ha='center', fontsize=12)
ax5.legend(loc='upper left')
ax5_tw = ax5.twinx()
for i in range(len(df_focus) - 1):
    current_date = df_focus.index[i]
    next_date = df_focus.index[i+1]
    current_price = df_focus['Market_Price_A'].iloc[i]
    next_price = df_focus['Market_Price_A'].iloc[i+1]
    if current_date in y_test.index:
        target_val = y_test[current_date]
    else:
        target_val = 0
    linewidth = 2.0 if target_val == 1 else 1.0
    color = 'red' if target_val == 1 else 'black'
    ax5_tw.plot([current_date, next_date], [current_price, next_price], color=color, linewidth=linewidth, alpha=0.7)
ax5_tw.set_ylabel('Price')
ax5_tw.legend(loc='upper right')

# X軸設定
ax5.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))

plt.tight_layout()
fname6 = os.path.join(OUTPUT_DIR, f'fig5_13b_micro_data_index1_scenario2_tariff.png')
plt.savefig(fname6, dpi=300)
print(f"✅ Saved Figure 6: {fname6}")
plt.close()

# === 図6データのCSV保存 ===
df_figure6 = pd.DataFrame({
    'Date': df_focus.index,
    'Prob_A': probs_a_series.values,
    'Prob_B': probs_b_series.values,
    'Prob_C': probs_c_series.values,
    'RMT_Raw_L': df_focus['RMT_Raw_L'].values,
    'E_pot': df_focus['E_pot'].values,
    'Market_Price_A': df_focus['Market_Price_A'].values,
    'Target': [y_test.get(d, 0) for d in df_focus.index],
    'Weight': [df_ml.loc[d, 'Sample_Weight'] if 'Sample_Weight' in df_ml.columns and d in df_ml.index else 1.0 for d in df_focus.index]
})
figure6_csv_path = os.path.join(OUTPUT_DIR, f'fig5_13b_micro_data_index1_scenario2_tariff.csv')
df_figure6.to_csv(figure6_csv_path, index=False)
print(f"📊 Saved Figure 6 data to: {figure6_csv_path}")
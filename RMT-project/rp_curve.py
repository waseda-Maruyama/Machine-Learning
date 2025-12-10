import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import os
from config import scenarios # 設定ファイル

# =========================================================
# 1. データ読み込み & 特徴量定義
# =========================================================
INPUT_FILE = "dataset_ml_dual.csv"
print(f"📊 データセット {INPUT_FILE} を読み込んでいます...")

if not os.path.exists(INPUT_FILE):
    print("❌ ファイルが見つかりません。dataset_ml_dual.csv が必要です。")
    exit()

df_ml = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)

# 特徴量グループ
feats_Tech = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14']
feats_RMT_S = ['RMT_Raw_S', 'RMT_Vel_S', 'RMT_Accel_S', 'RMT_Zscore_S']
feats_RMT_L = ['RMT_Raw_L', 'RMT_Vel_L', 'RMT_Accel_L', 'RMT_Zscore_L']

# 比較する3つのモデル構成
# 1. Base
feats_Model_A = feats_Tech
# 2. Single RMT (Short)
feats_Model_B = feats_Tech + feats_RMT_S
# 3. Dual RMT (Short + Long)
feats_Model_C = feats_Tech + feats_RMT_S + feats_RMT_L

# =========================================================
# 2. 全シナリオでの予測スコア収集
# =========================================================
print("🤖 全シナリオで学習・予測を実行中...")

y_true_all = []
probs_a_all = []
probs_b_all = [] # 追加
probs_c_all = []

for name, (start_str, end_str) in scenarios.items():
    test_start = pd.to_datetime(start_str)
    test_end = pd.to_datetime(end_str)
    
    # 時系列分割
    train_mask = df_ml.index < test_start
    test_mask = (df_ml.index >= test_start) & (df_ml.index <= test_end)
    
    X_train = df_ml.loc[train_mask]
    y_train = df_ml.loc[train_mask, 'Target']
    X_test = df_ml.loc[test_mask]
    y_test = df_ml.loc[test_mask, 'Target']
    
    if y_test.sum() == 0 or y_train.sum() == 0:
        continue

    # 不均衡データの重み付け
    pos_weight = len(y_train) / (2 * y_train.sum())
    
    # 共通パラメータ
    lgb_params = {
        'random_state': 42,
        'scale_pos_weight': pos_weight,
        'verbose': -1,
        'n_jobs': 1
    }
    
    # --- Model A (Tech Only) ---
    clf_a = lgb.LGBMClassifier(**lgb_params)
    clf_a.fit(X_train[feats_Model_A], y_train)
    probs_a = clf_a.predict_proba(X_test[feats_Model_A])[:, 1]
    
    # --- Model B (Single RMT) ---
    clf_b = lgb.LGBMClassifier(**lgb_params)
    clf_b.fit(X_train[feats_Model_B], y_train)
    probs_b = clf_b.predict_proba(X_test[feats_Model_B])[:, 1]
    
    # --- Model C (Dual RMT) ---
    clf_c = lgb.LGBMClassifier(**lgb_params)
    clf_c.fit(X_train[feats_Model_C], y_train)
    probs_c = clf_c.predict_proba(X_test[feats_Model_C])[:, 1]
    
    # リストに追加
    y_true_all.extend(y_test.values)
    probs_a_all.extend(probs_a)
    probs_b_all.extend(probs_b)
    probs_c_all.extend(probs_c)

print(f"✅ 収集完了: データ数 {len(y_true_all)}件")

# =========================================================
# 3. PR曲線の計算と描画
# =========================================================
print("📈 PR曲線を作成中...")

# APスコア (曲線の下側面積: 高いほど良い)
ap_a = average_precision_score(y_true_all, probs_a_all)
ap_b = average_precision_score(y_true_all, probs_b_all)
ap_c = average_precision_score(y_true_all, probs_c_all)

# 曲線データ生成
precision_a, recall_a, _ = precision_recall_curve(y_true_all, probs_a_all)
precision_b, recall_b, thresholds_b = precision_recall_curve(y_true_all, probs_b_all)
precision_c, recall_c, _ = precision_recall_curve(y_true_all, probs_c_all)

# --- プロット ---
plt.figure(figsize=(10, 8))

# Model A (グレー: ベースライン)
plt.plot(recall_a, precision_a, linestyle='--', color='gray', 
         label=f'Model A (Tech Only) AP={ap_a:.3f}')

# Model B (オレンジ: RMT単体)
plt.plot(recall_b, precision_b, linestyle='-', color='tab:orange', linewidth=2,
         label=f'Model B (Single RMT) AP={ap_b:.3f}')

# Model C (赤: Dual RMT)
plt.plot(recall_c, precision_c, linestyle='-', color='tab:red', linewidth=3,
         label=f'Model C (Dual RMT) AP={ap_c:.3f}')

# No Skill ライン
no_skill = sum(y_true_all) / len(y_true_all)
plt.plot([0, 1], [no_skill, no_skill], linestyle=':', label='Random', color='blue')

# Model b の閾値0.20の位置をマーク
idx_020 = np.argmin(np.abs(thresholds_b - 0.20))
plt.scatter(recall_b[idx_020], precision_b[idx_020], marker='o', color='black', s=100, zorder=10, 
            label='Threshold 0.20 (Model B)')

plt.title('Precision-Recall Curve: Step-wise Improvement', fontsize=14)
plt.xlabel('Recall (Sensitivity)', fontsize=12)
plt.ylabel('Precision (Reliability)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
output_img = "pr_curve_3models.png"
plt.savefig(output_img)
plt.show()

# =========================================================
# 4. 閾値ごとの詳細テーブル (Model Cについて表示)
# =========================================================
print(f"\n🏆 Model Comparison (Average Precision):")
print(f"   A (Tech)   : {ap_a:.3f}")
print(f"   B (+Single): {ap_b:.3f} (+{ap_b-ap_a:.3f})")
print(f"   C (+Dual)  : {ap_c:.3f} (+{ap_c-ap_b:.3f})")

print("\n📋 閾値感度テーブル (Model b):")
print(f"{'Threshold':<10} | {'Recall':<10} | {'Precision':<10} | {'F1-Score':<10}")
print("-" * 50)

for th in np.arange(0.1, 0.95, 0.05):
    preds = (np.array(probs_b_all) >= th).astype(int)
    tp = np.sum((preds == 1) & (np.array(y_true_all) == 1))
    fp = np.sum((preds == 1) & (np.array(y_true_all) == 0))
    fn = np.sum((preds == 0) & (np.array(y_true_all) == 1))
    
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
    print(f"{th:<10.2f} | {rec:<10.3f} | {prec:<10.3f} | {f1:<10.3f}")
    

print("-" * 50)
print(f"✅ 保存完了: {output_img}")
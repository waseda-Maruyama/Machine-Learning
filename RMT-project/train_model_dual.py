import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import recall_score
import os
from config import scenarios # 設定ファイル

# =========================================================
# ⚙️ 設定エリア
# =========================================================
INPUT_FILE = "dataset_ml_dual.csv" # create_dataset.py で作ったファイル
DECISION_THRESHOLD = 0.15          # 危険度 15% を超えたら警報

# =========================================================
# 1. データ読み込み
# =========================================================
print(f"🛠️ データセット {INPUT_FILE} を読み込んでいます...")

if not os.path.exists(INPUT_FILE):
    print("❌ ファイルが見つかりません。create_dataset.py を先に実行してください。")
    exit()

df_ml = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)

# ---------------------------------------------------------
# 2. 特徴量リストの定義 (3段階)
# ---------------------------------------------------------
# (1) テクニカル指標 (ベースライン)
feats_Tech = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14']

# (2) Single RMT (短期窓のみ追加)
# 従来のRMT研究に近い構成
feats_RMT_S = [
    'RMT_Raw_S', 'RMT_Vel_S', 'RMT_Accel_S', 'RMT_Zscore_S'
]
feats_Model_B = feats_Tech + feats_RMT_S

# (3) Dual RMT (短期 + 長期)
# 今回の提案手法 (マルチスケール)
feats_RMT_L = [
    'RMT_Raw_L', 'RMT_Vel_L', 'RMT_Accel_L', 'RMT_Zscore_L'
]
feats_Model_C = feats_Tech + feats_RMT_S + feats_RMT_L

# 安全装置: 列チェック
missing = [c for c in feats_Model_C if c not in df_ml.columns]
if missing:
    print(f"❌ エラー: CSVに以下の列がありません: {missing}")
    print("   create_dataset.py を再実行してください。")
    exit()

print(f"📊 データ形状: {df_ml.shape}")
print(f"   ターゲット数: {df_ml['Target'].sum()} (率: {df_ml['Target'].mean():.2%})")

# =========================================================
# 3. 学習ループ (3モデル比較)
# =========================================================
print(f"\n🤖 3つのモデルで学習開始 (閾値: {DECISION_THRESHOLD:.0%})...")
print("   Model A: Tech Only")
print("   Model B: Tech + Single RMT (Short)")
print("   Model C: Tech + Dual RMT (Short + Long)")

results = []
last_model_c = None 

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

    # 不均衡データの重み
    pos_weight = len(y_train) / (2 * y_train.sum())
    
    # 共通パラメータ
    lgb_params = {
        'random_state': 42,
        'scale_pos_weight': pos_weight,
        'verbose': -1,
        'n_jobs': 1
    }
    
    # --- Model A (Tech) ---
    clf_a = lgb.LGBMClassifier(**lgb_params)
    clf_a.fit(X_train[feats_Tech], y_train)
    probs_a = clf_a.predict_proba(X_test[feats_Tech])[:, 1]
    rec_a = recall_score(y_test, (probs_a >= DECISION_THRESHOLD).astype(int), zero_division=0)
    
    # --- Model B (Single RMT) ---
    clf_b = lgb.LGBMClassifier(**lgb_params)
    clf_b.fit(X_train[feats_Model_B], y_train)
    probs_b = clf_b.predict_proba(X_test[feats_Model_B])[:, 1]
    rec_b = recall_score(y_test, (probs_b >= DECISION_THRESHOLD).astype(int), zero_division=0)
    
    # --- Model C (Dual RMT) ---
    clf_c = lgb.LGBMClassifier(**lgb_params)
    clf_c.fit(X_train[feats_Model_C], y_train)
    probs_c = clf_c.predict_proba(X_test[feats_Model_C])[:, 1]
    rec_c = recall_score(y_test, (probs_c >= DECISION_THRESHOLD).astype(int), zero_division=0)
    
    # 結果保存
    last_model_c = clf_c
    results.append({
        'Scenario': name,
        'Recall_A': rec_a,
        'Recall_B': rec_b,
        'Recall_C': rec_c
    })
    
    print(f"📍 {name.ljust(12)} | A={rec_a:.1%} -> B={rec_b:.1%} -> C={rec_c:.1%}")

# =========================================================
# 4. 最終評価 & 重要度分析
# =========================================================
if results:
    df_res = pd.DataFrame(results)
    
    mean_a = df_res['Recall_A'].mean()
    mean_b = df_res['Recall_B'].mean()
    mean_c = df_res['Recall_C'].mean()
    
    print("\n🏆 最終成績 (平均再現率 Recall):")
    print(f"   Model A (Tech Only)   : {mean_a:.2%}")
    print(f"   Model B (Single RMT)  : {mean_b:.2%}  (vs A: {mean_b-mean_a:+.2%})")
    print(f"   Model C (Dual RMT)    : {mean_c:.2%}  (vs B: {mean_c-mean_b:+.2%})")
    
    # Model C (全部入り) の特徴量重要度を見る
    print("\n🔍 Dual Modelの重要度ランキング (Gain):")
    imp = pd.DataFrame({
        'Feature': feats_Model_C,
        'Gain': last_model_c.feature_importances_
    })
    imp = imp.sort_values('Gain', ascending=False).reset_index(drop=True)
    print(imp)
    
    # 上位5つに _S と _L のどっちが入っているか？
    top_5 = imp.head(5)['Feature'].tolist()
    print(f"\n💡 分析のヒント: 上位5つ {top_5} に、S(短期)とL(長期)の両方が入っていれば、マルチスケールの成功です。")
    
    imp.to_csv("feature_importance_dual.csv", index=False)
    
else:
    print("\n⚠️ 有効な結果が得られませんでした。")
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import recall_score, precision_score
import os
from config import scenarios # 設定ファイル

# =========================================================
# ⚙️ 設定エリア
# =========================================================
INPUT_FILE = "dataset_ml.csv" # 前処理済みのCSV
DECISION_THRESHOLD = 0.30     # 危険度 15% を超えたら警報を鳴らす (感度調整)

# =========================================================
# 1. データ読み込み
# =========================================================
print(f"🛠️ データセット {INPUT_FILE} を読み込んでいます...")

if not os.path.exists(INPUT_FILE):
    print("❌ ファイルが見つかりません。create_dataset.py を先に実行してください。")
    exit()

# CSVには [Market_Price, Target, RMT_Raw, RMT_Vel, RSI_14...] が全部入っています
df_ml = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)

# ---------------------------------------------------------
# ★追加: 安全装置 (列の存在チェック)
# ---------------------------------------------------------
# 使用する特徴量リスト
features_Tech = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14']
features_RMT = features_Tech + ['RMT_Raw', 'RMT_Vel', 'RMT_Accel', 'RMT_Zscore']

# CSVに必要な列があるか確認
missing_cols = [col for col in features_RMT if col not in df_ml.columns]
if missing_cols:
    print(f"❌ エラー: CSVファイルに以下の列が見つかりません: {missing_cols}")
    print("   create_dataset.py を再実行して、テクニカル指標を計算・保存してください。")
    exit()

print(f"📊 学習用データ最終形状: {df_ml.shape}")
print(f"   ターゲット(暴落)数: {df_ml['Target'].sum()} (率: {df_ml['Target'].mean():.2%})")

# =========================================================
# 3. モデル学習 (Walk-Forward Validation)
# =========================================================
print(f"\n🤖 学習開始 (閾値: {DECISION_THRESHOLD:.0%})...")

results = []
last_model = None 

for name, (start_str, end_str) in scenarios.items():
    test_start = pd.to_datetime(start_str)
    test_end = pd.to_datetime(end_str)
    
    # 時系列に沿って分割 (未来の情報をリークさせない)
    train_mask = df_ml.index < test_start
    test_mask = (df_ml.index >= test_start) & (df_ml.index <= test_end)
    
    X_train = df_ml.loc[train_mask]
    y_train = df_ml.loc[train_mask, 'Target']
    X_test = df_ml.loc[test_mask]
    y_test = df_ml.loc[test_mask, 'Target']
    
    # テスト期間に正解がない、または学習データ不足の場合はスキップ
    if y_test.sum() == 0 or y_train.sum() == 0:
        # print(f"⚠️ {name}: データ不足のためスキップ")
        continue

    # 不均衡データの重み付け (Scale Pos Weight)
    # 正解(暴落)が少ないので、見逃さないように重みを乗せる
    pos_weight = len(y_train) / (2 * y_train.sum())
    
    # -----------------------------------------------------
    # Model A: テクニカルのみ (旧来手法)
    # -----------------------------------------------------
    clf_a = lgb.LGBMClassifier(random_state=42, scale_pos_weight=pos_weight, verbose=-1, n_jobs=1)
    clf_a.fit(X_train[features_Tech], y_train)
    
    # 確率を出して閾値判定
    probs_a = clf_a.predict_proba(X_test[features_Tech])[:, 1]
    pred_a = (probs_a >= DECISION_THRESHOLD).astype(int)
    recall_a = recall_score(y_test, pred_a, zero_division=0)
    
    # -----------------------------------------------------
    # Model B: テクニカル + RMT (提案手法)
    # -----------------------------------------------------
    clf_b = lgb.LGBMClassifier(random_state=42, scale_pos_weight=pos_weight, verbose=-1, n_jobs=1)
    clf_b.fit(X_train[features_RMT], y_train)
    
    # 確率を出して閾値判定
    probs_b = clf_b.predict_proba(X_test[features_RMT])[:, 1]
    pred_b = (probs_b >= DECISION_THRESHOLD).astype(int)
    recall_b = recall_score(y_test, pred_b, zero_division=0)
    
    # 結果表示
    max_prob = probs_b.max()
    print(f"📍 {name.ljust(12)} | MaxProb: {max_prob:.1%} | Recall: Tech={recall_a:.1%} -> RMT={recall_b:.1%} ({recall_b-recall_a:+.1%})")
    
    last_model = clf_b
    results.append({
        'Scenario': name,
        'Recall_Tech': recall_a,
        'Recall_RMT': recall_b,
        'Improvement': recall_b - recall_a
    })

# =========================================================
# 4. 最終評価 & 重要度分析
# =========================================================
if results:
    df_res = pd.DataFrame(results)
    
    print("\n🏆 最終成績 (平均再現率 Recall):")
    print(f"   Model A (Tech Only)  : {df_res['Recall_Tech'].mean():.2%}")
    print(f"   Model B (RMT Enhanced): {df_res['Recall_RMT'].mean():.2%}")
    
    diff = df_res['Recall_RMT'].mean() - df_res['Recall_Tech'].mean()
    if diff > 0:
        print(f"   ✅ RMTの導入で精度が {diff:+.2%}向上しました！")
    
    # 特徴量重要度
    print("\n🔍 AIは『どの指標』を重視したか？ (Gainランキング):")
    imp = pd.DataFrame({
        'Feature': features_RMT,
        'Gain': last_model.feature_importances_
    })
    imp = imp.sort_values('Gain', ascending=False).reset_index(drop=True)
    print(imp)
    
    # CSV保存
    imp.to_csv("feature_importance.csv", index=False)
    print("\n✅ 分析完了。'feature_importance.csv' を確認してください。")
    
else:
    print("\n⚠️ 有効な結果が得られませんでした。閾値(DECISION_THRESHOLD)をもっと下げる必要があるかもしれません。")
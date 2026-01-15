import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import os
from config import scenarios  # 設定ファイル

# =========================================================
# 1. データ読み込み & 特徴量定義
# =========================================================
INPUT_FILE = "dataset_ml_weighted.csv"
print(f"📊 データセット {INPUT_FILE} を読み込んでいます...")

if not os.path.exists(INPUT_FILE):
    print("❌ ファイルが見つかりません。dataset_ml_dual.csv が必要です。")
    # 補足: Target列が含まれている結合済みファイルであることを前提としています
    exit()

df_ml = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)

# 特徴量グループ
feats_Tech = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14']

# 【修正】Zscoreを削除しました（KeyError回避）
feats_RMT_S = ['RMT_Raw_S', 'RMT_Vel_S', 'RMT_Accel_S']
feats_RMT_L = ['RMT_Raw_L', 'RMT_Vel_L', 'RMT_Accel_L']

# 比較する3つのモデル構成
# 1. Base
feats_Model_A = feats_Tech
# 2. Single RMT (Short)
feats_Model_B = feats_Tech + feats_RMT_L
# 3. Dual RMT (Short + Long)
feats_Model_C = feats_Tech + feats_RMT_S + feats_RMT_L

# =========================================================
# 2. "2025 Tariff"シナリオで3モデルの比較を復元
# =========================================================
scenario_name = "2025 Tariff"

if scenario_name in scenarios:
    test_start, test_end = [pd.to_datetime(date) for date in scenarios[scenario_name]]
    test_mask = (df_ml.index >= test_start) & (df_ml.index <= test_end)
    train_mask = df_ml.index < test_start

    X_train = df_ml.loc[train_mask]
    y_train = df_ml.loc[train_mask, 'Target']
    X_test = df_ml.loc[test_mask]
    y_test = df_ml.loc[test_mask, 'Target']

    if y_test.sum() > 0 and y_train.sum() > 0:
        pos_weight = len(y_train) / (2 * y_train.sum())
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

        # PR曲線とAPスコアの計算
        precision_a, recall_a, _ = precision_recall_curve(y_test, probs_a)
        ap_a = average_precision_score(y_test, probs_a)

        precision_b, recall_b, _ = precision_recall_curve(y_test, probs_b)
        ap_b = average_precision_score(y_test, probs_b)

        precision_c, recall_c, _ = precision_recall_curve(y_test, probs_c)
        ap_c = average_precision_score(y_test, probs_c)

        # --- プロット 1: PR曲線 ---
        plt.figure(figsize=(12, 9)) # サイズを少し大きく

        # Model A (グレー: ベースライン)
        plt.plot(recall_a, precision_a, linestyle='--', color='gray', 
                 label=f'Model A (Tech Only) AP={ap_a:.3f}')

        # Model B (オレンジ: RMT単体)
        plt.plot(recall_b, precision_b, linestyle='-', color='tab:orange', linewidth=2,
                 label=f'Model B (Single RMT) AP={ap_b:.3f}')

        # Model C (赤: Dual RMT)
        plt.plot(recall_c, precision_c, linestyle='-', color='tab:red', linewidth=3,
                 label=f'Model C (Dual RMT) AP={ap_c:.3f}')

        # 【修正】文字サイズを大きく設定
        plt.title(f'Precision-Recall Curve: {scenario_name}', fontsize=24)
        plt.xlabel('Recall (Sensitivity)', fontsize=20)
        plt.ylabel('Precision (Reliability)', fontsize=20)
        plt.legend(fontsize=18, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=16) # 軸の数字を大きく
        plt.tight_layout()
        plt.show()

        # 閾値感度テーブル (Model Bについて、"2025 Tariff"シナリオ単体の結果を表示)
        print(f"\n📋 閾値感度テーブル (Model B, シナリオ: {scenario_name}):")
        print(f"{'Threshold':<10} | {'Recall':<10} | {'Precision':<10}")
        print("-" * 50)

        thresholds_to_check = sorted(list(set(np.arange(0.1, 0.95, 0.05).tolist() + [0.30])))

        # 閾値ごとのメトリクスを保存するリスト
        threshold_metrics = []
        for th in thresholds_to_check:
            preds = (np.array(probs_b) >= th).astype(int)
            tp = np.sum((preds == 1) & (np.array(y_test) == 1))
            fp = np.sum((preds == 1) & (np.array(y_test) == 0))
            fn = np.sum((preds == 0) & (np.array(y_test) == 1))
            
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * rec * prec / (rec + prec) if (rec + prec) > 0 else 0
            
            threshold_metrics.append({'Threshold': th, 'Recall': rec, 'Precision': prec, 'F1-Score': f1})
            
            # コンソール出力用
            print(f"{th:<10.2f} | {rec:<10.3f} | {prec:<10.3f}")
        
        print("-" * 50)
        df_threshold_metrics = pd.DataFrame(threshold_metrics)

        # --- プロット 2: 閾値感度 ---
        plt.figure(figsize=(12, 8)) # サイズを少し大きく
        plt.plot(df_threshold_metrics['Threshold'], df_threshold_metrics['Recall'], label='Recall', marker='o', linestyle='-', linewidth=2)
        plt.plot(df_threshold_metrics['Threshold'], df_threshold_metrics['Precision'], label='Precision', marker='s', linestyle='--', linewidth=2)

        # 【修正】文字サイズを大きく設定
        plt.xlabel('Prediction Threshold', fontsize=20)
        plt.ylabel('Score', fontsize=20)
        plt.title(f'Threshold Sensitivity for Model B ({scenario_name})', fontsize=24)
        plt.legend(fontsize=18)
        plt.grid(True, alpha=0.3)
        plt.xticks(np.arange(0.1, 1.0, 0.1), fontsize=16)
        plt.yticks(fontsize=16)
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.show()

    else:
        print(f"❌ {scenario_name}シナリオのデータが不足しています（正例またはデータなし）。")
else:
    print(f"❌ シナリオが見つかりません: {scenario_name}")
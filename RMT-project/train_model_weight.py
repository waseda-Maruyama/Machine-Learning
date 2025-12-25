import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
from config import scenarios # シナリオ定義

# =========================================================
# ⚙️ 設定エリア
# =========================================================
INPUT_FILE = "dataset_ml_weighted.csv"
OUTPUT_FILE = "prediction_comparison.csv"  # 3モデル分の結果
DIR_A = "analysis_Model_A"
DIR_B = "analysis_Model_B"

# =========================================================
# 1. 特徴量の定義 (3段階の進化)
# =========================================================
# Model A: Tech Only (ベースライン)
feats_A = ['Return', 'Vol_20', 'Momentum_10', 'RSI_14']

# Model B: Tech + Single RMT (Short Only)
feats_B = feats_A + ['RMT_Raw_L', 'RMT_Vel_L', 'RMT_Accel_L']

# Model C: Tech + Dual RMT (Short + Long) -> 提案手法
feats_C = feats_B + ['RMT_Raw_S', 'RMT_Vel_S', 'RMT_Accel_S']

models = {
    "Model A (Tech)": feats_A,
    "Model B (Short)": feats_B,
    "Model C (Dual)": feats_C
}

# =========================================================
# 2. データ読み込み & 準備
# =========================================================
if not os.path.exists(INPUT_FILE):
    print("❌ データファイルがありません")
    exit()

df_ml = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)

# 結果格納用のDataFrame
df_result = pd.DataFrame(index=df_ml.index)
df_result['Market_Price'] = df_ml['Market_Price']
df_result['Target'] = df_ml['Target']

# 予測値を格納する列を初期化
for key in ["Prob_A", "Prob_B", "Prob_C"]:
    df_result[key] = np.nan

print(f"🤖 3モデル同時比較学習を開始します...")
print(f"   Model A Features: {len(feats_A)}")
print(f"   Model B Features: {len(feats_B)}")
print(f"   Model C Features: {len(feats_C)}")

importance_listA = []
importance_listB = []

for d in [DIR_A, DIR_B]:
    if os.path.exists(d): shutil.rmtree(d)
    os.makedirs(d)

# =========================================================
# 3. Walk-Forward Loop
# =========================================================
for name, (start_str, end_str) in scenarios.items():
    test_start = pd.to_datetime(start_str)
    test_end = pd.to_datetime(end_str)
    
    print(f"\n📍 Scenario: {name} ({start_str} ~ {end_str})")
    
    # データの切り出し
    train_mask = df_ml.index < test_start
    test_mask = (df_ml.index >= test_start) & (df_ml.index <= test_end)
    
    y_train = df_ml.loc[train_mask, 'Target']
    w_train = df_ml.loc[train_mask, 'Sample_Weight']
    
    if y_train.sum() == 0 or not any(test_mask):
        print("   ⚠️ Skip (データ不足)")
        continue

    # 不均衡データの調整 (共通)
    pos_ratio = y_train.sum() / len(y_train)
    scale_pos_weight = 1.0 / pos_ratio
    
    params = {
        'random_state': 42, 'verbose': -1, 'n_jobs': 1,
        'scale_pos_weight': scale_pos_weight,
        'n_estimators': 1000, 'learning_rate': 0.05
    }

    # --- 3つのモデルを順番に学習 & 予測 ---
    # Model A
    clf_a = lgb.LGBMClassifier(**params)
    clf_a.fit(df_ml.loc[train_mask, feats_A], y_train, sample_weight=w_train)
    probs_a = clf_a.predict_proba(df_ml.loc[test_mask, feats_A])[:, 1]
    df_result.loc[test_mask, 'Prob_A'] = probs_a
    # 重要度取得
    imp_gain = clf_a.booster_.feature_importance(importance_type='gain')
    # DataFrame化してリストに追加
    tmp_df_a = pd.DataFrame({
        'Feature': feats_A, 
        'Importance': imp_gain,
        'Scenario': name
    })
    importance_listA.append(tmp_df_a)
    
    # Model B
    clf_b = lgb.LGBMClassifier(**params)
    clf_b.fit(df_ml.loc[train_mask, feats_B], y_train, sample_weight=w_train)
    probs_b = clf_b.predict_proba(df_ml.loc[test_mask, feats_B])[:, 1]
    df_result.loc[test_mask, 'Prob_B'] = probs_b
    # 重要度取得
    imp_gain = clf_b.booster_.feature_importance(importance_type='gain')
    # DataFrame化してリストに追加
    tmp_df_b = pd.DataFrame({
        'Feature': feats_B,
        'Importance': imp_gain,
        'Scenario': name
    })
    importance_listB.append(tmp_df_b)
    
    # Model C
    clf_c = lgb.LGBMClassifier(**params)
    clf_c.fit(df_ml.loc[train_mask, feats_C], y_train, sample_weight=w_train)
    probs_c = clf_c.predict_proba(df_ml.loc[test_mask, feats_C])[:, 1]
    df_result.loc[test_mask, 'Prob_C'] = probs_c

    
    print(f"   ✅ Done. Max Probs -> A:{probs_a.max():.2f}, B:{probs_b.max():.2f}, C:{probs_c.max():.2f}")

# =========================================================
# 4. 保存
# =========================================================
df_final = df_result.dropna(subset=['Prob_A']) # 予測した期間だけ残す
df_final.to_csv(OUTPUT_FILE)
print(f"\n💾 比較結果を保存しました: {OUTPUT_FILE}")

# =========================================================
# 4. 分析結果の出力関数
# =========================================================
def save_analysis(imp_list, output_dir, model_name):
    if not imp_list:
        return
    
    df_all = pd.concat(imp_list)
    
    # 1. マトリックスCSV保存
    df_matrix = df_all.pivot(index='Feature', columns='Scenario', values='Importance')
    df_matrix['Average'] = df_matrix.mean(axis=1)
    df_matrix = df_matrix.sort_values(by='Average', ascending=False)
    
    csv_path = os.path.join(output_dir, "importance_matrix.csv")
    df_matrix.to_csv(csv_path)
    print(f"   📄 [{model_name}] Matrix saved: {csv_path}")
    
    # 2. シナリオ別グラフ保存
    unique_scenarios = df_all['Scenario'].unique()
    for scenario in unique_scenarios:
        subset = df_all[df_all['Scenario'] == scenario].sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(8, 5))
        try:
            import seaborn as sns
            sns.barplot(data=subset, x='Importance', y='Feature', hue='Feature', legend=False, palette='viridis')
        except ImportError:
            plt.barh(subset['Feature'], subset['Importance'])
            plt.gca().invert_yaxis()
            
        plt.title(f"{model_name} Importance: {scenario}")
        plt.tight_layout()
        
        safe_name = scenario.replace(" ", "_").replace("/", "-")
        plt.savefig(os.path.join(output_dir, f"imp_{safe_name}.png"))
        plt.close()

# 重要度分析 (A)
print("\n📊 Model A Analysis:")
save_analysis(importance_listA, DIR_A, "Model_A")

# 重要度分析 (B)
print("\n📊 Model B Analysis:")
save_analysis(importance_listB, DIR_B, "Model_B")
print("\n🎉 全ての処理が完了しました。")
print(f"   - Model Aの結果: ./{DIR_A}/")
print(f"   - Model Bの結果: ./{DIR_B}/")
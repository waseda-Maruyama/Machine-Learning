import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import japanize_matplotlib

# --- 1. データセットの読み込み ---
try:
    dataset = pd.read_csv('analysis_dataset_v1_RATIO.csv')
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset = dataset.set_index('Date')
    print("✅ 分析用データセット v1 (比率のみ) の読み込み完了。")
except FileNotFoundError:
    print("❌ エラー: 'analysis_dataset_v1_RATIO.csv' が見つかりません。")
    exit()

# --- 2. 特徴量 (X) と 目的変数 (y) の準備 ---
# ★★★ 特徴量リストを「比率」のみに修正 ★★★
features = [
    'ROE', 
    'SelfCapitalRatio'
]
X = dataset[features]
y = dataset['target']

# --- 3. データの分割 (正しい時系列分割) ---
split_point = int(len(X) * 0.8)
split_date = X.index[split_point]
print(f"\n--- データの分割 ---")
print(f"分割日: {split_date}")

X_train = X[X.index < split_date]
y_train = y[y.index < split_date]
X_test = X[X.index >= split_date]
y_test = y[y.index >= split_date]

print(f"学習データ数: {len(X_train)}")
print(f"テストデータ数: {len(X_test)}")

# --- 4. XGBoostモデルの学習 ---
print("\nXGBoostモデルの学習を開始します...")
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
model.fit(X_train, y_train)
print("✅ 学習完了。")

# --- 5. モデルの評価 ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📈 *** 真のベースライン正解率 (Accuracy): {accuracy:.4f} ***")

# --- 6. 特徴量の重要度の可視化 ---
print("\n特徴量の重要度を分析しています...")
importance_df = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\n--- 特徴量の重要度 ---")
print(importance_df)

# グラフを描画
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('重要度 (Feature Importance)')
plt.ylabel('財務特徴量')
plt.title('財務特徴量の重要度分析 (比率のみ)')
plt.gca().invert_yaxis() # 重要度が高い順に上から表示
plt.tight_layout()

# グラフをファイルに保存
graph_filename = 'feature_importance_v1_RATIO.png'
plt.savefig(graph_filename)
print(f"\n📊 特徴量の重要度グラフを '{graph_filename}' として保存しました。")
plt.show()
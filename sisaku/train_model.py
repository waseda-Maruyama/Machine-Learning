import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import japanize_matplotlib # 日本語表示のためのライブラリ

# --- 1. データセットの読み込み ---
try:
    dataset = pd.read_csv('analysis_dataset.csv', index_col='Date', parse_dates=True)
    print("✅ 分析用データセットの読み込み完了。")
except FileNotFoundError:
    print("❌ エラー: 'analysis_dataset.csv' が見つかりません。")
    exit()

# --- 2. 特徴量 (X) と 目的変数 (y) の準備 ---
# 予測に使う特徴量のカラムを指定
features = ['NetSales', 'OperatingProfit', 'Profit', 'TotalAssets', 'Equity', 'ROE', 'SelfCapitalRatio']
X = dataset[features]
y = dataset['target']

# --- 3. データの分割 (時系列を考慮) ---
# データをシャッフルせず、時間で分割する (最初の80%を学習用、残りの20%をテスト用)
test_size = int(len(dataset) * 0.2)
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

print(f"\n学習データ数: {len(X_train)}")
print(f"テストデータ数: {len(X_test)}")

# --- 4. XGBoostモデルの学習 ---
print("\nXGBoostモデルの学習を開始します...")
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, n_estimators=100, learning_rate=0.01)
model.fit(X_train, y_train)
print("✅ 学習完了。")

# --- 5. モデルの評価 ---
# テストデータで予測を実行
y_pred = model.predict(X_test)

# 正解率を計算
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📈 モデルの正解率 (Accuracy): {accuracy:.4f}")

# --- 6. 特徴量の重要度の可視化 ---
print("\n特徴量の重要度を分析しています...")

# 重要度をデータフレームにまとめる
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
plt.title('財務特徴量の重要度分析')
plt.gca().invert_yaxis() # 重要度が高い順に上から表示
plt.tight_layout()

# グラフをファイルに保存
graph_filename = 'feature_importance.png'
plt.savefig(graph_filename)
print(f"\n📊 特徴量の重要度グラフを '{graph_filename}' として保存しました。")
plt.show()
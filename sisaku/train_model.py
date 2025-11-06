import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import japanize_matplotlib

# --- 1. データセットの読み込み ---
try:
    dataset = pd.read_csv('analysis_dataset_v2_close.csv')
    print("✅ 分析用データセット v2 ('Close + 株式数'版) の読み込み完了。")
except FileNotFoundError:
    print("❌ エラー: 'analysis_dataset_v2_close.csv' が見つかりません。")
    exit()

# --- 2. 特徴量 (X) と 目的変数 (y) の準備 ---
# 長い特徴量を短い別名で扱う
long_name_shares = 'NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock'
short_name_shares = 'Issued Shares'

features = [
    'NetSales', 'OperatingProfit', 'Profit', 'TotalAssets', 'Equity', 
    'ROE', 'SelfCapitalRatio',
    'Close', 
    long_name_shares
]
X = dataset[features]
y = dataset['target']

# --- 3. データの分割 (時系列を考慮) ---
test_size = int(len(dataset) * 0.2)
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]

print(f"\n学習データ数: {len(X_train)}")
print(f"テストデータ数: {len(X_test)}")

# --- 4. XGBoostモデルの学習 ---
print("\nXGBoostモデルの学習を開始します...")
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
model.fit(X_train, y_train)
print("✅ 学習完了。")

# --- 5. モデルの評価 ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n📈 モデルの正解率 (Accuracy): {accuracy:.4f}")

# --- 6. 特徴量の重要度の可視化 ---
print("\n特徴量の重要度を分析しています...")
importance_df = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# グラフ表示のために長い特徴量を別名に置換
importance_df['feature'] = importance_df['feature'].replace({long_name_shares: short_name_shares})

print("\n--- 特徴量の重要度 ---")
print(importance_df)

# グラフ描画
plt.figure(figsize=(10, 8))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('重要度')
plt.ylabel('特徴量')
plt.title('特徴量の重要度 (Close + 株式数)')
plt.gca().invert_yaxis()  # 重要度が高いものが上に来るように
plt.tight_layout()  # レイアウトを調整

graph_filename = 'feature_importance_v2_close.png'
plt.savefig(graph_filename)
print(f"\n📊 特徴量の重要度グラフを '{graph_filename}' として保存しました。")
plt.show()
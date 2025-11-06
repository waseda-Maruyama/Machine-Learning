import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import japanize_matplotlib

# --- 1. データセットの読み込み ---
try:
    dataset = pd.read_csv('analysis_dataset_v3_technical.csv')
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset = dataset.set_index('Date')
    
    print("✅ 分析用データセット v3 (テクニカル指標追加) の読み込み完了。")
except FileNotFoundError:
    print("❌ エラー: 'analysis_dataset_v3_technical.csv' が見つかりません。")
    exit()

# --- 2. 特徴量の準備 (別名で扱う) ---
LONG_NAME = 'NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock'
SHORT_NAME = 'IssuedShares'
if LONG_NAME in dataset.columns:
    dataset = dataset.rename(columns={LONG_NAME: SHORT_NAME})
    print(f"特徴量 '{LONG_NAME}' を '{SHORT_NAME}' という別名に変更しました。")

features = [
    'NetSales', 'OperatingProfit', 'Profit', 'TotalAssets', 'Equity', 
    'ROE', 'SelfCapitalRatio',
    'Close', 
    SHORT_NAME,
    'MAdivergence'
]
X = dataset[features]
y = dataset['target']

# --- 3. データの分割 (正しい時系列分割) ---
# ★★★ ここからが最重要修正点 ★★★
# 80%地点の「行番号」を計算
split_point = int(len(X) * 0.8)

# その「行番号」にある「日付」を取得
split_date = X.index[split_point]
# ★★★ 修正完了 ★★★

print(f"\n--- データの分割 ---")
print(f"全データ期間: {X.index.min()} から {X.index.max()} まで")
print(f"分割日: {split_date} (この日を境に学習用とテスト用に分けます)")

# 分割日より「前」のデータを学習用にする
X_train = X[X.index < split_date]
y_train = y[y.index < split_date]

# 分割日「以降」のデータをテスト用にする
X_test = X[X.index >= split_date]
y_test = y[y.index >= split_date]

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
print("\n--- 特徴量の重要度 ---")
print(importance_df)

# グラフを描画
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('重要度 (Feature Importance)')
plt.ylabel('特徴量')
plt.title('財務・テクニカル特徴量の重要度分析 (V3)')
plt.gca().invert_yaxis()
plt.tight_layout()

graph_filename = 'feature_importance_v3_technical.png'
plt.savefig(graph_filename)
print(f"\n📊 特徴量の重要度グラフを '{graph_filename}' として保存しました。")
plt.show()
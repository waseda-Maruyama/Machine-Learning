import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import japanize_matplotlib # 日本語表示のためのライブラリ

# --- 1. データセットの読み込み ---
try:
    # ★★★ 修正点 ★★★
    # 読み込むファイルを v2 に変更
    dataset = pd.read_csv('analysis_dataset_v2.csv', index_col='Date', parse_dates=True)
    print("✅ 分析用データセット v2 (PER, PBR追加版) の読み込み完了。")
except FileNotFoundError:
    print("❌ エラー: 'analysis_dataset_v2.csv' が見つかりません。")
    print("   'create_dataset.py' を実行してファイルを作成してください。")
    exit()

# --- 2. 特徴量 (X) と 目的変数 (y) の準備 ---
# ★★★ 修正点 ★★★
# 予測に使う特徴量に PER と PBR を追加
features = [
    'NetSales', 'OperatingProfit', 'Profit', 'TotalAssets', 'Equity', 
    'ROE', 'SelfCapitalRatio',
    'PER', 'PBR' # 新しく追加した特徴量
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

print("\n--- 特徴量の重要度 ---")
print(importance_df)

# グラフを描画
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('重要度 (Feature Importance)')
plt.ylabel('財務特徴量')
plt.title('財務特徴量の重要度分析 (PER, PBR 追加版)')
plt.gca().invert_yaxis()
plt.tight_layout()

# グラフをファイルに保存
graph_filename = 'feature_importance_v2.png'
plt.savefig(graph_filename)
print(f"\n📊 特徴量の重要度グラフを '{graph_filename}' として保存しました。")
plt.show()
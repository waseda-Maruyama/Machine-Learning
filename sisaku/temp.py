import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import japanize_matplotlib

# --- 1. データセットの読み込み ---
try:
    dataset = pd.read_csv('analysis_dataset_v6_multi_window.csv')
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset = dataset.set_index('Date')
    print("✅ 分析用データセット v6 (マルチテクニカル) の読み込み完了。")
except FileNotFoundError:
    print("❌ エラー: 'analysis_dataset_v6_multi_window.csv' が見つかりません。")
    exit()

# --- 2. 特徴量 (X) と 目的変数 (y) の準備 ---
features = [
    'ROE', 'SelfCapitalRatio', 'OpProfitMargin',
    'PER', 'PBR',
     'MAdivergence_25', 'MAdivergence_50', 'MAdivergence_75',
     'RSI_25', 'RSI_50', 'RSI_75',
      'Volatility_50','Volatility_75',
      'Momentum_75','Momentum_50','Volatility_25','Momentum_25',
]
X = dataset[features]
y = dataset['target']

# --- 3. データの分割 (60-20-20の時系列分割) ---
# ★★★ ここからが最重要修正点 ★★★
# 60%地点と80%地点の「行番号」を計算
split_point_1 = int(len(X) * 0.6)
split_point_2 = int(len(X) * 0.8)

# その「行番号」にある「日付」を取得
split_date_1 = X.index[split_point_1]
split_date_2 = X.index[split_point_2]

print(f"\n--- データの分割 ---")
print(f"全データ期間: {X.index.min()} から {X.index.max()} まで")
print(f"学習データ (〜60%): 〜 {split_date_1}")
print(f"テスト1 (60%〜79%): {split_date_1} 〜 {split_date_2}")
print(f"テスト2 (79%〜100%): {split_date_2} 〜")

# 学習データ (最初の60%)
X_train = X[X.index < split_date_1]
y_train = y[y.index < split_date_1]

# テストデータ1 (次の20%)
X_test1 = X[(X.index >= split_date_1) & (X.index < split_date_2)]
y_test1 = y[(y.index >= split_date_1) & (y.index < split_date_2)]

# テストデータ2 (最後の20%)
X_test2 = X[X.index >= split_date_2]
y_test2 = y[y.index >= split_date_2]
# ★★★ 修正完了 ★★★

print(f"\n学習データ数: {len(X_train)}")
print(f"テストデータ1 数: {len(X_test1)}")
print(f"テストデータ2 数: {len(X_test2)}")

# --- 4. XGBoostモデルの学習 ---
print("\nXGBoostモデルの学習を開始します...")
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
model.fit(X_train, y_train)
print("✅ 学習完了。")

# --- 5. モデルの評価 ---
# ★★★ 評価を2回実行 ★★★
print("\n--- テスト1の評価 ---")
y_pred1 = model.predict(X_test1)
accuracy1 = accuracy_score(y_test1, y_pred1)
print(f"📈 *** テスト1 正解率 (Accuracy): {accuracy1:.4f} ***")

print("\n--- テスト2の評価 ---")
y_pred2 = model.predict(X_test2)
accuracy2 = accuracy_score(y_test2, y_pred2)
print(f"📈 *** テスト2 正解率 (Accuracy): {accuracy2:.4f} ***")

# --- 6. 特徴量の重要度の可視化 ---
print("\n特徴量の重要度を分析しています...")
# (グラフ描画部分は変更なし。モデルは1つなので重要度も1つ)
importance_df = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\n--- 特徴量の重要度 ---")
print(importance_df)

plt.figure(figsize=(10, 10))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('重要度 (Feature Importance)')
plt.ylabel('特徴量')
plt.title('財務・バリュー・テクニカル特徴量の重要度分析 (V6)')
plt.gca().invert_yaxis()
plt.tight_layout()

graph_filename = 'feature_importance_v6_multi_window.png'
plt.savefig(graph_filename)
print(f"\n📊 特徴量の重要度グラフを '{graph_filename}' として保存しました。")
plt.show()
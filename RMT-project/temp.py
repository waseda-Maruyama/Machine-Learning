import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. データ読み込み (日付インデックスを厳密に管理)
# ---------------------------------------------------------
print("🛠️ 比較実験用データセット構築を開始...")

# CSV1: 株価データ (期間: 2016-01-01 ~ ) -> これをマスターとする
try:
    df_prices = pd.read_csv("stock_prices_topix100_simple.csv", index_col=0, parse_dates=True)
    market_index = df_prices.mean(axis=1)
    print(f"✅ 株価データ読み込み: {df_prices.index.min().date()} ~ {df_prices.index.max().date()}")
except:
    print("❌ stock_prices_topix100_simple.csv がありません")
    exit()

# CSV2: RMTデータ (期間: 窓幅分遅れて開始 ~ )
try:
    ts_rmt = pd.read_csv("feature_rmt_eigen_98.csv", index_col=0, parse_dates=True)
    # Seriesとして読み込まれない場合があるので整形
    if isinstance(ts_rmt, pd.DataFrame):
        ts_rmt = ts_rmt.iloc[:, 0]
    ts_rmt.name = "RMT_Raw"
    print(f"✅ RMTデータ読み込み : {ts_rmt.index.min().date()} ~ {ts_rmt.index.max().date()}")
except:
    print("❌ feature_rmt_eigen_98.csv がありません")
    exit()

# ---------------------------------------------------------
# 2. データ結合と位置合わせ (Alignment Fix)
# ---------------------------------------------------------
# 【修正点】
# 株価データ(market_index)の日付をベース（マスター）にします。
# ここにRMTを代入することで、日付が自動的にマッチングされます。
# RMTが存在しない最初の期間(窓枠分)は自動的に NaN になります。

df_dataset = pd.DataFrame(index=market_index.index)
df_dataset['Market_Price'] = market_index
df_dataset['RMT_Raw'] = ts_rmt # <--- Pandasが日付インデックスを見て自動結合します

# 結合後のチェック
print(f"📊 結合後データ形状: {df_dataset.shape}")
print(f"   (先頭のNaN数: {df_dataset['RMT_Raw'].isna().sum()} 行 -> これが窓枠分です)")

# ---------------------------------------------------------
# 3. 特徴量エンジニアリング (平滑化あり vs なし)
# ---------------------------------------------------------
# === パターンA: 平滑化なし (No Smoothing) ===
# 単純な差分
df_dataset['Vel_NoSmooth'] = df_dataset['RMT_Raw'].diff()
df_dataset['Accel_NoSmooth'] = df_dataset['Vel_NoSmooth'].diff()

# === パターンB: 平滑化あり (With Smoothing) ===
# 5日移動平均をかけてから微分
smooth_window = 5
rmt_smooth = df_dataset['RMT_Raw'].rolling(window=smooth_window).mean()

df_dataset['Vel_Smooth'] = rmt_smooth.diff()
df_dataset['Accel_Smooth'] = df_dataset['Vel_Smooth'].diff()

# ---------------------------------------------------------
# 4. ターゲット生成
# ---------------------------------------------------------
LOOKAHEAD = 10
THRESHOLD = -0.07

future_min = df_dataset['Market_Price'].rolling(LOOKAHEAD).min().shift(-LOOKAHEAD)

# ドローダウン計算
drawdown = (future_min - df_dataset['Market_Price']) / df_dataset['Market_Price']
df_dataset['Target'] = (drawdown <= THRESHOLD).astype(int)

# NaNを含む行（先頭の窓枠分 + 末尾のターゲット計算分）を削除して、クリーンな学習データにする
df_final = df_dataset.dropna()

print(f"📉 最終学習用データ: {df_final.shape}")
print(f"   期間: {df_final.index.min().date()} ~ {df_final.index.max().date()}")

# ---------------------------------------------------------
# 5. 比較結果の可視化
# ---------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# 1. 元データ
axes[0].plot(df_final.index, df_final['RMT_Raw'], color='black', label='Raw Eigenvalue', alpha=0.8)
axes[0].plot(df_final.index, df_final['RMT_Raw'].rolling(5).mean(), color='cyan', label='Smoothed (MA5)', linewidth=2, alpha=0.6)
axes[0].set_title('1. Base Signal: Raw vs Smoothed')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# 2. 速度 (Velocity)
axes[1].plot(df_final.index, df_final['Vel_NoSmooth'], color='gray', label='No Smooth (Noise)', alpha=0.5)
axes[1].plot(df_final.index, df_final['Vel_Smooth'], color='blue', label='Smoothed (Trend)', linewidth=1.5)
axes[1].set_title('2. Velocity')
axes[1].legend(loc='upper left')
axes[1].grid(True, alpha=0.3)

# 3. 加速度 (Acceleration)
axes[2].plot(df_final.index, df_final['Accel_NoSmooth'], color='lightgray', label='No Smooth (Pure Noise)', alpha=0.8)
axes[2].plot(df_final.index, df_final['Accel_Smooth'], color='red', label='Smoothed (Force Signal)', linewidth=2.0)
axes[2].set_title('3. Acceleration (The Crucial Difference)')
axes[2].legend(loc='upper left')
axes[2].set_ylim(-10.0, 10.0) # ノイズが大きすぎるためY軸を制限
axes[2].grid(True, alpha=0.3)

# 暴落期間帯
y_true_dates = df_final[df_final['Target'] == 1].index
for date in y_true_dates:
    for ax in axes:
        ax.axvline(x=date, color='red', alpha=0.05)

plt.tight_layout()
plt.show()

# 定量比較
std_no = df_final['Accel_NoSmooth'].std()
std_yes = df_final['Accel_Smooth'].std()
print("-" * 50)
print(f"🔍 ノイズ倍率確認: 平滑化しないと {std_no/std_yes:.1f} 倍 のノイズが発生")
print("-" * 50)

df_final.to_csv("dataset_comparison_rmt.csv")
print("✅ 修正完了。'dataset_comparison_rmt.csv' を保存しました。")
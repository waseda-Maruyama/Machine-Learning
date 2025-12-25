import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# 1. データ読み込み
# ---------------------------------------------------------
input_file = "dataset_ml_multi.csv" if os.path.exists("dataset_ml_multi.csv") else "dataset_ml.csv"

if not os.path.exists(input_file):
    print("❌ データファイルが見つかりません。")
    # デモデータ生成
    dates = pd.date_range("2020-01-01", "2025-01-01")
    # シグナル(ゆっくりした波) + ノイズ
    signal = np.sin(np.linspace(0, 20, len(dates))) * 2
    noise = np.random.normal(0, 0.5, len(dates))
    raw_data = pd.Series(signal + noise, index=dates)
    print("⚠️ デモデータで実行します")
else:
    print(f"📂 {input_file} を読み込んでいます...")
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    # RMTの生データ (140日窓があればそれを優先)
    target_col = 'RMT_Raw_140' if 'RMT_Raw_140' in df.columns else 'RMT_Raw'
    raw_data = df[target_col]

# ---------------------------------------------------------
# 2. 計算: 単純 vs スムージング
# ---------------------------------------------------------

# --- A. 単純 (Simple) ---
# そのまま微分
vel_simple = raw_data.diff()
# さらに微分 (加速度)
acc_simple = vel_simple.diff()

# --- B. スムージング (Smoothed) ---
# あなたのモデルの手法 (5日移動平均 -> 微分)
smooth_data = raw_data.rolling(window=5).mean()
vel_smooth = smooth_data.diff()
acc_smooth = vel_smooth.diff()

# ---------------------------------------------------------
# 3. 可視化 (重ねて表示)
# ---------------------------------------------------------
# 直近1.5年分くらいにズーム (全体だと潰れて見えないため)
subset_days = 400
dates_sub = raw_data.index[-subset_days:]

# スライス
v_simp = vel_simple.iloc[-subset_days:]
v_smth = vel_smooth.iloc[-subset_days:]
a_simp = acc_simple.iloc[-subset_days:]
a_smth = acc_smooth.iloc[-subset_days:]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# --- 上段: 速度 (Velocity) ---
# ノイズ (単純微分) を薄いグレーで背景に描画
ax1.plot(dates_sub, v_simp, color='grey', alpha=0.5, linewidth=1, label='Simple Diff (Noise)')
# シグナル (スムージング) を赤色で強調
ax1.plot(dates_sub, v_smth, color='red', alpha=0.9, linewidth=2, label='Smoothed Diff (Signal)')

ax1.set_title(f"1. Velocity Comparison (1st Derivative)\nNoise Ratio: {v_simp.std()/v_smth.std():.1f}x", fontweight='bold')
ax1.set_ylabel('RMT Velocity')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# --- 下段: 加速度 (Acceleration) ---
# 加速度はノイズがさらに激しいので、範囲を少し制限して見やすくする
ax2.plot(dates_sub, a_simp, color='grey', alpha=0.4, linewidth=1, label='Simple Diff (Noise Explosion)')
ax2.plot(dates_sub, a_smth, color='blue', alpha=0.9, linewidth=2, label='Smoothed Diff (Signal)')

ax2.set_title(f"2. Acceleration Comparison (2nd Derivative)\nNoise Ratio: {a_simp.std()/a_smth.std():.1f}x", fontweight='bold')
ax2.set_ylabel('RMT Acceleration')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
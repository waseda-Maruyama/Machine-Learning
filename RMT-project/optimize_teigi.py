import pandas as pd
import numpy as np

# 1. データ読み込み
try:
    df_prices = pd.read_csv("stock_prices_topix100_simple.csv", index_col=0, parse_dates=True)
except FileNotFoundError:
    print("❌ ファイルが見つかりません。")
    exit()

market_index = df_prices.mean(axis=1)

# 2. 定義を変えて実験
# LOOKAHEAD(日数) と THRESHOLD(深さ) の組み合わせ
settings = [
    (10, -0.05), # 今の設定 (甘口)
    (10, -0.07), # 中辛
    (10, -0.10), # 激辛 (真の暴落)
    (5,  -0.05), # 短期急落 (スピード重視)
]

# 検証したいイベント期間
events = {
    "2016 Trump/China": ("2016-01-01", "2016-12-31"),
    "2018 VIX Shock":   ("2018-01-01", "2018-03-31"),
    "2020 Covid-19":    ("2020-01-01", "2020-04-30"),
    "2024 Ueda Shock":  ("2024-07-01", "2024-09-30"),
    "2025 Tariff":      ("2025-01-01", "2025-08-30") # データにあれば
}

print("📊 暴落定義の厳格化シミュレーション\n")

for lookahead, threshold in settings:
    print(f"--- 設定: {lookahead}日後に {threshold:.0%} の下落 ---")
    
    # ターゲット計算
    future_min = market_index.rolling(lookahead).min().shift(-lookahead)
    target = ((future_min - market_index) / market_index <= threshold).astype(int)
    
    total_count = target.sum()
    print(f"   全期間の発生数: {total_count} 日 (発生率: {total_count/len(target):.2%})")
    
    # イベントごとの内訳
    print("   [イベント別発生数]")
    active_events = []
    for name, (start, end) in events.items():
        # 期間内のターゲット数
        subset = target.loc[start:end]
        count = subset.sum()
        if count > 0:
            active_events.append(f"{name}({count})")
        else:
            active_events.append(f"{name}(0)")
            
    print(f"   👉 {', '.join(active_events)}\n")
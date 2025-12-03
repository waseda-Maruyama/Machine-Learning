import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. データ準備
# ---------------------------------------------------------
print("📊 データを準備中...")
try:
    df_prices = pd.read_csv("stock_prices_topix100_simple.csv", index_col=0, parse_dates=True)
except FileNotFoundError:
    print("❌ ファイルが見つかりません。")
    exit()

# 市場平均とボラティリティ計算
market_index = df_prices.mean(axis=1)
returns = market_index.pct_change()
# ボラティリティ基準は一般的な「20日(約1ヶ月)の標準偏差」を使います
vol_20 = returns.rolling(20).std()

# 検証したいイベント期間
events = {
    "2016 Trump/China": ("2016-01-01", "2016-12-31"),
    "2018 VIX Shock":   ("2018-01-01", "2018-03-31"),
    "2020 Covid-19":    ("2020-01-01", "2020-04-30"),
    "2024 Ueda Shock":  ("2024-07-01", "2024-09-30"),
    "2025 Tariff":      ("2025-01-01", "2025-08-30")
}

# ---------------------------------------------------------
# 2. シミュレーション実行
# ---------------------------------------------------------
# 試行するパラメータの組み合わせ
# 期間 (Days): 短期決戦(3日) 〜 スイング(10日)
lookaheads = [3, 5, 10]
# 倍率 (Sigma): 1.5倍(注意報) 〜 3.0倍(警報)
multipliers = [1.5, 2.0, 2.5, 3.0]

print(f"\n🔎 動的ターゲット(Sigma-Scaling) 発生数シミュレーション")
print(f"   データ総数: {len(market_index)} 日\n")

print(f"| 期間(日) | 倍率(σ) | 発生数 (率) | イベント内訳 (主要ショックでの検知日数) |")
print(f"| :---: | :---: | :--- | :--- |")

for days in lookaheads:
    # 未来N日間の最小リターンを計算
    # rolling(N).min() は過去を見るので、shiftで未来に合わせる
    # (t+1日 〜 t+days日の間の最安値を見る)
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=days)
    future_min_return = returns.rolling(window=indexer).min().shift(-1)
    
    for m in multipliers:
        # 動的閾値: -m * sigma
        threshold = -m * vol_20
        
        # 判定 (NaNは除外)
        target = (future_min_return < threshold).astype(int)
        # ボラティリティ計算不可の先頭などを除外してカウント
        valid_target = target[vol_20.notna()]
        
        count = valid_target.sum()
        rate = count / len(valid_target)
        
        # イベント内訳文字列作成
        event_str = []
        for name, (start, end) in events.items():
            ev_count = valid_target.loc[start:end].sum()
            if ev_count > 0:
                # 短縮名で表示
                short_name = name.split()[1] 
                event_str.append(f"{short_name}:{ev_count}")
        
        event_msg = ", ".join(event_str) if event_str else "なし"
        
        print(f"| {days:2d}日後 | -{m:.1f}σ | {count:4d} ({rate:.2%}) | {event_msg} |")

    print("|-------|-------|-------------|-----------------------------------|")
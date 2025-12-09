import pandas as pd
import numpy as np
import itertools
from config import scenarios

# 1. データ読み込み
try:
    df_prices = pd.read_csv("stock_prices_topix100_simple.csv", index_col=0, parse_dates=True)
except FileNotFoundError:
    print("❌ ファイルが見つかりません。")
    exit()

market_index = df_prices.mean(axis=1)

# ---------------------------------------------------------
# 2. 新しい暴落定義: 「複合条件（AND条件）」
# 短期条件
short_days_list = [3, 5]          # 3日後、5日後
short_drop_list = [-0.02, -0.03,-0.04]  # -2%, -3%

# 長期条件 (継続下落)
long_days_list = [10]         # 10日後、20日後
long_drop_list = [-0.06,-0.07,-0.08]   # -5%, -8%

# 組み合わせを生成 (itertools.product)
# これで 2x2x2x2 = 16通りの組み合わせが自動生成されます
param_grid = list(itertools.product(short_days_list, short_drop_list, long_days_list, long_drop_list))

print(f"🧪 全 {len(param_grid)} 通りの定義でシミュレーションを開始します...\n")

results = []

for s_days, s_drop, l_days, l_drop in param_grid:
    
    # --- A. ターゲット計算 (継続下落) ---
    ret_short = market_index.shift(-s_days) / market_index - 1.0
    ret_long = market_index.shift(-l_days) / market_index - 1.0
    
    # 複合条件
    raw_target = (ret_short <= s_drop) & (ret_long <= l_drop)
    raw_target = raw_target.astype(int)
    
    # --- B. フィルタリング (Onset: 最初の5日だけ採用) ---
    if raw_target.sum() == 0:
        # 発生ゼロなら計算する意味なし
        results.append({
            "Params": f"{s_days}d{s_drop:.0%}_{l_days}d{l_drop:.0%}",
            "Count": 0,
            "Events": "None"
        })
        continue

    event_id = (raw_target.diff() != 0).cumsum()
    days_since = raw_target.groupby(event_id).cumcount()
    
    # 最初の5日間だけ残す
    final_target = raw_target.copy()
    mask_late = (raw_target == 1) & (days_since >= 5)
    final_target[mask_late] = 0
    
    total_count = final_target.sum()
    
    # --- C. 有名イベントを拾えているかチェック ---
    caught_events = []
    for name, (start, end) in scenarios.items():
        if final_target.loc[start:end].sum() > 0:
            caught_events.append(name)
            
    # 結果を辞書にまとめる
    results.append({
        "Short": f"{s_days}d {s_drop:.0%}",
        "Long": f"{l_days}d {l_drop:.0%}",
        "Count": total_count, # 全期間での発生日数
        "Rate": f"{total_count / len(market_index):.2%}", # 発生率
        "Caught_Events": ", ".join(caught_events) # どの暴落を検知したか
    })

# ---------------------------------------------------------
# 3. 結果をDataFrameにして比較
# ---------------------------------------------------------
df_results = pd.DataFrame(results)

# 発生日数が多すぎる(ノイズ)順ではなく、程よい順に見たいのでソート
# ここでは「発生数」でソートしてみます
df_results = df_results.sort_values("Count", ascending=False)

print("📊 パラメータ比較結果 (上位10件):")
# 見やすく表示
pd.set_option('display.max_colwidth', None)
print(df_results.head(10))

# CSVにも保存してじっくり選べるようにする
df_results.to_csv("target_definition_comparison.csv", index=False)
print("\n✅ 'target_definition_comparison.csv' に全結果を保存しました。")
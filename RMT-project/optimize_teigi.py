import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
from config import scenarios # 設定ファイル

# =========================================================
# 1. データ読み込み & 市場インデックス作成
# =========================================================
price_file = "stock_prices.csv"
cap_file = "market_caps.csv"

print(f"📊 データを読み込んでいます...")

if not os.path.exists(price_file):
    print("❌ stock_prices.csv が見つかりません。get_data.py を実行してください。")
    exit()

df_prices = pd.read_csv(price_file, index_col=0, parse_dates=True)

# 時価総額データがあればそれを使う（推奨）
if os.path.exists(cap_file):
    print("⚖️ market_caps.csv を検知 -> 時価総額加重平均(TOPIX型)で検証します")
    df_caps = pd.read_csv(cap_file, index_col=0, parse_dates=True)
    # 株価データと日付を合わせる
    df_caps = df_caps.reindex(df_prices.index).ffill()
    market_index = df_caps.sum(axis=1)
else:
    print("⚠️ 時価総額データなし -> 単純平均で検証します")
    market_index = df_prices.mean(axis=1)

# 正規化 (開始日=1.0)
market_index = market_index / market_index.iloc[0]

# =========================================================
# 2. 実験パラメータの設定
# =========================================================
# 短期条件
short_days_list = [3, 5]            # 3日後、5日後
short_drop_list = [-0.02, -0.03, -0.04, -0.05]  # -2%、-3%、-4%、-5%

# 長期条件 (継続下落)
long_days_list = [7,10]               # 10日後
long_drop_list = [-0.05, -0.06, -0.07, -0.08]  # -5%、-6%、-7%、-8%

# 組み合わせ生成
param_grid = list(itertools.product(short_days_list, short_drop_list, long_days_list, long_drop_list))

print(f"🧪 全 {len(param_grid)} 通りの定義でシミュレーションを開始します...\n")

results = []

for s_days, s_drop, l_days, l_drop in param_grid:
    
    # --- A. 未来のリターン計算 ---
    # Shift(-N) で未来の価格を今の行に持ってくる
    ret_short = market_index.shift(-s_days) / market_index - 1.0
    ret_long = market_index.shift(-l_days) / market_index - 1.0
    
    # --- B. 複合条件判定 (AND) ---
    raw_target = (ret_short <= s_drop) & (ret_long <= l_drop)
    raw_target = raw_target.astype(int)
    
    # --- C. フィルタリング (Onset: 最初の5日だけ採用) ---
    if raw_target.sum() == 0:
        results.append({
            "Short": f"{s_days}d {s_drop:.0%}",
            "Long": f"{l_days}d {l_drop:.0%}",
            "Count": 0,
            "Rate": "0.00%",
            "Caught_Events": "None"
        })
        continue

    # 暴落開始イベントごとにIDを振る
    event_id = (raw_target.diff() != 0).cumsum()
    days_since = raw_target.groupby(event_id).cumcount()
    
    # 最初の5日間だけ残す
    final_target = raw_target.copy()
    mask_late = (raw_target == 1) & (days_since >= 5)
    final_target[mask_late] = 0
    
    total_count = final_target.sum()
    
    # --- D. 有名イベントを拾えているかチェック ---
    caught_events = []
    for name, (start, end) in scenarios.items():
        # 日付文字列をTimestampに変換してスライス
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        
        # 期間内のフラグ数を確認
        # グラフ範囲外などでデータがない場合はスキップ
        try:
            if final_target.loc[start_ts:end_ts].sum() > 0:
                caught_events.append(name)
        except KeyError:
            continue
            
    # 結果保存
    results.append({
        "Short": f"{s_days}d {s_drop:.0%}",
        "Long": f"{l_days}d {l_drop:.0%}",
        "Count": total_count,
        "Rate": f"{total_count / len(market_index):.2%}",
        "Caught_Events": ", ".join(caught_events)
    })

# =========================================================
# 3. 結果の表示と保存
# =========================================================
df_results = pd.DataFrame(results)

# 発生数が多い順にソート（多すぎず少なすぎずを選ぶため）
df_results = df_results.sort_values("Count", ascending=False)

print("📊 パラメータ比較結果 (上位10件):")
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 20)
print(df_results.head(10))

# CSV保存
output_csv = "target_definition_comparison.csv"
df_results.to_csv(output_csv, index=False)
print(f"\n✅ 結果を保存しました: {output_csv}")


target_s_days = 3
target_s_drop = -0.02
target_l_days = 10
target_l_drop = -0.08

print(f"\n📈 可視化チェック: Short({target_s_days}d, {target_s_drop:.0%}) & Long({target_l_days}d, {target_l_drop:.0%})")

# 再計算 (ループ内と同じロジック)
ret_s = market_index.shift(-target_s_days) / market_index - 1.0
ret_l = market_index.shift(-target_l_days) / market_index - 1.0
raw = (ret_s <= target_s_drop) & (ret_l <= target_l_drop)
raw = raw.astype(int)

event_id = (raw.diff() != 0).cumsum()
days_since = raw.groupby(event_id).cumcount()
final_target = raw.copy()
final_target[(raw == 1) & (days_since >= 5)] = 0

# プロット
plt.figure(figsize=(15, 6))
plt.plot(market_index.index, market_index, label="TOPIX100 Index", color="gray", alpha=0.6)

# 暴落発生箇所を赤点でプロット
crash_dates = final_target[final_target == 1].index
plt.scatter(crash_dates, market_index.loc[crash_dates], color="red", s=10, label="Crash Onset", zorder=5)

# 有名イベントの期間をシャドウで表示
colors = ['blue', 'green', 'orange', 'purple']
for i, (name, (start, end)) in enumerate(scenarios.items()):
    plt.axvspan(pd.to_datetime(start), pd.to_datetime(end), color=colors[i%4], alpha=0.1, label=name)

plt.title(f"Crash Definition: Short{target_s_drop:.0%} & Long{target_l_drop:.0%} (Total: {final_target.sum()} days)")
plt.legend()
plt.grid(True, alpha=0.3)

# 保存
plt.savefig("crash_definition_check.png")
print("🖼️ チャートを 'crash_definition_check.png' に保存しました。")
# plt.show() # 環境によってはコメントアウトを外す
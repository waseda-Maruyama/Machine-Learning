import pandas as pd
import numpy as np
import itertools
import os

# =========================================================
# ⚙️ 設定エリア
# =========================================================
PRICE_FILE = "stock_prices.csv"
CAP_FILE = "market_caps.csv"
RESULT_CSV = "grid_search_results_weighted.csv"  # ファイル名も少し変更

# 重みパラメータ (データセット作成コードと合わせる)
DECAY_RATE = 0.5     # 減衰スピード
BOOST_FACTOR = 10.0  # 初動の基本ブースト値

# 検証用シナリオ（実際に起きたイベント）
scenarios = {
    "2020 Covid": ("2020-02-01", "2020-04-01"),
    "2024 Ueda":  ("2024-07-20", "2024-08-15"),
    "2025 Tariff": ("2025-01-01", "2025-02-28")
}

# =========================================================
# 1. データ準備
# =========================================================
print(f"📊 データを読み込んでいます...")

if not os.path.exists(PRICE_FILE):
    # ダミーデータ（テスト用）
    print("⚠️ stock_prices.csv がないためダミーデータで動作確認します。")
    dates = pd.date_range("2013-01-01", "2025-12-31", freq="B")
    returns = np.random.normal(0.0002, 0.01, size=len(dates))
    shock_idx = np.random.choice(len(dates), 10)
    returns[shock_idx] = -0.05 
    data = np.cumprod(1 + returns)
    df_prices = pd.DataFrame(data, index=dates, columns=["Dummy"])
    market_index = df_prices["Dummy"]
else:
    df_prices = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True)
    
    if os.path.exists(CAP_FILE):
        print("   -> 時価総額データを使用 (TOPIX型)")
        df_caps = pd.read_csv(CAP_FILE, index_col=0, parse_dates=True)
        df_caps = df_caps.reindex(df_prices.index).ffill()
        common_cols = df_prices.columns.intersection(df_caps.columns)
        market_index = (df_prices[common_cols] * df_caps[common_cols]).sum(axis=1) / df_caps[common_cols].sum(axis=1)
    else:
        print("   -> 単純平均データを使用")
        market_index = df_prices.mean(axis=1)

# 正規化
market_index = market_index / market_index.iloc[0]

# =========================================================
# 2. グリッドサーチ設定
# =========================================================
# 短期条件 (初速)
short_days_list = [3, 5]
short_drop_list = [-0.02, -0.03, -0.04]

# 長期条件 (深度)
long_days_list = [7,10]
long_drop_list = [-0.05, -0.06, -0.07, -0.08]

param_grid = list(itertools.product(short_days_list, short_drop_list, long_days_list, long_drop_list))

print(f"🧪 全 {len(param_grid)} 通りのパラメータを検証します...\n")

# =========================================================
# 3. 実行ループ
# =========================================================
results = []

for s_days, s_drop, l_days, l_drop in param_grid:
    
    # --- A. ロジック計算 ---
    ret_short = market_index.shift(-s_days) / market_index - 1.0
    ret_long = market_index.shift(-l_days) / market_index - 1.0
    
    # 複合条件 (AND)
    is_crash = (ret_short <= s_drop) & (ret_long <= l_drop)
    is_crash = is_crash.fillna(False).astype(int)
    
    # 全く検知されなければスキップ
    if is_crash.sum() == 0:
        continue

    # --- B. 連続イベントのグループ化 ---
    # 暴落が連続している区間を1つのイベントとみなす（Onset判定用）
    event_group = (is_crash != is_crash.shift()).cumsum()
    
    # 各暴落日について「発生から何日目か」を計算 (0, 1, 2, ...)
    days_since_start = is_crash.groupby(event_group).cumcount()
    
    # 【変更点】5日以降も除外せず、全ての is_crash=1 を対象とする
    # ただし、統計情報として「イベント数（塊の数）」と「総日数」は分けて記録する
    
    # イベント数（塊の数）
    # is_crash=1 のグループIDのユニーク数をカウント（偶数・奇数の並びによる簡易計算）
    # 単純に「diff() != 0 で is_crash==1 になった回数」を数える
    num_events = ((is_crash == 1) & (is_crash.shift(1) != 1)).sum()
    
    # 総検知日数
    total_days = is_crash.sum()

    # --- C. 重み計算 (Decay x Severity) ---
    # 1. 時間減衰 (Time Decay)
    decay_comp = BOOST_FACTOR * np.exp(-DECAY_RATE * days_since_start)
    
    # 2. 被害規模 (Severity Ratio)
    # 現在設定している閾値(l_drop)に対して、実際のリターン(ret_long)が何倍酷いか
    severity_ratio = ret_long.abs() / abs(l_drop)
    
    # 3. 結合
    # is_crash=0 の場所は計算不要なのでマスクする
    raw_weights = decay_comp * severity_ratio
    final_weights = raw_weights[is_crash == 1] # 暴落日のみ抽出
    
    avg_weight = final_weights.mean() if len(final_weights) > 0 else 0
    max_weight = final_weights.max() if len(final_weights) > 0 else 0
    sum_weight = final_weights.sum() # 総学習エネルギー的な指標
    
    # --- D. イベント捕捉確認 ---
    caught_list = []
    for name, (start, end) in scenarios.items():
        try:
            # 期間内に1日でもフラグが立っていればOK
            sub = is_crash.loc[start:end]
            if sub.sum() > 0:
                caught_list.append(name)
        except:
            pass
            
    # 結果保存
    results.append({
        "Short_Cond": f"{s_days}d {s_drop:.0%}",
        "Long_Cond": f"{l_days}d {l_drop:.0%}",
        "Events": num_events,        # 暴落の「回数」
        "Days": total_days,          # 暴落の「総日数」
        "Avg_Weight": round(avg_weight, 2),
        "Max_Weight": round(max_weight, 2),
        "Sum_Weight": round(sum_weight, 2), # 今回はこれも参考になるかも
        "Caught": ", ".join(caught_list),
        "_sort_key": max_weight
    })

# =========================================================
# 4. 結果出力
# =========================================================
df_res = pd.DataFrame(results)

if df_res.empty:
    print("❌ 条件に合うイベントが見つかりませんでした。閾値を緩めてください。")
else:
    # Avg_Weightが高い順にソート
    df_res = df_res.sort_values(by="_sort_key", ascending=False).drop(columns=["_sort_key"])

    print(f"📊 分析結果 (Avg_Weight順 Top 15):")
    # 見やすいようにカラム順序を整理
    cols = ["Short_Cond", "Long_Cond", "Events", "Days", "Avg_Weight", "Max_Weight", "Caught"]
    df_res = df_res[cols]
    
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)
    print(df_res.head(15))

    # CSV保存
    df_res.to_csv(RESULT_CSV, index=False)
    print(f"\n💾 全結果を保存しました: {RESULT_CSV}")
    print("👉 Avg_Weightが高く、Events/Daysが過剰でない（ノイズを拾いすぎていない）設定を選んでください。")
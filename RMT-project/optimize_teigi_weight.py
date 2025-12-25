import pandas as pd
import numpy as np
import itertools
import os

# =========================================================
# ⚙️ 設定エリア
# =========================================================
PRICE_FILE = "stock_prices.csv"
CAP_FILE = "market_caps.csv"
RESULT_CSV = "grid_search_results.csv"  # 分析結果の保存先

# 検証用シナリオ（実際に起きたイベント）
scenarios = {
    "2020 Covid": ("2020-02-01", "2020-04-01"),
    "2024 Ueda":  ("2024-07-20", "2024-08-15"),
    "2025 Tariff": ("2025-01-01", "2025-02-28") # 未来の日付が含まれていてもエラーにならないように処理します
}

# =========================================================
# 1. データ準備
# =========================================================
print(f"📊 データを読み込んでいます...")

if not os.path.exists(PRICE_FILE):
    # ダミーデータ（テスト用）
    print("⚠️ stock_prices.csv がないためダミーデータで動作確認します。")
    dates = pd.date_range("2013-01-01", "2025-12-31", freq="B")
    # ランダムウォークに時々ショックを与える
    returns = np.random.normal(0.0002, 0.01, size=len(dates))
    # 擬似的なショック挿入
    shock_idx = np.random.choice(len(dates), 10)
    returns[shock_idx] = -0.05 
    data = np.cumprod(1 + returns)
    df_prices = pd.DataFrame(data, index=dates, columns=["Dummy"])
    market_index = df_prices["Dummy"]
else:
    df_prices = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True)
    
    # 時価総額加重平均 (TOPIX型) または 単純平均
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
long_days_list = [10, 20]
long_drop_list = [-0.05, -0.06, -0.07, -0.08]

param_grid = list(itertools.product(short_days_list, short_drop_list, long_days_list, long_drop_list))

print(f"🧪 全 {len(param_grid)} 通りのパラメータを検証します...\n")

# 重み計算関数 (案1: 未来の最大ドローダウン)
def get_drawdown_weight(idx_date, series, lookahead=20):
    try:
        current_val = series.loc[idx_date]
        future_idx = series.index.get_loc(idx_date)
        # lookahead期間内の最安値を探す
        future_vals = series.iloc[future_idx : future_idx + lookahead]
        min_val = future_vals.min()
        dd = (min_val - current_val) / current_val
        return abs(dd) * 100
    except:
        return 0.0

# =========================================================
# 3. 実行ループ
# =========================================================
results = []

for s_days, s_drop, l_days, l_drop in param_grid:
    
    # --- ロジック計算 ---
    ret_short = market_index.shift(-s_days) / market_index - 1.0
    ret_long = market_index.shift(-l_days) / market_index - 1.0
    
    # 複合条件
    is_crash = (ret_short <= s_drop) & (ret_long <= l_drop)
    is_crash = is_crash.fillna(False).astype(int)
    
    if is_crash.sum() == 0:
        continue

    # Onsetフィルタリング (連続する暴落期間の最初の5日だけを抽出)
    # 1. 変化点を見つけてグループ化
    event_group = (is_crash != is_crash.shift()).cumsum()
    # 2. グループ内連番を作成
    days_since_start = is_crash.groupby(event_group).cumcount()
    
    # 3. 最初の5日(0~4)かつフラグが1の場所だけ残す
    onset_mask = (is_crash == 1) & (days_since_start < 5)
    
    # 対象となる日付リスト
    target_dates = market_index.index[onset_mask]
    total_count = len(target_dates)
    
    if total_count < 10: # あまりに少なすぎる定義はスキップ
        continue

    # --- 重みシミュレーション ---
    # 検知した各イベントが、その後どれくらい深堀りしたか計算
    weights = [get_drawdown_weight(d, market_index) for d in target_dates]
    
    avg_weight = np.mean(weights) if weights else 0
    max_weight = np.max(weights) if weights else 0
    
    # --- イベント捕捉確認 ---
    caught_list = []
    for name, (start, end) in scenarios.items():
        # 期間内に1つでもフラグが立っていればOK
        try:
            sub = onset_mask.loc[start:end]
            if sub.sum() > 0:
                caught_list.append(name)
        except:
            pass # 日付範囲外などは無視
            
    # 結果保存
    results.append({
        "Short_Cond": f"{s_days}d {s_drop:.0%}",
        "Long_Cond": f"{l_days}d {l_drop:.0%}",
        "Count": total_count,
        "Avg_Weight": round(avg_weight, 2),  # 平均の深刻度
        "Max_Weight": round(max_weight, 2),  # コロナ級なら30くらいになるはず
        "Caught": ", ".join(caught_list),
        # ソート用に数値も持っておく
        "_sort_key": avg_weight
    })

# =========================================================
# 4. 結果出力
# =========================================================
df_res = pd.DataFrame(results)

if df_res.empty:
    print("❌ 条件に合うイベントが見つかりませんでした。閾値を緩めてください。")
else:
    # 平均重み（深刻度）が高い順にソートして表示
    # ※ 数が多すぎず、かつ重みが大きいものが「良質な定義」
    df_res = df_res.sort_values(by="_sort_key", ascending=False).drop(columns=["_sort_key"])

    print(f"📊 分析結果 (Avg_Weight順 Top 15):")
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)
    print(df_res.head(15))

    # CSV保存
    df_res.to_csv(RESULT_CSV, index=False)
    print(f"\n💾 全結果を保存しました: {RESULT_CSV}")
    print("👉 Avg_Weightが高く、かつCountが極端に少なくない（50〜100程度）設定を選んでください。")
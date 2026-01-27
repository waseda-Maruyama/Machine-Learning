import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os

# =========================================================
# ⚙️ 設定エリア
# =========================================================
PRICE_FILE = "stock_close.csv"
CAP_FILE = "market_caps.csv"
RESULT_CSV = "grid_search_comparison.csv"

# 手動で最良条件を固定する場合はタプルを設定する: (short_days, short_drop, long_days, long_drop)
# 例: {"Index 1 (Eco)": (5, -0.02, 10, -0.08)}
MANUAL_BEST = {
    "Index 1 (Eco)": (3,-0.01,10,-0.04),
    "Index 2 (Phy)": (3,-0.03,10,-0.08)
}

# 重みパラメータ
DECAY_RATE = 0.5     # 減衰スピード
BOOST_FACTOR = 5.0  # 初動の基本ブースト値

# 検証用シナリオ (config.pyから読み込む)
try:
    from config import scenarios
except ImportError:
    print("⚠️ config.pyが見つかりません。ダミーシナリオを使用します。")
    scenarios = {
        "2020 Covid": ("2020-02-01", "2020-04-01"),
        "2024 Ueda":  ("2024-07-20", "2024-08-15"),
        "2025 Tariff": ("2025-01-01", "2025-03-28")
    }

# シナリオ開始日における的中有無を保存する（最初に存在する営業日で判定）

# =========================================================
# 1. データ準備 (Index 1 & Index 2 生成)
# =========================================================
print(f"📊 データを読み込んでいます...")

if not os.path.exists(PRICE_FILE):
    print("⚠️ ファイルがないため、ダミーデータで動作確認します。")
    np.random.seed(42)
    dates = pd.date_range("2016-01-01", "2026-01-01", freq="B")
    
    # 銘柄A: 値がさ株 (High Price)
    r_A = np.random.normal(0.0002, 0.015, len(dates))
    price_A = 50000 * np.cumprod(1 + r_A)
    cap_A = price_A * 100000 # 株数一定
    
    # 銘柄B: 大型株 (Low Price, High Cap)
    r_B = np.random.normal(0.0001, 0.008, len(dates))
    price_B = 2000 * np.cumprod(1 + r_B)
    cap_B = price_B * 5000000 # 株数多い
    
    df_close = pd.DataFrame({'A': price_A, 'B': price_B}, index=dates)
    df_caps = pd.DataFrame({'A': cap_A, 'B': cap_B}, index=dates)
    
else:
    df_close = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True)
    if os.path.exists(CAP_FILE):
        df_caps = pd.read_csv(CAP_FILE, index_col=0, parse_dates=True)
        df_caps = df_caps.reindex(df_close.index).ffill()
    else:
        # 時価総額ファイルがない場合は単純平均などを代用
        df_caps = pd.DataFrame(1, index=df_close.index, columns=df_close.columns)

# 共通カラム抽出
common_cols = df_close.columns.intersection(df_caps.columns)
df_close = df_close[common_cols]
df_caps = df_caps[common_cols]

# --- インデックス作成 ---
# Index 1: 経済的価値 (時価総額加重)
raw_idx1 = df_caps.sum(axis=1)
market_index1 = raw_idx1 / raw_idx1.iloc[0] * 100

# Index 2: 物理的エネルギー (時価総額 × 生株価)
# 定義: Σ (M * P)
raw_idx2 = (df_caps **2).sum(axis=1)
market_index2 = raw_idx2 / raw_idx2.iloc[0] * 100

targets = {
    "Index 1 (Eco)": market_index1,
    "Index 2 (Phy)": market_index2
}

# =========================================================
# 2. グリッドサーチ設定
# =========================================================
# パラメータ範囲（必要に応じて変更してください）
short_days_list = [3, 5]
short_drop_list = [-0.01, -0.02,  -0.03]
long_days_list = [7,10]
long_drop_list = [ -0.02, -0.025,-0.03, -0.04, -0.06, -0.08, -0.10]

param_grid = list(itertools.product(short_days_list, short_drop_list, long_days_list, long_drop_list))
print(f"🧪 全 {len(param_grid) * 2} 通りのパラメータを検証します (Index 1 & 2)...\n")

# =========================================================
# 3. 実行ループ (Target x Grid)
# =========================================================
results = []

for target_name, series in targets.items():
    for s_days, s_drop, l_days, l_drop in param_grid:
        
        # --- A. ロジック計算 ---
        ret_short = series.shift(-s_days) / series - 1.0
        ret_long = series.shift(-l_days) / series - 1.0
        
        # 複合条件
        is_crash = (ret_short <= s_drop) & (ret_long <= l_drop)
        is_crash = is_crash.fillna(False).astype(int)
        
        if is_crash.sum() == 0:
            continue

        # --- B. イベント集計 ---
        event_group = (is_crash != is_crash.shift()).cumsum()
        days_since_start = is_crash.groupby(event_group).cumcount()
        
        # イベント数（塊の数）
        num_events = ((is_crash == 1) & (is_crash.shift(1) != 1)).sum()
        total_days = is_crash.sum()

        # --- C. 重み計算 ---
        decay_comp = BOOST_FACTOR * np.exp(-DECAY_RATE * days_since_start)
        
        # イベント全体の最大下落を基準にした深刻度スコア
        event_max_drop = ret_long.groupby(event_group).transform('min')
        event_severity_score = (event_max_drop.abs() / abs(l_drop))
        
        raw_weights = decay_comp * event_severity_score
        final_weights = raw_weights[is_crash == 1]
        
        avg_weight = final_weights.mean() if len(final_weights) > 0 else 0
        max_weight = final_weights.max() if len(final_weights) > 0 else 0

        # --- D. シナリオ捕捉 ---
        caught_list = []
        start_hits = {}
        for name, (start, end) in scenarios.items():
            try:
                sub = is_crash.loc[start:end]

                # 最初の時点からシナリオ開始日までの累積ヒット日数
                start_dt = pd.to_datetime(start)
                idx_pos = is_crash.index.searchsorted(start_dt, side="right") - 1
                if idx_pos >= 0:
                    start_hits[name] = int(is_crash.iloc[: idx_pos + 1].sum())
                else:
                    start_hits[name] = 0

                if sub.sum() > 0:
                    caught_list.append(name)
            except:
                # 範囲外などの場合は0件扱い
                start_hits[name] = 0
        
        results.append({
            "Target": target_name,
            "Short": f"{s_days}d {s_drop:.1%}",
            "Long": f"{l_days}d {l_drop:.1%}",
            "Events": num_events,
            "Days": total_days,
            "Avg_Weight": round(avg_weight, 2),
            "Max_Weight": round(max_weight, 2),
            "Caught": ", ".join(caught_list),
            "Start_Hits": "; ".join([f"{k}:{v}" for k, v in start_hits.items()]),
            # 生の値も保存（プロット選択用）
            "_s_days": s_days, "_s_drop": s_drop,
            "_l_days": l_days, "_l_drop": l_drop
        })

# =========================================================
# 4. 結果出力
# =========================================================
df_res = pd.DataFrame(results)

if df_res.empty:
    print("❌ 条件に合うイベントが見つかりませんでした。")
else:
    # ターゲットごとにAvg_Weightが高い順にソートして表示
    df_res = df_res.sort_values(by=["Target", "Short", "Long"], ascending=[True, False, False])
    
    print(f"📊 分析結果 (各Indexの上位5件):")
    cols = ["Target", "Short", "Long", "Events", "Days", "Avg_Weight", "Caught"]
    print(df_res[cols].groupby("Target").head(10)) # 各ターゲットの上位5行を表示
    
    df_res.to_csv(RESULT_CSV, index=False)
    print(f"\n💾 全結果を保存しました: {RESULT_CSV}")
# =========================================================
# 5. 比較プロット (1軸統合版: Index 1 & 2 を重ねる)
# =========================================================
print("\n📈 グラフを作成中...")

# --- A. 条件の選択 ---
def _manual_cond(target_name: str, manual_tuple):
    s_days, s_drop, l_days, l_drop = manual_tuple
    return pd.Series({
        "Target": target_name,
        "Short": f"{s_days}d {s_drop:.1%}",
        "Long": f"{l_days}d {l_drop:.1%}",
        "_s_days": s_days,
        "_s_drop": s_drop,
        "_l_days": l_days,
        "_l_drop": l_drop,
    })

# Index 1 (Eco) のベスト条件
if MANUAL_BEST["Index 1 (Eco)"]:
    best_cond1 = _manual_cond("Index 1 (Eco)", MANUAL_BEST["Index 1 (Eco)"])
elif not df_res[df_res["Target"] == "Index 1 (Eco)"].empty:
    best_cond1 = df_res[df_res["Target"] == "Index 1 (Eco)"].iloc[0]
else:
    best_cond1 = None

# Index 2 (Phy) のベスト条件
if MANUAL_BEST["Index 2 (Phy)"]:
    best_cond2 = _manual_cond("Index 2 (Phy)", MANUAL_BEST["Index 2 (Phy)"])
elif not df_res[df_res["Target"] == "Index 2 (Phy)"].empty:
    best_cond2 = df_res[df_res["Target"] == "Index 2 (Phy)"].iloc[0]
else:
    best_cond2 = None

# --- B. プロット実行 ---
fig, ax = plt.subplots(figsize=(15, 8))

# 1. Index 1 (経済指標: 黒実線)
# メインとなる市場価格の動き
ax.plot(market_index1.index, market_index1, color='black', alpha=0.7, linewidth=1.5, label='Index 1 (Eco: Price)')

# 2. Index 2 (物理指標: オレンジ破線) - 同じ軸に追加
# 両方とも100スタートで正規化されているため、乖離が直接比較できます
ax.plot(market_index2.index, market_index2, color='tab:orange', alpha=0.6, linewidth=1.2, linestyle='--', label='Index 2 (Phy: Energy)')

# 3. 暴落検知ポイントのプロット
# 視認性を高めるため、検知マークはすべて「Index 1（実際の価格線）」の上に表示します。
# 「価格はここにあるが、エネルギー側で警報が出た」という状況を可視化するためです。

if best_cond1 is not None:
    # Index 1 検知 (再計算)
    r_s1 = market_index1.shift(-int(best_cond1["_s_days"])) / market_index1 - 1.0
    r_l1 = market_index1.shift(-int(best_cond1["_l_days"])) / market_index1 - 1.0
    mask1 = (r_s1 <= best_cond1["_s_drop"]) & (r_l1 <= best_cond1["_l_drop"])
    
    crashes1 = market_index1[mask1]
    label1 = f"Index 1 Detect\n({best_cond1['Short']} & {best_cond1['Long']})"
    # 青丸: 価格自体が崩れたポイント
    ax.scatter(crashes1.index, crashes1, color='blue', s=50, label=label1, zorder=5)

if best_cond2 is not None:
    # Index 2 検知 (再計算)
    r_s2 = market_index2.shift(-int(best_cond2["_s_days"])) / market_index2 - 1.0
    r_l2 = market_index2.shift(-int(best_cond2["_l_days"])) / market_index2 - 1.0
    mask2 = (r_s2 <= best_cond2["_s_drop"]) & (r_l2 <= best_cond2["_l_drop"])
    
    # タイミングはIndex 2で判定するが、y座標はIndex 1に合わせる
    crashes2_dates = market_index2[mask2].index
    crashes2_values = market_index1.loc[crashes2_dates]
    
    label2 = f"Index 2 Detect\n({best_cond2['Short']} & {best_cond2['Long']})"
    # 赤バツ: エネルギーが崩壊したポイント（価格への予兆）
    ax.scatter(crashes2_dates, crashes2_values, color='red', marker='x', s=80, label=label2, zorder=6)

plt.title("Crash Detection Comparison: Economic Price vs Physical Energy", fontsize=16)
plt.ylabel("Normalized Index Value (Start=100)")
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

img_file = "crash_comparison_single_axis.pdf"
plt.savefig(img_file)
print(f"🖼️ 画像を保存しました: {img_file}")
plt.show()
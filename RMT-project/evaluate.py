import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================================================
# ⚙️ シミュレーション設定
# =========================================================
INPUT_FILE = "prediction_comparison.csv"
THRESHOLD_EXIT = 0.3   # この確率を超えたら「逃げる（Cash）」
THRESHOLD_ENTRY = 0.3  # この確率を下回ったら「戻る（Invest）」
COST_RATE = 0.001      # 取引コスト（片道0.1% = 10bps）
INITIAL_CAPITAL = 1.0  # 初期資産 (1.0 = 100%)

# 評価対象イベント期間
events = {
    "2020 Covid":  ("2020-01-01", "2020-12-31"), # 回復まで含めて評価するため期間を長めに
    "2024 Ueda":   ("2024-06-01", "2024-12-31"),
    "2025 Tariff": ("2025-01-01", "2025-10-31")
}

# =========================================================
# 1. データ読み込み
# =========================================================
if not os.path.exists(INPUT_FILE):
    # テスト用にダミー生成（丸山は自分のファイルを読んでね）
    dates = pd.date_range("2020-01-01", "2025-12-31", freq='B')
    df = pd.DataFrame(index=dates)
    df['Market_Price'] = 100 * (1 + np.random.randn(len(dates))*0.01).cumprod() # ランダムウォーク
    df['Return'] = df['Market_Price'].pct_change().fillna(0)
    # ダミー予測確率
    df['Prob_A'] = np.random.uniform(0, 1, len(dates)) # Tech
    df['Prob_C'] = np.random.uniform(0, 1, len(dates)) # Dual RMT (Proposed)
else:
    df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    # Return列がない場合は計算
    if 'Return' not in df.columns:
        df['Return'] = df['Market_Price'].pct_change().fillna(0)

# =========================================================
# 2. バックテスト関数 (これが心臓部よ)
# =========================================================
def run_backtest(df_subset, prob_col, threshold=0.5, cost=0.001):
    """
    時系列に沿って資産推移を計算する関数
    """
    # ポジション計算: 確率が閾値以上なら0 (Cash), 未満なら1 (Invest)
    # shift(1)を入れるのは「今日の予測で明日のポジションを決める」ため（リーク防止）
    signal = (df_subset[prob_col] < threshold).astype(int)
    position = signal.shift(1).fillna(1) # 初日はHoldと仮定
    
    # ポジション変化フラグ (売買発生日)
    # abs(diff) > 0 の日が取引日
    trade_occurred = position.diff().abs().fillna(0)
    
    # 資産推移の計算
    # 資産変動 = (株を持ってる日のリターン) - (売買コスト)
    strategy_returns = (df_subset['Return'] * position) - (trade_occurred * cost)
    
    # 累積リターン (Equity Curve)
    equity_curve = (1 + strategy_returns).cumprod()
    
    # パフォーマンス指標
    total_return = equity_curve.iloc[-1] - 1
    
    # 最大ドローダウン (MDD)
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return equity_curve, total_return, max_drawdown, position

# =========================================================
# 3. イベントごとの評価実行
# =========================================================
results_list = []

print(f"💰 資産推移シミュレーション (閾値: {THRESHOLD_EXIT:.0%}, コスト: {COST_RATE:.1%})")
print("=" * 80)

for name, (s_str, e_str) in events.items():
    s_dt, e_dt = pd.to_datetime(s_str), pd.to_datetime(e_str)
    
    # データ抽出
    subset = df.loc[s_dt:e_dt].copy()
    if len(subset) == 0: continue
    
    # 1. Buy & Hold (何もしない)
    bnh_curve = (1 + subset['Return']).cumprod()
    bnh_ret = bnh_curve.iloc[-1] - 1
    bnh_mdd = ((bnh_curve - bnh_curve.cummax()) / bnh_curve.cummax()).min()
    
    # 2. Model A (Tech Only)
    eq_a, ret_a, mdd_a, pos_a = run_backtest(subset, 'Prob_A', THRESHOLD_EXIT, COST_RATE)
    
    # 3. Model C (Dual RMT - Proposed)
    eq_c, ret_c, mdd_c, pos_c = run_backtest(subset, 'Prob_C', THRESHOLD_EXIT, COST_RATE)
    
    # 結果保存
    results_list.append({
        "Event": name,
        "Strategy": "Buy & Hold",
        "Return": bnh_ret,
        "MDD": bnh_mdd
    })
    results_list.append({
        "Event": name,
        "Strategy": "Model A (Tech)",
        "Return": ret_a,
        "MDD": mdd_a
    })
    results_list.append({
        "Event": name,
        "Strategy": "Model C (RMT)",
        "Return": ret_c,
        "MDD": mdd_c
    })

    # --- グラフ描画 (各イベントごとにpng保存してもいいわね) ---
    plt.figure(figsize=(10, 5))
    plt.plot(subset.index, bnh_curve, label='Buy & Hold', color='gray', linestyle='--')
    plt.plot(subset.index, eq_a, label='Model A (Tech)', color='orange', alpha=0.7)
    plt.plot(subset.index, eq_c, label='Model C (RMT)', color='red', linewidth=2)
    
    # キャッシュポジションの期間を塗る (Model C)
    cash_dates = subset.index[pos_c == 0]
    if len(cash_dates) > 0:
        # 簡易的に塗る（実際はfill_between推奨）
        plt.scatter(cash_dates, eq_c.loc[cash_dates], color='red', s=10, marker='x', label='Cash Position')

    plt.title(f"Equity Curve: {name}")
    plt.ylabel("Cumulative Return (Base=1.0)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# =========================================================
# 4. 結果の集計テーブル表示
# =========================================================
results_df = pd.DataFrame(results_list)

# 見やすく整形
pivot_df = results_df.pivot(index="Event", columns="Strategy", values=["Return", "MDD"])

print("\n🏆 最終パフォーマンス比較表")
print("-" * 80)
# カラムの階層を整理して表示
print(pivot_df)
print("-" * 80)

# CSVに保存 (これを論文の表にするのよ)
pivot_df.to_csv("simulation_results_final.csv")
print("💾 結果を simulation_results_final.csv に保存しました。")
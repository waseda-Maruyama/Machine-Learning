import pandas as pd
import numpy as np

# =========================================================
# 設定
# =========================================================
NAN_THRESHOLD = 2  # これ以上の欠損は削除

print("🧹 データクリーニング & 特定銘柄の救済処理を開始します...")

try:
    df_prices = pd.read_csv("stock_close.csv", index_col=0, parse_dates=True)
    df_mc = pd.read_csv("market_caps.csv", index_col=0, parse_dates=True)
    print(f"📥 読み込み完了: {len(df_prices.columns)} 銘柄")
except FileNotFoundError:
    print("❌ ファイルが見つかりません。")
    exit()

# =========================================================
# 1. インデックスの完全同期 (前提処理)
# =========================================================
# 時価総額データに存在しない日付（行）を作成し、補完可能な状態にする
df_mc = df_mc.reindex(df_prices.index)

# =========================================================
# 2. 【調査結果に基づく救済】 直近データの消失(Null Update)への処置
# =========================================================
# 調査結果:
# 以下の2銘柄において、2025年後半の決算発表以降、発行済株式数が取得できず
# 時価総額が直近（データの末尾）まで欠損していることが判明。
#
# 対象銘柄と欠損開始日:
#   1. 商船三井 (91040): 2025-08-01 (1Q決算) 〜 現在 (約110日分)
#   2. アシックス (79360): 2025-11-12 (3Q決算) 〜 現在 (約42日分)
#
# 原因: APIからの最新財務データが NaN で更新されたことによる連鎖的な空白。
# 対策: 過去の有効な値を現在まで延長(ffill)することで修復する。

print("🚑 調査済み欠損箇所の修復を実行中...")

target_repair_codes = ["91040", "79360"] # ログ出力用

for code in df_mc.columns:
    # 株価が存在する有効期間（上場期間）を取得
    valid_prices = df_prices[code].dropna()
    
    if valid_prices.empty:
        continue

    # 株価の「開始日」と「終了日」
    start_date = valid_prices.index[0]
    end_date = valid_prices.index[-1]

    # この期間の時価総額データを切り出す
    mc_subset = df_mc.loc[start_date:end_date, code]

    # -------------------------------------------------------------
    # 修復ロジック
    # -------------------------------------------------------------
    if mc_subset.isnull().any():
        # 【重要】 ffill() が今回の主役
        # 直近の欠損（商船三井・アシックス型）は、直前の正しい値を
        # 末尾までコピーすることで解決する。
        # ※念のため bfill() も付けて、出だしのわずかなズレもケアする
        mc_subset_repaired = mc_subset.ffill().bfill()
        
        # 本体に書き戻す
        df_mc.loc[start_date:end_date, code] = mc_subset_repaired

    # 株価がない期間（上場前など）は NaN でマスクする（余計な埋めすぎ防止）
    mask_no_price = df_prices[code].isnull()
    df_mc.loc[mask_no_price, code] = np.nan

# =========================================================
# 3. 修復結果の検証
# =========================================================
print("-" * 50)
print("🧐 特定銘柄の修復状況チェック:")
for code in target_repair_codes:
    if code in df_mc.columns:
        # 末尾5日間の欠損状況を確認
        last_5_days = df_mc[code].tail(5)
        null_count = last_5_days.isnull().sum()
        status = "✅ 修復成功" if null_count == 0 else "❌ まだ欠損あり"
        print(f"  {code}: 直近5日間の欠損数 = {null_count} -> {status}")
        if null_count == 0:
            print(f"     ↳ 最新値: {last_5_days.iloc[-1]:,.0f}")

# =========================================================
# 4. 最終削除判定 (救えなかった銘柄の削除)
# =========================================================
nan_counts_prices = df_prices.isnull().sum()
nan_counts_mc = df_mc.isnull().sum()

drop_tickers = nan_counts_prices[nan_counts_prices >= NAN_THRESHOLD].index.union(
               nan_counts_mc[nan_counts_mc >= NAN_THRESHOLD].index)

if len(drop_tickers) > 0:
    print("-" * 50)
    print(f"🗑️ 最終削除対象: {len(drop_tickers)} 銘柄")
    for code in drop_tickers:
        print(f"  ❌ {code} (株価欠損:{nan_counts_prices[code]}, MC欠損:{nan_counts_mc[code]})")
    
    df_prices_clean = df_prices.drop(columns=drop_tickers)
    df_mc_clean = df_mc.drop(columns=drop_tickers)
else:
    print("-" * 50)
    print("✅ 削除対象なし。全銘柄が維持されました。")
    df_prices_clean = df_prices
    df_mc_clean = df_mc

# =========================================================
# 5. 保存
# =========================================================
# 最終的な欠損埋め（念のため）
df_prices_clean = df_prices_clean.ffill()
df_mc_clean = df_mc_clean.ffill()

df_prices_clean.to_csv("stock_close.csv")
df_mc_clean.to_csv("market_caps.csv")

print(f"\n💾 保存完了: {len(df_prices_clean.columns)} 銘柄")
print("   stock_adj_close.csv / market_caps.csv を更新しました。")

import pandas as pd

# 1. 読み込み (あえて日付パースせずに生の状態も見ます)
print("📂 ファイルを読み込んでいます...")
try:
    df_prices = pd.read_csv("stock_prices1.csv", index_col=0, parse_dates=True)
    df_mc = pd.read_csv("market_caps1.csv", index_col=0, parse_dates=True)
except FileNotFoundError:
    print("❌ ファイルが見つかりません")
    exit()

target = "54010"  # 日本製鉄

# ---------------------------------------------------------
# 検査 1: そもそもデータフレームにいるか？
# ---------------------------------------------------------
if target not in df_mc.columns:
    print(f"❌ エラー: {target} が market_caps.csv に存在しません。")
    exit()

# ---------------------------------------------------------
# 検査 2: シンプルな欠損数カウント (条件なし)
# ---------------------------------------------------------
mc_series = df_mc[target]
price_series = df_prices[target] if target in df_prices.columns else pd.Series(dtype=float)

mc_null_count = mc_series.isnull().sum()
price_null_count = price_series.isnull().sum()

print(f"\n📊 {target} の基本ステータス:")
print(f"   - 全期間の日数: {len(mc_series)}")
print(f"   - 時価総額(MC) のNaN数: {mc_null_count} <--- ここが 0 ならファイルは『無欠損』です")
print(f"   - 株価(Price) のNaN数: {price_null_count}")

# ---------------------------------------------------------
# 検査 3: レポートとの矛盾の原因を探る
# ---------------------------------------------------------
# 「株価はあるのに、MCがない」日を探す (さっきのtemp.pyのロジック)
# インデックスを合わせてから比較
df_mc_aligned = df_mc.reindex(df_prices.index)
culprit_mask = df_prices[target].notnull() & df_mc_aligned[target].isnull()
culprit_days = df_prices.index[culprit_mask]

print(f"\n🕵️‍♀️ 犯人捜しロジック (Price!=NaN かつ MC==NaN) の結果:")
print(f"   - 該当日数: {len(culprit_days)} 日")

if len(culprit_days) > 0:
    print("   🚨 発見しました！以下のような日付です:")
    print(culprit_days[:5])
else:
    print("   ✅ 該当なし (つまり、株価がある日は必ずMCも入っています)")

# ---------------------------------------------------------
# 検査 4: もし「MC欠損数」が 0 なら...
# ---------------------------------------------------------
if mc_null_count == 0:
    print("\n💡 結論:")
    print("  このファイル上の日本製鉄は【完全に埋まっています】。")
    print("  レポートで「111日欠損」と出た理由は以下の可能性が高いです：")
    print("  1. 生成スクリプト内の `bfill()` が保存時に効いて、CSV化する段階で直った。")
    print("  2. レポート計算時点ではNaNだったが、その後の処理で埋まった。")
    
# ---------------------------------------------------------
# 検査 5: もし「MC欠損数」が > 0 なのに「犯人」が 0 なら...
# ---------------------------------------------------------
elif mc_null_count > 0 and len(culprit_days) == 0:
    print("\n💡 結論:")
    print("  時価総額の欠損はありますが、その日は【株価も欠損】しています。")
    print("  そのため「株価はあるのに...」という検索には引っかかりませんでした。")
    
    # 実際に両方NaNの日を表示
    both_nan = df_prices[target].isnull() & df_mc_aligned[target].isnull()
    print(f"  (株価と時価総額が共にNaNの日数: {both_nan.sum()} 日)")
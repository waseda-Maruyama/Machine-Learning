import os
import time
import pandas as pd
import numpy as np
import jquantsapi
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

# 1. 認証
load_dotenv(find_dotenv('J-Quants.env'))
cli = jquantsapi.Client(mail_address=os.getenv("JQUANTS_EMAIL"), password=os.getenv("JQUANTS_PASSWORD"))

# 2. TOPIX 100 銘柄リスト作成
print("📋 銘柄リストを作成中...")
df_list = cli.get_listed_info()
target_scales = ["TOPIX Core30", "TOPIX Large70"]
# 市場区分などのフィルタリング
df_topix100 = df_list[
    (df_list['MarketCodeName'].astype(str).str.contains('プライム')) & 
    (df_list['ScaleCategory'].isin(target_scales))
]
target_tickers = df_topix100['Code'].tolist()
print(f"🎯 対象: {len(target_tickers)} 銘柄")

# 3. データ取得 (APIの値をそのまま使う)
START_DATE = "20160101"
END_DATE = "20250830"

print(f"\n📥 データ取得開始...")
combined_data = {}

for code in tqdm(target_tickers):
    try:
        df = cli.get_prices_daily_quotes(code=code, from_yyyymmdd=START_DATE, to_yyyymmdd=END_DATE)
        
        if df.empty: continue

        # 日付変換
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        
        # 数値変換 (念のため)
        # ここで公式の 'AdjustmentClose' を数値化して採用します
        if 'AdjustmentClose' in df.columns:
            target_col = 'AdjustmentClose'
        else:
            # 万が一カラムがない場合はCloseを使う(リスクあり)
            print( f"⚠️ {code} に 'AdjustmentClose' カラムがありません")
            
        series = pd.to_numeric(df[target_col], errors='coerce')
        
        # 辞書に格納
        combined_data[code] = series
        
        time.sleep(0.1)

    except Exception as e:
        print(f"❌ Error at {code}: {e}")

# 4. 結合・保存
print("\n🔄 結合中...")
df_result = pd.concat(combined_data, axis=1)

# ※これをしないと、この日のせいで老舗企業まで全滅します
error_date = pd.to_datetime("2020-10-01")
if error_date in df_result.index:
    print("⚠️ 2020-10-01 (システム障害日) を除外します")
    df_result = df_result.drop(index=error_date)

# 生存者バイアス対策 (データが足りない銘柄は列ごと削除)
df_clean = df_result.dropna(axis=1)

# 保存
filename = "stock_prices_topix100_simple.csv"
df_clean.to_csv(filename)
# 1. 生存している列（銘柄）と、元の列（銘柄）を比較して、差分を出す
removed_tickers = df_result.columns.difference(df_clean.columns)

print("-" * 40)
print(f"🗑️ 削除された銘柄: {len(removed_tickers)} 件")
print("-" * 40)

# 2. なぜ消されたのか？（各銘柄のデータ開始日を調べる）
for code in removed_tickers:
    # first_valid_index() は、その列で最初に NaN じゃなくなる日付を返します
    start_date = df_result[code].first_valid_index()
    
    # もし全期間 NaN なら "データなし"
    if start_date is None:
        status = "全期間データなし (取得失敗)"
    else:
        status = f"データ開始: {start_date.date()}"
        
    print(f"❌ {code} : {status}")

print("-" * 40)

print("-" * 40)
print(f"✅ 完了: {filename}")
print(f"📊 データ形状: {df_clean.shape}")
print(f"   期間: {df_clean.index.min().date()} ~ {df_clean.index.max().date()}")
print("-" * 40)

# -----------------------------------------------------
# 【重要】 答え合わせ (プロット)
# -----------------------------------------------------
# もしこのグラフで「垂直落下」している銘柄があったら、
# APIの AdjustmentClose は「過去への遡及調整」をしていないことになります。
# その場合は、前の「田村流ロジック」に戻す必要があります。
if not df_clean.empty:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    # 正規化してプロット
    (df_clean.iloc[:, :5] / df_clean.iloc[0, :5]).plot(alpha=0.7)
    plt.title("Check: Is 'AdjustmentClose' really adjusted retroactively?")
    plt.grid(True)
    plt.show()
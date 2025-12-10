import os
import time
import pandas as pd
import numpy as np
import jquantsapi
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm

# =========================================================
# 1. 初期設定 & 認証
# =========================================================
load_dotenv(find_dotenv('J-Quants.env'))
try:
    cli = jquantsapi.Client(mail_address=os.getenv("JQUANTS_EMAIL"), password=os.getenv("JQUANTS_PASSWORD"))
except Exception as e:
    print(f"❌ 認証エラー: {e}")
    exit()

START_DATE = "20160101"
END_DATE = "20251215"
START_DATE_FIN = "20151201"

print("📋 TOPIX 100 銘柄リストを作成中...")
df_list = cli.get_listed_info()
df_topix100 = df_list[
    (df_list['MarketCodeName'].astype(str).str.contains('プライム')) & 
    (df_list['ScaleCategory'].isin(["TOPIX Core30", "TOPIX Large70"]))
]
target_tickers = df_topix100['Code'].tolist()
print(f"🎯 対象: {len(target_tickers)} 銘柄")

# =========================================================
# 2. データ取得 (株価 & 株式数)
# =========================================================
print(f"\n📥 データを取得中...")
data_adj = {}   # 分析用 (調整後株価)
data_raw = {}   # 計算用 (生株価)
data_shares = {} # 計算用 (株式数)

for code in tqdm(target_tickers):
    try:
        # --- (A) 株価取得 ---
        df = cli.get_prices_daily_quotes(code=code, from_yyyymmdd=START_DATE, to_yyyymmdd=END_DATE)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            
            if 'AdjustmentClose' in df.columns:
                data_adj[code] = pd.to_numeric(df['AdjustmentClose'], errors='coerce')
            if 'Close' in df.columns:
                data_raw[code] = pd.to_numeric(df['Close'], errors='coerce')
        
        time.sleep(0.5)
        
        # --- (B) 株式数取得 ---
        try:
            df_fins = cli.get_fins_statements(code=code)
            if not df_fins.empty:
                df_fins['Date'] = pd.to_datetime(df_fins['DisclosedDate'])
                df_fins = df_fins[(df_fins['Date'] >= pd.to_datetime(START_DATE_FIN)) & 
                                  (df_fins['Date'] <= pd.to_datetime(END_DATE))]
                
                # 【重要】あなたの調査で見つかった正しい列名を使用
                target_col = 'NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock'
                
                if target_col in df_fins.columns:
                    s = df_fins.set_index('Date')[target_col]
                    s = pd.to_numeric(s, errors='coerce')
                    if not s.dropna().empty:
                        # 重複日は最新を採用
                        data_shares[code] = s[~s.index.duplicated(keep='last')]
        except:
            print(f"⚠️ {code} 財務データ取得失敗")
            
    except Exception as e:
        print(f"❌ Error {code}: {e}")

# =========================================================
# 3. データ結合と保存
# =========================================================
print("\n⚙️ データ加工中...")

# 株価結合
df_adj = pd.concat(data_adj, axis=1)
error_date = pd.to_datetime("2020-10-01")
if error_date in df_adj.index:
    df_adj = df_adj.drop(index=error_date)
df_adj = df_adj.dropna(axis=1)

# 保存1: 株価データ (分析用)
df_adj.to_csv("stock_prices.csv")
print("✅ stock_prices.csv を保存しました")

# 時価総額計算
if len(data_shares) > 0:
    print("🧮 時価総額データの作成を試みます...")
    try:
        df_raw = pd.concat(data_raw, axis=1)
        df_shares_raw = pd.concat(data_shares, axis=1)
        
        # 銘柄合わせ
        valid_tickers = df_adj.columns.intersection(df_shares_raw.columns)
        
        if len(valid_tickers) > 0:
            df_raw = df_raw[valid_tickers]
            df_shares_raw = df_shares_raw[valid_tickers]
            
            # 日次展開 (ffill)
            df_shares_daily = df_shares_raw.reindex(df_adj.index, method='ffill').bfill()
            
            # 計算: 生株価 * 株式数
            df_market_cap = df_raw * df_shares_daily
            
            # 保存2: 時価総額データ
            df_market_cap.to_csv("market_caps.csv")
            print("✅ market_caps.csv を保存しました (時価総額加重用)")
        else:
            print("⚠️ 株価と株式数の銘柄が一致しませんでした。")
    except Exception as e:
        print(f"⚠️ 時価総額計算エラー: {e}")
else:
    print("⚠️ 株式数データが取得できませんでした。")

print("-" * 40)
print("完了。これで次のステップに進めます！")
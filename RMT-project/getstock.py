import os
import time
import pandas as pd
import numpy as np
import jquantsapi
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
import traceback

# =========================================================
# 1. 初期設定 & 認証
# =========================================================
print("🚀 初期設定を開始します...")
load_dotenv(find_dotenv('J-Quants.env'))
try:
    cli = jquantsapi.Client(mail_address=os.getenv("JQUANTS_EMAIL"), password=os.getenv("JQUANTS_PASSWORD"))
    print("✅ J-Quants API 認証成功")
except Exception as e:
    print(f"❌ 認証エラー: {e}")
    exit()

START_DATE = "20160130"
END_DATE = "20260115"
START_DATE_FIN = "20160130"

def fetch_with_retry(func, name, max_retries=3, wait_sec=2, **kwargs):
    for i in range(max_retries):
        try:
            return func(**kwargs)
        except Exception as e:
            if i == max_retries - 1:
                print(f"  ❌ {name} APIエラー (最終試行失敗): {e}")
                return pd.DataFrame()
            time.sleep(wait_sec)
    return pd.DataFrame()

print("📋 対象銘柄リストを作成中...")
try:
    df_list = cli.get_listed_info()
    df_topix100 = df_list[
        (df_list['MarketCodeName'].astype(str).str.contains('プライム')) & 
        (df_list['ScaleCategory'].isin(["TOPIX Core30", "TOPIX Large70"]))
    ]
    target_tickers = df_topix100['Code'].tolist()
    print(f"🎯 対象: {len(target_tickers)} 銘柄")
except Exception as e:
    print(f"❌ 銘柄リスト取得エラー: {e}")
    exit()

# =========================================================
# 2. データ取得ループ
# =========================================================
print(f"\n📥 データ取得と計算を開始します...")

data_market_cap = {} # 時価総額 (Close * Shares)
data_close = {}      # 株価 (Close)
data_adj_close = {}     # 分析用株価 (AdjustmentClose)
data_adj_shares = {}    # 調整後発行株式数
error_logs = []

for code in tqdm(target_tickers):
    try:
        # --- (A) 株価取得 ---
        df_price = fetch_with_retry(cli.get_prices_daily_quotes, f"株価({code})", code=code, from_yyyymmdd=START_DATE, to_yyyymmdd=END_DATE)
        
        if df_price.empty:
            continue

        df_price['Date'] = pd.to_datetime(df_price['Date'])
        df_price = df_price.set_index('Date').sort_index()

        # 必要なカラムの確保
        if 'Close' not in df_price.columns or 'AdjustmentClose' not in df_price.columns:
            msg = f"⚠️ {code}: 必要な株価カラムが不足しています"
            error_logs.append(msg)
            continue
            
        series_close = pd.to_numeric(df_price['Close'], errors='coerce')
        series_adj_close = pd.to_numeric(df_price['AdjustmentClose'], errors='coerce')

        # ★ここで「分析用株価」を保存！
        data_adj_close[code] = series_adj_close
        data_close[code] = series_close

        # 調整係数 (分割対応用)
        if 'AdjustmentFactor' in df_price.columns:
            series_factor = pd.to_numeric(df_price['AdjustmentFactor'], errors='coerce').fillna(1.0)
        else:
            series_factor = pd.Series(1.0, index=df_price.index)

        # --- (B) 財務データ取得 ---
        df_fins = fetch_with_retry(cli.get_fins_statements, f"財務({code})", code=code)
        series_shares_raw = None
        
        if not df_fins.empty:
            target_col = 'NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock'
            if target_col in df_fins.columns:
                try:
                    df_fins['Date'] = pd.to_datetime(df_fins['DisclosedDate'])
                    df_fins = df_fins.sort_values('Date')
                    df_fins = df_fins[df_fins['Date'] >= pd.to_datetime(START_DATE_FIN)]
                    s_fin = df_fins.set_index('Date')[target_col]
                    s_fin = pd.to_numeric(s_fin, errors='coerce')
                    s_fin = s_fin[~s_fin.index.duplicated(keep='last')]
                    series_shares_raw = s_fin.reindex(df_price.index, method='ffill').bfill()
                    
                except: pass

        # 財務データがない場合の救済 (最新値)
        if series_shares_raw is None or series_shares_raw.dropna().empty:
            latest_info = df_topix100[df_topix100['Code'] == code]
            if not latest_info.empty:
                latest_shares = latest_info['NumberOfIssuedAndOutstandingShares'].iloc[0]
                series_shares_raw = pd.Series(latest_shares, index=df_price.index)
            else:
                continue # データなしのためスキップ

        # --- (C) 分割ラグ補正 (時価総額用) ---
        adjusted_shares = series_shares_raw.copy()
        try:
            # 修正前: split_dates = series_factor[series_factor < 1.0].index
            # 修正後: 1.0以外（分割も併合も）すべて対象にする
            # J-Quants等の調整係数は、分割(1→2)なら約0.5、併合(10→1)なら約10.0が入る想定
            
            # 変更点1: 1.0以外のすべての変更点を取得
            action_dates = series_factor[series_factor != 1.0].index
            
            for action_date in action_dates:
                factor = series_factor.loc[action_date]
                
                # ゴミデータ除外
                if factor <= 0 or pd.isna(factor): continue
                
                # 変更点2: 係数から倍率を計算
                # 分割(factor=0.5) -> multiplier=2.0 (株数2倍)
                # 併合(factor=10.0) -> multiplier=0.1 (株数1/10)
                multiplier = 1.0 / factor
                
                # 変更基準日時点の「補正前」の株数を取得
                base_shares = adjusted_shares.loc[action_date]
                if pd.isna(base_shares): continue
                
                # その日以降のデータを確認
                future_shares = series_shares_raw.loc[action_date:]
                
                # 「財務データの更新がまだ来ていない期間」を特定する
                # 併合の場合、新しい財務データが来るまで株数は「多いまま(base_sharesに近い)」
                # 分割の場合、新しい財務データが来るまで株数は「少ないまま(base_sharesに近い)」
                
                # 許容誤差範囲（±10%以内なら「まだ更新されてない」とみなす）
                mask_unchanged = (future_shares >= base_shares * 0.9) & (future_shares <= base_shares * 1.1)
                
                target_period = future_shares[mask_unchanged].index
                
                if not target_period.empty:
                    # その期間だけ、株数に倍率を掛けて強制補正
                    adjusted_shares.loc[target_period] = adjusted_shares.loc[target_period] * multiplier
                    
        except Exception as e:
            # エラー時はログを出してスルー（厳密なエラー処理はお好みで）
            # print(f"Warning share adjustment {code}: {e}") 
            pass
        # --- (D) 時価総額計算 ---
        data_adj_shares[code] = adjusted_shares
        mc = series_close * adjusted_shares
        data_market_cap[code] = mc



    except Exception as e:
        error_logs.append(f"❌ CRITICAL {code}: {e}")

# =========================================================
# 3. 結合・整形・保存
# =========================================================
print("\n⚙️ データを整形・保存中...")

if len(data_market_cap) > 0 and len(data_adj_close) > 0 and len(data_close) > 0:
    # 1. 結合
    df_mc = pd.concat(data_market_cap, axis=1)
    df_prices = pd.concat(data_adj_close, axis=1)
    df_prices_raw = pd.concat(data_close, axis=1)
    df_shares = pd.concat(data_adj_shares, axis=1)
    
    # 2. 共通の銘柄のみ残す (整合性確保)
    common_tickers = df_mc.columns.intersection(df_prices.columns)
    df_mc = df_mc[common_tickers]
    df_prices = df_prices[common_tickers]
    df_prices_raw = df_prices_raw[common_tickers]
    df_shares = df_shares[common_tickers]

    # 3. 共通の日付のみ残す
    common_dates = df_mc.index.intersection(df_prices.index)
    df_mc = df_mc.loc[common_dates].sort_index()
    df_prices = df_prices.loc[common_dates].sort_index()
    df_prices_raw = df_prices_raw.loc[common_dates].sort_index()
    df_shares = df_shares.loc[common_dates].sort_index()

    # 4. 保存
    df_mc.to_csv("market_caps.csv")
    df_prices.to_csv("stock_adj_close.csv")
    df_prices_raw.to_csv("stock_close.csv")
    df_shares.to_csv("stock_shares.csv")

    print(f"✅ 保存完了:")
    print(f"   - market_caps.csv (時価総額: ウェイト計算用)")
    print(f"   - stock_adj_close.csv (調整後株価: リターン計算用)")
    print(f"   - stock_close.csv (通常株価: リターン計算用)")
    print(f"   - stock_shares.csv (調整後発行株式数: 時価総額計算用)")
    print(f"📊 銘柄数: {len(common_tickers)}, 期間: {common_dates.min().date()} ~ {common_dates.max().date()}")

else:
    print("❌ データが十分に集まりませんでした。")


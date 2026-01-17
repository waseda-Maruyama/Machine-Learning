import os
import time
import pandas as pd
import numpy as np
import jquantsapi
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
import traceback  # エラー詳細表示用

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
    # traceback.print_exc() # 必要であれば詳細なスタックトレースを表示
    exit()

START_DATE = "20160130"
END_DATE = "20250115"
START_DATE_FIN = "20150101"

# APIリトライ用の関数（エラー時にも内容を表示）
def fetch_with_retry(func, name, max_retries=3, wait_sec=2, **kwargs):
    for i in range(max_retries):
        try:
            return func(**kwargs)
        except Exception as e:
            if i == max_retries - 1:
                print(f"  ❌ {name} APIエラー (最終試行失敗): {e}")
                return pd.DataFrame()
            print(f"  ⚠️ {name} APIエラー (リトライ {i+1}/{max_retries}): {e}")
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
data_market_cap = {}

# エラー集計用リスト
error_logs = []

for code in tqdm(target_tickers):
    try:
        # --- (A) 株価取得 ---
        df_price = fetch_with_retry(cli.get_prices_daily_quotes, f"株価({code})", code=code, from_yyyymmdd=START_DATE, to_yyyymmdd=END_DATE)
        
        if df_price.empty:
            msg = f"⚠️ {code}: 株価データが取得できませんでした (Empty DataFrame)"
            print(msg)
            error_logs.append(msg)
            continue
            
        df_price['Date'] = pd.to_datetime(df_price['Date'])
        df_price = df_price.set_index('Date').sort_index()
        
        if 'Close' not in df_price.columns:
            msg = f"⚠️ {code}: 'Close'カラムが存在しません"
            print(msg)
            error_logs.append(msg)
            continue
            
        series_close = pd.to_numeric(df_price['Close'], errors='coerce')
        
        # 調整係数
        if 'AdjustmentFactor' in df_price.columns:
            series_factor = pd.to_numeric(df_price['AdjustmentFactor'], errors='coerce').fillna(1.0)
        else:
            series_factor = pd.Series(1.0, index=df_price.index)

        # --- (B) 財務データ取得 ---
        df_fins = fetch_with_retry(cli.get_fins_statements, f"財務({code})", code=code)
        
        series_shares_raw = None
        
        # 財務データの処理
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
                    
                    # ffill -> bfill で欠損を埋める
                    series_shares_raw = s_fin.reindex(df_price.index, method='ffill').bfill()
                except Exception as e:
                    print(f"  ⚠️ {code}: 財務データ処理中にエラー: {e}")
                    # 処理続行のためにNoneのままにする
            else:
                 print(f"  ⚠️ {code}: 財務データにカラム '{target_col}' が見つかりません")

        # 財務データNG時の救済策
        if series_shares_raw is None or series_shares_raw.dropna().empty:
            try:
                # 銘柄リストから最新値を取得
                latest_info = df_topix100[df_topix100['Code'] == code]
                if not latest_info.empty:
                    latest_shares = latest_info['NumberOfIssuedAndOutstandingShares'].iloc[0]
                    series_shares_raw = pd.Series(latest_shares, index=df_price.index)
                    # print(f"  ℹ️ {code}: 最新株式数({latest_shares})で代用")
                else:
                    raise ValueError("銘柄リストに情報なし")
            except Exception as e:
                msg = f"❌ {code}: 株式数データの取得に失敗しました (財務履歴なし & 最新値取得エラー: {e})"
                print(msg)
                error_logs.append(msg)
                continue

        # --- (C) 分割ラグ補正 ---
        try:
            adjusted_shares = series_shares_raw.copy()
            split_dates = series_factor[series_factor < 1.0].index
            
            for split_date in split_dates:
                factor = series_factor.loc[split_date]
                if factor <= 0: continue
                
                multiplier = 1.0 / factor
                base_shares = adjusted_shares.loc[split_date]
                
                if pd.isna(base_shares): continue
                
                future_shares = series_shares_raw.loc[split_date:]
                # 10%以内の変動なら「まだ値が変わっていない」とみなす
                mask_unchanged = (future_shares >= base_shares * 0.9) & (future_shares <= base_shares * 1.1)
                target_period = future_shares[mask_unchanged].index
                
                if not target_period.empty:
                    adjusted_shares.loc[target_period] = adjusted_shares.loc[target_period] * multiplier
        except Exception as e:
             print(f"  ⚠️ {code}: 分割ラグ補正処理中にエラー (スキップして生データを使用): {e}")
             adjusted_shares = series_shares_raw # エラー時は補正なしで続行

        # --- (D) 時価総額計算 ---
        try:
            mc = series_close * adjusted_shares
            # 全てNaNになっていないか確認
            if mc.dropna().empty:
                 msg = f"⚠️ {code}: 計算結果が全てNaNです"
                 print(msg)
                 error_logs.append(msg)
            else:
                data_market_cap[code] = mc
        except Exception as e:
            msg = f"❌ {code}: 計算エラー: {e}"
            print(msg)
            error_logs.append(msg)

    except Exception as e:
        # 想定外の根本的なエラー
        msg = f"❌ CRITICAL ERROR {code}: {e}"
        print(msg)
        traceback.print_exc() # 詳細ログ
        error_logs.append(msg)

# =========================================================
# 保存
# =========================================================
print("\n⚙️ データを保存中...")
if len(data_market_cap) > 0:
    df_mc = pd.concat(data_market_cap, axis=1)
    df_mc = df_mc.dropna(how='all', axis=1)
    
    print(f"  📊 取得成功銘柄数: {df_mc.shape[1]}")
    print(f"  📅 データ期間: {df_mc.index.min()} ~ {df_mc.index.max()}")
    
    if len(error_logs) > 0:
        print("\n⚠️ 発生したエラー一覧:")
        for log in error_logs[:10]: # 最初の10件を表示
            print(log)
        if len(error_logs) > 10:
            print(f"...他 {len(error_logs)-10} 件")

    df_mc.to_csv("market_caps.csv")
    print("✅ market_caps.csv を保存しました！")
else:
    print("❌ データが1件も作成されませんでした。エラーログを確認してください。")
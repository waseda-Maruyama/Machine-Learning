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

START_DATE = "20160130"  # ユーザー指定の期間に合わせて変更
END_DATE = "20250115"
START_DATE_FIN = "20160130" # 財務データは少し前から取っておく

# APIリトライ用の関数
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
error_logs = []

# ★追加: 欠損値レポート用のリスト
missing_stats = []

for code in tqdm(target_tickers):
    # 各銘柄ごとの欠損カウンタ
    stat = {
        "Code": code,
        "Total_Days": 0,
        "Missing_Close": 0,       # 株価データ自体がない日
        "Missing_Shares_Raw": 0,  # 財務データ結合直後の欠損（前方参照前）
        "Missing_MC_Final": 0,    # 最終的に計算できなかった日
        "Status": "OK"
    }

    try:
        # --- (A) 株価取得 ---
        df_price = fetch_with_retry(cli.get_prices_daily_quotes, f"株価({code})", code=code, from_yyyymmdd=START_DATE, to_yyyymmdd=END_DATE)
        
        if df_price.empty:
            msg = f"⚠️ {code}: 株価データ取得失敗"
            print(msg)
            error_logs.append(msg)
            stat["Status"] = "No Price Data"
            missing_stats.append(stat)
            continue
            
        df_price['Date'] = pd.to_datetime(df_price['Date'])
        df_price = df_price.set_index('Date').sort_index()
        
        # 数値変換 & 欠損チェック
        series_close = pd.to_numeric(df_price['Close'], errors='coerce')
        
        # ★ここで株価自体の欠損をカウント
        stat["Total_Days"] = len(series_close)
        stat["Missing_Close"] = series_close.isna().sum()

        # 調整係数
        if 'AdjustmentFactor' in df_price.columns:
            series_factor = pd.to_numeric(df_price['AdjustmentFactor'], errors='coerce').fillna(1.0)
        else:
            series_factor = pd.Series(1.0, index=df_price.index)

        # --- (B) 財務データ取得 ---
        df_fins = fetch_with_retry(cli.get_fins_statements, f"財務({code})", code=code)
        
        series_shares_raw = None
        target_col = 'NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock'

        if not df_fins.empty and target_col in df_fins.columns:
            try:
                df_fins['Date'] = pd.to_datetime(df_fins['DisclosedDate'])
                df_fins = df_fins.sort_values('Date')
                df_fins = df_fins[df_fins['Date'] >= pd.to_datetime(START_DATE_FIN)]
                
                s_fin = df_fins.set_index('Date')[target_col]
                s_fin = pd.to_numeric(s_fin, errors='coerce')
                s_fin = s_fin[~s_fin.index.duplicated(keep='last')]
                
                # ★検証用: 前方参照(ffill)する前に、単純に結合して「どれくらいデータがないか」を確認
                # (財務データは四半期ごとなので、本来はほとんどの日がNaNになるはず)
                shares_check = s_fin.reindex(df_price.index)
                stat["Missing_Shares_Raw"] = shares_check.isna().sum()
                
                # 本番処理: ffill -> bfill で欠損を埋める
                series_shares_raw = s_fin.reindex(df_price.index, method='ffill').bfill()

            except Exception as e:
                print(f"  ⚠️ {code}: 財務データ処理エラー: {e}")
        
        # 財務データNG時の救済
        if series_shares_raw is None or series_shares_raw.dropna().empty:
            latest_info = df_topix100[df_topix100['Code'] == code]
            if not latest_info.empty:
                latest_shares = latest_info['NumberOfIssuedAndOutstandingShares'].iloc[0]
                series_shares_raw = pd.Series(latest_shares, index=df_price.index)
                stat["Status"] = "Used Latest Shares (No History)"
            else:
                msg = f"❌ {code}: 株式数データ完全欠損"
                error_logs.append(msg)
                stat["Status"] = "Failed (No Shares)"
                missing_stats.append(stat)
                continue

        # --- (C) 分割ラグ補正 (省略せずそのまま実行) ---
        adjusted_shares = series_shares_raw.copy()
        try:
            split_dates = series_factor[series_factor < 1.0].index
            for split_date in split_dates:
                factor = series_factor.loc[split_date]
                if factor <= 0: continue
                multiplier = 1.0 / factor
                base_shares = adjusted_shares.loc[split_date]
                if pd.isna(base_shares): continue
                future_shares = series_shares_raw.loc[split_date:]
                mask_unchanged = (future_shares >= base_shares * 0.9) & (future_shares <= base_shares * 1.1)
                target_period = future_shares[mask_unchanged].index
                if not target_period.empty:
                    adjusted_shares.loc[target_period] = adjusted_shares.loc[target_period] * multiplier
        except:
            pass # エラー時は補正なし

        # --- (D) 時価総額計算 ---
        mc = series_close * adjusted_shares
        
        # ★最終的な欠損チェック
        nan_count = mc.isna().sum()
        stat["Missing_MC_Final"] = nan_count
        
        if nan_count > 0:
            stat["Status"] = f"Partial Missing ({nan_count} days)"
        
        data_market_cap[code] = mc
        missing_stats.append(stat)

    except Exception as e:
        msg = f"❌ CRITICAL ERROR {code}: {e}"
        print(msg)
        error_logs.append(msg)
        stat["Status"] = "Critical Error"
        missing_stats.append(stat)

# =========================================================
# 保存 & レポート出力
# =========================================================
print("\n⚙️ データを保存中...")

# 1. 時価総額CSV
if len(data_market_cap) > 0:
    df_mc = pd.concat(data_market_cap, axis=1)
    df_mc = df_mc.dropna(how='all', axis=1) # 全てNaNの列は削除
    df_mc.to_csv("market_caps.csv")
    print(f"✅ 'market_caps.csv' を保存しました (銘柄数: {df_mc.shape[1]})")

# 2. 欠損値レポートCSV (★新規追加)
df_report = pd.DataFrame(missing_stats)
if not df_report.empty:
    # 見やすいように列の順序を整理
    cols = ["Code", "Status", "Total_Days", "Missing_Close", "Missing_MC_Final", "Missing_Shares_Raw"]
    df_report = df_report[cols]
    
    output_report_name = "missing_data_report.csv"
    df_report.to_csv(output_report_name, index=False)
    print(f"✅ 欠損状況レポートを '{output_report_name}' に保存しました。")
    
    # 簡易表示: 問題があった銘柄だけ表示
    problematic = df_report[df_report["Missing_MC_Final"] > 0]
    if not problematic.empty:
        print("\n⚠️ 最終的にデータが欠けている銘柄があります:")
        print(problematic[["Code", "Missing_MC_Final"]].to_string(index=False))
    else:
        print("\n✨ 最終データ(MC)に欠損値はありませんでした（全期間計算完了）。")

print("\n処理完了")
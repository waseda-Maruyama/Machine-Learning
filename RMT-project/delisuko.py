import os
import pandas as pd
import jquantsapi
from dotenv import load_dotenv, find_dotenv

# =========================================================
# 設定
# =========================================================
TARGET_CODES = ["54010", "94340"]  # 日本製鉄, ディスコ
START_DATE_CHECK = "20160130"      # 欠損が疑われる時期（2025年中盤以降）
END_DATE_CHECK   = "20260117"      # 現在

print("🚀 J-Quants API 生データ診断を開始します...")

# 認証
load_dotenv(find_dotenv('J-Quants.env'))
try:
    cli = jquantsapi.Client(mail_address=os.getenv("JQUANTS_EMAIL"), password=os.getenv("JQUANTS_PASSWORD"))
    print("✅ 認証成功\n")
except Exception as e:
    print(f"❌ 認証エラー: {e}")
    exit()

for code in TARGET_CODES:
    print(f"🔎 銘柄コード: {code} の調査")
    print("-" * 50)

    # -----------------------------------------------------
    # 1. 株価データ (Price) の確認
    # -----------------------------------------------------
    try:
        df_price = cli.get_prices_daily_quotes(code=code, from_yyyymmdd=START_DATE_CHECK, to_yyyymmdd=END_DATE_CHECK)
        if df_price.empty:
            print("   ❌ 株価データ: 取得できませんでした (Empty)")
        else:
            df_price['Date'] = pd.to_datetime(df_price['Date'])
            df_price = df_price.set_index('Date').sort_index()
            last_date = df_price.index[-1].date()
            print(f"   📈 株価データ: {len(df_price)} 行取得")
            print(f"      最新日付: {last_date}")
            print(f"      直近5日の終値: {df_price['Close'].tail(5).tolist()}")
    except Exception as e:
        print(f"   ❌ 株価APIエラー: {e}")

    # -----------------------------------------------------
    # 2. 財務データ (Shares) の確認
    # -----------------------------------------------------
    try:
        # 財務情報はキャッシュが効くことがあるので念のため広めに取る
        df_fins = cli.get_fins_statements(code=code)
        
        target_col = 'NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock'
        
        if df_fins.empty:
            print("   ❌ 財務データ: 取得できませんでした (Empty)")
        else:
            # 日付整理
            df_fins['Date'] = pd.to_datetime(df_fins['DisclosedDate'])
            df_fins = df_fins.sort_values('Date')
            
            # 対象期間（2025年以降）に絞る
            df_fins_recent = df_fins[df_fins['Date'] >= pd.to_datetime("2025-01-01")]
            
            print(f"   🏢 財務データ(発行済株式数):")
            
            if df_fins_recent.empty:
                print("      ⚠️ 2025年以降の開示データが見つかりません！")
            else:
                # 重要な列だけ表示
                display_cols = ['Date', target_col]
                # もし株式分割係数の情報があればそれも見たいが、finsには直接ないので株式数推移を見る
                
                print(f"      直近の開示リスト (2025年以降):")
                for _, row in df_fins_recent.iterrows():
                    d_str = row['Date'].strftime('%Y-%m-%d')
                    val = row.get(target_col, "None")
                    print(f"      - {d_str}: {val}")

            # -----------------------------------------------------
            # 3. 結合テスト (Merge Simulation)
            # -----------------------------------------------------
            # 株価はあるが、財務データの最新日付が古すぎて届いていないかチェック
            if not df_price.empty and not df_fins_recent.empty:
                last_price_date = df_price.index[-1]
                last_fin_date = df_fins_recent['Date'].iloc[-1]
                
                print(f"\n   ⚖️ 結合判定:")
                print(f"      株価の最新日: {last_price_date.date()}")
                print(f"      財務の最新日: {last_fin_date.date()}")
                
                if last_fin_date < last_price_date:
                    diff_days = (last_price_date - last_fin_date).days
                    print(f"      ⚠️ 財務データの更新が {diff_days} 日止まっています。")
                    print(f"      → ffill(前方穴埋め) をしないと、この期間が欠損(NaN)になります。")
                else:
                    print(f"      ✅ 日付はカバーされています。")

    except Exception as e:
        print(f"   ❌ 財務APIエラー: {e}")
    
    print("\n" + "="*50 + "\n")
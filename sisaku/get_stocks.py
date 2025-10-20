import pandas as pd
import os
import jquantsapi
from dotenv import load_dotenv, find_dotenv
import time

# .envファイルから認証情報を読み込む
load_dotenv(find_dotenv('J-Quants.env'))
JQUANTS_MAIL = os.getenv("JQUANTS_EMAIL")
JQUANTS_PASS = os.getenv("JQUANTS_PASSWORD")

if not JQUANTS_MAIL or not JQUANTS_PASS:
    print("❌ 環境変数ファイル 'J-Quants.env' に認証情報が設定されていません")
    exit()

# 1. APIクライアントを初期化
print("J-Quants APIクライアントを初期化しています...")
cli = jquantsapi.Client(mail_address=JQUANTS_MAIL, password=JQUANTS_PASS)
print("クライアントの初期化が完了しました。")

# 2. 分析対象の証券コードリスト
tickers = ["4755", "4689", "4385", "4443", "4751", "3697"]

# 3. データを取得する期間 (君が指定した日付)
start_date = "20230727"
end_date = "20250727"

# 4. jQuants APIで株価データを取得
try:
    print("jQuants APIからデータのダウンロードを開始します...")
    
    # 各社の終値データを格納するリストを準備
    all_price_series = []

    # 個別銘柄の株価を取得
    for code in tickers:
        df = cli.get_prices_daily_quotes(code=code, from_yyyymmdd=start_date, to_yyyymmdd=end_date)
        # 'Date'列を日付型に変換し、インデックスに設定
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
        # 'Close'列を銘柄コード名に変更してリストに追加
        all_price_series.append(df['Close'].rename(code))
        print(f"✅ {code} のデータを取得完了...")
        time.sleep(1) # サーバーへの負荷を考慮して1秒待つ

    # 全てのデータを日付を基準に結合する
    final_df = pd.concat(all_price_series, axis=1)

    if final_df.empty:
        print("\n❌ エラー: データのダウンロードに失敗しました。")
    else:
        print("\n🎉 データのダウンロードと結合に成功しました！")
        # 5. 取得したデータをCSVファイルに保存
        output_filename = 'stock_prices.csv'
        try:
            final_df.to_csv(output_filename)
            print(f"\n📄 データを '{output_filename}' として保存しました。")
        except Exception as e:
            print(f"\n❌ ファイルの保存中にエラーが発生しました: {e}")
        print("--- 終値データ (始め) ---")
        print(final_df.head())
        print("\n--- 終値データ (終わり) ---")
        print(final_df.tail())

except Exception as e:
    print(f"\n❌ 予期せぬエラーが発生しました: {e}")
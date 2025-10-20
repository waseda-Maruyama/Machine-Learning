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
    print("❌ 環境変数ファイル 'J-Quants.env' が見つかりません")
    exit()

# 1. APIクライアントを初期化
print("J-Quants APIクライアントを初期化しています...")
cli = jquantsapi.Client(mail_address=JQUANTS_MAIL, password=JQUANTS_PASS)
print("クライアントの初期化が完了しました。")

# 2. 分析対象の証券コードリスト
tickers = ["4755", "4689", "4385", "4443", "4751", "3697"]

# 3. jQuants APIで財務諸表データを取得
try:
    print("財務諸表データのダウンロードを開始します...")
    
    all_fins_data = []
    for code in tickers:
        df = cli.get_fins_statements(code=code)
        all_fins_data.append(df)
        print(f"✅ {code} の財務データを取得完了...")
        time.sleep(1)

    # 取得した全社のデータフレームを縦に連結
    final_fins_df = pd.concat(all_fins_data)
    
    if final_fins_df.empty:
        print("\n❌ エラー: 財務データのダウンロードに失敗しました。")
    else:
        print("\n🎉 全企業の財務データのダウンロードと結合に成功しました！")
        
        # ★★★ ここが最重要修正点 ★★★
        # 'LocalCode'列を文字列に変換し、末尾1文字を削除して4桁コードに整形する
        print("\n銘柄コードを4桁に整形しています... (例: 47550 -> 4755)")
        final_fins_df['LocalCode'] = final_fins_df['LocalCode'].astype(str).str[:-1]
        
        # 4. 取得したデータをCSVファイルに保存
        output_filename = 'financial_statements.csv'
        final_fins_df.to_csv(output_filename, index=False)
        print(f"\n📄 整形後のデータを '{output_filename}' として保存しました。")
        
        print("\n--- 整形後の楽天グループ(4755)のデータの一部 ---")
        print(final_fins_df[final_fins_df['LocalCode'] == '4755'].head())

except Exception as e:
    print(f"\n❌ 予期せぬエラーが発生しました: {e}")
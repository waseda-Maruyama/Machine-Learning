import pandas as pd
import numpy as np

# --- 1. データの読み込み ---
print("CSVファイルを読み込んでいます...")
try:
    prices_df = pd.read_csv('stock_prices.csv', index_col='Date', parse_dates=True)
    financials_df = pd.read_csv('financial_statements.csv', parse_dates=['DisclosedDate'])
    print("✅ ファイルの読み込み完了。")
except FileNotFoundError as e:
    print(f"❌ エラー: {e.filename} が見つかりません。")
    exit()

# ★★★ ここからが最重要修正点 ★★★
# --- 1.5. 分析期間の調整 ---
# 最初の財務開示日を取得
first_disclosure_date = financials_df['DisclosedDate'].min()
print(f"最初の財務開示日: {first_disclosure_date}")

# 株価データを、最初の財務開示日以降に限定する
prices_df = prices_df[prices_df.index >= first_disclosure_date]
print(f"分析対象の株価データ期間を {prices_df.index.min()} 以降に調整しました。")
# ★★★ ここまでが最重要修正点 ★★★

# --- 2. 目的変数 (y) の作成 ---
print("\n目的変数（3ヶ月後の勝ち/負け）を作成しています...")
future_return = prices_df.shift(-60) / prices_df
target_df = (future_return > 1).astype(int)

# --- 3. 財務特徴量 (X) の作成 ---
print("財務特徴量を作成しています...")
required_columns = ['LocalCode', 'DisclosedDate', 'NetSales', 'OperatingProfit', 'Profit', 'TotalAssets', 'Equity']
fins = financials_df[required_columns].copy()
fins['ROE'] = np.where(fins['Equity'] > 0, fins['Profit'] / fins['Equity'], np.nan)
fins['SelfCapitalRatio'] = np.where(fins['TotalAssets'] > 0, fins['Equity'] / fins['TotalAssets'], np.nan)
fins = fins.rename(columns={'LocalCode': 'code', 'DisclosedDate': 'Date'})
fins['code'] = fins['code'].astype(str)

# --- 4. データセットの結合 ---
print("株価データと財務データを結合しています...")
all_dfs = []
for code_str in prices_df.columns:
    y = target_df[[code_str]].rename(columns={code_str: 'target'})
    X = fins[fins['code'] == code_str].sort_values('Date')
    merged_df = pd.merge_asof(y.sort_index(), X, on='Date', direction='backward')
    merged_df['code'] = code_str
    all_dfs.append(merged_df)

# ★★★ dropnaの修正 ★★★
# 結合後、全ての行を消すのではなく、未来のデータがない末尾60行などを削除する
final_dataset = pd.concat(all_dfs).dropna()

# --- 5. 結果の確認と保存 ---
if final_dataset.empty:
    print("\n❌ エラー: 最終データセットが空です。")
else:
    print("\n🎉 最終分析用データセットの作成に成功しました！")
    output_filename = 'analysis_dataset.csv'
    final_dataset.to_csv(output_filename)
    print(f"\n📄 データを '{output_filename}' として保存しました。")
    print("\n--- 最終データセット（始め） ---")
    print(final_dataset.head())
    print("\n--- 最終データセット（終わり） ---")
    print(final_dataset.tail())
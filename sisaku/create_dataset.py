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

# --- 2. 分析期間の調整 ---
first_disclosure_date = financials_df['DisclosedDate'].min()
prices_df = prices_df[prices_df.index >= first_disclosure_date]
print(f"分析対象の株価データ期間を {prices_df.index.min()} 以降に調整しました。")

# --- 3. 目的変数 (y) の作成 ---
print("\n目的変数（3ヶ月後の勝ち/負け）を作成しています...")
future_return = prices_df.shift(-60) / prices_df
target_df = (future_return > 1).astype(int)

# --- 4. 財務特徴量 (X) の作成 ---
print("財務特徴量を作成しています...")

# ★★★ ここからが修正・追加点 ★★★
# 'NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock' が
# jQuantsの財務データに含まれている必要がある。
required_columns = [
    'LocalCode', 'DisclosedDate', 'NetSales', 'OperatingProfit', 'Profit', 
    'TotalAssets', 'Equity', 
    'NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock'
]

# カラムが存在するかチェック
if not all(col in financials_df.columns for col in required_columns):
    print("❌ エラー: 'NumberOfIssuedAndOutstandingShares...'カラムが financial_statements.csv に見つかりません。")
    print("   PBR/PERの計算に必要です。jQuantsから取得するデータを再確認してください。")
    # 代替カラム名（もしあれば）: 'NumberOfOutstandingShares' など
    exit()

fins = financials_df[required_columns].copy()

# 1株あたり利益 (EPS) と 1株あたり純資産 (BPS) を計算
fins['EPS'] = fins['Profit'] / fins['NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock']
fins['BPS'] = fins['Equity'] / fins['NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock']

# ROE と 自己資本比率も計算
fins['ROE'] = np.where(fins['Equity'] > 0, fins['Profit'] / fins['Equity'], np.nan)
fins['SelfCapitalRatio'] = np.where(fins['TotalAssets'] > 0, fins['Equity'] / fins['TotalAssets'], np.nan)

# 扱いやすいようにカラム名をリネーム
fins = fins.rename(columns={'LocalCode': 'code', 'DisclosedDate': 'Date'})
fins['code'] = fins['code'].astype(str)

# --- 5. データセットの結合 ---
print("株価データと財務データを結合しています...")
all_dfs = []

for code_str in prices_df.columns:
    # 株価データ
    price_col = prices_df[[code_str]].rename(columns={code_str: 'Close'})
    # 目的変数データ
    y = target_df[[code_str]].rename(columns={code_str: 'target'})
    
    # 株価と目的変数を結合
    price_target_df = pd.concat([price_col, y], axis=1)
    
    # 財務データ
    X_fins = fins[fins['code'] == code_str].sort_values('Date')
    
    # 株価(y含む)と財務(X)を結合
    merged_df = pd.merge_asof(price_target_df.sort_index(), X_fins, on='Date', direction='backward')
    merged_df['code'] = code_str
    
    # PER と PBR を計算 (日々の株価と直近の財務データで)
    merged_df['PER'] = merged_df['Close'] / merged_df['EPS']
    merged_df['PBR'] = merged_df['Close'] / merged_df['BPS']
    
    all_dfs.append(merged_df)

# 全銘柄のデータを縦に結合し、NaNを含む行を削除
final_dataset = pd.concat(all_dfs).dropna(subset=['target'])

# --- 6. 結果の確認と保存 ---
if final_dataset.empty:
    print("\n❌ エラー: 最終データセットが空です。")
else:
    print("\n🎉 最終分析用データセットの作成に成功しました！ (PER, PBR 追加)")
    output_filename = 'analysis_dataset_v2.csv'
    final_dataset.to_csv(output_filename)
    print(f"\n📄 データを '{output_filename}' として保存しました。")
    print("\n--- 最終データセット（始め） ---")
    print(final_dataset.head())
    print("\n--- 最終データセット（終わり） ---")
    print(final_dataset.tail())

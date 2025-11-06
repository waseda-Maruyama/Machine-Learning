import pandas as pd
import numpy as np

print("--- V2 (財務比率 + バリュー比率) 公正比較版 ---")
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
print("財務特徴量（比率 + バリュー比率）を作成しています...")
required_columns = [
    'LocalCode', 'DisclosedDate', 'Profit', 'TotalAssets', 'Equity',
    'NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock'
]
fins = financials_df[required_columns].copy()

# 「比率」を計算
fins['ROE'] = np.where(fins['Equity'] > 0, fins['Profit'] / fins['Equity'], np.nan)
fins['SelfCapitalRatio'] = np.where(fins['TotalAssets'] > 0, fins['Equity'] / fins['TotalAssets'], np.nan)
# 「部品」を計算
fins['EPS'] = fins['Profit'] / fins['NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock']
fins['BPS'] = fins['Equity'] / fins['NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock']

fins = fins.rename(columns={'LocalCode': 'code', 'DisclosedDate': 'Date'})
fins['code'] = fins['code'].astype(str)

# ★★★ 必要な「比率」と「部品」だけを残す ★★★
fins = fins[['code', 'Date', 'ROE', 'SelfCapitalRatio', 'EPS', 'BPS']]

# --- 5. データセットの結合 ---
print("株価データと財務データを結合しています...")
all_dfs = []
for code_str in prices_df.columns:
    y = target_df[[code_str]].rename(columns={code_str: 'target'})
    price_col = prices_df[[code_str]].rename(columns={code_str: 'Close'})
    
    X_fins = fins[fins['code'] == code_str].sort_values('Date')
    
    y_price_df = pd.concat([y, price_col], axis=1)
    
    merged_df = pd.merge_asof(
        y_price_df.sort_index().reset_index(), # 'Date'をインデックスから列に戻す
        X_fins, 
        on='Date', 
        direction='backward'
    )
    
    # ★★★ PER, PBRを計算 ★★★
    merged_df['PER'] = merged_df['Close'] / merged_df['EPS']
    merged_df['PBR'] = merged_df['Close'] / merged_df['BPS']
    
    merged_df['code'] = code_str
    all_dfs.append(merged_df)

# --- 修正C：公正な 'dropna' (ごちゃまzeルール) ---
final_dataset = pd.concat(all_dfs).dropna(subset=['target','ROE'])

# --- 6. 結果の確認と保存 ---
if final_dataset.empty:
    print("\n❌ エラー: 最終データセットが空です。")
else:
    print("\n🎉 V2 (比率+バリュー) 公正比較版データセットの作成に成功！")
    output_filename = 'analysis_dataset_v2_VALUE.csv'
    final_dataset.to_csv(output_filename, index=False)
    print(f"\n📄 データを '{output_filename}' として保存しました。")
    print(final_dataset.head())
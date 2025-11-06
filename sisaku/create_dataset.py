import pandas as pd
import numpy as np

print("--- V1 (比率のみ) 公正比較版 ---")
# --- 1. データの読み込み ---
print("CSVファイルを読み込んでいます...")
try:
    prices_df = pd.read_csv('stock_prices.csv', index_col='Date', parse_dates=True)
    financials_df = pd.read_csv('financial_statements.csv', parse_dates=['DisclosedDate'])
    print("✅ ファイルの読み込み完了。")
except FileNotFoundError as e:
    print(f"❌ エラー: {e.filename} が見つかりません。")
    exit()

# --- 2. 修正A：分析期間の調整 ---
first_disclosure_date = financials_df['DisclosedDate'].min()
prices_df = prices_df[prices_df.index >= first_disclosure_date]
print(f"分析対象の株価データ期間を {prices_df.index.min()} 以降に調整しました。")

# --- 3. 目的変数 (y) の作成 ---
print("\n目的変数（3ヶ月後の勝ち/負け）を作成しています...")
future_return = prices_df.shift(-60) / prices_df
target_df = (future_return > 1).astype(int)

# --- 4. 財務特徴量 (X) の作成 ---
print("財務特徴量（比率のみ）を作成しています...")
required_columns = [
    'LocalCode', 'DisclosedDate', 'Profit', 'TotalAssets', 'Equity'
]
fins = financials_df[required_columns].copy()

# 「比率」を計算
fins['ROE'] = np.where(fins['Equity'] > 0, fins['Profit'] / fins['Equity'], np.nan)
fins['SelfCapitalRatio'] = np.where(fins['TotalAssets'] > 0, fins['Equity'] / fins['TotalAssets'], np.nan)
fins = fins.rename(columns={'LocalCode': 'code', 'DisclosedDate': 'Date'})
fins['code'] = fins['code'].astype(str)

# ★★★ ここが最重要修正点 ★★★
# 必要な「比率」と「キー」だけを残し、「生の数字」はすべて捨てる
fins = fins[['code', 'Date', 'ROE', 'SelfCapitalRatio']]

# --- 5. データセットの結合 ---
print("株価データと財務データを結合しています...")
all_dfs = []
for code_str in prices_df.columns:
    y = target_df[[code_str]].rename(columns={code_str: 'target'})
    X_fins = fins[fins['code'] == code_str].sort_values('Date')
    
    merged_df = pd.merge_asof(
        y.sort_index().reset_index(), # 'Date'をインデックスから列に戻す
        X_fins, 
        on='Date', 
        direction='backward'
    )
    merged_df['code'] = code_str
    all_dfs.append(merged_df)

# --- 修正C：公正な 'dropna' (ごちゃまzeルール) ---
final_dataset = pd.concat(all_dfs).dropna(subset=['target'])

# --- 6. 結果の確認と保存 ---
if final_dataset.empty:
    print("\n❌ エラー: 最終データセットが空です。")
else:
    print("\n🎉 V1 (比率のみ) 公正比較版データセットの作成に成功！")
    output_filename = 'analysis_dataset_v1_RATIO.csv'
    final_dataset.to_csv(output_filename, index=False)
    print(f"\n📄 データを '{output_filename}' として保存しました。")
    print(final_dataset.head())
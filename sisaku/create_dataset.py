import pandas as pd
import numpy as np

print("--- V3 (テクニカル指標) 公正比較版 ---")
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

# --- 4. テクニカル特徴量 (X_tech) の作成 ---
print("テクニカル特徴量（60日移動平均乖離率）を作成しています...")
# 60日移動平均を計算 (君が65%を出した期間)
ma_60 = prices_df.rolling(window=60).mean()
# 乖離率を計算 = (現在の株価 - 60日平均) / 60日平均
tech_features_df = (prices_df - ma_60) / ma_60
# 扱いやすいように列名を変更
tech_features_df = tech_features_df.rename(columns=lambda c: f"{c}_MAdivergence60")

# --- 5. 財務特徴量 (X_fin) の作成 ---
print("財務特徴量（ROE, 自己資本比率）を作成しています...")
# ★★★ 特徴量問題の解決 ★★★
# Profit, NetSalesのような「生の数字」は、結合する前に落とす
required_columns = [
    'LocalCode', 'DisclosedDate', 'Profit', 'TotalAssets', 'Equity'
]
fins = financials_df[required_columns].copy()
fins['ROE'] = np.where(fins['Equity'] > 0, fins['Profit'] / fins['Equity'], np.nan)
fins['SelfCapitalRatio'] = np.where(fins['TotalAssets'] > 0, fins['Equity'] / fins['TotalAssets'], np.nan)
fins = fins.rename(columns={'LocalCode': 'code', 'DisclosedDate': 'Date'})
fins['code'] = fins['code'].astype(str) # get_financials.pyで修正済みのはずだが、念のため

# ★★★ 必要な「比率」だけを残す ★★★
fins = fins[['code', 'Date', 'ROE', 'SelfCapitalRatio']]

# --- 6. データセットの結合 ---
print("全データを結合しています...")
all_dfs = []
for code_str in prices_df.columns:
    y = target_df[[code_str]].rename(columns={code_str: 'target'})
    # ★★★ Close（生の株価）はもう使わない ★★★
    tech_col = tech_features_df[[f"{code_str}_MAdivergence60"]].rename(columns={f"{code_str}_MAdivergence60": 'MAdivergence60'})
    
    X_fins = fins[fins['code'] == code_str].sort_values('Date')
    
    # [y, Tech] をまず結合
    y_tech_df = pd.concat([y, tech_col], axis=1)
    
    # [y, Tech] と [財務] を結合
    merged_df = pd.merge_asof(
        y_tech_df.sort_index().reset_index(), # Dateを列に戻す
        X_fins, 
        on='Date', 
        direction='backward'
    )
    all_dfs.append(merged_df)

# --- 7. 最終化 (ごちゃまぜルール) ---
final_dataset = pd.concat(all_dfs).dropna(subset=['target'])

# --- 8. 結果の確認と保存 ---
if final_dataset.empty:
    print("\n❌ エラー: 最終データセットが空です。")
else:
    print("\n🎉 最終分析用データセット V3 (比率のみ) の作成に成功！")
    output_filename = 'analysis_dataset_v3_technical.csv'
    final_dataset.to_csv(output_filename, index=False)
    print(f"\n📄 データを '{output_filename}' として保存しました。")
    print(final_dataset.head())
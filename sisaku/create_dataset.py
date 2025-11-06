import pandas as pd
import numpy as np

# ★★★ テクニカル指標の期間をここで定義 ★★★
# 短期・中期・長期（田村氏の論文も参考に 25, 60, 75日を採用）
WINDOWS = [25, 50,75] 

print(f"--- V6 (マルチウィンドウ) 賢いフィルター版 ---")
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
print("テクニカル特徴量（3期間 x 4種類）を作成しています...")

# 結合するテクニカル特徴量を格納するリスト
tech_features_list = []
tech_feature_names = [] # 特徴量名を管理するリスト

# 1. 複数の移動平均乖離率
for window in WINDOWS:
    ma = prices_df.rolling(window=window).mean()
    tech_ma = (prices_df - ma) / ma
    col_name = f'MAdivergence_{window}'
    tech_ma = tech_ma.rename(columns=lambda c: f"{c}_{col_name}")
    tech_features_list.append(tech_ma)
    tech_feature_names.append(col_name)

# 2. 複数のRSI
delta = prices_df.diff(1)
for window in WINDOWS:
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    tech_rsi = 100 - (100 / (1 + rs))
    col_name = f'RSI_{window}'
    tech_rsi = tech_rsi.rename(columns=lambda c: f"{c}_{col_name}")
    tech_features_list.append(tech_rsi)
    tech_feature_names.append(col_name)
    
# 3. 複数のボラティリティ
for window in WINDOWS:
    tech_vol = prices_df.pct_change().rolling(window=window).std()
    col_name = f'Volatility_{window}'
    tech_vol = tech_vol.rename(columns=lambda c: f"{c}_{col_name}")
    tech_features_list.append(tech_vol)
    tech_feature_names.append(col_name)

# 4. 複数のモメンタム
for window in WINDOWS:
    tech_mom = prices_df.pct_change(periods=window)
    col_name = f'Momentum_{window}'
    tech_mom = tech_mom.rename(columns=lambda c: f"{c}_{col_name}")
    tech_features_list.append(tech_mom)
    tech_feature_names.append(col_name)

# --- 5. 財務特徴量 (X_fin) の作成 ---
print("財務特徴量を作成しています...")
# (財務特徴量の作成部分はV5と同じ)
required_columns = [
    'LocalCode', 'DisclosedDate', 'NetSales', 'OperatingProfit', 'Profit', 
    'TotalAssets', 'Equity',
    'NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock'
]
fins = financials_df[required_columns].copy()
fins['ROE'] = np.where(fins['Equity'] > 0, fins['Profit'] / fins['Equity'], np.nan)
fins['SelfCapitalRatio'] = np.where(fins['TotalAssets'] > 0, fins['Equity'] / fins['TotalAssets'], np.nan)
fins['OpProfitMargin'] = np.where(fins['NetSales'] > 0, fins['OperatingProfit'] / fins['NetSales'], np.nan)
fins['EPS'] = fins['Profit'] / fins['NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock']
fins['BPS'] = fins['Equity'] / fins['NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock']
fins = fins.rename(columns={'LocalCode': 'code', 'DisclosedDate': 'Date'})
fins['code'] = fins['code'].astype(str)
fins = fins[['code', 'Date', 'ROE', 'SelfCapitalRatio', 'OpProfitMargin', 'EPS', 'BPS']]

# --- 6. データセットの結合 ---
print("全データを結合しています...")
all_dfs = []
for code_str in prices_df.columns:
    y = target_df[[code_str]].rename(columns={code_str: 'target'})
    price_col = prices_df[[code_str]].rename(columns={code_str: 'Close'})
    
    # 全テクニカル指標を動的に結合
    tech_cols = [
        tech_df[[f"{code_str}_{name}"]].rename(columns={f"{code_str}_{name}": name})
        for tech_df, name in zip(tech_features_list, tech_feature_names)
    ]
    
    X_fins = fins[fins['code'] == code_str].sort_values('Date')
    
    y_price_tech_df = pd.concat([y, price_col] + tech_cols, axis=1)
    
    merged_df = pd.merge_asof(
        y_price_tech_df.sort_index().reset_index(), # Dateを列に戻す
        X_fins, 
        on='Date', 
        direction='backward'
    )
    
    merged_df['PER'] = merged_df['Close'] / merged_df['EPS']
    merged_df['PBR'] = merged_df['Close'] / merged_df['BPS']
    
    all_dfs.append(merged_df)

# --- 7. 最終化 (賢いフィルター) ---
print("計算不能な行と未来の行を削除しています...")
final_dataset = pd.concat(all_dfs).dropna(
    subset=['target', 'ROE','MAdivergence_75'] # 君の「賢いフィルター」ルール
)

# --- 8. 結果の確認と保存 ---
if final_dataset.empty:
    print("\n❌ エラー: 最終データセットが空です。")
else:
    print("\n🎉 最終分析用データセット V6 (マルチテクニカル) の作成に成功！")
    output_filename = 'analysis_dataset_v6_multi_window.csv'
    final_dataset.to_csv(output_filename, index=False)
    print(f"\n📄 データを '{output_filename}' として保存しました。")
    print(final_dataset.head())
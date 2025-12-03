import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. データ読み込み
# ---------------------------------------------------------
input_file = "stock_prices_topix100_simple.csv"
df_prices = pd.read_csv(input_file, index_col=0, parse_dates=True)

# ---------------------------------------------------------
# 2. 「ハイテク・グロース」セクターの定義
# ---------------------------------------------------------
# 金利上昇局面(2022)で売られやすい銘柄群を手動定義
# (半導体, 電機, キーエンス, リクルート, SBGなど)
tech_growth_tickers = [
    "80350", # 東京エレクトロン (半導体)
    "68570", # アドバンテスト (半導体)
    "69200", # レーザーテック (半導体)
    "67230", # ルネサス (半導体)
    "68610", # キーエンス (FAセンサ/高PER)
    "69540", # ファナック (FA)
    "69810", # 村田製作所 (電子部品)
    "67580", # ソニーG (電機/エンタメ)
    "65940", # ニデック (電機)
    "99840", # ソフトバンクG (投資会社/ハイテク)
    "60980", # リクルート (サービス/高PER)
    "79740", # 任天堂 (ゲーム)
    "96130", # NTTデータ (SIer)
    "67020", # 富士通 (IT)
    "67620", # TDK (電子部品)
    "77410", # HOYA (精密)
    "24130", # エムスリー (医療IT)
    "46890"  # LINEヤフー (IT)
]

# データセットにある銘柄だけに絞る (生存確認)
available_tickers = [t for t in tech_growth_tickers if t in df_prices.columns]
print(f"🎯 分析対象: ハイテク・グロース株 {len(available_tickers)} 銘柄")
print(f"   (リスト: {available_tickers})")

if len(available_tickers) < 5:
    print("❌ 銘柄数が少なすぎます。データセットを確認してください。")
    exit()

# 抽出
df_sector = df_prices[available_tickers]

# ---------------------------------------------------------
# 3. RMT計算 (セクター内部相関)
# ---------------------------------------------------------
df_log_returns = np.log(df_sector / df_sector.shift(1)).dropna()

def get_max_eigenvalue(window_df):
    corr_mat = window_df.corr().fillna(0)
    vals, _ = np.linalg.eigh(corr_mat)
    return vals[-1]

WINDOW = 60
rmt_dates = []
max_eigenvalues = []

print("🧮 セクターRMTを計算中...")

for i in range(WINDOW, len(df_log_returns)):
    window_data = df_log_returns.iloc[i-WINDOW : i]
    lambda_max = get_max_eigenvalue(window_data)
    rmt_dates.append(df_log_returns.index[i])
    max_eigenvalues.append(lambda_max)

# 正規化された固有値 (銘柄数Nで割ることで、全銘柄版と比較可能にする)
# Max Eigenvalue / N = 「説明率(%)」に近い概念になります
ts_rmt_sector = pd.Series(max_eigenvalues, index=rmt_dates)
ts_rmt_normalized = ts_rmt_sector / len(available_tickers)

# ---------------------------------------------------------
# 4. 可視化 (2021年〜2023年にフォーカス)
# ---------------------------------------------------------
# 比較用にセクター平均株価も作成
sector_index = df_sector.mean(axis=1)
sector_index = sector_index / sector_index.iloc[0]

fig, ax1 = plt.figure(figsize=(14, 8)), plt.gca()

# 期間を絞ってズーム (ウクライナ/利上げ局面)
zoom_start = "2021-01-01"
zoom_end = "2023-12-31"

ax1.set_xlim(pd.to_datetime(zoom_start), pd.to_datetime(zoom_end))

# [左軸] セクター平均株価
color_price = 'tab:blue'
ax1.plot(sector_index.index, sector_index, color=color_price, alpha=0.6, label='Sector Index (Tech/Growth)')
ax1.set_ylabel('Sector Price', color=color_price)

# [右軸] RMT最大固有値 (正規化済み)
ax2 = ax1.twinx()
color_rmt = 'tab:red'
# わかりやすくするため移動平均をかけてスムージングしても良いが、今回は生で
ax2.plot(ts_rmt_normalized.index, ts_rmt_normalized, color=color_rmt, alpha=0.9, linewidth=2, label='Sector Sync (Eigen/N)')
ax2.set_ylabel('Sector Synchronization Strength', color=color_rmt, fontweight='bold')

# イベント
events = {
    '2021-11-01': 'Fed Tapering Talk', # 緩和縮小示唆
    '2022-01-05': 'Fed Minutes (Shock)', # 利上げ加速懸念
    '2022-02-24': 'Ukraine Invasion',
    '2022-06-15': '0.75% Rate Hike', # ジャイアント・キリング
    '2022-10-13': 'CPI Shock (Bottom)'
}

for date, label in events.items():
    try:
        d = pd.to_datetime(date)
        plt.axvline(x=d, color='black', linestyle='--', alpha=0.5)
        plt.text(d, ax2.get_ylim()[1]*0.95, f' {label}', rotation=90)
    except: pass

plt.title("Hypothesis Validation: High-Tech/Growth Sector Synchronization (2021-2023)")
plt.show()
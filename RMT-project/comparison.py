import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 日本語フォント設定（環境に合わせて変更してください。豆腐化する場合はEnglish推奨）
# import japanize_matplotlib 
plt.rcParams['figure.figsize'] = (12, 6)

def analyze_topix_reproduction():
    print("📊 データを読み込んでいます...")
    
    # 1. データの読み込み
    try:
        # 自作の時価総額データ
        df_mc = pd.read_csv("market_caps.csv", index_col=0, parse_dates=True)
        # 公式TOPIXデータ
        df_real = pd.read_csv("real_topix.csv", index_col=0, parse_dates=True)
    except FileNotFoundError as e:
        print(f"❌ ファイルが見つかりません: {e}")
        print("前のステップでCSVが正しく保存されているか確認してください。")
        return

    # 2. 自作インデックスの合成 (単純時価総額加重)
    # 行方向(axis=1)に合計して、その日の「全銘柄時価総額」を算出
    my_index_series = df_mc.sum(axis=1)
    
    # 3. データの結合 (日付を合わせる)
    # 本物TOPIXのCloseカラムを使う (名前を Real_TOPIX に変更)
    real_series = df_real.iloc[:, 0] # 1列目を採用
    
    df_compare = pd.DataFrame({
        'My_Proxy': my_index_series,
        'Real_TOPIX': real_series
    }).dropna() # 両方のデータがある日だけ残す

    if df_compare.empty:
        print("❌ 共通する日付のデータがありません。")
        return

    # =========================================================
    # 4. 統計指標の計算
    # =========================================================
    
    # 日次リターン (変化率) の計算
    df_returns = df_compare.pct_change().dropna()
    
    # (A) 相関係数 (Correlation)
    # 1.0に近いほど「値動きの方向」が一致している
    correlation = df_returns.corr().iloc[0, 1]
    
    # (B) 年率標準偏差 (Annualized Volatility)
    # 日次標準偏差 × √250日
    volatility = df_returns.std() * np.sqrt(250) * 100
    
    # (C) ベータ値 (Beta)
    # Myポートフォリオの感応度 (Covariance / Variance)
    cov_matrix = df_returns.cov()
    beta = cov_matrix.iloc[0, 1] / df_returns['Real_TOPIX'].var()

    print("\n" + "="*40)
    print(f" 🧪 検証結果 ({len(df_compare)} 営業日)")
    print("="*40)
    print(f"🔹 相関係数 (Correlation) : {correlation:.4f}  (1.0に近いほど完璧)")
    print(f"🔹 ベータ値 (Beta)        : {beta:.4f}  (市場に対する感応度)")
    print("-" * 40)
    print(f"🔹 年率リスク (標準偏差):")
    print(f"   - My Proxy   : {volatility['My_Proxy']:.2f}%")
    print(f"   - Real TOPIX : {volatility['Real_TOPIX']:.2f}%")
    print("="*40 + "\n")

    # =========================================================
    # 5. 正規化と可視化
    # =========================================================
    
    # 開始日を100として正規化 (Normalize to 100)
    df_normalized = (df_compare / df_compare.iloc[0]) * 100
    
    # グラフ描画
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # 上段: 正規化チャート
    ax1.plot(df_normalized.index, df_normalized['Real_TOPIX'], label='Real TOPIX (Official)', color='black', linewidth=1.5, alpha=0.8)
    ax1.plot(df_normalized.index, df_normalized['My_Proxy'], label='My Reproduction (Proxy)', color='dodgerblue', linewidth=1.5)
    
    ax1.set_title(f"Replication Check: TOPIX vs My Proxy (Normalized)\nCorr: {correlation:.3f} | Beta: {beta:.3f}", fontsize=14)
    ax1.set_ylabel("Normalized Value (Start=100)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 下段: 乖離率 (Tracking Errorの視覚化)
    # (My / Real) - 1 で、何%ズレているかを表示
    spread = (df_normalized['My_Proxy'] / df_normalized['Real_TOPIX']) - 1
    
    ax2.plot(spread.index, spread, color='crimson', linewidth=1, label='Spread (My / Real - 1)')
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_title("Spread / Deviation Ratio", fontsize=10)
    ax2.set_ylabel("Deviation")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # 軸のフォーマット
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_topix_reproduction()
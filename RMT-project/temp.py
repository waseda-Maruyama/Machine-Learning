import pandas as pd
import numpy as np

def find_outliers_2017():
    print("🔍 2017年のデータ異常を調査します...")
    
    # 1. データ読み込み
    try:
        df_mc = pd.read_csv("market_caps.csv", index_col=0, parse_dates=True)
    except Exception as e:
        print(f"❌ 読み込みエラー: {e}")
        return

    # 2. 2017年に絞り込む
    df_2017 = df_mc.loc['2017-01-01':'2017-12-31']
    
    if df_2017.empty:
        print("⚠️ 2017年のデータがありません。")
        return

    # 3. 全体の時価総額と、前日比(差額)を計算
    total_mc = df_2017.sum(axis=1)
    
    # 前日との差額 (Total Market Cap Diff)
    total_diff = total_mc.diff()
    
    # 変動が大きかった日トップ5を抽出 (絶対値ベース)
    # ※乖離がいきなり+10%とかになった日を探すため
    top_volatiles = total_diff.abs().nlargest(5)
    
    print(f"\n📅 2017年で時価総額が激しく動いた日 TOP5")
    print("="*60)
    
    for date, change_amount in top_volatiles.items():
        date_str = date.strftime('%Y-%m-%d')
        
        # その日の「個別銘柄の変動額」を計算
        # (当日 - 前日)
        day_idx = df_2017.index.get_loc(date)
        if day_idx == 0: continue
        
        prev_date = df_2017.index[day_idx - 1]
        
        # 全銘柄の差分を計算
        stock_diffs = df_2017.loc[date] - df_2017.loc[prev_date]
        
        # その日の変動要因になった銘柄トップ3 (絶対値でソート)
        culprits = stock_diffs.abs().sort_values(ascending=False).head(3)
        
        # 表示
        direction = "📈 増加" if total_diff.loc[date] > 0 else "📉 減少"
        print(f"\n🗓 {date_str} | 全体変動: {change_amount:,.0f} ({direction})")
        print(f"   👇 主な要因銘柄:")
        
        for code, diff_val in culprits.items():
            # 寄与率: その銘柄の変動 / 全体の変動
            contribution = (diff_val / total_diff.loc[date]) * 100
            
            # その銘柄の時価総額が前日からどう変わったか
            val_prev = df_2017.loc[prev_date, code]
            val_curr = df_2017.loc[date, code]
            pct = ((val_curr - val_prev) / val_prev) * 100 if val_prev != 0 else np.nan
            
            print(f"     Code {code}: 変動額 {diff_val:,.0f} (前日比 {pct:+.1f}%) | 寄与率: {contribution:.1f}%")
            print(f"         (前日: {val_prev:,.0f} -> 当日: {val_curr:,.0f})")

    print("\n" + "="*60)
    print("💡 ヒント:")
    print("もし '前日比 +100%' や '+900%' (10倍) のような銘柄があれば、")
    print("株式分割や併合のデータ反映タイミングが、株価と発行済み株式数でズレています。")

if __name__ == "__main__":
    find_outliers_2017()
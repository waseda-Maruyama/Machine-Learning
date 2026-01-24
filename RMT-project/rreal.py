import os
import time
import pandas as pd
import numpy as np
import jquantsapi
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
import traceback

# =========================================================
# 1. 初期設定 & 認証
# =========================================================
print("🚀 初期設定を開始します...")
load_dotenv(find_dotenv('J-Quants.env'))
try:
    cli = jquantsapi.Client(mail_address=os.getenv("JQUANTS_EMAIL"), password=os.getenv("JQUANTS_PASSWORD"))
    print("✅ J-Quants API 認証成功")
except Exception as e:
    print(f"❌ 認証エラー: {e}")
    exit()

START_DATE = "20160130"
END_DATE = "20260115"
START_DATE_FIN = "20160130"

def fetch_with_retry(func, name, max_retries=3, wait_sec=2, **kwargs):
    for i in range(max_retries):
        try:
            return func(**kwargs)
        except Exception as e:
            if i == max_retries - 1:
                print(f"  ❌ {name} APIエラー (最終試行失敗): {e}")
                return pd.DataFrame()
            time.sleep(wait_sec)
    return pd.DataFrame()




  # =========================================================
# 4. 正解データ (TOPIX指数) の取得と保存
# =========================================================
print("\n📈 公式TOPIX指数を取得して保存します...")

try:
    # J-Quants APIには指数取得用の専用エンドポイントがあります
    # TOPIXのコードは一般的に省略されますが、API仕様に従い取得します
    df_topix_real = cli.get_indices_topix(
        from_yyyymmdd=START_DATE,
        to_yyyymmdd=END_DATE
    )
    
    if not df_topix_real.empty:
        # 日付型変換と整形
        df_topix_real['Date'] = pd.to_datetime(df_topix_real['Date'])
        df_topix_real = df_topix_real.set_index('Date').sort_index()
        
        # 終値 (Close) だけを取り出す
        series_topix_real = pd.to_numeric(df_topix_real['Close'], errors='coerce')
        
        # 保存
        series_topix_real.to_csv("real_topix.csv")
        print("✅ real_topix.csv を保存しました (比較用正解データ)")
        
        # ---------------------------------------------------------
        # (オプション) 簡易検証: 再現データの合計 vs 本物TOPIX の相関を表示
        # ---------------------------------------------------------
        if 'df_mc' in locals() and not df_mc.empty:
            # 取得した銘柄（TOPIX100など）の時価総額合計を計算
            my_topix_proxy = df_mc.sum(axis=1)
            
            # 日付を合わせて結合
            comparison = pd.DataFrame({
                'My_Sum': my_topix_proxy,
                'Real_TOPIX': series_topix_real
            }).dropna()
            
            if not comparison.empty:
                # リターン（変化率）での相関を確認
                # ※時価総額の絶対値は浮動株調整や除数の関係でTOPIX値とは一致しないため、動き(リターン)で比較します
                corr = comparison.pct_change().corr().iloc[0, 1]
                print(f"📊 [検証] 作成した時価総額合計とTOPIXの相関係数: {corr:.4f}")
                print("   (1.0に近いほど値動きが完全に一致しています)")
            else:
                print("⚠️ 日付がマッチせず比較できませんでした")
            
    else:
        print("⚠️ TOPIXデータが空でした。期間設定などを確認してください。")

except Exception as e:
    print(f"❌ TOPIX取得エラー: {e}")
    # エラー詳細表示
    traceback.print_exc()

print("\n🎉 全工程が完了しました！")  
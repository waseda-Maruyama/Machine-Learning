import pandas as pd
import numpy as np
import os

# =========================================================
# ⚙️ 設定
# =========================================================
INPUT_FILE = "prediction_comparison.csv"
THRESHOLD = 0.3  # 危険度50%で「売り抜け（Exit）」と判断

# 評価対象イベント
events = {
    "2020 Covid":  ("2020-02-01", "2020-04-30"),
    "2024 Ueda":   ("2024-07-01", "2024-08-31"),
    "2025 Tariff": ("2025-03-01", "2025-05-31")
}

# =========================================================
# 1. データ読み込み
# =========================================================
if not os.path.exists(INPUT_FILE):
    print("❌ ファイルがありません。train_comparison.py を実行してください。")
    exit()

df = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)

print(f"💰 損失回避シミュレーション (脱出閾値: {THRESHOLD:.0%})")
print("=" * 100)
# 表のヘッダー
print(f"{'Event':<12} | {'Model':<8} | {'Result':<8} | {'Peak':<8} | {'Exit':<8} | {'Bottom':<8} | {'Avoided Loss':<12}")
print("-" * 100)

# =========================================================
# 2. イベントごとの損益計算
# =========================================================
for name, (s_str, e_str) in events.items():
    s_dt, e_dt = pd.to_datetime(s_str), pd.to_datetime(e_str)
    
    # 期間外チェック
    if s_dt > df.index.max(): continue
    subset = df.loc[s_dt:e_dt]
    if len(subset) == 0: continue
    
    # この期間の「天井(Peak)」と「大底(Bottom)」
    peak_price = subset['Market_Price'].max()
    bottom_price = subset['Market_Price'].min()
    peak_date = subset['Market_Price'].idxmax()
    bottom_date = subset['Market_Price'].idxmin()
    
    # 最大下落幅 (Max Drawdown)
    max_drop = (bottom_price - peak_price) / peak_price
    
    # 3モデル比較
    for model_char in ['A', 'B', 'C']:
        col = f"Prob_{model_char}"
        model_name = f"Model {model_char}"
        
        # アラートが出た日を探す
        alerts = subset[subset[col] >= THRESHOLD]
        
        result_str = "❌ Hold"   # 逃げ遅れ
        exit_price_str = "-"
        avoided_str = f"Lost {max_drop:.1%}" # そのままホールドして食らった
        
        if len(alerts) > 0:
            # 最初にアラートが出た日を「脱出日」とする
            first_alert_date = alerts.index[0]
            
            # ただし、底を打った後のアラートは意味がないので除外判定してもいいが
            # ここでは純粋に「その日の価格」で計算する
            exit_price = subset.loc[first_alert_date, 'Market_Price']
            
            # 回避できた損失 = (脱出価格 - 大底価格) / 脱出価格
            # プラスなら「安くなる前に逃げられた」
            # マイナスなら「底で狼狽売りした（最悪）」
            avoided_loss = (exit_price - bottom_price) / exit_price
            
            # 天井からどれくらい下がったところで気づいたか？
            dd_at_exit = (exit_price - peak_price) / peak_price
            
            # 表示用フォーマット
            exit_price_str = f"{exit_price:.0f}"
            
            if avoided_loss > 0:
                result_str = "✅ Exit"
                avoided_str = f"SAVE {avoided_loss:.1%}"
            else:
                result_str = "⚠️ Late"
                avoided_str = f"Bad Exit"
        
        # 結果出力
        # Peak: 天井価格, Exit: 脱出価格, Bottom: 大底価格
        print(f"{name:<12} | {model_name:<8} | {result_str:<8} | {peak_price:<8.0f} | {exit_price_str:<8} | {bottom_price:<8.0f} | {avoided_str:<12}")
        
    print("-" * 100)
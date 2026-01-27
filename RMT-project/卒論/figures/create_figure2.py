import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
import japanize_matplotlib  # 日本語化ライブラリ

# 論文用にフォントサイズなどを調整
plt.rcParams.update({'font.size': 12})

# --- データ作成 ---
x = np.linspace(-10, 10, 2000)
x_log = np.logspace(0, 2, 1000)

# 1. ガウス分布 (指数関数的に減衰 / べき乗則なし)
pdf_gauss = norm.pdf(x, 0, 1)
ccdf_gauss = 2 * (1 - norm.cdf(x_log))

# 2. 逆三乗則 (実証的な分布: alpha=3 / 分散は有限)
# 自由度3のt分布は、裾において alpha=3 のべき乗減衰を示す
pdf_alpha3 = t.pdf(x, df=3)
ccdf_alpha3 = 2 * (1 - t.cdf(x_log, df=3))

# 3. レヴィ安定分布 (数学的な定義: alpha=1.5 / 分散は無限大)
# 自由度1.5のt分布は、裾において alpha=1.5 のべき乗減衰を示す
pdf_levy = t.pdf(x, df=1.5)
ccdf_levy = 2 * (1 - t.cdf(x_log, df=1.5))

# 6シグマの確率（両側）
p_gauss_6s = 2 * (1 - norm.cdf(6))
p_alpha3_6s = 2 * (1 - t.cdf(6, df=3))

# ---------------------------------------------------------
# 図1: 確率密度関数 (PDF) の比較 - 線形スケール
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(x, pdf_gauss, label='ガウス分布 (正規分布)', color='black', linestyle='--')
plt.plot(x, pdf_alpha3, label=r'逆三乗則 ($\alpha \approx 3$)', color='blue', linewidth=2)
plt.plot(x, pdf_levy, label=r'レヴィ安定分布 ($\alpha = 1.5$)', color='red', linestyle=':')

# 6σ境界
plt.axvline(6, color='blue', linestyle=':', label='6σ')
plt.axvline(-6, color='blue', linestyle=':')

# 注釈（6σの発生確率）
plt.annotate(
	f'ガウス: {p_gauss_6s:.1e}',
	xy=(6, norm.pdf(6, 0, 1)),
	xytext=(3.0, 0.20),
	arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5)
)
plt.annotate(
	f'逆三乗: {p_alpha3_6s:.1e}',
	xy=(6, t.pdf(6, df=3)),
	xytext=(3.0, 0.08),
	arrowprops=dict(facecolor='blue', shrink=0.05, width=1, headwidth=5)
)


plt.xlabel('$x$ (標準偏差 $\sigma$)')
plt.ylabel('確率密度')
plt.ylim(0, 0.45)
plt.xlim(-8, 8)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_2_1a.pdf')  # 論文用PDF保存
plt.show()

# ---------------------------------------------------------
# 図2: 裾の確率の両対数プロット - べき乗則の証明
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.loglog(x_log, ccdf_gauss, label='ガウス分布 (指数減衰 / 下方に曲落)', color='black', linestyle='--')
plt.loglog(x_log, ccdf_alpha3, label=r'逆三乗則 ($\propto x^{-3}$ / 直線的)', color='blue', linewidth=2)
plt.loglog(x_log, ccdf_levy, label=r'レヴィ安定分布 ($\propto x^{-1.5}$ / 直線的)', color='red', linestyle=':')


plt.xlabel('$x$ (対数スケール)')
plt.ylabel('超過確率 (対数スケール)')
plt.legend()
plt.grid(True, which="both", linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig('fig_2_1b.pdf')
plt.show()
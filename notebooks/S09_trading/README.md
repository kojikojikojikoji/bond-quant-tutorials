# S9 相対価値・トレーディング

| Notebook | テーマ |
|---|---|
| `nb_0901_rich_cheap` | カーブ対比 rich/cheap 分析（NSS残差・AR(1)半減期） |
| `nb_0902_butterfly` | バタフライ取引の構築と PnL 分解（50-50/risk/PCA） |
| `nb_0903_carry_roll` | キャリーとロールダウン（スポット−フォワード・BEI） |
| `nb_0904_futures_basis` | 国債先物ベーシスと CTD（コンバージョンファクター・インプライドレポ） |
| `nb_0905_backtest` | バックテスト基盤と戦略評価（ルックアヘッド検査・コスト感応度） |

`bondlab.bt`（backtest / performance / conversion_factor）と `bondlab.analytics`（pca /
duration_convexity）、`bondlab.curve`（fit_nss）を使う。バックテスターは翌営業日約定で
先読みを防ぎ、PnL 分解は説明率で検証する。合成 JGB ユニバースは data/samples に置く。

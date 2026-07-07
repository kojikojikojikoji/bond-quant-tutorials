# S3 リスク指標

| Notebook | テーマ |
|---|---|
| `nb_0301_duration_convexity` | デュレーション（マコーレー/修正/実効）とコンベクシティ |
| `nb_0302_dv01_hedge` | DV01・BPV・PV01 とヘッジ比率、非平行シフトの残存PnL |
| `nb_0303_key_rate_duration` | キーレートデュレーション（テナー別分解・バーベル/ブレット） |
| `nb_0304_scenario_pca` | シナリオ分析・PCA（レベル/スロープ/曲率）・ストレステスト |
| `nb_0305_var_es` | VaR と Expected Shortfall、劣加法性、Kupiec 検定 |

`bondlab.analytics`（duration_convexity / effective_duration / bump_curve / pca）と
`bondlab.risk`（VaR/ES/Kupiec）を使う。デュレーション・コンベクシティは QuantLib と
機械精度で突合し、KRD は「合計=並行DV01」で整合を確認する。

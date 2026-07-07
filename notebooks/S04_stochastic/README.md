# S4 確率過程の数学

| Notebook | テーマ |
|---|---|
| `nb_0401_brownian_motion` | ブラウン運動・2次変分・スケーリング極限 |
| `nb_0402_ito` | 伊藤積分・伊藤の公式・GBM/OU |
| `nb_0403_sde_schemes` | SDE数値解法（Euler-Maruyama / Milstein・強/弱収束） |
| `nb_0404_monte_carlo` | モンテカルロ法と分散削減（対照変量/制御変量/Sobol） |
| `nb_0405_risk_neutral` | リスク中立評価・ニュメレール変換・Girsanov |

`bondlab.sim`（brownian_paths / simulate_sde / mc_stats / control_variate）を使う。
解析解（GBM/OU/Black-Scholes）と QuantLib（OU過程・欧州オプション）で突合し、
収束次数（EM 0.5 / Milstein 1.0）と 1/√n 収束を数値で確認する。乱数は seed 固定。

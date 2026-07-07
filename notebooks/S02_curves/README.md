# S2 イールドカーブ構築

| Notebook | テーマ |
|---|---|
| `nb_0201_bootstrap` | パー利回りからのブートストラップ（ゼロ/フォワード/パーの関係） |
| `nb_0202_interpolation` | 補間法の比較（線形/log-DF/三次スプライン/単調凸 Hagan-West） |
| `nb_0203_nelson_siegel` | Nelson-Siegel / Svensson パラメトリックフィット |
| `nb_0204_multicurve` | マルチカーブ（OIS割引・デュアルブートストラップ・SOFR先物凸性調整） |
| `nb_0205_pipeline` | 実データ日次カーブ構築パイプライン（品質検査・フォールバック） |

`bondlab.curve`（DiscountCurve / bootstrap_par / nss / fit_nss）を使い、
ブートストラップはパー再現、NSS は解析 round-trip、補間は QuantLib と突合して検証する。
合成サンプルは `data/samples/synthetic_ust_par_curve.csv` と `..._par_panel.csv`。

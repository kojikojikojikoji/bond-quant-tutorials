# S10 ポートフォリオ

| Notebook | テーマ |
|---|---|
| `nb_1001_immunization` | イミュナイゼーションと ALM（Redington 3条件・キーレート免疫化） |
| `nb_1002_index_replication` | インデックス複製（層化サンプリング・セル法・KRD マッチング） |
| `nb_1003_optimization` | 債券ポートフォリオ最適化（ファクターリスクモデル・シャドープライス） |

`bondlab.analytics`（duration_convexity / bump_curve）と scipy.optimize / cvxpy を使う。
免疫化は ±100bp でサープラス保存を確認し、複製は in/out-of-sample のトラッキングエラーで
評価、最適化はシャドープライスの相補スラック性で検証する。

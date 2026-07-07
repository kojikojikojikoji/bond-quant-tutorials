# S8 証券化商品（オプション回・米系運用志望向け）

| Notebook | テーマ |
|---|---|
| `nb_0801_mbs_prepayment` | MBS の仕組みとプリペイメント（CPR/SMM/PSA・WAL・ネガコン） |
| `nb_0802_mc_oas` | モンテカルロ OAS（金利パス×プリペイメント・オプションコスト） |
| `nb_0803_japan_rmbs` | 日本の証券化市場（機構 MBS・団信・日米プリペイメント差） |

`bondlab.mbs`（psa_cpr / cpr_to_smm / mbs_cashflows / weighted_average_life）を使う。
PSA ramp と WAL は独立計算で検算し、MC-OAS はプリペイメントゼロで Z スプレッドと
一致することを確認する。S8 はコアの一部ではなくオプション（志望先に応じて選択）。

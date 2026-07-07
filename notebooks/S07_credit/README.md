# S7 クレジット

| Notebook | テーマ |
|---|---|
| `nb_0701_hazard` | ハザードレートとサバイバル確率・リスキー割引 |
| `nb_0702_cds` | CDS プライシングとハザードカーブのブートストラップ・CS01 |
| `nb_0703_merton` | Merton 構造モデル（距離 to default・PD・KMV） |
| `nb_0704_credit_spread` | クレジットスプレッド分析（G/I/OAS/ASW） |
| `nb_0705_cva` | CVA 入門（オプション回・銀行XVA志望向け） |

`bondlab.credit`（HazardCurve / cds_par_spread / bootstrap_hazard / Merton）を使う。
CDS は QuantLib の CreditDefaultSwap と 0.02bp 以内で一致、ハザードブートストラップは
往復で厳密再現する。S7-5 CVA は S5 の金利 MC と S7 のハザードを組み合わせた総合演習。

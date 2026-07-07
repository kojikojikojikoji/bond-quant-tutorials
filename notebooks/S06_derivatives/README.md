# S6 金利デリバティブ

| Notebook | テーマ |
|---|---|
| `nb_0601_swaps` | FRA と金利スワップ（パーレート・マルチカーブ・DV01） |
| `nb_0602_caps_floors` | キャップ/フロア（Black'76・Bachelier・フラット/スポットボラ） |
| `nb_0603_swaptions` | スワップション（アニュイティ測度・Jamshidian 分解） |
| `nb_0604_callable_oas` | コーラブル債と OAS（後退帰納・負のコンベクシティ・後半の山場） |
| `nb_0605_sabr` | ボラティリティサーフェスと SABR（Hagan 近似・スマイル） |

`bondlab.pricing`（swap / black76 / bachelier / swaption_black / sabr_vol）を使う。
Black'76・Bachelier・SABR・スワップは QuantLib と機械精度で突合、コーラブル債は
自作 HW ツリーと QuantLib CallableFixedRateBond を数セント以内で突合する。

# S5 短期金利モデル

| Notebook | テーマ |
|---|---|
| `nb_0501_vasicek` | Vasicek（アフィンZCB解析解・OU厳密サンプリング） |
| `nb_0502_cir` | CIR（Feller条件・非中心カイ二乗・厳密サンプリング） |
| `nb_0503_hull_white_tree` | Hull-White① 三項ツリーと初期カーブ再現 |
| `nb_0504_hw_calibration` | Hull-White② スワップションキャリブレーション（前半の山場） |
| `nb_0505_model_comparison` | モデル比較とモデルリスク（Vasicek/CIR/HW・多因子の見取り図） |

`bondlab.models`（Vasicek / CIR / HullWhite）を使う。Vasicek・CIR の ZCB は
QuantLib と機械精度で突合、HW は初期カーブを厳密再現し三項ツリーの評価が解析 ZCB と
一致する。S5-4 のキャリブレーションは合成ボラ面で既知パラメータの復元を主検証に据える。

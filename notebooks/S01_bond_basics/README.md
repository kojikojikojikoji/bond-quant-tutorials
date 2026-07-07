# S1 債券の基礎

| Notebook | テーマ |
|---|---|
| `nb_0101_compounding` | 現在価値と複利規約（年/半年/連続の変換、割引係数） |
| `nb_0102_daycount` | Day Count Conventions（ACT/360・ACT/365F・30/360・ACT/ACT）と日付計算 |
| `nb_0103_price_yield` | 債券価格とYTM（価格式の導出、Newton/Brent 求解、凸性） |
| `nb_0104_clean_dirty` | クリーン/ダーティ価格・経過利子・国際慣行（ACT/ACT の ISDA vs ISMA） |

各 notebook は jupytext の `.py`（percent・レビュー対象）と実行済み `.ipynb` を両方置く。
`bondlab.rates` / `bondlab.daycount` / `bondlab.bond` を使い、すべて QuantLib と機械精度で突合する。

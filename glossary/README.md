# 用語集

`glossary/` を用語の唯一の正とする。各 notebook 末尾の用語表は、ここから
スクリプトで自動生成して埋め込む（手書きしない）。

各用語は次の構造で記す。カード学習ツールへ変換しやすいよう、用語と定義の
ペアを崩さない。

```
### 用語（English）
- 定義: 一文の定義。
- 数式: 必要なら LaTeX。
- 関連: 別の用語
- 初出: S2-1
```

## 分野別ファイル

| ファイル | 分野 |
|---|---|
| [00_tooling.md](00_tooling.md) | 環境・データ基盤（S0） |
| [01_bond_basics.md](01_bond_basics.md) | 債券の基礎 |
| [02_curves.md](02_curves.md) | イールドカーブ構築 |
| [03_risk.md](03_risk.md) | リスク指標 |
| [04_stochastic.md](04_stochastic.md) | 確率過程の数学 |
| [05_rate_models.md](05_rate_models.md) | 短期金利モデル |
| [06_derivatives.md](06_derivatives.md) | 金利デリバティブ |
| [07_credit.md](07_credit.md) | クレジット |
| [08_securitization.md](08_securitization.md) | 証券化商品 |
| [09_trading.md](09_trading.md) | 相対価値・トレーディング |
| [10_portfolio.md](10_portfolio.md) | ポートフォリオ |

## 索引

全用語の五十音索引・英語索引は `scripts/build_glossary_index.py` で
このファイル末尾に生成する（現時点は雛形のため未生成）。

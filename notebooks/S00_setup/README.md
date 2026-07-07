# S0 環境構築・ツールキット

| Notebook | テーマ |
|---|---|
| `nb_0001_environment` | Python環境・ライブラリ概観・bondlab レイヤー・QuantLib 日付/営業日 |
| `nb_0002_market_data` | 市場データ取得基盤（FRED・財務省・JSDA・BoE）、.env・キャッシュ・欠損処理 |

各 notebook は jupytext のペア `.ipynb` と `.py`（percent 形式）を両方コミットする。
レビューは `.py` の差分で行う。`.ipynb` は実行済み出力つき。

## 実行方法

```bash
pip install -e ".[dev]"
cp .env.example .env          # FRED_API_KEY は任意（無くても合成サンプルで動く）
export PYTHONPATH=$PWD
jupyter lab notebooks/S00_setup/
```

鍵が無い場合、`nb_0002` は `data/samples/` の合成サンプルで動作する。

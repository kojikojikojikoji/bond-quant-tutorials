# bond-quant-tutorials

債券クオンツに必要な理論と実装を、スクラッチ実装と QuantLib 検証の両輪で学ぶ
Jupyter チュートリアル。全 notebook を進めると、自作ライブラリ `bondlab`
（カーブ構築・金利モデル・デリバティブ評価・リスク指標・バックテスト）が
手元に残る。

> **English summary**
> A Jupyter tutorial series for fixed-income quant skills. Every model is
> implemented from scratch in NumPy/SciPy, then cross-checked against QuantLib.
> Working through the notebooks builds `bondlab`, a small pricing library
> covering curve construction, short-rate models, rate derivatives, risk
> metrics, relative-value analytics, and backtesting.

## 構成

| シリーズ | テーマ |
|---|---|
| S0 | 環境構築・市場データ取得基盤（FRED / 財務省 / JSDA / BoE） |
| S1 | 債券の基礎（価格⇔利回り・日付計算） |
| S2 | イールドカーブ構築（ブートストラップ〜マルチカーブ） |
| S3 | リスク指標（DV01・KRD・シナリオ・VaR/ES） |
| S4 | 確率過程の数学（伊藤・SDE・モンテカルロ・リスク中立評価） |
| S5 | 短期金利モデル（Vasicek・CIR・Hull-White） |
| S6 | 金利デリバティブ（スワップ〜コーラブル債 OAS） |
| S7 | クレジット（ハザード・CDS・Merton、CVA はオプション） |
| S8 | 証券化商品（MBS・プリペイメント。オプション） |
| S9 | 相対価値・トレーディング（rich/cheap・バタフライ・ベーシス・バックテスト） |
| S10 | ポートフォリオ（免疫化・複製・最適化） |
| S11 | Capstone（ライブラリ整理・RFQ プライサー・ダッシュボード） |

全 50 notebook（コア 46 ＋ オプション 4：S7-5 CVA と S8 証券化 3 本）。各シリーズの
詳細は各フォルダの README を参照。ライブラリ API は
[`docs/bondlab_api.md`](docs/bondlab_api.md)。

すべての notebook は実行済み出力つき（jupytext の `.ipynb`＋`.py` ペア）で、
自作実装は QuantLib と機械精度で突合する。`bondlab` は 61 個のユニットテストで
守られている（`pytest -q`）。

## 使い方

```bash
git clone <this repo>
cd bond-quant-tutorials
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export PYTHONPATH=$PWD

cp .env.example .env      # FRED_API_KEY は任意（無くても合成サンプルで動く）
pytest -q                 # 61 tests
jupyter lab notebooks/
```

各 notebook は先頭のデータ取得セルで、キャッシュがあればキャッシュを、なければ
API から取得する。FRED キーが無くても `data/samples/` の合成データで動く。

## アプリ（Capstone）

- `app/rfq_pricer.py` — RFQ プライサー（FastAPI）: `uvicorn app.rfq_pricer:app`
- `app/dashboard.py` — 日次モニタリング（Streamlit）: `streamlit run app/dashboard.py`

## 学習の補助

- `glossary/` — 分野別の用語集（定義の正）
- `interview/` — 各シリーズの面接想定問答
- `solutions/` — 演習の解答

## データと再現性

- 生の市場データは再配布しない。リポジトリには取得スクリプトのみ置く。
- 入手できないデータ（スワップションボラ・市販インデックス構成）は合成データ
  生成コードで代替する。`data/samples/` に出所・生成方法を明記する。

## ライセンス

- コード（`bondlab/`, `tests/`, `scripts/`, notebook のコードセル）: MIT（`LICENSE`）
- 文章（markdown セル・`glossary/`・`interview/`・README）: CC BY 4.0（`LICENSE-docs`）

## 秘密情報

API キーは `.env`（`.gitignore` 対象）にのみ置く。notebook では `python-dotenv`
経由で読み、コード・出力セルに書き込まない。

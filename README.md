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

コア 46 本 ＋ オプション 4 本。詳細な制作要件・週次スケジュールは企画書
（別管理）にまとめている。

## 使い方

```bash
git clone <this repo>
cd bond-quant-tutorials
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env      # FRED_API_KEY を記入
```

各 notebook は先頭のデータ取得セルで、キャッシュがあればキャッシュを、
なければ API から取得する。取得データは `data/cache/`（Git 管理外）に置く。

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

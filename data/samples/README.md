# data/samples

鍵の無い読者でも notebook が動くよう、再配布可能な教材データだけを置く。
生の市場データ（FRED・JSDA 等）はここに置かず、`data/cache/`（Git 管理外）に
取得する。

## 収録データ

| ファイル | 種別 | 出所・生成方法 | 単位 | 追加日 |
|---|---|---|---|---|
| `synthetic_us10y.csv` | 合成 | `scripts/make_sample_yields.py`（seed=1）が生成。実データではない | % | 2026-07-07 |
| `synthetic_jp10y.csv` | 合成 | `scripts/make_sample_yields.py`（seed=2）が生成。実データではない | % | 2026-07-07 |

## 注意

- これらは平均回帰つきランダムウォークで作った**合成系列**であり、実際の
  市場水準・変動を再現するものではない。水準そのものに意味はない。
- 決定論的に生成しているため、`scripts/make_sample_yields.py` を実行すれば
  同じファイルが再生成される。
- 実データで再現したい場合は、`.env` に `FRED_API_KEY` を設定し、各 notebook の
  `USE_LIVE` パスを使う。

# S11 Capstone

| Notebook | テーマ |
|---|---|
| `nb_1101_bondlab_library` | bondlab ライブラリの整理とテスト（全層の通し実演・QuantLib突合一覧） |
| `nb_1102_rfq_pricer` | RFQ プライサー（ミッド推定・在庫スキュー・FastAPI アプリ） |
| `nb_1103_dashboard` | 日次カーブモニタリングダッシュボード（Streamlit アプリ） |

ここまでの全層（data/rates/daycount/bond/curve/analytics/risk/sim/models/pricing/
credit/mbs/bt）を1つのライブラリとして通しで使い、就職活動で見せられる形にまとめる。
API リファレンスは `docs/bondlab_api.md`。アプリは `app/rfq_pricer.py`（FastAPI）と
`app/dashboard.py`（Streamlit）。

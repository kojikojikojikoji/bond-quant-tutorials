# app

Capstone（S11）で作るアプリ。ライブラリ bondlab を実サービスの形に載せた例。

## RFQ プライサー（FastAPI）

```bash
pip install -e ".[app]"
uvicorn app.rfq_pricer:app --reload
# POST /quote に {bond_id, size, side} を投げると bid/ask が返る
```

## モニタリングダッシュボード（Streamlit）

```bash
pip install -e ".[app]"
streamlit run app/dashboard.py
```

いずれも合成サンプル（data/samples）で動く。実データは .env に FRED_API_KEY を設定。

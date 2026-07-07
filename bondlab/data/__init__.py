"""市場データ取得・キャッシュ層。

S0-2 で構築する最小のデータアクセス層。設計方針は次の3点。

1. 秘密情報は .env からのみ読む（コード・出力に書かない）。
2. 取得結果はローカルにキャッシュし、再実行と再現性を担保する。
3. CI や鍵の無い環境では API を叩かず、キャッシュのみで動く
   （cache_only モード）。キャッシュも無ければ明示的に例外を出す。

現状は FRED（米セントルイス連銀）の時系列取得に対応する。財務省・JSDA・BoE
のダウンローダは後続シリーズで同じインターフェースの上に足していく。
"""
from __future__ import annotations

import io
import os
import time
from pathlib import Path

import pandas as pd

# リポジトリ直下の data/cache/ を既定のキャッシュ先とする。
_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[2]
CACHE_DIR = REPO_ROOT / "data" / "cache"

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def _load_dotenv_key(name: str) -> str | None:
    """.env から鍵を読む。python-dotenv があれば使い、無ければ簡易パーサで読む。"""
    val = os.getenv(name)
    if val:
        return val
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == name:
            return v.strip().strip('"').strip("'") or None
    return None


def _cache_path(series_id: str) -> Path:
    return CACHE_DIR / f"fred_{series_id}.csv"


def _rel(path: Path) -> str:
    """絶対パスをリポジトリ相対に丸める。ローカルパスの漏えいを防ぐ。"""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def fred_series(
    series_id: str,
    *,
    cache_only: bool | None = None,
    max_retries: int = 3,
    timeout: float = 15.0,
) -> pd.Series:
    """FRED の時系列を取得して pandas.Series（index=日付, 値=float）で返す。

    Parameters
    ----------
    series_id : str
        FRED の系列 ID（例 "DGS10" は米国債10年 CMT 利回り）。
    cache_only : bool, optional
        True ならネットワークを使わずキャッシュだけを読む。None のときは
        環境変数 BONDLAB_DATA_MODE == "cache_only" を見る（CI 用）。
    max_retries : int
        API 一時障害時の再試行回数（指数バックオフ）。
    timeout : float
        1リクエストのタイムアウト秒。

    Returns
    -------
    pandas.Series
        欠損（FRED は "." で表す）は除外済み。float 型。

    Notes
    -----
    値の単位は系列に依存する（利回り系列はパーセント）。呼び出し側で
    単位を把握すること。
    """
    if cache_only is None:
        cache_only = os.getenv("BONDLAB_DATA_MODE") == "cache_only"

    cache = _cache_path(series_id)
    if cache.exists():
        cached = _read_cache(cache)
        if cache_only:
            return cached
        # キャッシュがあれば再取得せずそのまま使う（再現性優先）。
        return cached

    if cache_only:
        raise RuntimeError(
            f"cache_only モードだがキャッシュが無い: {_rel(cache)} . "
            f"先にネットワーク接続のある環境で fred_series({series_id!r}) を実行するか、"
            f"data/samples の合成データを使うこと。"
        )

    api_key = _load_dotenv_key("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY が未設定。.env に FRED_API_KEY を記入するか、"
            "cache_only=True で合成データ/キャッシュを使うこと。"
        )

    raw = _download_fred(series_id, api_key, max_retries=max_retries, timeout=timeout)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(raw, encoding="utf-8")
    return _read_cache(cache)


def _download_fred(series_id: str, api_key: str, *, max_retries: int, timeout: float) -> str:
    """FRED API を叩いて CSV 文字列を返す。requests をこの場で import する。"""
    import requests  # 依存を関数内に閉じ込め、オフライン検証を容易にする

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(FRED_BASE, params=params, timeout=timeout)
            resp.raise_for_status()
            obs = resp.json()["observations"]
            df = pd.DataFrame(obs)[["date", "value"]]
            return df.to_csv(index=False)
        except Exception as err:  # noqa: BLE001 一時障害を握って再試行する
            last_err = err
            time.sleep(2 ** attempt)
    raise RuntimeError(f"FRED 取得に失敗: {series_id} ({last_err})")


def _read_cache(path: Path) -> pd.Series:
    """キャッシュ CSV を読み、スキーマ検証のうえ Series を返す。"""
    df = pd.read_csv(path)
    expected = {"date", "value"}
    if set(df.columns) != expected:
        raise ValueError(f"キャッシュのスキーマ不正 {path}: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"])
    # FRED は欠損を "." で表す。数値化して落とす。
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).set_index("date")["value"].astype(float)
    df.name = path.stem.replace("fred_", "")
    return df


def load_sample(name: str) -> pd.Series:
    """data/samples に置いた合成/再配布可データを読む（鍵不要）。

    公開リポジトリで鍵の無い読者でも notebook が動くよう、教材用の小さな
    サンプルを CSV（列: date,value）で用意しておき、ここから読む。
    """
    path = REPO_ROOT / "data" / "samples" / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"サンプルが無い: {path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["value"].astype(float)
    s.name = name
    return s

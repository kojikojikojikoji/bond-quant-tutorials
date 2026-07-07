"""bondlab.data の回帰テスト（S0-2）。ネットワークは使わない。"""
import pytest

from bondlab import data


def test_load_sample():
    s = data.load_sample("synthetic_us10y")
    assert len(s) > 1000
    assert s.index.is_monotonic_increasing
    assert s.dtype.kind == "f"


def test_cache_only_without_cache_raises(tmp_path, monkeypatch):
    # 存在しない系列を cache_only で読むとキャッシュが無く例外になる。
    monkeypatch.setattr(data, "CACHE_DIR", tmp_path)
    with pytest.raises(RuntimeError):
        data.fred_series("NONEXISTENT_SERIES_ID", cache_only=True)


def test_missing_sample_raises():
    with pytest.raises(FileNotFoundError):
        data.load_sample("does_not_exist")

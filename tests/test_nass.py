import json
import os

import pytest

from src import nass

GRAIN = "CORN, GRAIN - PROGRESS, MEASURED IN PCT HARVESTED"
SILAGE = "CORN, SILAGE - PROGRESS, MEASURED IN PCT HARVESTED"


def record(state="IA", week="2026-10-18", value="45", desc=GRAIN, year="2026"):
    return {"short_desc": desc, "state_alpha": state, "week_ending": week, "Value": value, "year": year}


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self.payload = payload or {}
        self.text = text

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.last_params = None

    def get(self, url, params, timeout):
        self.calls += 1
        self.last_params = params
        return self.response


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(nass, "CACHE_DIR", tmp_path / "nass")


def test_normalize_keeps_only_grain_and_parses_values():
    df = nass.normalize([
        record(value="45"),
        record(week="2026-10-25", value="1,000"),
        record(desc=SILAGE),
        record(value="(D)"),
    ])
    assert len(df) == 2
    assert df["pct_harvested"].tolist() == [45.0, 1000.0]
    assert str(df["week_ending"].dtype).startswith("datetime64")


def test_normalize_drops_weeks_before_the_harvest_season():
    df = nass.normalize([
        record(state="ND", week="2026-04-12", value="95"),
        record(state="ND", week="2026-10-18", value="30"),
    ])
    assert df["week_ending"].tolist() == [nass.pd.Timestamp("2026-10-18")]


def test_normalize_drops_duplicate_state_weeks():
    df = nass.normalize([record(), record()])
    assert len(df) == 1


def test_normalize_empty_returns_expected_columns():
    df = nass.normalize([])
    assert list(df.columns) == ["state_code", "year", "week_ending", "pct_harvested"]
    assert df.empty


def test_fetch_year_sends_filters_and_caches():
    session = FakeSession(FakeResponse(payload={"data": [record()]}))
    first = nass.fetch_year(2026, session, api_key="secret-key")
    second = nass.fetch_year(2026, session, api_key="secret-key")

    assert first == second
    assert session.calls == 1
    assert session.last_params["commodity_desc"] == "CORN"
    assert session.last_params["unit_desc"] == "PCT HARVESTED"
    assert session.last_params["year"] == 2026
    assert session.last_params["agg_level_desc"] == "STATE"


def test_fetch_year_national_level_uses_its_own_request_and_cache():
    session = FakeSession(FakeResponse(payload={"data": [record(state="US")]}))
    nass.fetch_year(2026, session, api_key="k", level="STATE")
    nass.fetch_year(2026, session, api_key="k", level="NATIONAL")
    assert session.calls == 2
    assert session.last_params["agg_level_desc"] == "NATIONAL"


def test_fetch_progress_combines_state_and_national_series():
    session = FakeSession(FakeResponse(payload={"data": [record(state="IA")]}))
    df = nass.fetch_progress([2026], session, api_key="k")
    assert session.calls == 2
    assert set(df["state_code"]) == {"IA"}


def test_fetch_year_does_not_store_api_key_in_cache():
    session = FakeSession(FakeResponse(payload={"data": [record()]}))
    nass.fetch_year(2026, session, api_key="secret-key")
    cached = (nass.CACHE_DIR / "corn_harvest_state_2026.json").read_text()
    assert "secret-key" not in cached


def test_fetch_year_treats_no_data_response_as_empty():
    session = FakeSession(FakeResponse(status_code=400, text='{"error":["bad request - no data"]}'))
    assert nass.fetch_year(2026, session, api_key="k") == []


def test_fetch_year_error_message_does_not_leak_key():
    session = FakeSession(FakeResponse(status_code=500, text="boom"))
    with pytest.raises(RuntimeError) as error:
        nass.fetch_year(2026, session, api_key="secret-key")
    assert "secret-key" not in str(error.value)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("NASS_API_KEY", raising=False)
    monkeypatch.setattr(nass, "load_dotenv", lambda: None)
    with pytest.raises(RuntimeError, match="NASS_API_KEY"):
        nass.get_api_key()


@pytest.mark.network
@pytest.mark.skipif(not os.getenv("NASS_API_KEY"), reason="NASS_API_KEY not set")
def test_nass_service_returns_2026_corn_harvest_progress():
    df = nass.normalize(nass.fetch_year(2026))
    assert not df.empty
    assert df["pct_harvested"].between(0, 100).all()
    assert {"IA", "IL"} <= set(df["state_code"])


@pytest.mark.network
@pytest.mark.skipif(not os.getenv("NASS_API_KEY"), reason="NASS_API_KEY not set")
def test_nass_service_returns_national_series():
    df = nass.normalize(nass.fetch_year(2026, level="NATIONAL"))
    assert not df.empty
    assert set(df["state_code"]) == {"US"}

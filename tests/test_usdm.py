from datetime import date

import pytest

from src import usdm


def record(fips="19153", map_date="20260707", d0=80.0, d1=60.0, d2=30.0, d3=5.0, d4=0.0, **extra):
    return {"MapDate": map_date, "FIPS": fips, "None": 100 - d0, "D0": d0, "D1": d1, "D2": d2, "D3": d3, "D4": d4, **extra}


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text="[]"):
        self.status_code = status_code
        self.payload = payload if payload is not None else []
        self.text = text

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = 0
        self.last_params = None
        self.last_headers = None

    def get(self, url, params, headers, timeout):
        self.calls += 1
        self.last_params = params
        self.last_headers = headers
        return self.response


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(usdm, "CACHE_DIR", tmp_path / "usdm")


def test_normalize_parses_levels_dates_and_dsci():
    df = usdm.normalize([record(ValidStart="2026-07-07", ValidEnd="2026-07-13")])
    row = df.iloc[0]
    assert row["county_fips"] == "19153"
    assert row["d1"] == 60.0
    assert row["week_start"] == usdm.pd.Timestamp("2026-07-07")
    assert row["week_end"] == usdm.pd.Timestamp("2026-07-13")
    assert row["dsci"] == pytest.approx(175.0)


def test_normalize_falls_back_to_map_date_when_valid_dates_missing():
    df = usdm.normalize([record(map_date="20260707")])
    assert df.iloc[0]["week_start"] == usdm.pd.Timestamp("2026-07-07")
    assert df.iloc[0]["week_end"] == usdm.pd.Timestamp("2026-07-13")


def test_normalize_pads_short_fips_and_accepts_lowercase_keys():
    lowercase = {k.lower(): v for k, v in record(fips="1001").items()}
    df = usdm.normalize([lowercase])
    assert df.iloc[0]["county_fips"] == "01001"


def test_normalize_empty_returns_expected_columns():
    df = usdm.normalize([])
    assert list(df.columns) == usdm.COLUMNS
    assert df.empty


def test_fetch_county_sends_expected_request_and_caches():
    session = FakeSession(FakeResponse(payload=[record()], text="[{}]"))
    start, end = date(2026, 1, 1), date(2026, 10, 30)

    first = usdm.fetch_county("19153", start, end, session)
    second = usdm.fetch_county("19153", start, end, session)

    assert first == second
    assert session.calls == 1
    assert session.last_params == {"aoi": "19153", "startdate": "1/1/2026", "enddate": "10/30/2026", "statisticsType": 1}
    assert session.last_headers == {"Accept": "application/json"}


def test_fetch_county_treats_empty_body_as_no_records():
    session = FakeSession(FakeResponse(payload=None, text=""))
    assert usdm.fetch_county("19153", date(2026, 1, 1), date(2026, 2, 1), session) == []


def test_fetch_county_raises_on_http_error():
    session = FakeSession(FakeResponse(status_code=500))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        usdm.fetch_county("19153", date(2026, 1, 1), date(2026, 2, 1), session)


def test_fetch_weekly_combines_counties_and_skips_empty_ones():
    class PerCountySession:
        def get(self, url, params, headers, timeout):
            if params["aoi"] == "99999":
                return FakeResponse(payload=[], text="")
            return FakeResponse(payload=[record(fips=params["aoi"])], text="[{}]")

    df = usdm.fetch_weekly(["19153", "31109", "19153", "99999"], date(2026, 1, 1), date(2026, 2, 1), PerCountySession())
    assert sorted(df["county_fips"].unique()) == ["19153", "31109"]


@pytest.mark.network
def test_usdm_service_returns_cumulative_county_statistics():
    df = usdm.normalize(usdm.fetch_county("19153", date(2026, 6, 1), date(2026, 8, 31)))
    assert not df.empty
    assert (df["county_fips"] == "19153").all()
    assert (df["d0"] >= df["d1"]).all() and (df["d1"] >= df["d2"]).all()
    assert df[usdm.LEVELS].apply(lambda col: col.between(0, 100)).all().all()

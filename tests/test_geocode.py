import pandas as pd
import pytest

from src import geocode

IOWA = {"County": {"FIPS": "19153", "name": "Polk"}, "State": {"FIPS": "19", "code": "IA"}, "status": "OK"}
OUTSIDE_US = {"County": {"FIPS": None, "name": None}, "State": {"FIPS": None, "code": None}, "status": "OK"}


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def get(self, url, params, timeout):
        self.calls += 1
        return FakeResponse(self.payload)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(geocode, "CACHE_DIR", tmp_path / "fcc")


def test_parse_response_extracts_county_and_state():
    assert geocode.parse_response(IOWA) == {
        "county_fips": "19153", "county_name": "Polk", "state_fips": "19", "state_code": "IA",
    }


def test_parse_response_handles_point_outside_us():
    assert geocode.parse_response(OUTSIDE_US)["county_fips"] is None


def test_lookup_caches_response_on_disk():
    session = FakeSession(IOWA)
    first = geocode.lookup(41.59, -93.60, session)
    second = geocode.lookup(41.59, -93.60, session)
    assert first == second
    assert session.calls == 1


def test_add_geography_flags_trials_outside_us():
    trials = pd.DataFrame({
        "trial_id": [1, 2],
        "latitude": [41.59, 49.90],
        "longitude": [-93.60, -97.14],
        "dq_flags": ["", "invalid_maturity_zone"],
    })
    fake = lambda lat, lon: geocode.parse_response(IOWA if lat < 45 else OUTSIDE_US)
    result = geocode.add_geography(trials, fetch=fake)

    assert result["is_us"].tolist() == [True, False]
    assert result["state_code"].iloc[0] == "IA"
    assert pd.isna(result["state_code"].iloc[1])
    assert result["dq_flags"].tolist() == ["", "invalid_maturity_zone;not_in_us"]


@pytest.mark.network
def test_fcc_service_resolves_real_coordinates(monkeypatch):
    iowa = geocode.lookup(41.5911, -93.6037)
    assert iowa["state_code"] == "IA"
    assert iowa["county_fips"].startswith("19")

    winnipeg = geocode.lookup(49.8951, -97.1384)
    assert winnipeg["county_fips"] is None

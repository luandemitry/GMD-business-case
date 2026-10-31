from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src import geocode, pipeline

FIXTURE = Path(__file__).parent / "fixtures" / "trials_sample.csv"
AS_OF = date(2026, 10, 30)


def fake_geo(lat, lon):
    if lon > -60:
        return geocode.parse_response({"County": {"FIPS": None}, "State": {"FIPS": None, "code": None}})
    return geocode.parse_response({"County": {"FIPS": "19153", "name": "Polk"}, "State": {"FIPS": "19", "code": "IA"}})


def fake_progress(years):
    return pd.DataFrame({
        "state_code": "IA",
        "year": 2026,
        "week_ending": pd.to_datetime(["2026-09-27", "2026-10-04", "2026-10-11", "2026-10-18", "2026-10-25"]),
        "pct_harvested": [5.0, 15.0, 40.0, 60.0, 80.0],
    })


def fake_drought(counties, start, end):
    return pd.DataFrame({
        "county_fips": "19153",
        "week_start": pd.to_datetime(["2026-07-07", "2026-07-14"]),
        "week_end": pd.to_datetime(["2026-07-13", "2026-07-20"]),
        "d0": [90.0, 95.0], "d1": [60.0, 70.0], "d2": [10.0, 20.0], "d3": [0.0, 0.0], "d4": [0.0, 0.0],
        "dsci": [160.0, 185.0],
    })


def build_database(tmp_path: Path) -> tuple[dict, Path]:
    db = tmp_path / "breeder.duckdb"
    tables = pipeline.run(
        raw_path=FIXTURE, season_year=2026, as_of=AS_OF, db_path=db, processed_dir=tmp_path / "processed",
        geo_fetch=fake_geo, progress_fetch=fake_progress, drought_fetch=fake_drought,
    )
    return tables, db


@pytest.fixture
def insights_db(tmp_path) -> Path:
    return build_database(tmp_path)[1]

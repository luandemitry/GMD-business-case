from datetime import date
from pathlib import Path

import duckdb
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


@pytest.fixture
def built(tmp_path):
    db = tmp_path / "breeder.duckdb"
    tables = pipeline.run(
        raw_path=FIXTURE, season_year=2026, as_of=AS_OF, db_path=db, processed_dir=tmp_path / "processed",
        geo_fetch=fake_geo, progress_fetch=fake_progress, drought_fetch=fake_drought,
    )
    return tables, db, tmp_path


def query(db, sql):
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.sql(sql).df()
    finally:
        con.close()


def test_view_has_one_row_per_clean_trial(built):
    tables, db, _ = built
    assert query(db, "SELECT count(*) AS n FROM trial_insights")["n"][0] == len(tables["trials"]) == 9


def test_view_answers_the_three_business_questions(built):
    _, db, _ = built
    rows = query(db, "SELECT * FROM trial_insights").set_index("trial_id")

    assert rows.loc[1, "drought_status"] == "drought"
    assert rows.loc[9, "drought_status"] == "no_data"
    assert rows.loc[9, "drought_note"] == "outside the US"
    assert rows.loc[1, "harvest_timing_vs_area"] == "typical"
    assert rows.loc[1, "usda_pct_at_trial_harvest"] == pytest.approx(36.4, abs=0.1)
    assert pd.notna(rows.loc[2, "pred_harvest_date"])
    assert pd.isna(rows.loc[1, "pred_harvest_date"])


def test_outside_us_trial_is_flagged_and_has_no_usda_context(built):
    _, db, _ = built
    row = query(db, "SELECT * FROM trial_insights WHERE trial_id = 9").iloc[0]
    assert "not_in_us" in row["dq_flags"]
    assert pd.isna(row["nass_geo"])


def test_all_supporting_tables_are_written(built):
    _, db, tmp_path = built
    tables = set(query(db, "SELECT table_name FROM information_schema.tables")["table_name"])
    assert {"trials", "drought_pollination", "harvest_comparison", "harvest_prediction",
            "harvest_backtest", "usda_harvest_progress", "usdm_weekly", "data_quality",
            "pipeline_run", "trial_insights"} <= tables
    assert (tmp_path / "processed" / "trials.parquet").exists()


def test_temporary_database_file_is_replaced_atomically(built):
    _, db, _ = built
    assert db.exists()
    assert not db.with_suffix(".building").exists()


def test_rerun_overwrites_existing_database(built, tmp_path):
    _, db, _ = built
    pipeline.run(
        raw_path=FIXTURE, season_year=2026, as_of=AS_OF, db_path=db, processed_dir=tmp_path / "processed",
        geo_fetch=fake_geo, progress_fetch=fake_progress, drought_fetch=fake_drought,
    )
    assert query(db, "SELECT count(*) AS n FROM pipeline_run")["n"][0] == 1

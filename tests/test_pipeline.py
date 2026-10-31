import duckdb
import pandas as pd
import pytest

from tests.conftest import build_database


@pytest.fixture
def built(tmp_path):
    tables, db = build_database(tmp_path)
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
            "harvest_backtest", "harvest_backtest_errors", "usda_harvest_progress", "usdm_weekly",
            "data_quality", "pipeline_run", "trial_insights"} <= tables
    assert (tmp_path / "processed" / "trials.parquet").exists()


def test_temporary_database_file_is_replaced_atomically(built):
    _, db, _ = built
    assert db.exists()
    assert not db.with_suffix(".building").exists()


def test_rerun_overwrites_existing_database(built):
    _, db, tmp_path = built
    build_database(tmp_path)
    assert query(db, "SELECT count(*) AS n FROM pipeline_run")["n"][0] == 1

import os
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv

from src import clean_trials, drought, geocode, harvest, nass, predict, usdm

RAW_PATH = Path("data/raw/trials.csv")
PROCESSED_DIR = Path("data/processed")
DB_PATH = Path("data/breeder.duckdb")

VIEW_SQL = """
CREATE VIEW trial_insights AS
SELECT
    t.trial_id, t.trial_name, t.maturity_zone, t.state_code AS state, t.county_name AS county,
    t.latitude, t.longitude, t.is_us, t.planting_date, t.harvest_date, t.harvest_status,
    t.days_to_harvest, t.dq_flags,
    d.pollination_date, d.window_start AS pollination_window_start, d.window_end AS pollination_window_end,
    d.drought_status, d.drought_during_pollination, d.max_pct_d0, d.max_pct_d1, d.max_pct_d2,
    d.max_dsci, d.drought_note,
    h.nass_geo, h.nass_national_fallback, h.usda_pct_at_trial_harvest, h.days_vs_usda_50pct,
    h.harvest_timing_vs_area, h.usda_date_50pct, h.usda_latest_pct, h.usda_latest_week,
    p.pred_harvest_date, p.pred_agronomic_date, p.pred_usda_date, p.pred_confidence, p.pred_status
FROM trials t
LEFT JOIN drought_pollination d ON d.trial_id = t.trial_id
LEFT JOIN harvest_comparison h ON h.trial_id = t.trial_id
LEFT JOIN harvest_prediction p ON p.trial_id = t.trial_id
"""


def build_tables(
    raw_path: Path,
    season_year: int,
    as_of: date,
    geo_fetch=geocode.lookup,
    progress_fetch=nass.fetch_progress,
    drought_fetch=usdm.fetch_weekly,
) -> dict[str, pd.DataFrame]:
    raw = clean_trials.load_raw(raw_path)
    trials = geocode.add_geography(clean_trials.clean(raw), fetch=geo_fetch)

    progress = progress_fetch([season_year])
    counties = trials.loc[trials["is_us"], "county_fips"].tolist()
    weekly = drought_fetch(counties, date(season_year, 1, 1), min(date(season_year, 12, 31), as_of))

    comparison = harvest.compare(trials, progress)
    run_info = pd.DataFrame([{
        "run_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season_year": season_year,
        "as_of_date": as_of.isoformat(),
        "trial_count": len(trials),
    }])

    return {
        "trials": trials,
        "drought_pollination": drought.drought_at_pollination(trials, weekly),
        "harvest_comparison": comparison,
        "harvest_prediction": predict.predict_pending(trials, comparison, progress, as_of),
        "harvest_backtest": predict.backtest(trials, comparison, progress, as_of),
        "usda_harvest_progress": progress,
        "usdm_weekly": weekly,
        "data_quality": clean_trials.quality_summary(raw, trials),
        "pipeline_run": run_info,
    }


def write_outputs(tables: dict[str, pd.DataFrame], db_path: Path, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = db_path.with_suffix(".building")
    temp_path.unlink(missing_ok=True)
    con = duckdb.connect(str(temp_path))
    for name, table in tables.items():
        table.to_parquet(processed_dir / f"{name}.parquet", index=False)
        con.register("source_table", table)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM source_table")
        con.unregister("source_table")
    con.execute(VIEW_SQL)
    con.close()
    temp_path.replace(db_path)


def run(
    raw_path: Path = RAW_PATH,
    season_year: int = 2026,
    as_of: date | None = None,
    db_path: Path = DB_PATH,
    processed_dir: Path = PROCESSED_DIR,
    **fetchers,
) -> dict[str, pd.DataFrame]:
    tables = build_tables(raw_path, season_year, as_of or date.today(), **fetchers)
    write_outputs(tables, db_path, processed_dir)
    return tables


def print_summary(tables: dict[str, pd.DataFrame]) -> None:
    at_harvest = tables["harvest_comparison"]["usda_pct_at_trial_harvest"].dropna()
    sections = {
        "data quality": tables["data_quality"].set_index("check")["rows"],
        "% of area harvested when each trial was harvested": at_harvest.describe().round(1),
        f"trials harvested before the area started (<{predict.MIN_TARGET_PCT}%)": pd.Series({"trials": int((at_harvest < predict.MIN_TARGET_PCT).sum())}),
        "geocoding: trials in the US": tables["trials"]["is_us"].value_counts(),
        "drought status at pollination": tables["drought_pollination"]["drought_status"].value_counts(),
        "harvest timing vs USDA area": tables["harvest_comparison"]["harvest_timing_vs_area"].value_counts(dropna=False),
        "NASS series used (True = national fallback)": tables["harvest_comparison"]["nass_national_fallback"].value_counts(dropna=False),
        "prediction status (pending trials)": tables["harvest_prediction"]["pred_status"].value_counts(),
        "prediction confidence (pending trials)": tables["harvest_prediction"]["pred_confidence"].value_counts(dropna=False),
    }
    for title, counts in sections.items():
        print(f"\n== {title}")
        print(counts.to_string())
    print("\n== backtest (leave-one-out, days)")
    print(tables["harvest_backtest"].to_string(index=False))


def main() -> None:
    load_dotenv()
    print_summary(run(season_year=int(os.getenv("SEASON_YEAR", "2026"))))


if __name__ == "__main__":
    main()

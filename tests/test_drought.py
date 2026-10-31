import pandas as pd
import pytest

from src.drought import classify, drought_at_pollination


def trials_frame(**overrides):
    base = {
        "trial_id": 1,
        "planting_date": pd.Timestamp("2026-05-01"),
        "county_fips": "19153",
        "is_us": True,
    }
    return pd.DataFrame([{**base, **overrides}])


def weekly_frame(rows):
    columns = ["county_fips", "week_start", "week_end", "d0", "d1", "d2", "d3", "d4", "dsci"]
    return pd.DataFrame(rows, columns=columns).assign(
        week_start=lambda df: pd.to_datetime(df["week_start"]),
        week_end=lambda df: pd.to_datetime(df["week_end"]),
    )


def week(start, end, d0=0, d1=0, d2=0, d3=0, d4=0, fips="19153"):
    return (fips, start, end, d0, d1, d2, d3, d4, d0 + d1 + d2 + d3 + d4)


def test_pollination_date_is_planting_plus_70_days():
    result = drought_at_pollination(trials_frame(), weekly_frame([]))
    assert result.loc[0, "pollination_date"] == pd.Timestamp("2026-07-10")
    assert result.loc[0, "window_start"] == pd.Timestamp("2026-07-03")
    assert result.loc[0, "window_end"] == pd.Timestamp("2026-07-17")


def test_classify_uses_highest_level_above_threshold():
    assert classify({"d0": 90, "d1": 70, "d2": 55, "d3": 0, "d4": 0}, 50) == "severe_drought"
    assert classify({"d0": 90, "d1": 70, "d2": 10, "d3": 0, "d4": 0}, 50) == "drought"
    assert classify({"d0": 90, "d1": 20, "d2": 0, "d3": 0, "d4": 0}, 50) == "abnormally_dry"
    assert classify({"d0": 10, "d1": 0, "d2": 0, "d3": 0, "d4": 0}, 50) == "no_drought"


def test_drought_uses_peak_across_weeks_overlapping_the_window():
    weekly = weekly_frame([
        week("2026-06-23", "2026-06-29", d0=100, d1=100, d2=100),
        week("2026-07-07", "2026-07-13", d0=80, d1=30),
        week("2026-07-14", "2026-07-20", d0=90, d1=60),
    ])
    result = drought_at_pollination(trials_frame(), weekly).iloc[0]
    assert result["usdm_weeks"] == 2
    assert result["max_pct_d1"] == 60
    assert result["max_pct_d2"] == 0
    assert result["drought_status"] == "drought"
    assert bool(result["drought_during_pollination"])


def test_weeks_of_other_counties_are_ignored():
    weekly = weekly_frame([week("2026-07-07", "2026-07-13", d0=100, d1=100, fips="31109")])
    result = drought_at_pollination(trials_frame(), weekly).iloc[0]
    assert result["drought_status"] == "no_data"


def test_trial_outside_us_has_no_data_but_keeps_pollination_dates():
    result = drought_at_pollination(trials_frame(is_us=False, county_fips=None), weekly_frame([])).iloc[0]
    assert result["drought_status"] == "no_data"
    assert result["drought_note"] == "outside the US"
    assert result["pollination_date"] == pd.Timestamp("2026-07-10")
    assert result["drought_during_pollination"] is None


def test_missing_planting_date_has_no_data():
    result = drought_at_pollination(trials_frame(planting_date=pd.NaT), weekly_frame([])).iloc[0]
    assert result["drought_status"] == "no_data"
    assert pd.isna(result["pollination_date"])


def test_output_has_one_row_per_trial_with_stable_columns():
    trials = pd.concat([trials_frame(trial_id=1), trials_frame(trial_id=2, is_us=False)], ignore_index=True)
    result = drought_at_pollination(trials, weekly_frame([]))
    assert result["trial_id"].tolist() == [1, 2]
    assert "max_pct_d4" in result.columns and "drought_note" in result.columns


@pytest.mark.parametrize("threshold,expected", [(50, "no_drought"), (20, "drought")])
def test_threshold_is_configurable(threshold, expected):
    weekly = weekly_frame([week("2026-07-07", "2026-07-13", d0=40, d1=30)])
    result = drought_at_pollination(trials_frame(), weekly, threshold=threshold).iloc[0]
    assert result["drought_status"] == expected

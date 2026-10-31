import numpy as np
import pandas as pd
import pytest

from src import harvest


def progress_frame(series: dict[str, list[tuple[str, float]]]) -> pd.DataFrame:
    rows = [
        {"state_code": geo, "year": 2026, "week_ending": pd.Timestamp(week), "pct_harvested": pct}
        for geo, points in series.items()
        for week, pct in points
    ]
    return pd.DataFrame(rows)


IOWA = [("2026-09-27", 5), ("2026-10-04", 15), ("2026-10-11", 40), ("2026-10-18", 60), ("2026-10-25", 80)]
NATIONAL = [("2026-10-04", 10), ("2026-10-11", 30), ("2026-10-18", 50), ("2026-10-25", 70)]


def trials_frame(rows):
    defaults = {"state_code": "IA", "is_us": True, "harvest_date": pd.NaT}
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_pct_on_interpolates_linearly_between_weekly_reports():
    days, pct = harvest.series_for(progress_frame({"IA": IOWA}), "IA")
    assert harvest.pct_on(days, pct, pd.Timestamp("2026-10-14")) == pytest.approx(40 + 3 / 7 * 20)


def test_pct_on_is_zero_before_first_report_and_unknown_after_last():
    days, pct = harvest.series_for(progress_frame({"IA": IOWA}), "IA")
    assert harvest.pct_on(days, pct, pd.Timestamp("2026-08-01")) == 0.0
    assert np.isnan(harvest.pct_on(days, pct, pd.Timestamp("2026-11-15")))


def test_pct_on_holds_at_100_after_harvest_completes():
    days, pct = harvest.series_for(progress_frame({"IA": [("2026-10-04", 90), ("2026-10-11", 100)]}), "IA")
    assert harvest.pct_on(days, pct, pd.Timestamp("2026-11-15")) == 100.0


def test_date_reaching_exact_and_interpolated_targets():
    days, pct = harvest.series_for(progress_frame({"IA": IOWA}), "IA")
    assert harvest.date_reaching(days, pct, 60) == pd.Timestamp("2026-10-18")
    assert harvest.date_reaching(days, pct, 50) in (pd.Timestamp("2026-10-14"), pd.Timestamp("2026-10-15"))


def test_date_reaching_returns_nat_when_target_not_reached_yet():
    days, pct = harvest.series_for(progress_frame({"IA": IOWA}), "IA")
    assert pd.isna(harvest.date_reaching(days, pct, 95))


@pytest.mark.parametrize("pct,label", [(10, "early"), (50, "typical"), (90, "late"), (np.nan, None)])
def test_timing_label(pct, label):
    assert harvest.timing_label(pct) == label


def test_compare_places_trial_harvest_on_state_curve():
    trials = trials_frame([{"trial_id": 1, "harvest_date": pd.Timestamp("2026-10-18")}])
    result = harvest.compare(trials, progress_frame({"IA": IOWA, "US": NATIONAL})).iloc[0]

    assert result["nass_geo"] == "IA"
    assert not result["nass_national_fallback"]
    assert result["usda_pct_at_trial_harvest"] == 60.0
    assert result["harvest_timing_vs_area"] == "typical"
    assert result["days_vs_usda_50pct"] in (3, 4)
    assert result["usda_latest_pct"] == 80
    assert result["usda_latest_week"] == pd.Timestamp("2026-10-25")


def test_compare_uses_national_curve_when_state_is_not_surveyed():
    trials = trials_frame([{"trial_id": 1, "state_code": "MS", "harvest_date": pd.Timestamp("2026-10-18")}])
    result = harvest.compare(trials, progress_frame({"IA": IOWA, "US": NATIONAL})).iloc[0]
    assert result["nass_geo"] == "US"
    assert result["nass_national_fallback"]
    assert result["usda_pct_at_trial_harvest"] == 50.0


def test_compare_trial_outside_us_has_no_usda_context():
    trials = trials_frame([{"trial_id": 1, "state_code": None, "is_us": False, "harvest_date": pd.Timestamp("2026-10-18")}])
    result = harvest.compare(trials, progress_frame({"IA": IOWA, "US": NATIONAL})).iloc[0]
    assert pd.isna(result["nass_geo"])
    assert pd.isna(result["usda_pct_at_trial_harvest"])


def test_compare_harvest_after_last_report_has_unknown_percentage():
    trials = trials_frame([{"trial_id": 1, "harvest_date": pd.Timestamp("2026-10-30")}])
    result = harvest.compare(trials, progress_frame({"IA": IOWA})).iloc[0]
    assert pd.isna(result["usda_pct_at_trial_harvest"])
    assert result["harvest_timing_vs_area"] is None


def test_compare_pending_trial_still_gets_state_context():
    trials = trials_frame([{"trial_id": 1}])
    result = harvest.compare(trials, progress_frame({"IA": IOWA})).iloc[0]
    assert result["nass_geo"] == "IA"
    assert pd.isna(result["usda_pct_at_trial_harvest"])
    assert result["usda_latest_pct"] == 80

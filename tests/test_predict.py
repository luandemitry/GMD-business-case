from datetime import date

import pandas as pd
import pytest

from src import predict

AS_OF = date(2026, 10, 30)
PLANTED = pd.Timestamp("2026-05-10")
IOWA = pd.DataFrame({
    "state_code": "IA",
    "week_ending": pd.to_datetime(["2026-09-27", "2026-10-04", "2026-10-11", "2026-10-18", "2026-10-25"]),
    "pct_harvested": [5, 15, 40, 60, 80],
})


def trial(trial_id, zone=5, days=None, planted=PLANTED, state="IA", is_us=True):
    harvest = planted + pd.Timedelta(days=days) if days else pd.NaT
    return {
        "trial_id": trial_id, "maturity_zone": zone, "planting_date": planted, "harvest_date": harvest,
        "days_to_harvest": float(days) if days else float("nan"), "state_code": state, "is_us": is_us,
    }


def comparison(pcts: dict[int, float]) -> pd.DataFrame:
    return pd.DataFrame({"trial_id": list(pcts), "usda_pct_at_trial_harvest": list(pcts.values())})


@pytest.fixture
def harvested():
    return [trial(1, days=150), trial(2, days=152), trial(3, days=154)]


@pytest.fixture
def pcts():
    return {1: 40.0, 2: 50.0, 3: 60.0}


def test_fit_uses_zone_median_when_enough_samples(harvested, pcts):
    data = predict.with_usda_pct(pd.DataFrame(harvested), comparison(pcts))
    model = predict.fit(data)
    assert model["days_by_zone"] == {5: 152.0}
    assert model["pct_by_zone"] == {5: 50.0}
    assert model["days_all"] == 152.0


def test_fit_ignores_zones_with_too_few_samples(pcts):
    rows = [trial(1, zone=5, days=150), trial(2, zone=5, days=152), trial(3, zone=8, days=160)]
    data = predict.with_usda_pct(pd.DataFrame(rows), comparison(pcts))
    assert 8 not in predict.fit(data)["days_by_zone"]


def test_fit_ignores_non_positive_cycle_lengths(pcts):
    rows = [trial(1, days=150), trial(2, days=152), {**trial(3, days=154), "days_to_harvest": -10.0}]
    data = predict.with_usda_pct(pd.DataFrame(rows), comparison(pcts))
    assert predict.fit(data)["days_all"] == 151.0


def test_combine_confidence_depends_on_agreement():
    base = pd.Timestamp("2026-10-10")
    assert predict.combine(base, base + pd.Timedelta(days=6))[1] == "high"
    assert predict.combine(base, base + pd.Timedelta(days=20))[1] == "medium"
    assert predict.combine(base, base + pd.Timedelta(days=40))[1] == "low"
    assert predict.combine(base, pd.NaT) == (base, "low")
    assert predict.combine(pd.NaT, pd.NaT)[1] is None


def test_status_for_relative_to_today():
    assert predict.status_for(pd.Timestamp("2026-10-01"), AS_OF) == "overdue_verify_record"
    assert predict.status_for(pd.Timestamp("2026-10-25"), AS_OF) == "ready_now"
    assert predict.status_for(pd.Timestamp("2026-11-10"), AS_OF) == "upcoming"
    assert predict.status_for(pd.NaT, AS_OF) == "cannot_predict"


def test_pending_trial_gets_agronomic_and_usda_predictions(harvested, pcts):
    trials = pd.DataFrame([*harvested, trial(4)])
    result = predict.predict_pending(trials, comparison(pcts), IOWA, AS_OF)

    assert result["trial_id"].tolist() == [4]
    row = result.iloc[0]
    assert row["pred_agronomic_date"] == pd.Timestamp("2026-10-09")
    assert row["pred_usda_date"] in (pd.Timestamp("2026-10-14"), pd.Timestamp("2026-10-15"))
    assert row["pred_confidence"] == "high"
    assert row["pred_status"] == "overdue_verify_record"


def test_invalid_zone_falls_back_to_overall_medians(harvested, pcts):
    trials = pd.DataFrame([*harvested, trial(5, zone=0)])
    row = predict.predict_pending(trials, comparison(pcts), IOWA, AS_OF).iloc[0]
    assert row["pred_agronomic_date"] == PLANTED + pd.Timedelta(days=152)


def test_trial_without_any_basis_cannot_be_predicted(pcts):
    trials = pd.DataFrame([trial(4, planted=pd.NaT, state=None, is_us=False)])
    row = predict.predict_pending(trials, comparison(pcts), IOWA, AS_OF).iloc[0]
    assert row["pred_status"] == "cannot_predict"
    assert pd.isna(row["pred_harvest_date"])


def test_only_trials_without_harvest_date_are_predicted(harvested, pcts):
    result = predict.predict_pending(pd.DataFrame(harvested), comparison(pcts), IOWA, AS_OF)
    assert result.empty
    assert list(result.columns) == predict.COLUMNS


def test_backtest_hides_each_trial_before_predicting_it(harvested, pcts):
    result = predict.backtest(pd.DataFrame(harvested), comparison(pcts), IOWA, AS_OF).set_index("method")
    assert result.loc["agronomic", "n"] == 3
    assert result.loc["agronomic", "mae_days"] == 2.0
    assert result.loc["agronomic", "bias_days"] == 0.0
    assert set(result.index) == {"agronomic", "usda_curve", "combined"}

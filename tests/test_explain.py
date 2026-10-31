from src.agent import run_query
from src.explain import explain_trial


def load(db, trial_id):
    trials = run_query(db, "SELECT * FROM trial_insights")
    backtest = run_query(db, "SELECT * FROM harvest_backtest")
    return trials[trials["trial_id"] == trial_id].iloc[0], backtest


def test_harvested_trial_is_explained_against_usda(insights_db):
    row, backtest = load(insights_db, 1)
    sections = explain_trial(row, backtest)
    assert sections["Data quality"] == ["No issues found in the trial sheet."]
    assert any("planting + 70 days" in line for line in sections["Drought around pollination"])
    assert any("Classified as drought" in line for line in sections["Drought around pollination"])
    assert any("% of its corn harvested" in line for line in sections["Harvest"])


def test_pending_trial_explains_both_prediction_methods(insights_db):
    row, backtest = load(insights_db, 2)
    harvest = explain_trial(row, backtest)["Harvest"]
    assert any("Expected harvest" in line for line in harvest)
    assert any("Method A" in line for line in harvest)
    assert any("Method B" in line for line in harvest)


def test_trial_outside_us_explains_missing_data(insights_db):
    row, backtest = load(insights_db, 9)
    sections = explain_trial(row, backtest)
    assert any("outside the US" in line for line in sections["Data quality"])
    assert any("No Drought Monitor data" in line for line in sections["Drought around pollination"])
    assert any("outside the US" in line for line in sections["Harvest"])


def test_swapped_coordinates_are_explained(insights_db):
    row, backtest = load(insights_db, 4)
    assert any("swapped" in line for line in explain_trial(row, backtest)["Data quality"])

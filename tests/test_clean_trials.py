from pathlib import Path

import pytest

from src.clean_trials import clean, load_raw, quality_summary

FIXTURE = Path(__file__).parent / "fixtures" / "trials_sample.csv"


@pytest.fixture
def raw():
    return load_raw(FIXTURE)


@pytest.fixture
def cleaned(raw):
    return clean(raw)


def flags_of(cleaned, trial_id):
    value = cleaned.loc[cleaned["trial_id"] == trial_id, "dq_flags"].iloc[0]
    return set(value.split(";")) - {""}


def test_column_names_are_normalized(raw):
    assert {"trial_id", "maturity_zone", "planting_date", "harvest_date"} <= set(raw.columns)


def test_exact_duplicate_rows_are_removed(raw, cleaned):
    assert len(cleaned) == len(raw) - 1
    assert (cleaned["trial_id"] == 3).sum() == 1


def test_swapped_coordinates_are_fixed_and_flagged(cleaned):
    row = cleaned[cleaned["trial_id"] == 4].iloc[0]
    assert row["latitude"] == pytest.approx(44.3)
    assert row["longitude"] == pytest.approx(-95.8)
    assert "lat_lon_swapped_fixed" in flags_of(cleaned, 4)


def test_zone_zero_is_flagged_invalid(cleaned):
    assert "invalid_maturity_zone" in flags_of(cleaned, 5)
    assert "invalid_maturity_zone" not in flags_of(cleaned, 1)


def test_shared_coordinates_are_flagged_on_both_trials(cleaned):
    assert "shared_coordinates" in flags_of(cleaned, 6)
    assert "shared_coordinates" in flags_of(cleaned, 7)


def test_harvest_before_planting_is_flagged(cleaned):
    assert "harvest_before_planting" in flags_of(cleaned, 8)


def test_coordinates_outside_expected_area_are_flagged(cleaned):
    assert "outside_expected_area" in flags_of(cleaned, 9)
    assert "outside_expected_area" not in flags_of(cleaned, 1)


def test_clean_trial_has_no_flags(cleaned):
    assert flags_of(cleaned, 1) == set()


def test_harvest_status_and_days_to_harvest(cleaned):
    harvested = cleaned[cleaned["trial_id"] == 1].iloc[0]
    pending = cleaned[cleaned["trial_id"] == 2].iloc[0]
    assert harvested["harvest_status"] == "harvested"
    assert harvested["days_to_harvest"] == 153
    assert pending["harvest_status"] == "no_harvest_date"
    assert pending["days_to_harvest"] != pending["days_to_harvest"]


def test_quality_summary_counts(raw, cleaned):
    summary = quality_summary(raw, cleaned).set_index("check")["rows"]
    assert summary["raw_rows"] == 10
    assert summary["exact_duplicates_removed"] == 1
    assert summary["lat_lon_swapped_fixed"] == 1

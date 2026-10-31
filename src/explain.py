import pandas as pd

from src import drought, harvest

FLAG_TEXT = {
    "invalid_maturity_zone": "Maturity zone is outside the 1-10 legend, so zone-based estimates use all trials instead.",
    "not_in_us": "Coordinates fall outside the US (no county found), so USDA and Drought Monitor data do not apply.",
    "lat_lon_swapped_fixed": "Latitude and longitude were swapped in the sheet and were corrected automatically.",
    "shared_coordinates": "Another trial has exactly the same coordinates; the location should be checked.",
    "duplicate_trial_id": "Another row uses the same trial id with different values.",
    "missing_planting_date": "Planting date is missing, so pollination and cycle length cannot be estimated.",
    "harvest_before_planting": "Harvest date is before planting date; the record is likely wrong.",
    "outside_expected_area": "Coordinates are far outside the expected growing area.",
}
STATUS_TEXT = {
    "severe_drought": "severe drought (D2 or worse)",
    "drought": "drought (D1 or worse)",
    "abnormally_dry": "abnormally dry (D0)",
    "no_drought": "no drought",
}


def fmt_date(value) -> str:
    return pd.Timestamp(value).strftime("%b %d, %Y") if pd.notna(value) else "n/a"


def text(value) -> str:
    return "" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value)


def is_true(value) -> bool:
    return False if value is None or pd.isna(value) else bool(value)


def data_quality_lines(row) -> list[str]:
    flags = [flag for flag in text(row["dq_flags"]).split(";") if flag]
    return [FLAG_TEXT.get(flag, flag) for flag in flags] or ["No issues found in the trial sheet."]


def drought_lines(row) -> list[str]:
    if pd.isna(row["pollination_date"]):
        return [f"Pollination could not be estimated: {text(row['drought_note'])}."]
    lines = [
        f"Pollination estimated on {fmt_date(row['pollination_date'])} (planting + {drought.POLLINATION_OFFSET_DAYS} days), "
        f"window {fmt_date(row['pollination_window_start'])} to {fmt_date(row['pollination_window_end'])}."
    ]
    if row["drought_status"] == "no_data":
        return lines + [f"No Drought Monitor data: {text(row['drought_note'])}."]
    return lines + [
        f"Peak share of {text(row['county'])} County, {text(row['state'])} during the window: "
        f"D0+ {row['max_pct_d0']:.0f}%, D1+ {row['max_pct_d1']:.0f}%, D2+ {row['max_pct_d2']:.0f}%.",
        f"Classified as {STATUS_TEXT.get(row['drought_status'], row['drought_status'])} "
        f"(rule: at least {drought.AREA_THRESHOLD:.0f}% of the county area at that level).",
    ]


def harvest_lines(row, backtest: pd.DataFrame) -> list[str]:
    lines = []
    if is_true(row["nass_national_fallback"]):
        lines.append("USDA does not publish weekly progress for this state, so the national curve was used.")

    if pd.notna(row["harvest_date"]):
        if pd.isna(row["usda_pct_at_trial_harvest"]):
            reason = "the trial is outside the US" if not is_true(row["is_us"]) else "the harvest is after the latest USDA report"
            return lines + [f"Harvested on {fmt_date(row['harvest_date'])}. No USDA comparison because {reason}."]
        lines.append(
            f"Harvested on {fmt_date(row['harvest_date'])}, when {text(row['nass_geo'])} had "
            f"{row['usda_pct_at_trial_harvest']:.0f}% of its corn harvested (USDA NASS weekly progress)."
        )
        if pd.notna(row["days_vs_usda_50pct"]):
            days = int(row["days_vs_usda_50pct"])
            relation = "before" if days < 0 else "after"
            lines.append(f"That is {abs(days)} days {relation} the state reached 50% harvested.")
        lines.append(
            f"Timing vs area: {text(row['harvest_timing_vs_area'])} "
            f"(early < {harvest.EARLY_BELOW_PCT:.0f}%, late > {harvest.LATE_ABOVE_PCT:.0f}%)."
        )
        return lines

    if pd.isna(row["pred_harvest_date"]):
        return lines + ["No harvest date recorded and not enough information to predict one."]

    mae = backtest.set_index("method")["mae_days"].get("combined") if not backtest.empty else None
    lines += [
        f"No harvest date recorded. Expected harvest: {fmt_date(row['pred_harvest_date'])} "
        f"({text(row['pred_confidence'])} confidence; status: {text(row['pred_status']).replace('_', ' ')}).",
        f"Method A, typical cycle of the maturity zone: {fmt_date(row['pred_agronomic_date'])}.",
        f"Method B, USDA curve for {text(row['nass_geo']) or 'n/a'}: {fmt_date(row['pred_usda_date'])}.",
        "Confidence reflects how closely the two methods agree.",
    ]
    if mae is not None:
        lines.append(f"Backtest on harvested trials: {mae} days average error.")
    return lines


def explain_trial(row: pd.Series, backtest: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "Data quality": data_quality_lines(row),
        "Drought around pollination": drought_lines(row),
        "Harvest": harvest_lines(row, backtest),
    }

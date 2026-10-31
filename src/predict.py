from datetime import date

import numpy as np
import pandas as pd

from src.clean_trials import ZONE_RANGE
from src.harvest import date_reaching, resolve_geo, series_for

MIN_ZONE_SAMPLES = 2
MIN_TARGET_PCT = 5.0
HIGH_CONFIDENCE_SPREAD_DAYS = 10
MEDIUM_CONFIDENCE_SPREAD_DAYS = 25
OVERDUE_AFTER_DAYS = 14

COLUMNS = [
    "trial_id", "pred_harvest_date", "pred_agronomic_date", "pred_usda_date",
    "pred_confidence", "pred_status",
]
DATE_COLUMNS = ["pred_harvest_date", "pred_agronomic_date", "pred_usda_date"]
METHODS = {
    "agronomic": "pred_agronomic_date",
    "usda_curve": "pred_usda_date",
    "combined": "pred_harvest_date",
}


def zone_medians(frame: pd.DataFrame, column: str) -> dict:
    in_range = frame["maturity_zone"].between(*ZONE_RANGE).fillna(False).astype(bool)
    stats = frame[in_range].groupby("maturity_zone")[column].agg(["median", "count"])
    return stats.loc[stats["count"] >= MIN_ZONE_SAMPLES, "median"].to_dict()


def fit(harvested: pd.DataFrame) -> dict:
    train = harvested[harvested["days_to_harvest"] > 0]
    usable = train[train["usda_pct_at_trial_harvest"] >= MIN_TARGET_PCT]
    return {
        "days_by_zone": zone_medians(train, "days_to_harvest"),
        "days_all": float(train["days_to_harvest"].median()) if len(train) else np.nan,
        "pct_by_zone": zone_medians(usable, "usda_pct_at_trial_harvest"),
        "pct_all": float(usable["usda_pct_at_trial_harvest"].median()) if len(usable) else np.nan,
    }


def combine(agronomic: pd.Timestamp, usda: pd.Timestamp) -> tuple[pd.Timestamp, str | None]:
    dates = [d for d in (agronomic, usda) if pd.notna(d)]
    if not dates:
        return pd.NaT, None
    ordinals = [d.toordinal() for d in dates]
    combined = pd.Timestamp.fromordinal(int(round(float(np.mean(ordinals)))))
    if len(dates) == 1:
        return combined, "low"
    spread = max(ordinals) - min(ordinals)
    if spread <= HIGH_CONFIDENCE_SPREAD_DAYS:
        return combined, "high"
    return combined, "medium" if spread <= MEDIUM_CONFIDENCE_SPREAD_DAYS else "low"


def status_for(predicted: pd.Timestamp, as_of: date) -> str:
    if pd.isna(predicted):
        return "cannot_predict"
    today = pd.Timestamp(as_of)
    if predicted < today - pd.Timedelta(days=OVERDUE_AFTER_DAYS):
        return "overdue_verify_record"
    return "ready_now" if predicted <= today else "upcoming"


def predict_trial(trial, model: dict, progress: pd.DataFrame, as_of: date) -> dict:
    out = dict.fromkeys(COLUMNS)
    out["trial_id"] = trial.trial_id

    zone = trial.maturity_zone
    zone_is_valid = pd.notna(zone) and ZONE_RANGE[0] <= zone <= ZONE_RANGE[1]
    days = model["days_by_zone"].get(zone, model["days_all"]) if zone_is_valid else model["days_all"]
    target_pct = model["pct_by_zone"].get(zone, model["pct_all"]) if zone_is_valid else model["pct_all"]

    agronomic = pd.NaT
    if pd.notna(trial.planting_date) and pd.notna(days):
        agronomic = trial.planting_date + pd.Timedelta(days=int(round(days)))

    usda = pd.NaT
    geo = resolve_geo(trial.state_code, set(progress["state_code"]))[0] if trial.is_us else None
    if geo is not None and pd.notna(target_pct):
        usda = date_reaching(*series_for(progress, geo), target_pct)

    combined, confidence = combine(agronomic, usda)
    out.update(
        pred_harvest_date=combined,
        pred_agronomic_date=agronomic,
        pred_usda_date=usda,
        pred_confidence=confidence,
        pred_status=status_for(combined, as_of),
    )
    return out


def with_usda_pct(trials: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    return trials.merge(comparison[["trial_id", "usda_pct_at_trial_harvest"]], on="trial_id", how="left")


def predict_pending(trials: pd.DataFrame, comparison: pd.DataFrame, progress: pd.DataFrame, as_of: date) -> pd.DataFrame:
    data = with_usda_pct(trials, comparison)
    model = fit(data[data["harvest_date"].notna()])
    pending = data[data["harvest_date"].isna()]
    rows = [predict_trial(trial, model, progress, as_of) for trial in pending.itertuples()]
    result = pd.DataFrame(rows, columns=COLUMNS)
    for column in DATE_COLUMNS:
        result[column] = pd.to_datetime(result[column])
    return result


ERROR_COLUMNS = ["trial_id", "maturity_zone", "state_code", "method", "error_days"]


def backtest_errors(trials: pd.DataFrame, comparison: pd.DataFrame, progress: pd.DataFrame, as_of: date) -> pd.DataFrame:
    data = with_usda_pct(trials, comparison)
    harvested = data[data["harvest_date"].notna()]
    rows = []

    for trial in harvested.itertuples():
        model = fit(harvested[harvested["trial_id"] != trial.trial_id])
        predicted = predict_trial(trial, model, progress, as_of)
        for method, column in METHODS.items():
            if pd.notna(predicted[column]):
                rows.append({
                    "trial_id": trial.trial_id,
                    "maturity_zone": trial.maturity_zone,
                    "state_code": trial.state_code,
                    "method": method,
                    "error_days": (predicted[column] - trial.harvest_date).days,
                })
    return pd.DataFrame(rows, columns=ERROR_COLUMNS)


def summarize_errors(errors: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        values = errors.loc[errors["method"] == method, "error_days"]
        if len(values):
            rows.append({
                "method": method,
                "n": len(values),
                "mae_days": round(float(values.abs().mean()), 1),
                "bias_days": round(float(values.mean()), 1),
            })
    return pd.DataFrame(rows, columns=["method", "n", "mae_days", "bias_days"])


def backtest(trials: pd.DataFrame, comparison: pd.DataFrame, progress: pd.DataFrame, as_of: date) -> pd.DataFrame:
    return summarize_errors(backtest_errors(trials, comparison, progress, as_of))

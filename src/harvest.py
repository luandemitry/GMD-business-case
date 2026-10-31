import numpy as np
import pandas as pd

EARLY_BELOW_PCT = 25.0
LATE_ABOVE_PCT = 75.0

COLUMNS = [
    "trial_id", "nass_geo", "nass_national_fallback", "usda_pct_at_trial_harvest",
    "usda_date_50pct", "days_vs_usda_50pct", "harvest_timing_vs_area",
    "usda_latest_pct", "usda_latest_week",
]
DATE_COLUMNS = ["usda_date_50pct", "usda_latest_week"]


def resolve_geo(state_code: str | None, available: set[str]) -> tuple[str | None, bool]:
    if state_code in available:
        return state_code, False
    if "US" in available:
        return "US", True
    return None, True


def series_for(progress: pd.DataFrame, geo: str) -> tuple[np.ndarray, np.ndarray]:
    rows = progress[progress["state_code"] == geo].sort_values("week_ending")
    days = np.array([week.toordinal() for week in rows["week_ending"]], dtype=float)
    pct = np.maximum.accumulate(rows["pct_harvested"].to_numpy(dtype=float)) if len(rows) else np.array([])
    return days, pct


def pct_on(days: np.ndarray, pct: np.ndarray, date: pd.Timestamp) -> float:
    if len(days) == 0:
        return np.nan
    after_last = 100.0 if pct[-1] >= 100 else np.nan
    return float(np.interp(date.toordinal(), days, pct, left=0.0, right=after_last))


def date_reaching(days: np.ndarray, pct: np.ndarray, target: float) -> pd.Timestamp:
    reached = np.flatnonzero(pct >= target) if len(pct) else []
    if len(reached) == 0:
        return pd.NaT
    idx = reached[0]
    if idx == 0 or pct[idx] == pct[idx - 1]:
        return pd.Timestamp.fromordinal(int(days[idx]))
    fraction = (target - pct[idx - 1]) / (pct[idx] - pct[idx - 1])
    return pd.Timestamp.fromordinal(int(round(days[idx - 1] + fraction * (days[idx] - days[idx - 1]))))


def timing_label(pct: float) -> str | None:
    if np.isnan(pct):
        return None
    if pct < EARLY_BELOW_PCT:
        return "early"
    if pct > LATE_ABOVE_PCT:
        return "late"
    return "typical"


def compare_trial(trial, progress: pd.DataFrame, available: set[str]) -> dict:
    row = dict.fromkeys(COLUMNS)
    row["trial_id"] = trial.trial_id

    if not trial.is_us:
        return row
    geo, fallback = resolve_geo(trial.state_code, available)
    if geo is None:
        return row
    row.update(nass_geo=geo, nass_national_fallback=fallback)

    days, pct = series_for(progress, geo)
    date_50 = date_reaching(days, pct, 50.0)
    row.update(
        usda_date_50pct=date_50,
        usda_latest_pct=float(pct[-1]),
        usda_latest_week=pd.Timestamp.fromordinal(int(days[-1])),
    )

    if pd.notna(trial.harvest_date):
        at_harvest = pct_on(days, pct, trial.harvest_date)
        row["usda_pct_at_trial_harvest"] = None if np.isnan(at_harvest) else round(at_harvest, 1)
        row["harvest_timing_vs_area"] = timing_label(at_harvest)
        if pd.notna(date_50):
            row["days_vs_usda_50pct"] = (trial.harvest_date - date_50).days
    return row


def compare(trials: pd.DataFrame, progress: pd.DataFrame) -> pd.DataFrame:
    available = set(progress["state_code"])
    rows = [compare_trial(trial, progress, available) for trial in trials.itertuples()]
    result = pd.DataFrame(rows, columns=COLUMNS)
    for column in DATE_COLUMNS:
        result[column] = pd.to_datetime(result[column])
    for column in ["usda_pct_at_trial_harvest", "days_vs_usda_50pct", "usda_latest_pct"]:
        result[column] = pd.to_numeric(result[column])
    return result

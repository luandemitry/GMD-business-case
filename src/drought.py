import pandas as pd

from src.usdm import LEVELS

POLLINATION_OFFSET_DAYS = 70
WINDOW_DAYS = 7
AREA_THRESHOLD = 50.0

COLUMNS = [
    "trial_id", "pollination_date", "window_start", "window_end", "usdm_weeks",
    *[f"max_pct_{level}" for level in LEVELS], "max_dsci",
    "drought_status", "drought_during_pollination", "drought_note",
]
DATE_COLUMNS = ["pollination_date", "window_start", "window_end"]


def classify(peaks: dict[str, float], threshold: float) -> str:
    if peaks["d2"] >= threshold:
        return "severe_drought"
    if peaks["d1"] >= threshold:
        return "drought"
    if peaks["d0"] >= threshold:
        return "abnormally_dry"
    return "no_drought"


def assess_trial(trial, weekly: pd.DataFrame, offset: int, window: int, threshold: float) -> dict:
    row = dict.fromkeys(COLUMNS)
    row["trial_id"] = trial.trial_id

    if pd.isna(trial.planting_date):
        return {**row, "drought_status": "no_data", "drought_note": "missing planting date"}

    pollination = trial.planting_date + pd.Timedelta(days=offset)
    start, end = pollination - pd.Timedelta(days=window), pollination + pd.Timedelta(days=window)
    row.update(pollination_date=pollination, window_start=start, window_end=end)

    if not trial.is_us:
        return {**row, "drought_status": "no_data", "drought_note": "outside the US"}

    weeks = weekly[
        (weekly["county_fips"] == trial.county_fips)
        & (weekly["week_start"] <= end)
        & (weekly["week_end"] >= start)
    ]
    if weeks.empty:
        return {**row, "drought_status": "no_data", "drought_note": "no Drought Monitor weeks for the window"}

    peaks = {level: float(weeks[level].max()) for level in LEVELS}
    status = classify(peaks, threshold)
    row.update(
        usdm_weeks=len(weeks),
        max_dsci=float(weeks["dsci"].max()),
        drought_status=status,
        drought_during_pollination=status in ("drought", "severe_drought"),
    )
    row.update({f"max_pct_{level}": value for level, value in peaks.items()})
    return row


def drought_at_pollination(
    trials: pd.DataFrame,
    weekly: pd.DataFrame,
    offset: int = POLLINATION_OFFSET_DAYS,
    window: int = WINDOW_DAYS,
    threshold: float = AREA_THRESHOLD,
) -> pd.DataFrame:
    rows = [assess_trial(trial, weekly, offset, window, threshold) for trial in trials.itertuples()]
    result = pd.DataFrame(rows, columns=COLUMNS)
    for column in DATE_COLUMNS:
        result[column] = pd.to_datetime(result[column])
    return result

import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from src.geocode import build_session

USDM_URL = "https://usdmdataservices.unl.edu/api/CountyStatistics/GetDroughtSeverityStatisticsByAreaPercent"
CACHE_DIR = Path("data/cache/usdm")
TIMEOUT_SECONDS = 60
LEVELS = ["d0", "d1", "d2", "d3", "d4"]
COLUMNS = ["county_fips", "week_start", "week_end", *LEVELS, "dsci"]


def us_date(value: date) -> str:
    return f"{value.month}/{value.day}/{value.year}"


def fetch_county(fips: str, start: date, end: date, session: requests.Session | None = None) -> list[dict]:
    cache_file = CACHE_DIR / f"{fips}_{start:%Y%m%d}_{end:%Y%m%d}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    session = session or build_session()
    response = session.get(
        USDM_URL,
        params={"aoi": fips, "startdate": us_date(start), "enddate": us_date(end), "statisticsType": 1},
        headers={"Accept": "application/json"},
        timeout=TIMEOUT_SECONDS,
    )
    if response.status_code != 200:
        raise RuntimeError(f"USDM request for county {fips} failed with HTTP {response.status_code}")

    records = response.json() if response.text.strip() else []
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(records))
    return records


def parse_dates(df: pd.DataFrame, column: str, fallback: pd.Series) -> pd.Series:
    if column not in df:
        return fallback
    return pd.to_datetime(df[column], errors="coerce").fillna(fallback)


def normalize(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=COLUMNS)

    df = pd.DataFrame(records)
    df.columns = df.columns.str.lower()

    raw_map_date = df["mapdate"].astype(str)
    map_date = pd.to_datetime(raw_map_date, format="%Y%m%d", errors="coerce")
    map_date = map_date.fillna(pd.to_datetime(raw_map_date, errors="coerce"))

    out = pd.DataFrame({
        "county_fips": df["fips"].astype(str).str.zfill(5),
        "week_start": parse_dates(df, "validstart", map_date),
        "week_end": parse_dates(df, "validend", map_date + pd.Timedelta(days=6)),
    })
    for level in LEVELS:
        out[level] = pd.to_numeric(df[level], errors="coerce")
    out["dsci"] = out[LEVELS].sum(axis=1)
    return out.dropna(subset=["week_start", "week_end"]).reset_index(drop=True)


def fetch_weekly(county_fips: list[str], start: date, end: date, session: requests.Session | None = None) -> pd.DataFrame:
    frames = [normalize(fetch_county(fips, start, end, session)) for fips in sorted(set(county_fips))]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)

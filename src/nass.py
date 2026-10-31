import json
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.geocode import build_session

NASS_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
CACHE_DIR = Path("data/cache/nass")
TIMEOUT_SECONDS = 60

FILTERS = {
    "source_desc": "SURVEY",
    "sector_desc": "CROPS",
    "group_desc": "FIELD CROPS",
    "commodity_desc": "CORN",
    "statisticcat_desc": "PROGRESS",
    "unit_desc": "PCT HARVESTED",
    "domain_desc": "TOTAL",
    "format": "JSON",
}


def get_api_key() -> str:
    load_dotenv()
    key = os.getenv("NASS_API_KEY")
    if not key:
        raise RuntimeError("NASS_API_KEY is not set")
    return key


def fetch_year(
    year: int,
    session: requests.Session | None = None,
    api_key: str | None = None,
    level: str = "STATE",
) -> list[dict]:
    cache_file = CACHE_DIR / f"corn_harvest_{level.lower()}_{year}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    session = session or build_session()
    params = {**FILTERS, "agg_level_desc": level, "year": year, "key": api_key or get_api_key()}
    response = session.get(NASS_URL, params=params, timeout=TIMEOUT_SECONDS)

    if response.status_code == 400 and "no data" in response.text.lower():
        records = []
    elif response.status_code != 200:
        raise RuntimeError(f"NASS request for {year} failed with HTTP {response.status_code}")
    else:
        records = response.json().get("data", [])

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(records))
    return records


def normalize(records: list[dict]) -> pd.DataFrame:
    columns = ["state_code", "year", "week_ending", "pct_harvested"]
    if not records:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(records)
    df = df[df["short_desc"].str.contains("CORN, GRAIN", na=False)]
    df = df.assign(
        state_code=df["state_alpha"],
        year=df["year"].astype(int),
        week_ending=pd.to_datetime(df["week_ending"], errors="coerce"),
        pct_harvested=pd.to_numeric(df["Value"].str.replace(",", ""), errors="coerce"),
    )
    df = df.dropna(subset=["week_ending", "pct_harvested"])
    return (
        df[columns]
        .drop_duplicates(["state_code", "week_ending"])
        .sort_values(["state_code", "week_ending"])
        .reset_index(drop=True)
    )


def fetch_progress(
    years: list[int],
    session: requests.Session | None = None,
    levels: tuple[str, ...] = ("STATE", "NATIONAL"),
    api_key: str | None = None,
) -> pd.DataFrame:
    frames = [
        normalize(fetch_year(year, session, api_key=api_key, level=level))
        for year in years
        for level in levels
    ]
    return pd.concat(frames, ignore_index=True)

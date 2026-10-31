import json
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

FCC_URL = "https://geo.fcc.gov/api/census/block/find"
CACHE_DIR = Path("data/cache/fcc")
TIMEOUT_SECONDS = 30


def build_session() -> requests.Session:
    retry = Retry(total=4, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504))
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def parse_response(payload: dict) -> dict:
    county = payload.get("County") or {}
    state = payload.get("State") or {}
    return {
        "county_fips": county.get("FIPS"),
        "county_name": county.get("name"),
        "state_fips": state.get("FIPS"),
        "state_code": state.get("code"),
    }


def lookup(lat: float, lon: float, session: requests.Session | None = None) -> dict:
    cache_file = CACHE_DIR / f"{lat:.6f}_{lon:.6f}.json"
    if cache_file.exists():
        return parse_response(json.loads(cache_file.read_text()))

    session = session or build_session()
    response = session.get(
        FCC_URL,
        params={"latitude": lat, "longitude": lon, "format": "json"},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(payload))
    return parse_response(payload)


def add_geography(trials: pd.DataFrame, fetch=lookup) -> pd.DataFrame:
    geo = pd.DataFrame(
        [fetch(lat, lon) for lat, lon in zip(trials["latitude"], trials["longitude"])],
        index=trials.index,
    )
    out = trials.join(geo)
    out["is_us"] = out["county_fips"].notna()

    not_us = ~out["is_us"]
    flags = out["dq_flags"].where(~not_us, out["dq_flags"] + ";not_in_us")
    out["dq_flags"] = flags.str.strip(";")
    return out

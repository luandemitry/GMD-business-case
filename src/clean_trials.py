from pathlib import Path

import pandas as pd

DATE_FORMAT = "%m/%d/%Y"
ZONE_RANGE = (1, 10)
LAT_RANGE = (24.0, 50.0)
LON_RANGE = (-125.0, -66.0)


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def fix_swapped_coordinates(df: pd.DataFrame) -> pd.Series:
    swapped = df["latitude"].abs().gt(90) | (df["longitude"].gt(0) & df["latitude"].lt(0))
    df.loc[swapped, ["latitude", "longitude"]] = df.loc[swapped, ["longitude", "latitude"]].to_numpy()
    return swapped


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop_duplicates().copy()
    df["trial_name"] = df["trial_name"].str.strip()
    df["planting_date"] = pd.to_datetime(df["planting_date"], format=DATE_FORMAT, errors="coerce")
    df["harvest_date"] = pd.to_datetime(df["harvest_date"], format=DATE_FORMAT, errors="coerce")
    df["maturity_zone"] = pd.to_numeric(df["maturity_zone"], errors="coerce").astype("Int64")

    swapped = fix_swapped_coordinates(df)

    checks = pd.DataFrame({
        "lat_lon_swapped_fixed": swapped,
        "invalid_maturity_zone": ~df["maturity_zone"].between(*ZONE_RANGE).fillna(False).astype(bool),
        "outside_expected_area": ~(df["latitude"].between(*LAT_RANGE) & df["longitude"].between(*LON_RANGE)),
        "shared_coordinates": df.duplicated(["latitude", "longitude"], keep=False),
        "duplicate_trial_id": df.duplicated("trial_id", keep=False),
        "missing_planting_date": df["planting_date"].isna(),
        "harvest_before_planting": df["harvest_date"] <= df["planting_date"],
    })
    df["dq_flags"] = checks.apply(lambda row: ";".join(row.index[row]), axis=1)
    df["harvest_status"] = df["harvest_date"].notna().map({True: "harvested", False: "no_harvest_date"})
    df["days_to_harvest"] = (df["harvest_date"] - df["planting_date"]).dt.days
    return df.reset_index(drop=True)


def quality_summary(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    rows = {
        "raw_rows": len(raw),
        "exact_duplicates_removed": len(raw) - len(raw.drop_duplicates()),
        "clean_rows": len(cleaned),
        "no_harvest_date": int((cleaned["harvest_status"] == "no_harvest_date").sum()),
    }
    flags = cleaned["dq_flags"].str.split(";").explode()
    rows.update(flags[flags != ""].value_counts().to_dict())
    return pd.DataFrame({"check": rows.keys(), "rows": rows.values()})


def main(path: Path = Path("data/raw/trials.csv")) -> None:
    raw = load_raw(path)
    cleaned = clean(raw)
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(out_dir / "trials_clean.parquet", index=False)
    print(quality_summary(raw, cleaned).to_string(index=False))


if __name__ == "__main__":
    main()

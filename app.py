import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src import pipeline
from src.agent import Agent, connect
from src.explain import explain_trial
from src.llm import build_llm, get_secret

DB_PATH = Path(os.getenv("BREEDER_DB_PATH", "data/breeder.duckdb"))
RAW_PATH = DB_PATH.parent / "raw" / "trials.csv"
MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "20"))
STATUS_COLORS = {
    "severe_drought": "#b3261e",
    "drought": "#e8590c",
    "abnormally_dry": "#f2b705",
    "no_drought": "#2f9e44",
    "no_data": "#8a8f98",
}
EXAMPLES = [
    "Which trials had drought during pollination?",
    "How does our harvest compare with USDA by state?",
    "When will the unharvested trials be ready?",
    "Which trials have data-quality problems?",
]
TABLE_COLUMNS = [
    "trial_id", "trial_name", "maturity_zone", "state", "planting_date", "harvest_date",
    "drought_status", "harvest_timing_vs_area", "pred_harvest_date", "pred_confidence", "dq_flags",
]

st.set_page_config(page_title="Breeder Copilot", page_icon="🌽", layout="wide")


def display_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.select_dtypes(include=["datetime", "datetimetz"]).columns:
        out[column] = out[column].dt.date
    return out


@st.cache_data
def load_data(db_path: str, modified: float) -> dict[str, pd.DataFrame]:
    con = connect(Path(db_path))
    try:
        names = ["trial_insights", "harvest_backtest", "data_quality", "pipeline_run"]
        return {name: con.execute(f"SELECT * FROM {name}").df() for name in names}
    finally:
        con.close()


@st.cache_resource
def get_llm():
    return build_llm()


@st.cache_resource(show_spinner=False)
def build_database_from_secret(csv_text: str) -> None:
    if nass_key := get_secret("NASS_API_KEY"):
        os.environ["NASS_API_KEY"] = nass_key
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text(csv_text.strip() + "\n")
    pipeline.run(raw_path=RAW_PATH, db_path=DB_PATH, processed_dir=DB_PATH.parent / "processed")


def ensure_database() -> None:
    if DB_PATH.exists():
        return
    csv_text = get_secret("TRIALS_CSV")
    if not csv_text:
        st.error("No data yet. Run `python -m src.pipeline` to build the database.")
        st.stop()
    try:
        with st.spinner("First start: building the database from USDA NASS, Drought Monitor and FCC (about a minute)..."):
            build_database_from_secret(csv_text)
    except Exception as exc:
        build_database_from_secret.clear()
        st.error(f"Could not build the database: {type(exc).__name__}: {exc}"[:400])
        st.stop()


def require_password() -> None:
    password = get_secret("APP_PASSWORD")
    if not password or st.session_state.get("authenticated"):
        return
    entered = st.text_input("Password", type="password")
    if entered == password:
        st.session_state.authenticated = True
        st.rerun()
    if entered:
        st.error("Wrong password.")
    st.stop()


def sidebar_filters(trials: pd.DataFrame, run: pd.Series) -> pd.DataFrame:
    with st.sidebar:
        st.header("Filters")
        zones = st.multiselect("Maturity zone", sorted(trials["maturity_zone"].dropna().astype(int).unique()))
        states = st.multiselect("State", sorted(trials["state"].dropna().unique()))
        statuses = st.multiselect("Drought at pollination", [s for s in STATUS_COLORS if s in set(trials["drought_status"])])
        only_flagged = st.toggle("Only trials with data-quality flags")

        st.header("Data")
        st.caption(f"Season {run['season_year']} · data as of {run['as_of_date']}")
        st.caption(f"Pipeline run (UTC): {run['run_at_utc']}")
        st.caption("Sources: trial sheet, USDA NASS Quick Stats, U.S. Drought Monitor, FCC Area API.")

    mask = pd.Series(True, index=trials.index)
    if zones:
        mask &= trials["maturity_zone"].isin(zones)
    if states:
        mask &= trials["state"].isin(states)
    if statuses:
        mask &= trials["drought_status"].isin(statuses)
    if only_flagged:
        mask &= trials["dq_flags"].fillna("") != ""
    return trials[mask]


def show_kpis(trials: pd.DataFrame) -> None:
    in_us = trials[trials["is_us"]]
    cols = st.columns(4)
    cols[0].metric("Trials", len(trials))
    cols[1].metric("Drought at pollination", int(in_us["drought_during_pollination"].fillna(False).sum()),
                   help=f"Out of {len(in_us)} trials in the US")
    cols[2].metric("No harvest date", int(trials["harvest_date"].isna().sum()), help="Predicted dates in the table")
    cols[3].metric("Data-quality flags", int((trials["dq_flags"].fillna("") != "").sum()))


def trial_map(trials: pd.DataFrame):
    fig = px.scatter_geo(
        trials, lat="latitude", lon="longitude", color="drought_status",
        color_discrete_map=STATUS_COLORS, hover_name="trial_name", custom_data=["trial_id"],
        hover_data={"trial_id": True, "maturity_zone": True, "state": True, "latitude": False, "longitude": False},
        scope="north america",
    )
    fig.update_geos(fitbounds="locations", showsubunits=True, subunitcolor="#c9d1c9")
    fig.update_traces(marker={"size": 11, "line": {"width": 1, "color": "white"}})
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=360,
        legend={"title_text": "", "orientation": "h", "y": -0.05},
    )
    return fig


def selected_from_map(event) -> int | None:
    try:
        points = event.selection.points
        return int(points[0]["customdata"][0]) if points else None
    except (AttributeError, KeyError, IndexError, TypeError):
        return None


def show_trials(trials: pd.DataFrame, backtest: pd.DataFrame) -> None:
    st.subheader("Trials by drought status at pollination")
    event = st.plotly_chart(trial_map(trials), width="stretch", on_select="rerun", selection_mode="points", key="map")
    clicked = selected_from_map(event)
    if clicked in set(trials["trial_id"]):
        st.session_state["selected_trial"] = clicked

    labels = {row.trial_id: f"{row.trial_id} · {row.trial_name} · zone {row.maturity_zone} · {row.state if isinstance(row.state, str) else 'outside US'}"
              for row in trials.itertuples()}
    if st.session_state.get("selected_trial") not in labels:
        st.session_state["selected_trial"] = next(iter(labels))
    trial_id = st.selectbox("Selected trial (or click a dot)", list(labels), format_func=labels.get, key="selected_trial")

    row = trials[trials["trial_id"] == trial_id].iloc[0]
    with st.container(border=True):
        st.markdown("**How this was computed**")
        for title, lines in explain_trial(row, backtest).items():
            st.markdown(f"*{title}*")
            st.markdown("\n".join(f"- {line}" for line in lines))


def show_result(result) -> None:
    with st.chat_message("assistant"):
        if result.llm_unavailable:
            st.warning(result.answer, icon="⏳")
        else:
            st.markdown(result.answer)
        with st.expander("How I got this result"):
            if result.steps:
                st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(result.steps, 1)))
            for note in result.notes:
                st.markdown(f"- {note}")
            if result.confidence:
                st.markdown(f"**Confidence:** {result.confidence}. {result.confidence_reason}")
            st.caption(" · ".join(x for x in [result.data_as_of, result.model and f"model {result.model}"] if x))
            if result.sql:
                st.code(result.sql, language="sql")
            if result.rows is not None and not result.rows.empty:
                st.dataframe(display_dates(result.rows), hide_index=True, width="stretch")
            if result.error:
                st.caption(f"Error: {result.error}")


def show_ask() -> None:
    st.subheader("Ask about your trials")
    llm = get_llm()
    if llm is None:
        st.info("Questions are disabled: no GEMINI_API_KEY (or GROQ_API_KEY) is configured.")
        return

    history = st.session_state.setdefault("history", [])
    messages = st.container(height=560)
    with messages:
        if not history:
            st.caption("Try one of these:")
            clicked = next((q for q in EXAMPLES if st.button(q, width="stretch")), None)
        else:
            clicked = None
        for result in history:
            with st.chat_message("user"):
                st.markdown(result.question)
            show_result(result)

    question = st.chat_input("Ask a question in plain language") or clicked
    if not question:
        return
    if len(history) >= MAX_QUESTIONS:
        st.warning(f"Question limit for this session reached ({MAX_QUESTIONS}).")
        return

    with messages:
        with st.chat_message("user"):
            st.markdown(question)
        with st.spinner("Querying the trial data..."):
            result = Agent(llm, DB_PATH).ask(question, [(r.question, r.answer) for r in history])
        history.append(result)
        show_result(result)


def show_quality(data: dict[str, pd.DataFrame]) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Data quality")
        st.dataframe(data["data_quality"], hide_index=True, width="stretch")
    with right:
        st.subheader("Harvest prediction backtest")
        st.dataframe(data["harvest_backtest"], hide_index=True, width="stretch")
        st.caption("Leave-one-out on harvested trials: each trial is hidden, predicted, and compared with its real date.")
    st.subheader("Method")
    st.markdown(
        "- **Pollination** = planting + 70 days, window ±7 days.\n"
        "- **Drought** = at least 50% of the county in D1 or worse (D2 or worse = severe) in any U.S. Drought Monitor week in the window.\n"
        "- **Harvest vs USDA** = % of the state's corn harvested (USDA NASS weekly, interpolated) on the trial harvest date.\n"
        "- **Prediction** = average of the maturity-zone cycle and the date the state reaches the % at which similar trials were harvested."
    )


def main() -> None:
    st.title("Breeder Copilot · corn field trials")
    require_password()
    ensure_database()

    data = load_data(str(DB_PATH), DB_PATH.stat().st_mtime)
    run = data["pipeline_run"].iloc[0]
    trials = sidebar_filters(data["trial_insights"], run)
    show_kpis(trials)

    overview, quality = st.tabs(["Trials", "Data quality & method"])
    with overview:
        if trials.empty:
            st.info("No trials match the filters.")
        else:
            left, right = st.columns([3, 2], gap="large")
            with left:
                show_trials(trials, data["harvest_backtest"])
            with right:
                show_ask()
            st.subheader("All trials")
            st.dataframe(display_dates(trials[TABLE_COLUMNS]), hide_index=True, width="stretch")
    with quality:
        show_quality(data)

main()

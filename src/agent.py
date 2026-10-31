import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from src import drought, harvest
from src.llm import LLM

MAX_ROWS = 200
ROWS_SENT_TO_LLM = 50
HISTORY_TURNS = 3
ALLOWED_TABLES = {"trial_insights", "harvest_backtest", "data_quality", "pipeline_run", "usda_harvest_progress"}
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|pragma|install|load|export|import|call|set|reset|checkpoint|vacuum)\b",
    re.IGNORECASE,
)

SCHEMA = """
trial_insights (one row per clean trial):
  trial_id INT, trial_name TEXT, maturity_zone INT (valid 1-10; 0 = invalid code),
  state TEXT (2-letter, NULL outside US), county TEXT, latitude, longitude, is_us BOOLEAN,
  planting_date, harvest_date (NULL = no harvest date recorded), harvest_status ('harvested' | 'no_harvest_date'),
  days_to_harvest INT, dq_flags TEXT (semicolon list, '' = clean; e.g. invalid_maturity_zone, not_in_us,
  lat_lon_swapped_fixed, shared_coordinates),
  pollination_date, pollination_window_start, pollination_window_end,
  drought_status ('severe_drought' | 'drought' | 'abnormally_dry' | 'no_drought' | 'no_data'),
  drought_during_pollination BOOLEAN, max_pct_d0, max_pct_d1, max_pct_d2 (peak % of county area in D0+/D1+/D2+),
  max_dsci (0-500), drought_note TEXT,
  nass_geo TEXT (state used for USDA curve, 'US' = national), nass_national_fallback BOOLEAN,
  usda_pct_at_trial_harvest (% of state corn harvested on the trial harvest date),
  days_vs_usda_50pct INT (trial harvest minus date state reached 50%; negative = earlier),
  harvest_timing_vs_area ('early' | 'typical' | 'late'), usda_date_50pct, usda_latest_pct, usda_latest_week,
  pred_harvest_date, pred_agronomic_date, pred_usda_date (only for trials without harvest date),
  pred_confidence ('high' | 'medium' | 'low'), pred_status ('overdue_verify_record' | 'ready_now' | 'upcoming' | 'cannot_predict')
harvest_backtest: method, n, mae_days, bias_days
data_quality: check, rows
pipeline_run: run_at_utc, season_year, as_of_date, trial_count
usda_harvest_progress: state_code, year, week_ending, pct_harvested
"""

SQL_SYSTEM = f"""You translate questions from corn breeders into ONE DuckDB SQL query over these tables:
{SCHEMA}
Return JSON only:
{{"sql": "<query>", "reasoning": ["short step", "..."]}}
If the tables cannot answer the question (yield, genetics, rainfall, anything not listed), return
{{"sql": null, "answer": "<what is missing and what you can answer instead>", "reasoning": ["..."]}}
Rules:
- A single SELECT or WITH query. Never modify data.
- Prefer trial_insights. When listing trials, include trial_id, trial_name and the columns that justify the answer.
- Trials outside the US have no drought or USDA data; say so instead of counting them as 'no drought'.
- Use date_diff('day', start, end) for day differences.
Examples:
Q: Which trials had drought during pollination?
{{"sql": "SELECT trial_id, trial_name, state, drought_status, max_pct_d1, pollination_date FROM trial_insights WHERE drought_during_pollination ORDER BY max_pct_d1 DESC", "reasoning": ["drought_during_pollination flags D1+ on at least half the county"]}}
Q: How does our harvest compare with USDA in Iowa?
{{"sql": "SELECT trial_id, trial_name, harvest_date, usda_pct_at_trial_harvest, harvest_timing_vs_area, days_vs_usda_50pct FROM trial_insights WHERE state = 'IA' AND harvest_date IS NOT NULL", "reasoning": ["compare each Iowa harvest with the Iowa USDA curve"]}}
Q: When will unharvested trials be ready?
{{"sql": "SELECT trial_id, trial_name, state, maturity_zone, pred_harvest_date, pred_confidence, pred_status FROM trial_insights WHERE harvest_date IS NULL ORDER BY pred_harvest_date", "reasoning": ["predictions exist only for trials without a harvest date"]}}
"""

ANSWER_SYSTEM = """You help corn breeders read query results about their field trials.
Answer only from the result table given. Be concise: at most five sentences or a short list.
Cite trial ids. If the result is empty, say that nothing matched. Do not show SQL.
Answer in the same language as the question."""

FIX_PROMPT = """The previous query failed.
Question: {question}
Query: {sql}
Error: {error}
Return corrected JSON in the same format."""

OUT_OF_SCOPE = "I can only answer questions about the trials, drought around pollination, USDA harvest progress and harvest predictions."


@dataclass
class AgentResult:
    question: str
    answer: str
    sql: str | None = None
    rows: pd.DataFrame | None = None
    steps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    confidence: str | None = None
    confidence_reason: str | None = None
    data_as_of: str | None = None
    model: str | None = None
    error: str | None = None


def parse_json(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)


def validate_sql(sql: str) -> str:
    query = sql.strip().rstrip(";").strip()
    if not re.match(r"^(select|with)\b", query, re.IGNORECASE):
        raise ValueError("Only SELECT or WITH queries are allowed")
    if ";" in query:
        raise ValueError("Only one statement is allowed")
    if FORBIDDEN.search(query):
        raise ValueError("Query contains a forbidden keyword")
    ctes = {name.lower() for name in re.findall(r"(?:\bwith|,)\s*([a-z_]\w*)\s+as\s*\(", query, re.IGNORECASE)}
    without_from_functions = re.sub(r"\b(extract|substring|trim|overlay|position)\s*\([^()]*\)", "", query, flags=re.IGNORECASE)
    tables = {name.lower() for name in re.findall(r"\b(?:from|join)\s+([a-z_][\w.]*)", without_from_functions, re.IGNORECASE)}
    unknown = tables - ALLOWED_TABLES - ctes
    if unknown:
        raise ValueError(f"Unknown tables: {', '.join(sorted(unknown))}")
    return query


def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path), read_only=True, config={"enable_external_access": False})


def run_query(db_path: Path, sql: str) -> pd.DataFrame:
    con = connect(db_path)
    try:
        return con.execute(f"SELECT * FROM ({sql}) AS q LIMIT {MAX_ROWS}").df()
    finally:
        con.close()


def methodology_notes(sql: str, backtest: pd.DataFrame) -> list[str]:
    text = sql.lower()
    notes = []
    if "drought" in text or "max_pct" in text or "pollination" in text:
        notes.append(
            f"Pollination = planting + {drought.POLLINATION_OFFSET_DAYS} days, window ±{drought.WINDOW_DAYS} days. "
            f"Drought = at least {drought.AREA_THRESHOLD:.0f}% of the county in D1 or worse in a U.S. Drought Monitor "
            "week overlapping the window (D2 or worse = severe)."
        )
    if "usda" in text or "harvest_timing" in text or "nass" in text:
        notes.append(
            "Harvest timing places the trial harvest date on the weekly USDA NASS % of corn harvested in the "
            f"trial's state (linear between weekly reports): early < {harvest.EARLY_BELOW_PCT:.0f}%, "
            f"late > {harvest.LATE_ABOVE_PCT:.0f}%."
        )
    if "pred_" in text:
        mae = backtest.set_index("method")["mae_days"].get("combined") if not backtest.empty else None
        notes.append(
            "Predicted harvest = average of (a) planting + median cycle of the maturity zone and (b) the date the "
            "state reaches the % harvested at which similar trials were harvested."
            + (f" Leave-one-out backtest error: {mae} days on average." if mae is not None else "")
        )
    return notes


def assess_confidence(db_path: Path, rows: pd.DataFrame) -> tuple[str, str]:
    if rows.empty:
        return "low", "No rows matched the question."
    if "pred_confidence" in rows and rows["pred_confidence"].notna().any():
        level = rows["pred_confidence"].mode().iloc[0]
        return level, "Most common confidence of the predictions returned (agreement between the two methods)."
    if "trial_id" not in rows:
        return "high", "Aggregate computed directly from the validated tables."

    ids = ",".join(str(int(i)) for i in rows["trial_id"].dropna().unique())
    flags = run_query(db_path, f"SELECT trial_id, dq_flags FROM trial_insights WHERE trial_id IN ({ids})") if ids else pd.DataFrame()
    flagged = flags[flags["dq_flags"].fillna("") != ""] if not flags.empty else flags
    if len(flagged):
        listed = ", ".join(str(i) for i in flagged["trial_id"].head(5))
        return "medium", f"{len(flagged)} of the returned trials have data-quality flags (e.g. {listed})."
    return "high", "Returned trials have no data-quality flags."


class Agent:
    def __init__(self, llm: LLM, db_path: Path):
        self.llm = llm
        self.db_path = Path(db_path)

    def context(self) -> tuple[str | None, pd.DataFrame]:
        con = connect(self.db_path)
        try:
            run = con.execute("SELECT as_of_date, run_at_utc FROM pipeline_run").df()
            backtest = con.execute("SELECT * FROM harvest_backtest").df()
        finally:
            con.close()
        as_of = f"data as of {run['as_of_date'][0]} (pipeline run {run['run_at_utc'][0]})" if len(run) else None
        return as_of, backtest

    def plan(self, question: str, history: list[tuple[str, str]]) -> dict:
        recent = "\n".join(f"Q: {q}\nA: {a}" for q, a in history[-HISTORY_TURNS:])
        prompt = (f"Conversation so far:\n{recent}\n\n" if recent else "") + f"Question: {question}"
        return parse_json(self.llm.generate(SQL_SYSTEM, prompt, json_output=True))

    def execute(self, question: str, plan: dict) -> tuple[str, pd.DataFrame, list[str]]:
        steps = []
        sql = plan["sql"]
        for attempt in range(2):
            try:
                query = validate_sql(sql)
                return query, run_query(self.db_path, query), steps
            except (ValueError, duckdb.Error) as exc:
                if attempt == 1:
                    raise
                steps.append(f"First query failed ({type(exc).__name__}); asked the model to correct it.")
                fixed = parse_json(self.llm.generate(SQL_SYSTEM, FIX_PROMPT.format(question=question, sql=sql, error=exc), json_output=True))
                sql = fixed.get("sql") or sql
        raise RuntimeError("unreachable")

    def ask(self, question: str, history: list[tuple[str, str]] | None = None) -> AgentResult:
        history = history or []
        as_of, backtest = self.context()
        result = AgentResult(question=question, answer="", data_as_of=as_of, model=getattr(self.llm, "name", None))

        try:
            plan = self.plan(question, history)
        except Exception as exc:
            return self.failed(result, "The language model did not return a usable plan.", exc)

        result.steps = [f"Interpreted the question: {step}" for step in plan.get("reasoning", [])]
        if not plan.get("sql"):
            result.answer = plan.get("answer") or OUT_OF_SCOPE
            result.steps.append("No query run: the available data cannot answer this question.")
            return result

        try:
            sql, rows, retry_steps = self.execute(question, plan)
        except Exception as exc:
            return self.failed(result, "I could not build a valid query for this question. Try rephrasing it.", exc)

        result.sql, result.rows = sql, rows
        result.steps += retry_steps + [
            "Ran a read-only query on the curated view (no writes, no file or network access).",
            f"{len(rows)} row(s) returned" + (f" (capped at {MAX_ROWS})." if len(rows) == MAX_ROWS else "."),
        ]
        result.notes = methodology_notes(sql, backtest)
        result.confidence, result.confidence_reason = assess_confidence(self.db_path, rows)

        prompt = (
            f"Question: {question}\n\nResult ({len(rows)} rows, first {ROWS_SENT_TO_LLM} shown):\n"
            f"{rows.head(ROWS_SENT_TO_LLM).to_csv(index=False)}\nMethod notes:\n" + "\n".join(result.notes)
        )
        try:
            result.answer = self.llm.generate(ANSWER_SYSTEM, prompt).strip()
        except Exception as exc:
            result.answer = f"{len(rows)} row(s) matched. The summary could not be generated; see the table below."
            result.error = type(exc).__name__
        result.model = getattr(self.llm, "last_used", None) or result.model
        return result

    @staticmethod
    def failed(result: AgentResult, message: str, exc: Exception) -> AgentResult:
        result.answer = message
        result.error = f"{type(exc).__name__}: {exc}"[:300]
        return result

import json
import os

import pytest

from src import agent
from src.llm import build_llm


class ScriptedLLM:
    name = "scripted"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, system, prompt, json_output=False):
        self.prompts.append((system, prompt, json_output))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def plan(sql, *steps):
    return json.dumps({"sql": sql, "reasoning": list(steps)})


@pytest.mark.parametrize("sql", [
    "SELECT * FROM trial_insights",
    "select trial_id from trial_insights where state = 'IA';",
    "WITH d AS (SELECT * FROM trial_insights) SELECT * FROM d",
    "SELECT extract(month FROM harvest_date) AS m FROM trial_insights",
    "SELECT method, mae_days FROM harvest_backtest",
])
def test_validate_sql_accepts_read_queries(sql):
    assert agent.validate_sql(sql)


@pytest.mark.parametrize("sql", [
    "DROP TABLE trials",
    "SELECT 1; DROP TABLE trials",
    "DELETE FROM trial_insights",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM secrets_table",
    "COPY trial_insights TO 'out.csv'",
    "WITH x AS (SELECT 1) INSERT INTO trials SELECT * FROM x",
])
def test_validate_sql_rejects_unsafe_or_unknown_queries(sql):
    with pytest.raises(ValueError):
        agent.validate_sql(sql)


def test_database_connection_is_read_only_and_sandboxed(insights_db):
    with pytest.raises(Exception):
        agent.run_query(insights_db, "SELECT * FROM read_csv('/etc/hostname')")
    con = agent.connect(insights_db)
    try:
        with pytest.raises(Exception):
            con.execute("CREATE TABLE x (a INT)")
    finally:
        con.close()


def test_parse_json_tolerates_code_fences():
    assert agent.parse_json('```json\n{"sql": null}\n```') == {"sql": None}


def test_ask_runs_query_and_explains_how_the_answer_was_built(insights_db):
    llm = ScriptedLLM(
        plan("SELECT trial_id, trial_name, drought_status, max_pct_d1 FROM trial_insights WHERE drought_during_pollination",
             "drought flag column answers the question"),
        "Trials 1 and 3 had drought during pollination.",
    )
    result = agent.Agent(llm, insights_db).ask("Which trials had drought during pollination?")

    assert result.answer == "Trials 1 and 3 had drought during pollination."
    assert "drought_during_pollination" in result.sql
    assert len(result.rows) > 0
    assert any("Interpreted the question" in s for s in result.steps)
    assert any("read-only" in s for s in result.steps)
    assert any("70 days" in n for n in result.notes)
    assert result.confidence in {"high", "medium"}
    assert result.data_as_of.startswith("data as of 2026-10-30")
    assert "Result (" in llm.prompts[1][1]


def test_flagged_trials_lower_confidence(insights_db):
    llm = ScriptedLLM(plan("SELECT trial_id, trial_name FROM trial_insights WHERE trial_id = 4"), "Trial 4.")
    result = agent.Agent(llm, insights_db).ask("Show trial 4")
    assert result.confidence == "medium"
    assert "4" in result.confidence_reason


def test_prediction_questions_report_backtest_and_prediction_confidence(insights_db):
    llm = ScriptedLLM(
        plan("SELECT trial_id, pred_harvest_date, pred_confidence FROM trial_insights WHERE harvest_date IS NULL"),
        "Predictions listed.",
    )
    result = agent.Agent(llm, insights_db).ask("When will pending trials be ready?")
    assert any("backtest" in n for n in result.notes)
    assert result.confidence in {"high", "medium", "low"}
    assert "agreement" in result.confidence_reason


def test_out_of_scope_question_returns_explanation_without_query(insights_db):
    llm = ScriptedLLM(json.dumps({"sql": None, "answer": "Yield is not in the data.", "reasoning": ["no yield column"]}))
    result = agent.Agent(llm, insights_db).ask("What was the yield of trial 1?")
    assert result.answer == "Yield is not in the data."
    assert result.sql is None
    assert any("No query run" in s for s in result.steps)


def test_failed_query_is_corrected_once(insights_db):
    llm = ScriptedLLM(
        plan("SELECT missing_column FROM trial_insights"),
        plan("SELECT count(*) AS n FROM trial_insights"),
        "There are 9 trials.",
    )
    result = agent.Agent(llm, insights_db).ask("How many trials?")
    assert result.rows["n"][0] == 9
    assert any("asked the model to correct it" in s for s in result.steps)


def test_unsafe_query_from_model_is_never_executed(insights_db):
    llm = ScriptedLLM(plan("DROP TABLE trials"), plan("DELETE FROM trials"))
    result = agent.Agent(llm, insights_db).ask("Delete everything")
    assert result.sql is None
    assert result.error
    assert agent.run_query(insights_db, "SELECT count(*) AS n FROM trials")["n"][0] == 9


def test_model_failure_explains_the_free_tier_limit(insights_db):
    result = agent.Agent(ScriptedLLM(RuntimeError("quota")), insights_db).ask("Anything")
    assert result.llm_unavailable
    assert "free tier" in result.answer
    assert "quota" in result.error


def test_invalid_model_output_is_not_reported_as_limit(insights_db):
    result = agent.Agent(ScriptedLLM("not json"), insights_db).ask("Anything")
    assert not result.llm_unavailable
    assert "usable plan" in result.answer


def test_failure_while_writing_summary_still_returns_rows(insights_db):
    llm = ScriptedLLM(plan("SELECT count(*) AS n FROM trial_insights"), RuntimeError("quota"))
    result = agent.Agent(llm, insights_db).ask("How many trials?")
    assert result.llm_unavailable
    assert result.rows["n"][0] == 9
    assert "free-tier limit" in result.answer


def test_follow_up_questions_include_recent_history(insights_db):
    llm = ScriptedLLM(plan("SELECT count(*) AS n FROM trial_insights"), "9.")
    agent.Agent(llm, insights_db).ask("And in Iowa?", history=[("How many trials?", "There are 9 trials.")])
    assert "How many trials?" in llm.prompts[0][1]


@pytest.mark.network
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
def test_live_model_answers_a_drought_question(insights_db):
    result = agent.Agent(build_llm(), insights_db).ask("Which trials had drought during pollination?")
    assert result.sql and "trial_insights" in result.sql.lower()
    assert result.rows is not None and len(result.rows) > 0
    assert result.answer

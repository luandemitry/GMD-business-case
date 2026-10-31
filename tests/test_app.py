from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def app(insights_db, monkeypatch):
    monkeypatch.setenv("BREEDER_DB_PATH", str(insights_db))
    for key in ("GEMINI_API_KEY", "GROQ_API_KEY", "APP_PASSWORD"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("src.llm.load_dotenv", lambda: None)
    return AppTest.from_file(APP, default_timeout=30)


def test_app_renders_without_errors(app):
    app.run()
    assert not app.exception
    assert "Breeder Copilot" in app.title[0].value
    assert [m.label for m in app.metric] == ["Trials", "Drought at pollination", "No harvest date", "Data-quality flags"]
    assert app.metric[0].value == "9"


def test_filters_reduce_the_trial_count(app):
    app.run()
    app.sidebar.multiselect[2].select("no_data").run()
    assert not app.exception
    assert app.metric[0].value == "2"


def test_questions_are_disabled_without_llm_key(app):
    app.run()
    assert any("Questions are disabled" in info.value for info in app.info)


def test_missing_database_shows_instructions(monkeypatch, tmp_path):
    monkeypatch.setenv("BREEDER_DB_PATH", str(tmp_path / "missing.duckdb"))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert any("python -m src.pipeline" in error.value for error in at.error)

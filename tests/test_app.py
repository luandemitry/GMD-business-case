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
    monkeypatch.delenv("TRIALS_CSV", raising=False)
    monkeypatch.setattr("src.llm.load_dotenv", lambda: None)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert any("python -m src.pipeline" in error.value for error in at.error)


def test_database_is_built_from_trials_secret_on_first_start(monkeypatch, tmp_path):
    from src import pipeline
    from tests.conftest import FIXTURE, fake_drought, fake_geo, fake_progress

    real_run = pipeline.run
    monkeypatch.setattr(pipeline, "run", lambda **kw: real_run(
        **kw, geo_fetch=fake_geo, progress_fetch=fake_progress, drought_fetch=fake_drought))
    monkeypatch.setenv("BREEDER_DB_PATH", str(tmp_path / "data" / "breeder.duckdb"))
    monkeypatch.setenv("TRIALS_CSV", FIXTURE.read_text())
    monkeypatch.setattr("src.llm.load_dotenv", lambda: None)

    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert (tmp_path / "data" / "raw" / "trials.csv").exists()
    assert at.metric[0].value == "9"

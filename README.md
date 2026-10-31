# GDM Corn Trials: Breeder Copilot

Integrates internal corn field-trial data with USDA NASS harvest progress and the U.S. Drought Monitor, and exposes the result to breeders through a visual and conversational interface.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with your API keys. The NASS key is requested at https://quickstats.nass.usda.gov/api.

## Structure

```
src/        pipeline, analytics and agent code
tests/      unit tests
data/raw/   trial input data
docs/       architecture diagrams and technical documentation
```

Further sections (methodology, assumptions, limitations, production scenario) are added as each part is built.

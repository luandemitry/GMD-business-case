# Improvements

Ordered roughly by value for breeders.

## Methods
- **Pollination from growing degree days** instead of a fixed 70 days, using daily temperature (e.g. NOAA or PRISM) and the hybrid's relative maturity.
- **Field-level drought** with soil moisture or precipitation grids instead of county averages.
- **Prediction intervals** (e.g. from the backtest error distribution) instead of a single date, and more history (several seasons) to fit the zone cycles.
- **NASS 5-year average curve** to say whether the current season is early or late, not only where a trial sits on this year's curve.
- **Zone 0 sites:** confirm with the data owner whether they use another maturity system and map it.
- Investigate the Missouri trial that misses by 34 days; check the record before changing the method.

## Agent
- **Multi-agent design:** a router that sends questions to specialized agents (SQL analyst, methodology explainer, data-quality auditor) and a reviewer that checks the answer against the rows before replying.
- **Evaluation set** of reference questions with expected results, run in CI on every prompt/model change.
- Native tool calling and structured outputs, plus charts generated from results ("plot harvest timing by state").
- User feedback buttons (useful / wrong) stored with the question, SQL and answer.
- Semantic layer generated from table metadata instead of a hand-written schema prompt.

## Data and platform
- Scheduled pipeline (weekly after the NASS and Drought Monitor releases) instead of manual runs.
- Census geocoder as a second provider when the FCC service is down.
- Data contracts and a quarantine table for rejected rows.
- Store agent logs for monitoring and cost tracking.
- Move to the Azure architecture in [production.md](production.md).

## Interface
- Proactive alerts: trials expected ready in the next two weeks, trials overdue for a harvest record.
- Export of filtered tables and answers to Excel.
- Portuguese/Spanish interface for global teams (the agent already answers in the question's language).

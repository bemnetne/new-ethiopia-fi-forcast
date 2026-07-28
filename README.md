# Ethiopia Financial Inclusion Forecasting

A finance-focused data and machine learning project that analyzes historical financial inclusion in Ethiopia, models major market and policy events, forecasts Account Ownership and Digital Payment Usage for 2025–2027, and presents results through an interactive Streamlit dashboard with SHAP-based explainability.

[![CI](https://github.com/YOUR-USERNAME/YOUR-REPOSITORY/actions/workflows/unittests.yml/badge.svg)](https://github.com/YOUR-USERNAME/YOUR-REPOSITORY/actions/workflows/unittests.yml)

## Business Problem

Financial institutions, development organizations, mobile network operators, and policymakers need reliable evidence on how financial inclusion in Ethiopia may evolve.

Available information is scattered across surveys, regulatory reports, telecom announcements, and payment-system records. This makes it difficult to answer practical questions such as:

- Will Ethiopia reach its Account Ownership target?
- Will Digital Payment Usage grow faster than Access?
- Which policy and market events may influence progress?
- What risks could prevent forecast outcomes from being achieved?
- Which demographic groups remain underserved?

This project combines those sources into a structured analytical workflow that supports planning, scenario analysis, and risk-aware decision-making.

## Solution Overview

The project follows an end-to-end workflow:

1. Collect and validate financial inclusion observations and market events.
2. Standardize records into a unified dataset.
3. Explore Account Ownership, mobile-money penetration, gender, location, and payment-use trends.
4. Model event direction, magnitude, and expected lag.
5. Forecast Account Ownership and Digital Payment Usage for 2025–2027.
6. Build pessimistic, base, and optimistic scenarios.
7. Add uncertainty ranges and target-gap analysis.
8. Train an explainable Ridge regression model on Global Findex subgroup data.
9. Generate global and local SHAP explanations.
10. Present metrics, forecasts, risks, and business implications in a Streamlit dashboard.
11. Validate core functions with automated tests and GitHub Actions.

## Key Results

- Account Ownership base forecast:
  - 2025: 50.3%
  - 2026: 51.7%
  - 2027: 53.2%

- Digital Payment Usage base forecast:
  - 2025: 38.0%
  - 2026: 41.5%
  - 2027: 45.5%

- National Account Ownership target: 60%

- Automated testing:
  - 19 tests passed
  - 71.89% total coverage across the selected refactored core modules
  - `src/explainability.py`: 86% coverage

- SHAP explainability:
  - Historical trend accounts for approximately 44% of total global feature importance
  - Secondary education is the strongest demographic feature
  - Income, age, education, and gender contribute meaningful explanatory value

- Dashboard capabilities:
  - Historical metric exploration
  - Scenario comparison
  - Forecast uncertainty
  - Target-gap analysis
  - Event-impact review
  - Global and local SHAP explanations
  - CSV downloads

## Quick Start

```bash
git clone https://github.com/bemnetne/new-ethiopia-fi-forcast.git
cd new-ethiopia-fi-forcast

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

python -m streamlit run dashboard/app.py
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

python -m streamlit run dashboard/app.py
```

## Testing

Run all unit and integration tests with coverage:

```bash
pytest \
  --cov=src.data_utils \
  --cov=src.data_loader \
  --cov=src.event_model \
  --cov=src.forecast_utils \
  --cov=src.explainability \
  --cov-report=term-missing \
  --cov-fail-under=60
```

Current result:

```text
19 passed
Required test coverage of 60% reached
Total coverage: 71.89%
```

Run a single test file:

```bash
pytest tests/test_explainability.py -v
```

Run code-quality checks:

```bash
ruff check src tests dashboard
ruff format src tests dashboard
```

Check that the dashboard file compiles:

```bash
python -m py_compile dashboard/app.py
```

## Project Structure

```text
new-ethiopia-fi-forcast/
├── .github/
│   └── workflows/
│       └── unittests.yml
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
│       ├── access_forecasts.csv
│       ├── usage_forecasts.csv
│       ├── forecast_uncertainty.csv
│       ├── access_model_features.csv
│       ├── access_shap_global.csv
│       ├── access_shap_local.csv
│       └── access_shap_diagnostics.csv
├── models/
│   └── access_explainable_model.joblib
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_event_impact_modeling.ipynb
│   └── 04_access_usage_forecasting.ipynb
├── reports/
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── data_utils.py
│   ├── event_model.py
│   ├── explainability.py
│   └── forecast_utils.py
├── tests/
│   ├── test_data_utils.py
│   ├── test_event_model.py
│   ├── test_explainability.py
│   ├── test_forecast_utils.py
│   └── test_integration.py
├── .gitignore
├── README.md
└── requirements.txt
└── benchmark.py

```



## Technical Details

### Data

Main sources include:

- World Bank Global Findex
- National Bank of Ethiopia
- Ethio Telecom
- Safaricom Ethiopia and M-Pesa announcements
- EthSwitch
- Digital financial services and policy reports

The project uses a unified schema containing:

```text
record_id
record_type
category
pillar
indicator
indicator_code
value_numeric
observation_date
gender
location
source_name
source_url
confidence
related_indicator
impact_direction
impact_magnitude
lag_months
evidence_basis
collection_date
original_text
```

Preprocessing includes:

- column-name standardization
- numeric conversion
- date normalization
- duplicate review
- required-column validation
- observation, event, and target separation
- subgroup filtering
- processed CSV generation

### Forecast Model

The national forecasts use transparent trend and scenario methods because the number of direct historical observations is limited.

Forecast outputs include:

- pessimistic scenario
- base scenario
- optimistic scenario
- 80% uncertainty range
- gap to the 60% target

### Event-Impact Model

Events are represented through:

- affected indicator
- impact direction
- relative magnitude
- expected lag
- evidence confidence
- signed effect score

Event-effect scores represent expected direction and relative strength. They are not interpreted as causal percentage-point effects.

### Explainability Model

A separate Ridge regression model is used for historical subgroup explainability.

Model inputs include:

- survey trend
- gender subgroup
- income subgroup
- age subgroup
- education subgroup

Global Findex subgroup observations are downloaded from the World Bank API and converted into a model-ready feature table.

Hyperparameter:

```text
Ridge alpha = 2.0
```

SHAP outputs include:

- global mean absolute feature importance
- beeswarm feature-effect visualization
- local prediction contribution records
- waterfall explanation for a selected observation
- feature-concentration checks
- high-correlation diagnostics

The explainability model describes historical subgroup patterns. It does not replace the national forecasting model and does not establish causality.

### Evaluation

Current automated checks:

```text
19 passing tests
71.89% scoped test coverage
60% minimum CI coverage threshold
Ruff linting and formatting
Dashboard compile validation
GitHub Actions CI
```

Because Account Ownership and Digital Payment Usage have limited direct historical observations, forecasts are evaluated primarily through:

- scenario consistency
- bounded-value validation
- uncertainty ranges
- comparison with historical trends
- transparent assumptions
- sensitivity and risk interpretation

## CI/CD

GitHub Actions runs automatically on configured pushes and pull requests.

The workflow performs:

1. Python environment setup
2. dependency installation
3. Ruff linting
4. unit and integration tests
5. test coverage enforcement
6. dashboard compile validation

Workflow file:

```text
.github/workflows/unittests.yml
```

## Dashboard Performance

Dashboard performance was measured locally using the repeatable benchmark script:

```bash
python benchmark_dashboard.py
```

The script starts a fresh headless Streamlit process, waits until the application returns an HTTP 200 response, and then records 10 warm local HTTP requests.

Latest benchmark results:

```text
Cold startup time: 0.805 seconds
Warm requests measured: 10
Average warm response: 1.72 ms
Median warm response: 1.71 ms
Minimum warm response: 1.45 ms
Maximum warm response: 2.15 ms
Under-three-second startup target: PASS
```

The dashboard met the target of loading in under three seconds. This benchmark measures Streamlit server startup and local HTTP response time. It does not include complete browser-side rendering time for Plotly charts.

## Future Improvements

- Increase coverage for `src/data_loader.py` and remaining legacy analysis modules.
- Add more historical Account Ownership and Digital Payment Usage observations.
- Add time-aligned macroeconomic and infrastructure variables.
- Validate forecasts with new 2025–2027 observations as they become available.
- Add formal backtesting when enough survey waves exist.
- Measure and optimize dashboard page-load time.
- Deploy the dashboard publicly.
- Add automated data-refresh pipelines.
- Expand subgroup explainability with rural and urban observations where available.
- Add stronger forecast monitoring and model-drift checks.
- Improve accessibility and mobile responsiveness in the dashboard.

## Author

**Bemnet Bekele**

- LinkedIn: [Add LinkedIn profile URL]
- GitHub: [Add GitHub profile URL]
- Email: [Add contact email]

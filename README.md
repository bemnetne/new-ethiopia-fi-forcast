# Ethiopia Financial Inclusion Forecast

This project analyzes Ethiopia's financial inclusion landscape, models the expected influence of major market and policy events, forecasts Account Ownership and Digital Payment Usage for 2025–2027, and presents the results in an interactive Streamlit dashboard.

## Project Objectives

The project focuses on two main financial inclusion outcomes:

- **Account Ownership (Access):** Percentage of adults with an account at a financial institution or mobile money provider.
- **Digital Payment Usage:** Percentage of adults who made or received a digital payment.

The analysis covers:

- historical financial inclusion trends;
- mobile money adoption;
- gender and location gaps;
- infrastructure and payment-channel development;
- major policy and market events;
- event-impact modeling;
- Access and Usage forecasts;
- pessimistic, base, and optimistic scenarios;
- progress toward a 60% Account Ownership target.

## Project Structure

```text
new-ethiopia-fi-forcast/
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_task1_data_understanding.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_event_impact_modeling.ipynb
│   └── 04_access_usage_forecasting.ipynb
├── reports/
├── src/
├── README.md
├── requirements.txt
└── venv/
```

## Main Processed Files

The dashboard reads data from:

```text
data/processed/
```

Expected files include:

```text
ethiopia_fi_enriched.csv
refined_event_impacts.csv
event_indicator_association_summary.csv
access_forecast_scenarios.csv
usage_forecast_scenarios.csv
final_financial_inclusion_forecasts.csv
forecast_uncertainty_summary.csv
```

Run the notebooks before starting the dashboard when these files are missing.

Recommended notebook order:

```text
01_task1_data_understanding.ipynb
02_exploratory_data_analysis.ipynb
03_event_impact_modeling.ipynb
04_access_usage_forecasting.ipynb
```

## Key Findings

- Account Ownership increased from 14% in 2011 to 49% in 2024.
- Growth slowed between 2021 and 2024, increasing by only 3 percentage points.
- Mobile money account penetration increased from 4.7% in 2021 to 9.45% in 2024.
- Digital Payment Usage was estimated at 35% in 2024.
- The 2021 gender gap was approximately 20 percentage points.
- P2P transfers were the clearest available digital-payment use case.
- Mobile money registrations grew faster than unique adult account ownership.
- Usage is expected to grow faster than Access.

## Forecast Summary

### Base Forecast

| Target | 2024 | 2025 | 2026 | 2027 |
|---|---:|---:|---:|---:|
| Account Ownership | 49.0% | 50.3% | 51.7% | 53.2% |
| Digital Payment Usage | 35.0% | 38.0% | 41.5% | 45.5% |

### 2027 Scenario Range

| Target | Pessimistic | Base | Optimistic |
|---|---:|---:|---:|
| Account Ownership | 50.5% | 53.2% | 53.5% |
| Digital Payment Usage | 39.5% | 45.5% | 50.0% |

These forecasts are scenario-based planning estimates rather than exact predictions.

## Dashboard Features

### Overview

- Key metric cards
- Account Ownership
- Digital Payment Usage
- Mobile Money Account Rate
- P2P/ATM Crossover Ratio
- Growth highlights
- Interactive overview chart
- Data download

### Trends

- Interactive time-series charts
- Date-range selector
- Indicator selector
- Payment-channel comparison
- Downloadable filtered data

### Event Impacts

- Event and indicator filters
- Relative event-impact chart
- Refined event-impact table
- Confidence labels
- Download functionality

### Forecasts

- Baseline trend option
- Trend-with-events option
- Scenario comparison
- Account Ownership and Usage selector
- Forecast confidence or uncertainty bands
- Key milestones
- Forecast download

### Inclusion Projections

- Scenario selector
- Progress toward the 60% Account Ownership target
- Remaining-gap calculation
- Answers to key consortium questions
- Projection download

### Data Explorer

- Record-type filter
- Searchable dataset table
- Downloadable filtered records

## Requirements

The main dashboard dependencies are:

```text
streamlit
pandas
numpy
plotly
```

Install all project dependencies from:

```text
requirements.txt
```

## Running the Project Locally

### 1. Open the Project Folder

In Command Prompt or PowerShell:

```bash
cd C:\Users\bemnet\Desktop\10academy\newWeek11\new-ethiopia-fi-forcast
```

Run the dashboard from the project root, not from inside the `dashboard` folder.

### 2. Create a Virtual Environment

Skip this step when `venv` already exists.

#### Windows

```bash
python -m venv venv
```

#### Linux or macOS

```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment

#### Windows Command Prompt

```bash
venv\Scripts\activate
```

#### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

#### Linux or macOS

```bash
source venv/bin/activate
```

After activation, the terminal should show `(venv)` before the current path.

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To install only the dashboard dependencies manually:

```bash
python -m pip install streamlit pandas numpy plotly
```

### 5. Confirm the Processed Data

Before running the dashboard, confirm that the expected CSV files exist inside:

```text
data/processed/
```

When files are missing, run the related notebooks again.

### 6. Start the Dashboard

From the project root, run:

```bash
python -m streamlit run dashboard\app.py
```

You can also use:

```bash
streamlit run dashboard\app.py
```

Using `python -m streamlit` is safer when several Python installations are available.

### 7. Open the Dashboard

The local address is normally:

```text
http://localhost:8501
```

### 8. Stop the Dashboard

Return to the terminal and press:

```text
Ctrl + C
```

## Running from Visual Studio Code

1. Open the project folder in VS Code.
2. Open `dashboard/app.py`.
3. Open **Terminal → New Terminal**.
4. Confirm the terminal is in the project root.
5. Activate the virtual environment:

```bash
venv\Scripts\activate
```

6. Run:

```bash
python -m streamlit run dashboard\app.py
```

Do not use the normal **Run Python File** button for `app.py`.



## Notes on Interpretation

- Event-impact scores represent direction and relative strength.
- Event-impact scores are not percentage-point effects.
- The Access forecast is based on only five historical observations.
- The Usage forecast begins from one direct national observation.
- Scenario ranges should be used for planning rather than treated as exact confidence bounds.
- Registered accounts, active accounts, and unique adults are different measures.

## Future Work

- Add new Findex observations.
- Add more active-user and transaction data.
- Improve gender and rural analysis.
- Add merchant-payment and bill-payment indicators.
- Recalibrate event effects when more evidence becomes available.
- Deploy the Streamlit dashboard online.
- Update forecasts automatically when new data is added.

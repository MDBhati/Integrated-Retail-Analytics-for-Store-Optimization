# Integrated Retail Analytics for Store Optimization and Demand Forecasting

## Overview

This project applies machine learning, statistical analysis, and data visualization techniques to optimize retail store performance, forecast demand, and generate actionable business insights.

The solution analyzes historical sales data, store characteristics, promotional markdowns, and external economic indicators to identify anomalies, segment stores, forecast sales, and recommend inventory and marketing strategies.

---

## Business Objectives

* Detect unusual sales patterns across stores and departments.
* Forecast weekly sales to improve inventory planning.
* Understand the impact of economic and promotional factors on sales.
* Segment stores based on sales behavior and operational characteristics.
* Generate data-driven recommendations for marketing and store optimization.

---

## Key Features

### Sales Anomaly Detection

* Identified abnormal sales spikes and drops.
* Analyzed the effects of holidays, markdowns, and economic indicators.
* Improved data quality for downstream forecasting models.

### Time Series Analysis

* Explored seasonal trends and holiday impacts.
* Evaluated store and department performance over time.

### Data Preprocessing & Feature Engineering

* Handled missing values in markdown features.
* Created predictive features using store, sales, and economic data.

### Store Segmentation

* Grouped stores with similar sales and operational characteristics.
* Evaluated segmentation quality using clustering metrics.

### Market Basket Analysis

* Inferred department-level product associations.
* Generated cross-selling opportunities and promotional insights.

### Demand Forecasting

* Built predictive models for weekly sales forecasting.
* Incorporated sales history, markdowns, holidays, CPI, fuel prices, and unemployment rates.

### Business Strategy Recommendations

* Inventory optimization strategies.
* Segment-based marketing recommendations.
* Store performance improvement initiatives.

---

## Project Workflow

```text
Data Collection
      ↓
Data Cleaning & Feature Engineering
      ↓
Anomaly Detection
      ↓
Store Segmentation
      ↓
Market Basket Analysis
      ↓
Demand Forecasting
      ↓
Business Insights & Recommendations
```

## Quick start

```bash
# Create environment and install
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Optional: copy environment overrides
cp .env.example .env

# Validate raw data
retail-analytics validate-data

# Run full pipeline (train + score)
retail-analytics run-pipeline
```

Or use Make:

```bash
make install-dev
make pipeline
```

## CLI commands

| Command | Description |
|---------|-------------|
| `retail-analytics validate-data` | Schema and quality checks on raw CSVs |
| `retail-analytics ingest` | Merge and clean → `data/processed/` |
| `retail-analytics train` | Train segmentation + XGBoost; write artifacts |
| `retail-analytics score` | Batch score with champion model |
| `retail-analytics run-pipeline` | Validate → train → score |

## Project structure

```
```text
Integrated-Retail-Analytics/
│
├── config/
│   └── config.yaml                 # Pipeline parameters
│
├── src/
│   └── retail_analytics/
│       ├── data/                   # Ingestion + quality gates
│       ├── features/               # Feature engineering
│       ├── models/                 # Segmentation, forecasting, SARIMA, association rules
│       ├── pipeline/               # Orchestration
│       └── cli.py                  # Entry point
│
├── tests/                          # Unit + integration tests
│
├── docs/                           # Output schemas, runbooks
│
├── artifacts/                      # Models, manifests, evaluation reports (generated)
│
└── data/
    ├── raw/                        # raw data (csv files) 
    └── outputs/                    # Forecasts and batch scores (generated)
```

```

## Configuration

- **YAML:** `config/config.yaml` — model params, quality thresholds, feature lists
- **Environment:** prefix `RETAIL_` (see `.env.example`)
  - `RETAIL_DATA_DIR` — raw CSV location (default: `data/raw/Retail Datsets/`)
  - `RETAIL_ARTIFACTS_DIR`, `RETAIL_OUTPUTS_DIR`, `RETAIL_LOG_LEVEL`

## Docker

```bash
docker build -t retail-analytics .
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" -v "$(pwd)/data:/app/data" retail-analytics run-pipeline
```

## Testing

```bash
make test        # Fast unit tests (excludes full-dataset pipeline)
make test-all    # Includes slow integration test
make lint
```

CI runs on push/PR via `.github/workflows/ci.yml`.

## Production deliverables

| Deliverable | Location |
|-------------|----------|
| Champion model + preprocessors | `artifacts/models/xgboost_champion.joblib` |
| Run manifest (lineage) | `artifacts/manifests/{run_id}.json` |
| Evaluation metrics | `artifacts/reports/evaluation_{run_id}.json` |
| Batch forecasts | `data/outputs/forecasts_*.parquet` |
| Output schema | `docs/output_schema.md` |
| Operational runbooks | `docs/runbooks.md` |

## Analytical capabilities

1. **Data prep & ETL** — merge sales, stores, and macro features; imputation and negative-sales filtering
2. **Anomaly detection** — Z-score flags per store–department with holiday context
3. **Store segmentation** — K-Means (k=4) on scaled store profiles with PCA
4. **Demand forecasting** — XGBoost with time-based holdout (pre-2012 train, 2012 test)
5. **SARIMA baseline** — auto-ARIMA seasonal benchmark on aggregate sales
6. **Association rules** — department co-occurrence via Apriori (mlxtend)

---

## Strategic recommendations (from analysis)

1. **Dynamic supply** — higher safety stock at mega-flagship clusters; lean assortment at neighborhood stores
2. **Weather-responsive marketing** — tie promotions to temperature bands and regional forecasts
3. **Floor layout** — use association rules to co-locate or traffic-drive high-lift department pairs

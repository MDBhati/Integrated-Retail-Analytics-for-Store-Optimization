# Integrated Retail Analytics for Store Optimization

Production-grade retail analytics pipeline: data quality gates, feature engineering, store segmentation, demand forecasting, anomaly detection, and association-rule mining.

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
config/config.yaml          # Pipeline parameters (single source of truth)
src/retail_analytics/       # Production Python package
  data/                     # Ingestion + quality gates
  features/                 # Feature engineering
  models/                   # Segmentation, forecasting, SARIMA, association rules
  pipeline/                 # Orchestration
  cli.py                    # Entry point
tests/                      # Unit + integration tests
docs/                       # Output schemas, runbooks
artifacts/                  # Models, manifests, evaluation reports (generated)
data/processed/             # Curated tables (generated)
data/outputs/               # Forecasts and batch scores (generated)
Project_3.ipynb             # Original research notebook (reference)
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

See `Project description.md` for the full production specification and phased delivery roadmap.

## Research notebook

`Project_3.ipynb` contains the original Colab exploration. Production logic lives in `src/retail_analytics/` and should be treated as the source of truth for scheduled jobs.

---

## Strategic recommendations (from analysis)

1. **Dynamic supply** — higher safety stock at mega-flagship clusters; lean assortment at neighborhood stores
2. **Weather-responsive marketing** — tie promotions to temperature bands and regional forecasts
3. **Floor layout** — use association rules to co-locate or traffic-drive high-lift department pairs

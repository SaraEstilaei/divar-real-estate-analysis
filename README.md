# Divar Real Estate Analysis

An end-to-end data science and exploratory analysis of real estate listings from the Divar platform, covering data sanitation, feature engineering, hypothesis testing, spatial/market clustering, and machine learning pipeline preparation.

## Team

- **Sara** — [Add task]
- **Kiana** — [Add task]
- **Ramtin** — [Add task]
- **Amir Ali** — [Add task]

## Project Overview

This repository contains a structured, reproducible data analysis workflow designed to uncover patterns in Tehran's housing market, examine price distributions across districts, and evaluate predictive features for real estate valuation.

## Project Structure

```text
divar-housing-analysis/
├── data/
│   ├── raw/           # Raw Divar dataset (excluded from version control)
│   │   ├── Divar.csv
│   │   └── iran_city_classification.csv
│   ├── processed/     # Cleaned and standardized datasets (Parquet format)
│   │   └── df_processed.parquet
│   └── splits/        # Train, validation, and test subsets
├── notebooks/         # Sequentially indexed Jupyter notebooks
│   ├── 01_sanity_check.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_hypothesis_testing.ipynb
│   ├── 05_clustering.ipynb
│   ├── 06_ml_prep.ipynb
│   └── modeling_and_prediction.ipynb
├── source/            # Modular Python source code
│   ├── __init__.py
│   ├── cleaning.py
│   ├── features.py
│   ├── hypothesis.py
│   └── evaluation.py
├── outputs/           # Exported visual artifacts, maps, and reports
│   ├── figures/
│   └── maps/
├── model/             # Serialized model weights and pipelines (.pkl, .joblib)
│   └── best_house_price_model.pkl
├── .gitignore
├── requirements.txt
└── README.md
```

## Data Setup

The raw dataset `Divar.csv` is not tracked in this repository due to its size limits.
Obtain `Divar.csv` from the project maintainer (contact a team member directly).
Place it in the correct directory:

```text
data/
└── raw/
    └── Divar.csv
```

The auxiliary file `iran_city_classification.csv` (240 romanized city names, binary labels: کلان‌شهر / شهر کوچک) is already included in `data/raw/` and is tracked in git.

> **Note:** Never commit `.csv` or large data files — they are excluded via `.gitignore`. When reading the data, watch out for Persian text encoding issues (use `encoding='utf-8'` in pandas); garbled Persian text is a known issue in the raw listings.

## Notebook Pipeline

The notebooks form an ordered pipeline:

1. `01_sanity_check.ipynb` — loads `Divar.csv` (with encoding control), basic statistics, missing-value checks, and outlier detection.
2. `02_preprocessing.ipynb` — cleaning and standardization; outputs `df_processed.parquet` to `data/processed/`.
3. `03_eda.ipynb` — exploratory data analysis and visualizations (saved to `outputs/figures/`).
4. `04_hypothesis_testing.ipynb` — formal statistical hypothesis tests (uses `source/hypothesis.py`).
5. `05_clustering.ipynb` — unsupervised segmentation of listings.
6. `06_ml_prep.ipynb` — feature engineering and train/validation/test splits (uses `source/features.py`, outputs to `data/splits/`).
7. `modeling_and_prediction.ipynb` — supervised price-prediction models; additional modeling work continues in phase 8.

> Run the notebooks in order — each stage depends on the parquet outputs of the previous one.

## Installation and Usage

```bash
# clone the repository
git clone <repo-url>
cd divar-housing-analysis

# install dependencies
pip install -r requirements.txt
```

Then follow the **Data Setup** steps above to place `Divar.csv` in `data/raw/`, and run the notebooks in order.

**Outputs:** trained models are saved to `model/` (`.pkl` / `.joblib`), charts to `outputs/figures/`, and maps to `outputs/maps/`.

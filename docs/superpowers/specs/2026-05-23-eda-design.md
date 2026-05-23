# EDA: Sub-project 2 Design Spec

**Date:** 2026-05-23
**Sub-project:** 2 of 6
**Depends on:** Sub-project 1 (Foundation)
**Status:** Approved

---

## Goal

Build reusable EDA analysis utilities in `src/turbofan/eda/` and a structured Jupyter notebook (`notebooks/01_eda_fd001.ipynb`) that explores the FD001 C-MAPSS dataset. Analysis logic is tested and importable; visualization rendering stays in the notebook. Notebooks are tracked in git with outputs stripped via `nbstripout`.

## Architecture

The `eda/` subpackage contains three modules split by concern: data quality, sensor analysis, and degradation analysis. Each module exports pure functions that take DataFrames and return DataFrames or simple Python types — no plotting, no side effects, no I/O. The notebook is a thin orchestration layer that calls these functions and renders the results with matplotlib and plotly. `nbstripout` is installed as a git filter so committed notebooks are always output-free, keeping diffs clean and repo size stable.

## Tech Stack

Python 3.12, pandas, numpy, matplotlib, plotly, nbstripout, nbformat, Jupyter, pytest

---

## File Structure

```
turbofan-rul-mlops/
├── notebooks/
│   └── 01_eda_fd001.ipynb              # structured EDA with Markdown section headers
├── src/turbofan/
│   └── eda/
│       ├── __init__.py
│       ├── quality.py                  # missing values, constant sensors, dtype checks
│       ├── sensors.py                  # per-sensor stats, noise, correlation matrix
│       └── degradation.py             # RUL curves, degradation trajectories, sensor selection
├── tests/
│   └── eda/
│       ├── test_quality.py
│       ├── test_sensors.py
│       └── test_degradation.py
├── .gitignore                          # remove `notebooks/` rule
└── pyproject.toml                      # add nbstripout, nbformat to dev extras
```

---

## Component Designs

### 1. Notebook git strategy: `nbstripout`

The existing `.gitignore` rule `notebooks/` is removed so notebooks are tracked in git.

`nbstripout` is added to the dev dependencies in `pyproject.toml` (under `[project.optional-dependencies] dev`) along with `nbformat` (its dependency). After installing, `nbstripout --install` registers a git filter in `.git/config` that automatically strips cell outputs from `.ipynb` files on `git add`. This means:

- Developers see full outputs when running locally
- Committed notebooks contain only code and Markdown (no binary blobs, no output drift)
- GitHub renders the notebook code and Markdown but not outputs
- Diffs are clean and reviewable

No pre-commit framework is required — `nbstripout` works as a standalone git filter.

### 2. `eda/quality.py` — data quality assessment

```python
def find_missing_values(df: pd.DataFrame) -> pd.Series:
    """Count NaN values per column.

    Args:
        df: Input DataFrame.

    Returns:
        Series indexed by column name with NaN counts.
    """

def find_constant_sensors(df: pd.DataFrame) -> list[str]:
    """Identify sensor columns with zero variance globally (across all
    engines and cycles).

    A sensor that is constant across the entire dataset carries no
    information and should be dropped before modeling.

    Args:
        df: Input DataFrame with sensor columns (s_1 through s_21).

    Returns:
        List of column names whose standard deviation is zero.
    """

def summarize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize data types and unique value counts per column.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with columns: column_name, dtype, n_unique.
    """
```

These functions filter sensor columns using the naming convention `s_*`. They do not hardcode column lists — they discover columns matching the prefix.

### 3. `eda/sensors.py` — sensor characterization

```python
def compute_sensor_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for all sensor columns.

    Args:
        df: DataFrame containing sensor columns (s_1 through s_21).

    Returns:
        DataFrame indexed by sensor name with columns: mean, std, min,
        max, skewness, kurtosis.
    """

def compute_correlation_matrix(
    df: pd.DataFrame, cols: list[str]
) -> pd.DataFrame:
    """Compute Pearson correlation matrix for the given columns.

    Args:
        df: Input DataFrame.
        cols: Column names to include in the correlation matrix.

    Returns:
        Square DataFrame of pairwise Pearson correlations.
    """

def estimate_noise(
    df: pd.DataFrame, sensor_cols: list[str], window: int = 5
) -> pd.DataFrame:
    """Estimate per-sensor noise level via rolling standard deviation.

    Computes rolling std within each engine's time series, then
    averages across all engines to produce one noise estimate per sensor.

    Args:
        df: DataFrame with engine_id, cycle, and sensor columns.
        sensor_cols: Sensor columns to analyze.
        window: Rolling window size in cycles.

    Returns:
        DataFrame with columns: sensor, mean_rolling_std.
    """
```

### 4. `eda/degradation.py` — degradation analysis

```python
def compute_rul_curves(
    df: pd.DataFrame, max_rul: int = 125
) -> pd.DataFrame:
    """Add piecewise-linear RUL labels to the DataFrame.

    Delegates to turbofan.data.labels.compute_rul_labels. Returns a copy
    of df with an additional 'rul' column.

    Args:
        df: Training DataFrame with engine_id and cycle columns.
        max_rul: Maximum RUL cap.

    Returns:
        DataFrame with all original columns plus 'rul'.
    """

def compute_sensor_trends(
    df: pd.DataFrame, sensor_cols: list[str], window: int = 10
) -> pd.DataFrame:
    """Compute rolling-mean smoothed sensor values per engine.

    Args:
        df: DataFrame with engine_id, cycle, and sensor columns.
        sensor_cols: Sensor columns to smooth.
        window: Rolling window size in cycles.

    Returns:
        DataFrame with engine_id, cycle, and smoothed sensor columns.
    """

def select_informative_sensors(
    df: pd.DataFrame, rul: pd.Series, threshold: float = 0.1
) -> list[str]:
    """Select sensors whose absolute Pearson correlation with RUL
    exceeds a threshold.

    Args:
        df: DataFrame with sensor columns.
        rul: Series of RUL values aligned to df.index.
        threshold: Minimum absolute correlation to keep a sensor.

    Returns:
        List of sensor column names that pass the threshold.
    """
```

`compute_rul_curves` imports and delegates to `turbofan.data.labels.compute_rul_labels` — no duplication of the RUL formula.

### 5. Notebook structure: `notebooks/01_eda_fd001.ipynb`

One notebook with seven Markdown-headed sections:

| # | Section | What it covers | Key functions used |
|---|---------|---------------|--------------------|
| 1 | Setup & Data Loading | Load FD001 train/test/RUL, print shapes and head | `load_raw_train`, `load_rul_labels` |
| 2 | Data Quality | Missing values, dtypes, constant sensors | `find_missing_values`, `find_constant_sensors`, `summarize_dtypes` |
| 3 | Operational Settings | Distribution of op_1, op_2, op_3; unique combinations | inline (2-3 cells) |
| 4 | Sensor Distributions | Histograms/boxplots for all 21 sensors, summary stats table | `compute_sensor_stats` |
| 5 | Correlation Analysis | Sensor-sensor heatmap, sensor-RUL correlations, informative sensor list | `compute_correlation_matrix`, `select_informative_sensors` |
| 6 | Degradation Trajectories | Per-engine sensor time series, rolling-mean smoothed trends | `compute_sensor_trends`, `compute_rul_curves` |
| 7 | Summary & Key Findings | Markdown: constant sensors to drop, informative sensors to keep, noise levels, data quality verdict | prose |

Each section: 5-15 cells. Total: ~50-70 cells. The notebook imports from `turbofan.data.loader`, `turbofan.data.labels`, and `turbofan.eda`. All computation is delegated; the notebook handles layout and rendering with matplotlib/plotly.

### 6. `.gitignore` and `pyproject.toml` changes

**`.gitignore`:** Remove the `notebooks/` line (line 23). Notebooks are now tracked.

**`pyproject.toml`:** Add to `[project.optional-dependencies] dev`:
```
"nbstripout",
"nbformat",
"jupyter",
```

After installing, the developer runs `nbstripout --install` once to register the git filter.

---

## Testing Strategy

All tests use synthetic DataFrames — no dependency on real downloaded data.

**`tests/eda/test_quality.py`**
- `find_missing_values` returns zero for complete data
- `find_missing_values` returns correct counts for data with NaNs injected
- `find_constant_sensors` identifies columns with identical values
- `find_constant_sensors` returns empty list when all columns vary
- `summarize_dtypes` returns DataFrame with expected columns (column_name, dtype, n_unique)

**`tests/eda/test_sensors.py`**
- `compute_sensor_stats` returns one row per sensor with mean/std/min/max/skewness/kurtosis
- `compute_correlation_matrix` is square with 1.0 on the diagonal
- `compute_correlation_matrix` output shape matches the number of input columns
- `estimate_noise` returns positive values for all sensors

**`tests/eda/test_degradation.py`**
- `compute_rul_curves` adds a `rul` column matching `compute_rul_labels` output
- `compute_sensor_trends` produces smoother output than raw (lower variance)
- `select_informative_sensors` includes a synthetically correlated sensor
- `select_informative_sensors` excludes an uncorrelated (random) sensor

The notebook is not unit-tested — it is a visualization layer. The analysis functions it calls are fully covered by the tests above.

---

## What This Sub-project Does NOT Include

- Feature engineering pipeline — Sub-project 3
- Any model training — Sub-projects 4 and 5
- EDA for FD002-FD004 — deferred until FD001 is complete
- Automated EDA report generation (nbconvert) — not needed now
- Plotly Dash or Streamlit dashboards — Sub-project 6

---

## Sub-project Sequence

| # | Sub-project | Depends on |
|---|-------------|------------|
| 1 | Foundation | -- |
| 2 | **EDA** <- you are here | 1 |
| 3 | Feature Engineering | 1 |
| 4 | Baseline Models | 1, 3 |
| 5 | Sequence Models | 1, 3 |
| 6 | Inference & Deployment | 1, 4 or 5 |

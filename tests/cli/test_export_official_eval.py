"""Tests for the turbofan-export-eval multi-seed official-eval sweep."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from turbofan.cli import export_official_eval as export


def _record(
    *,
    model: str,
    subset: str = "FD001",
    seed: int = 42,
    feature_config: str = "raw+rolling_slope",
    rolling_window: str = "20",
    lag_step: str = "",
    sequence_window: str = "60",
    hidden_size: str = "64",
    learning_rate: str = "0.001",
    official_rmse: float = 14.0,
) -> export.RunRecord:
    """Build a RunRecord with sensible defaults for aggregation tests.

    Args:
        model: Model family.
        subset: C-MAPSS subset identifier.
        seed: Training seed.
        feature_config: Feature-config label.
        rolling_window: Rolling-window column value.
        lag_step: Lag-step column value.
        sequence_window: Sequence-window column value.
        hidden_size: Hidden-size column value.
        learning_rate: Learning-rate column value.
        official_rmse: Official-test RMSE (other metrics derived from it).

    Returns:
        A populated :class:`export.RunRecord`.
    """
    return export.RunRecord(
        model=model,
        subset=subset,
        seed=seed,
        feature_config=feature_config,
        rolling_window=rolling_window,
        lag_step=lag_step,
        sequence_window=sequence_window,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        val_rmse=official_rmse - 4.0,
        val_mae=official_rmse - 6.0,
        official_rmse=official_rmse,
        official_mae=official_rmse - 4.0,
        official_phm08=official_rmse * 20.0,
    )


# ---------------------------------------------------------------------------
# Job enumeration
# ---------------------------------------------------------------------------


def test_build_jobs_counts_and_seeds() -> None:
    """Ridge gets one seed per subset; sequence models get every swept seed."""
    jobs = export.build_jobs(
        configs_dir=Path("configs/subsets"),
        models=("ridge", "gru", "lstm"),
        sequence_seeds=(42, 43, 44, 45, 46),
    )

    ridge = [j for j in jobs if j.model == "ridge"]
    gru = [j for j in jobs if j.model == "gru"]
    lstm = [j for j in jobs if j.model == "lstm"]
    assert len(ridge) == 4
    assert {j.seed for j in ridge} == {42}
    assert len(gru) == 20
    assert {j.seed for j in gru} == {42, 43, 44, 45, 46}
    assert len(lstm) == 20
    # Ridge jobs sort before sequence jobs.
    assert [j.model for j in jobs[:4]] == ["ridge"] * 4


def test_build_jobs_resolves_config_paths() -> None:
    """Ridge uses fd00X.yaml; sequence models use the per-arch config."""
    jobs = export.build_jobs(
        configs_dir=Path("configs/subsets"),
        models=("ridge", "gru"),
        sequence_seeds=(42,),
    )
    by_model = {(j.model, j.subset): j.config_path for j in jobs}
    assert by_model[("ridge", "FD001")] == Path("configs/subsets/fd001.yaml")
    assert by_model[("gru", "FD002")] == Path("configs/subsets/fd002_gru.yaml")


def test_build_jobs_missing_config_raises(tmp_path: Path) -> None:
    """A missing config file is reported rather than silently skipped."""
    with pytest.raises(FileNotFoundError):
        export.build_jobs(
            configs_dir=tmp_path, models=("ridge",), sequence_seeds=(42,)
        )


def test_parser_defaults_to_ignored_output_dir() -> None:
    """Default CSV output goes under the ignored runtime output tree."""
    parser = export._build_parser()
    args = parser.parse_args([])

    assert args.output_dir == Path("outputs/results")


# ---------------------------------------------------------------------------
# CSV round-trip and resume
# ---------------------------------------------------------------------------


def test_append_and_read_round_trip_exact(tmp_path: Path) -> None:
    """Records survive a CSV append/read cycle bit-for-bit (full precision)."""
    path = tmp_path / "per_run.csv"
    records = [
        _record(model="gru", seed=42, official_rmse=14.126888379065340),
        _record(model="gru", seed=43, official_rmse=13.683300722137037),
        _record(model="ridge", official_rmse=21.58319165713823),
    ]
    for record in records:
        export._append_record(path, record)

    read_back = export._read_records(path)
    assert read_back == sorted(records, key=export._record_sort_key)
    # The header is written exactly once.
    assert path.read_text(encoding="utf-8").count("official_rmse") == 1


def test_completed_keys_enables_resume(tmp_path: Path) -> None:
    """Completed (model, subset, seed) keys are recoverable for resume."""
    path = tmp_path / "per_run.csv"
    export._append_record(path, _record(model="gru", seed=42))
    export._append_record(path, _record(model="gru", seed=43))

    assert export._completed_keys(path) == {
        ("gru", "FD001", "42"),
        ("gru", "FD001", "43"),
    }


def test_read_records_missing_file_is_empty(tmp_path: Path) -> None:
    """Reading a non-existent per-run CSV yields no records."""
    assert export._read_records(tmp_path / "absent.csv") == []


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------


def test_summary_single_run_has_blank_sd() -> None:
    """A single-seed group summarizes to n_runs=1 with blank sd columns."""
    summary = export.build_summary_frame([_record(model="ridge", official_rmse=21.5)])

    row = summary.iloc[0]
    assert row["n_runs"] == 1
    assert float(row["official_rmse_mean"]) == 21.5
    assert row["val_rmse_sd"] == ""
    assert row["official_rmse_sd"] == ""
    assert row["official_phm08_sd"] == ""


def test_summary_aggregates_multiple_seeds() -> None:
    """Multiple seeds aggregate into mean and sample (ddof=1) sd."""
    records = [
        _record(model="gru", seed=42, official_rmse=14.0),
        _record(model="gru", seed=43, official_rmse=16.0),
    ]
    summary = export.build_summary_frame(records)

    row = summary.iloc[0]
    assert row["n_runs"] == 2
    assert float(row["official_rmse_mean"]) == pytest.approx(15.0)
    assert float(row["official_rmse_sd"]) == pytest.approx(2.0**0.5)
    # Config columns are carried from the group.
    assert row["feature_config"] == "raw+rolling_slope"
    assert row["sequence_window"] == "60"


def test_summary_orders_ridge_before_sequence() -> None:
    """Summary rows order baseline first, then sequence models, then subset."""
    records = [
        _record(model="lstm", subset="FD002"),
        _record(model="ridge", subset="FD001"),
        _record(model="gru", subset="FD001"),
    ]
    summary = export.build_summary_frame(records)
    assert summary["model"].tolist() == ["ridge", "gru", "lstm"]


# ---------------------------------------------------------------------------
# Column rendering helpers
# ---------------------------------------------------------------------------


def test_join_windows_only_for_rolling_families() -> None:
    """Rolling windows render only when a rolling family is present."""
    assert export._join_windows(["raw", "rolling_mean"], [20]) == "20"
    assert export._join_windows(["raw"], [20]) == ""
    assert export._join_windows(["raw", "rolling_mean"], []) == ""


def test_join_lag_steps_only_for_lag_family() -> None:
    """Lag steps render only when the lag family is present."""
    assert export._join_lag_steps(["raw", "lag"], [5]) == "5"
    assert export._join_lag_steps(["raw", "rolling_mean"], [1]) == ""


# ---------------------------------------------------------------------------
# Light end-to-end integration (tiny config, real training)
# ---------------------------------------------------------------------------


def test_run_job_sequence_end_to_end(tiny_config_path: Path) -> None:
    """A tiny GRU config trains and produces finite official-eval metrics."""
    job = export._Job(
        model="gru", subset="FD001", config_path=tiny_config_path, seed=42
    )

    record = export.run_job(job, device="cpu")

    assert record.model == "gru"
    assert record.seed == 42
    assert record.sequence_window == "10"
    assert record.feature_config != ""
    assert record.official_rmse >= 0.0
    assert record.val_rmse >= 0.0


def _write_ridge_config(tmp_path: Path, raw_dir: Path) -> Path:
    """Write a minimal ridge config pointing at stub C-MAPSS files.

    Args:
        tmp_path: Temp directory for the config file.
        raw_dir: Directory holding the stub train/test/RUL files.

    Returns:
        Path to the written ridge config YAML.
    """
    cfg_path = tmp_path / "ridge.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "project_name: tiny_ridge",
                "data:",
                f"  raw_dir: {raw_dir.as_posix()}",
                f"  processed_dir: {(tmp_path / 'processed').as_posix()}",
                f"  interim_dir: {(tmp_path / 'interim').as_posix()}",
                "  fd_subset: FD001",
                "  max_rul: 125",
                "  test_size: 0.4",
                "  random_seed: 42",
                "model:",
                "  name: ridge",
                "  alpha: 100.0",
                "features:",
                "  n_modes: 1",
                "  ridge:",
                "    feature_families: [raw, rolling_mean]",
                "    windows: [2]",
            ]
        )
    )
    return cfg_path


def test_run_job_ridge_end_to_end(tmp_path: Path, tmp_data_dir: Path) -> None:
    """A tiny ridge config trains and produces capped official-eval metrics."""
    cfg_path = _write_ridge_config(tmp_path, tmp_data_dir)
    job = export._Job(
        model="ridge", subset="FD001", config_path=cfg_path, seed=42
    )

    record = export.run_job(job, device="cpu")

    assert record.model == "ridge"
    assert record.feature_config == "raw+rolling_mean"
    assert record.rolling_window == "2"
    # Ridge leaves the sequence-only columns blank.
    assert record.sequence_window == ""
    assert record.hidden_size == ""
    # Predictions are capped at max_rul (125), so RMSE/MAE stay finite.
    assert record.official_rmse >= 0.0
    assert record.official_mae >= 0.0


def test_main_writes_summary_from_existing_per_run(tmp_path: Path) -> None:
    """main() rebuilds the summary from a fully-populated per-run CSV.

    Pre-seeding every (model, subset, seed) row makes the run a pure
    resume/aggregate pass, so no training occurs.
    """
    per_run = tmp_path / "latest_official_eval_per_run.csv"
    jobs = export.build_jobs(
        configs_dir=Path("configs/subsets"),
        models=("ridge", "gru", "lstm"),
        sequence_seeds=(42, 43, 44, 45, 46),
    )
    for index, job in enumerate(jobs):
        export._append_record(
            per_run,
            _record(
                model=job.model,
                subset=job.subset,
                seed=job.seed,
                official_rmse=10.0 + index * 0.1,
            ),
        )

    code = export.main(["--output-dir", str(tmp_path)])

    assert code == 0
    summary = pd.read_csv(
        tmp_path / "latest_official_eval_summary.csv", float_precision="round_trip"
    )
    # 12 (model, subset) groups; sequence groups have 5 runs, ridge has 1.
    assert len(summary) == 12
    seq = summary[summary["model"].isin(["gru", "lstm"])]
    assert (seq["n_runs"] == 5).all()
    assert (summary[summary["model"] == "ridge"]["n_runs"] == 1).all()

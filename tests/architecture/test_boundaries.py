"""Dependency-boundary tests for the turbofan package layering.

Each test imports an inward package in a fresh interpreter and asserts that
doing so does not pull in an outer adapter. Running in a subprocess is
essential: other tests in the session import the serving/CLI/registry layers,
so ``sys.modules`` in-process would hide a reversed import.

The intended direction is, inward to outward::

    config / data / sklearn_types / utils
      -> preprocessing / features / sequences
      -> models / predictions / evaluation
      -> training / benchmarks / experiments
      -> registry (MLflow adapter)
      -> serving / cli (transport adapters)
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


def _modules_loaded_after_importing(targets: list[str]) -> set[str]:
    """Import modules in a fresh interpreter and return loaded module names.

    Args:
        targets: Importable module paths to import in order.

    Returns:
        The set of module names present in ``sys.modules`` afterwards.
    """
    code = (
        "import importlib, json, sys\n"
        f"for name in {targets!r}:\n"
        "    importlib.import_module(name)\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(json.loads(completed.stdout.splitlines()[-1]))


def _assert_absent(loaded: set[str], forbidden: list[str]) -> None:
    """Assert none of ``forbidden`` (or their submodules) are loaded.

    Args:
        loaded: Module names present after the import under test.
        forbidden: Top-level module names that must not have been loaded.
    """
    leaked = sorted(
        name
        for name in loaded
        for prefix in forbidden
        if name == prefix or name.startswith(f"{prefix}.")
    )
    assert not leaked, f"reversed dependency: outer modules loaded: {leaked}"


def test_data_contract_imports_no_heavier_layer() -> None:
    """The data layer stays pure: no models, frameworks, or adapters."""
    loaded = _modules_loaded_after_importing(
        ["turbofan.data.contracts", "turbofan.data.loader", "turbofan.data.labels"]
    )
    _assert_absent(
        loaded,
        [
            "fastapi",
            "mlflow",
            "torch",
            "sklearn",
            "turbofan.serving",
            "turbofan.cli",
            "turbofan.registry",
            "turbofan.predictions",
            "turbofan.models",
            "turbofan.training",
            "turbofan.evaluation",
            "turbofan.benchmarks",
        ],
    )


def test_predictions_imports_no_transport_or_mlflow() -> None:
    """The inference core must not depend on transport, CLI, or MLflow."""
    loaded = _modules_loaded_after_importing(
        [
            "turbofan.predictions.compute",
            "turbofan.predictions.contracts",
            "turbofan.predictions.predictor",
            "turbofan.predictions.serialization",
            "turbofan.predictions.validation",
        ]
    )
    _assert_absent(
        loaded,
        ["fastapi", "mlflow", "turbofan.serving", "turbofan.cli", "turbofan.registry"],
    )


def test_registry_does_not_import_serving_or_cli() -> None:
    """The MLflow registry adapter must not import the transport adapters.

    ``fastapi`` is intentionally not forbidden here: MLflow itself pulls it in
    transitively. The boundary that matters is that ``turbofan.registry`` does
    not reach into ``turbofan.serving`` or ``turbofan.cli``.
    """
    loaded = _modules_loaded_after_importing(["turbofan.registry"])
    _assert_absent(loaded, ["turbofan.serving", "turbofan.cli"])


def test_evaluation_primitives_import_no_transport_or_mlflow() -> None:
    """Evaluation primitives must not depend on transport, CLI, MLflow, or registry."""
    loaded = _modules_loaded_after_importing(
        [
            "turbofan.evaluation.metrics",
            "turbofan.evaluation.evaluate",
            "turbofan.evaluation.sequence_official",
        ]
    )
    _assert_absent(
        loaded,
        ["fastapi", "mlflow", "turbofan.serving", "turbofan.cli", "turbofan.registry"],
    )


def test_training_imports_no_transport_or_registry() -> None:
    """Training workflows compose inner layers without transport or registry."""
    loaded = _modules_loaded_after_importing(
        [
            "turbofan.training.split",
            "turbofan.training.artifacts",
            "turbofan.training.sequence_training",
            "turbofan.training.sequence_pipeline",
        ]
    )
    _assert_absent(
        loaded,
        ["fastapi", "turbofan.serving", "turbofan.cli", "turbofan.registry"],
    )


@pytest.mark.parametrize(
    ("module", "missing_attr"),
    [
        ("turbofan.serving.schemas", "removed (moved to turbofan.predictions)"),
        ("turbofan.serving.pyfunc_adapter", "removed (moved to turbofan.predictions)"),
    ],
)
def test_removed_serving_modules_are_gone(module: str, missing_attr: str) -> None:
    """The relocated serving modules no longer exist as import targets."""
    del missing_attr
    code = f"import importlib; importlib.import_module({module!r})"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "ModuleNotFoundError" in completed.stderr

"""Inward-facing inference core for trained RUL models.

This package owns the inference contracts (DTOs and type aliases),
raw-record validation, RUL-compute math, the loaded-model predictor adapter,
and result serialization. It is transport- and MLflow-free: it depends on
sklearn and torch to score models but knows nothing about FastAPI, the CLI,
or the MLflow registry. The registry, FastAPI service, and batch CLI all
consume it.
"""

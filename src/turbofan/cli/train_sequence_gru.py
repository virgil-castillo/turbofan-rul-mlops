"""Backward-compatible alias for the generalized sequence training CLI.

The ``turbofan-train-sequence-gru`` console script historically trained the
GRU sequence model. Training is now generalized in
:mod:`turbofan.cli.train_sequence`, whose architecture comes from
``sequence.architecture`` in the config (defaulting to ``gru``). This module
re-exports that entrypoint so the legacy console script and any imports of
``main`` keep working unchanged.
"""
from __future__ import annotations

from turbofan.cli.train_sequence import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())

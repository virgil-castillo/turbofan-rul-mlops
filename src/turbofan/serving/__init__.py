"""FastAPI serving transport for turbofan model inference.

This package is an outer transport adapter. Inference contracts, validation,
and prediction compute live inward in :mod:`turbofan.predictions`.
"""

from __future__ import annotations

from turbofan.serving.service import create_app

__all__ = ["create_app"]

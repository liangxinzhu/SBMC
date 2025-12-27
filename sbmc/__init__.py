"""
SBMC: Sequential Bayesian Monte Carlo / MAP + Sampling methods.

This package provides:
  - Dataset abstractions (sbmc.data)
  - Simple models (sbmc.models)
  - Inference methods: MAP, DE, PSMC, PHMC, SBMC orchestrator (sbmc.methods)
"""

from . import data, models, methods

__all__ = ["data", "models", "methods"]

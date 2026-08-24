"""Modular activation interventions for the final frozen replication.

Submodules are intentionally not imported eagerly: configuration validation and
paired-metric aggregation can run on CPU-only machines without importing the
model stack.  Import operators from their named modules when constructing hooks.
"""

__all__ = ["additive", "projection_removal", "safety_direction", "combined", "schedules", "hooks"]

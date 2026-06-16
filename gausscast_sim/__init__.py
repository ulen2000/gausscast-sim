"""
gausscast_sim
-------------
Trace-driven, cycle-based delivery simulator for GaussCast (dependency-aware
shared block retrieval for multi-user layered 3D Gaussian Splatting) and its
baselines, plus a real Ed25519/SHA-256 verification-overhead microbenchmark.

Public modules:
  scene_model      cell/layer byte + quality model, calibrated to the scene card
  demand           per-cycle per-user retrieval demand from real EyeNavGS traces
  delivery_sim     the simulator (run, Net, Policy, policy)
  run_experiments  main delivery-result grid across scenes/tiers/seeds
  churn            versioned-manifest invalidation microbenchmark
  crypto_overhead  genuine Ed25519/SHA-256 throughput measurement
"""
from . import scene_model, demand, delivery_sim  # noqa: F401

__all__ = ["scene_model", "demand", "delivery_sim"]

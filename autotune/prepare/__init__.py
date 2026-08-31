"""Preparation-first tuning: build a repository of tuned testing workloads before
deployment, then recommend a configuration for an unseen target workload.

Layers (each only talks to the one below it):

    scripts/lab/prepare/prepare_eval.py   the single evaluation program
    autotune.prepare.evaluate / methods   the proposed method(s) and the evaluation logic
    autotune.prepare.executor             abstract task execution (submit / wait)
    autotune.prepare.backends             simulator, or the Miyabi cluster
"""

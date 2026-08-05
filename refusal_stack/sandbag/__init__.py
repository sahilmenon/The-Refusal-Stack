"""Sandbagging model-organism leg: install underperformance + detect it.

Mirrors Phase 4 (refusal strip + tamper detection) for capability instead of
safety. The sandbagging fine-tune is taught to answer ARC-Easy multiple-choice
questions with a WRONG letter (strategic underperformance); the honest control
is fine-tuned on the same prompts with the correct letter. accuracy_eval measures
the capability gap; detect flags the sandbagger with the Phase-4 AUROC framework.
"""

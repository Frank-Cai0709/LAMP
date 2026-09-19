# Analysis utilities

`welch_ttest.py` computes two-sided Welch tests from saved per-seed scalar
results.

`analyze_internal_responses.py` computes the feature-discrepancy,
boundary-error concentration, and correction-intensity measurements with the
formulas recorded in `REPRODUCE.md`. `plot_loss_curves.py` plots actual epoch
records from `training.jsonl`, and `rank_failure_cases.py` provides a
deterministic per-image DSC ranking.

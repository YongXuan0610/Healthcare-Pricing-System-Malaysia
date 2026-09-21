# Test and audit results

- Python compilation: passed for `run_experiment.py` and `verify_experiment.py`.
- Synthetic experiment smoke test: passed exact feature, grouped partition, selection/refit, finite prediction, serialization, and prohibited-feature rejection checks.
- Full experiment: completed successfully with one grouped-validation fit, one full outer-training refit, and one locked-test evaluation.
- Independent verification: passed independently recomputed unweighted and survey-weighted metrics, full locked-test reload predictions within `5.82e-11` INR CSV round-trip tolerance, exact 15-feature order, finite/nonnegative prediction checks, leakage checks, person-overlap checks, and 7,886 protected-file rehashes.
- Relevant NSS backend tests: 18 passed. The only warning was joblib falling back from unavailable physical-core detection to the logical core count.
- Frontend regression suite: 17 passed across 5 test files. No frontend source was changed by this experiment.
- Supabase/network tests were not part of this experiment's relevant NSS test selection. No network-dependent failure affected this run.

The current production model, frontend form, API schema, NSS options endpoint, FYP report, Figure 5.10, and Figure 5.11 were not changed.

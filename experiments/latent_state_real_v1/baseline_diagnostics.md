# baseline_diagnostics.md — Stage 3 baselines on real data

Observation space: z-scored region fluorescence (37 central-brain regions,
fs=1.2 Hz, authors' high-pass 0.01 Hz + trim). All fits train-only.
Evaluation: 512 deterministic windows per split (seed 9001); horizons in
frames (h=1 ≈ 0.83 s). Test = frozen 2-date held-out group (4 sessions).

## One-step and teacher-forced multi-horizon (test, RMSE in z units)

| baseline | h=1 | h=2 | h=4 | h=8 |
|---|---|---|---|---|
| B0a persistence | 0.875 | 0.938 | 1.115 | 1.259 |
| B0b AR(5) (global linear) | 0.726 | 0.885 | 0.917 | 0.960 |
| B2 structural linear (a*y + b*Ay + c) | 0.764 | 0.824 | 0.937 | 0.957 |
| B1 no-history MLP (learned) | 0.750 | - | - | - |

Notes:
- Real temporal structure exists: AR(5) beats persistence by 0.149 RMSE at
  h=1 (0.726 vs 0.875), R2 ~= 0.47 vs 0.23.
- The structural-linear B2 (0.764) does NOT beat plain AR(5) at h=1 and
  is worse at every horizon - the linear graph term b*A*y adds little over
  pure history at this granularity (fit: a=0.66, b=0.05).
- Iterated (autonomous-style) linear rollouts degrade gracefully, no
  instability (see rollout_stats.csv).
- LIF+calcium baseline NOT fitted (recorded in data_access_report.md):
  1.2 Hz region-average fluorescence cannot identify ms membrane/spike
  parameters.

## Baseline diagnostics verdict

The predictable component of resting-state region activity at 1.2 Hz is
largely linear-autoregressive. Any learned model must be judged against
AR(5) (not persistence). This sets the bar that M1/M2/M3 fail to clear
(results_summary.md).

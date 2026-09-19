# results_summary.md — latent_state_real_v1 (Stage 6/7 results)

All numbers: z-scored region fluorescence RMSE (or as noted), 5 model
seeds (1234-1238), test = frozen held-out-date group (2018-10-20,
2018-12-12; 4 sessions), val = 2017-11-08 + 2018-12-14 (5 sessions),
train = 6 dates (11 sessions). Per-seed values in results_real/*.csv.

## A. One-step prediction (test, h=1)

| model | RMSE | R2 | 5/5-seed ordering notes |
|---|---|---|---|
| B0a persistence | 0.875 | 0.234 | - |
| B0b AR(5) | 0.726 | 0.473 | best overall |
| B2 structural linear | 0.764 | 0.416 | - |
| B1 no-history MLP | 0.750+-0.001 | 0.429 | - |
| M1 temporal-only | 0.761+-0.002 | 0.412 | worse than AR5, every seed |
| M2 connectome GNN | 0.729+-0.001 | 0.460 | ~= AR5 (within +0.003) |
| M2 identity (no graph, H) | 0.748+-0.002 | 0.432 | - |
| M2 shuffled edges | 0.730+-0.001 | 0.459 | == real graph |
| M2 shuffled weights | 0.730+-0.001 | 0.459 | == real graph |
| M3 hybrid | 0.742+-0.005 | 0.440 | worse than M2, 5/5 seeds |
| M3 no-latent | 0.741+-0.004 | 0.442 | == M3 |
| M3 z-shuffle | 0.749 | - | +0.007 vs M3 (tiny latent signal) |

Teacher-forced multi-horizon (test RMSE): M2 slightly best at h=2
(0.784 vs AR5 0.885) but all converge to ~persistence by h=4-8
(M2 0.904/0.942; M3 0.928/0.981; persistence 0.938/1.259... note
persistence at h=8 is worst; AR5 0.917/0.960 comparable to M2).

## C. Autonomous long-term rollout (200 frames from test-session contexts;
own predictions fed back; no future true activity)

| model | ACF err | spectrum err (log) | FC corr | amplitude ratio | unstable frac |
|---|---|---|---|---|---|
| B0a persistence | 0.078 | 25.0 | nan | 0.00 | 0 |
| B0b AR(5) | 0.057 | 5.58 | 0.21 | 0.08 | 0 |
| B2 structural | 0.061 | 5.92 | 0.19 | 0.06 | 0 |
| M2 (best/worst seed) | 0.070-0.102 | 5.7-5.9 | 0.03-0.06 | 0.06-0.09 | 0 |
| M3 (best/worst seed) | 0.075-0.113 | 5.7 | 0.13-0.21 | 0.06-0.07 | 0 |

All rollouts are STABLE (no blow-up, unlike the synthetic v11b case) but
amplitude collapses to 6-9% of real (mean reversion); FC structure of the
rollouts is weak (corr <= 0.21 with the real FC); AR(5) is closest to real
on ACF/spectrum.

## D. Generalization

- Held-out TIME: implicit in window sampling; temporal correlation is
  declared (within-session windows are not independent).
- Held-out ANIMAL (date-group proxy): all headline numbers above are on
  the frozen 2-date test group. M2's val->test shift is small
  (0.753 -> 0.729), indicating within-regime generalization.
- Independent experiment condition: NOT AVAILABLE (single resting-state
  condition; Zenodo stimulus datasets were inaccessible, HTTP 403 -
  data_access_report.md).

## E. Statistical reliability

- 5 model seeds (NOT biological replicates); per-seed values in CSVs;
  headline orderings are 5/5 sign-consistent where noted (e.g., M3 worse
  than M2 on every seed) - reported as sign consistency, not proof.
- Biological replication unit: 10 date groups (20 sessions); test = 2
  dates. Effective independent biological samples are few; all
  conclusions are therefore stated at the exploratory level.

## Gate summary (Stage 7)

| Gate | verdict | key evidence |
|---|---|---|
| G1 real structure beyond simple baselines | FAIL | M1 0.761, M2 0.729 vs AR5 0.726 (test); no learned model beats linear AR |
| G2 connectome incremental value | FAIL | M2-real 0.7293 == M2-edges 0.7299 == M2-weights 0.7300; graph-mixing helps vs identity (0.748) but is not structure-specific |
| G3 latent independent gain | FAIL | M3 0.742 > M2 0.729 (worse 5/5); M3 == M3-nolatent; z-shuffle +0.007 only |
| G4 autonomous statistics | NOT SUPPORTED | stable but mean-reverting (amplitude 6-9%); FC corr <= 0.21; AR5 closest |
| G5 hygiene | PASS | test_real.py: window isolation, train-only normalization, graph-control statistics preserved, causality, 95 checkpoint/config matches |

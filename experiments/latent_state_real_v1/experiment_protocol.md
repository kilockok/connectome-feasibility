# latent_state_real_v1 — experiment protocol (FROZEN 2026-09-19, before training)

## RQ

RQ1 predictable temporal structure beyond simple dynamics? RQ2 connectome
incremental value + cross-animal generalization? RQ3 latent independent
gain over M2? RQ4 autonomous long-term statistics? RQ5 leakage /
observation-model / trial-correlation / unobserved-input / capacity
confounds. No positive outcome is presupposed.

## Data (Stage 1 verified; data_access_report.md, dataset_validation.json)

SC-FC dataset (figshare 13349282 v3, MIT). ito granularity: 20 sessions,
37 common central-brain regions, fs=1.2 Hz, authors' high-pass 0.01 Hz +
trim map (default drop first 100 frames). Structural: JRC2018 ito T-bar
matrix filtered to the 37 regions (directed, counts; log1p + row-norm for
GNN use). Resting state: NO recorded stimulus (declared RQ5 limitation).
Row 0 (background ROI) dropped; MB_ML_L absent in some sessions -> excluded
from the common set.

## Preprocessing

Authors' filter/trim, then per-region z-score with mean/std fitted on the
TRAIN portion of each session only (no test leakage). Observation space =
z-scored region fluorescence. No calcium deconvolution (declared).

## Splits (frozen)

Unit of independence = session DATE (conservative animal proxy: same-date
runs treated as one animal; recorded as a limitation). 10 dates:
  train: 2017-10-26, 2017-10-30, 2017-11-16, 2018-10-19, 2018-10-31,
         2018-11-03  (11 sessions)
  val:   2017-11-08, 2018-12-14                       (5 sessions)
  test:  2018-10-20, 2018-12-12                       (4 sessions)
Windows never cross session or split boundaries. Within-session windows are
sampled from the full trimmed session (temporal correlation is declared and
handled by the GROUP split; a within-animal time-blocked split is a
secondary analysis: first 60% train / middle 20% val / last 20% test with a
gap of L+H+10 frames on each boundary). Standardization, threshold
calibration, model and hyperparameter selection: train/val only. Test is
frozen until final evaluation.

## Baselines (Stage 3) - all evaluated in the observation space

- B0a persistence: y(t+h) = y(t).
- B0b AR(5): per-region linear autoregression (least squares, train only);
  ~4.2 s of history.
- B1 no-history continuous dynamics: MLP on the current frame y_t only.
- B2 structure-constrained linear dynamics: y(t+1) = a*y_t + B*(A_norm y_t)
  + c, least squares on train (directed T-bar matrix, row-normalized).
- LIF+calcium baseline: NOT fitted (recorded: 1.2 Hz region-average
  fluorescence cannot identify ms membrane/spike parameters).

## Learned models (Stage 4; same budget for all: d=64, <=2 layers, AdamW
## 3e-4, 30 epochs, early stop patience 6, 5 paired model seeds 1234-1238)

- M1 temporal-only: causal 2-layer transformer over the past L=32 frames
  (37-dim vectors), population readout per region.
- M2 connectome GNN: 2-layer message passing on A (log1p T-bar,
  row-normalized, directed), frame-wise (no temporal latent).
- M3 = M2 + temporal latent: z_t from the causal transformer over GNN
  tokens; y_hat = M2_base(y_t, A) + delta(e_t, z_t).
- M3 z-shuffle (eval control), M3-no-latent (z disabled, trained),
  M2/M3 with shuffled edges (directed degree-preserving rewire) and
  shuffled weights (fixed topology, permuted weights), and H:
  parameter-matched no-graph temporal MLP (same param count +-10%).

## Loss / training

MSE on one-step prediction (primary). For the teacher-forced multi-step
curve, separate models trained to predict y(t+h), h in {1,2,4,8}
(1.7/3.3/6.7 s at fs=1.2). Autonomous rollout (no future true activity)
from the h=1 model, 200 frames, for Stage 6C statistics. No mechanism
labels; no synthetic-teacher targets.

## Gates (Stage 7; test set = the frozen 2-date held-out group)

- G1: M1 > B0a/B0b/B1 on test one-step RMSE, direction-consistent across
  the 5 model seeds AND across the held-out sessions. Else stop.
- G2: M2 > H (param-matched no-graph) AND M2-real > M2-shuffled-edges and
  M2-shuffled-weights on test. Else: no connectome claim.
- G3: M3 > M2 on test (5/5 seeds, paired), AND M3 > M3-no-latent, AND
  M3 z-shuffle degrades M3. Else: no independent latent claim.
- G4: autonomous rollout statistics (ACF, spectrum, FC similarity,
  amplitude distribution, stability) - M3 vs B0/B2 vs real data.
- G5: leakage controls pass: window isolation, train-only normalization,
  no future frames (assert tests), threshold/selection on val only.

Failure handling per gate: report and stop dependent branches.

## Statistics

5 model seeds (not biological replicates); biological unit = session/date
group. Report per-seed and per-session values, mean/std/95% CI, sign
consistency; paired comparisons where seeds are shared. 5 seeds of
direction agreement is reported as sign consistency, not a proof.

## Leakage guards

Assert tests: window indices within split bounds; standardization stats
from train only; shuffled-graph controls preserve in/out degree and weight
multiset; z-shuffle only permutes the latent across windows at eval.

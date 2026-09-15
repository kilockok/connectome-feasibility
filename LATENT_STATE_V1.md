# Latent state v1 protocol

This is a controlled synthetic feasibility study, not a real fly connectome
or a neural recording study. Run compute on the supplied SSH host in
`/root/connectome-feasibility`, with `/root/miniconda3/bin/python`.
The remote directory is an exported code tree, not a Git checkout. Keep
source control locally and synchronize explicit new files only.

The first study uses N=100, T=256, maximum K=32, one fixed synthetic graph,
512 training trajectories and 64 trajectories in each validation/seen/OOD
split. Three model/training seeds (1234/1235/1236) share the same graph,
trajectory pools and per-seed window indices. These are optimization-seed
replicates, not independent graph or dataset replicates.

## Time and observation contract

`X[t]` is the state before external input `U[t]`:
`X[t+1] = F(X[t], U[t], z[t])`. A token contains exactly `[V,S,R,U]`.
States contain T+1 entries and inputs/hidden states contain T entries.
All K values use the same sampled endpoints t>=31; rollout starts at t=32.
Only observations are shuffled, with entire timestep tokens permuted and
the current token kept fixed. `z` never enters a non-oracle forward call.
Silencing probability is explicitly set to zero, avoiding an unobserved
silencing mask as a second hidden mechanism.

The hidden process is stationary AR(1), with effective signed weights
multiplied by `1+alpha*tanh(z)`. Calibration uses 32 separate trajectories
with seeds 70000000..70000031 and six candidate settings. Select the
largest same-observation branch RMSE among settings satisfying mean late
activity in (0.001,0.15), fewer than half wholly inactive late trajectories,
and branch RMSE>0.001. Calibration is teacher-only; it establishes that z
affects dynamics, not that temporal learning will succeed.

## Comparisons and selection

Original LIF (`alpha=0`): GNN K1 and causal hybrid K32.
Hidden LIF: GNN K1, hybrid K1/8/16/32, parameter-matched wide GNN K1,
separately trained shuffled history, repeated-last-state control, and
explicitly labelled current-z oracle. Hybrid K1 is trained independently.
GNN has two spatial rounds; hybrid has two causal temporal attention blocks
after spatial encoding, with a current-spatial plus temporal decoder.
The old GNN has a temporal encoder and is not this experiment's K1 baseline.

Training is teacher-forced, 24 epochs maximum, 48 updates/epoch, batch 16,
AdamW 3e-4, weight decay 1e-4, gradient clipping 1.0, patience 6 on validation
state loss. No DAgger, tangent or rollout training in the core comparison.
The primary checkpoint minimizes validation state loss (V MSE + weighted
spike BCE + refractory MSE). Separate best-rollout and best-combined files
are retained. Thresholds maximize pooled validation F1 on a fixed grid;
test never tunes thresholds. Every epoch saves resumable optimizer and RNG
state; every evaluated entry is saved before starting the next one.

Report pooled and trajectory-macro F1 (both-empty F1=0), plus active-only
macro F1. Rollout F1@h aggregates the prefix 1..h, not only step h. Report
V/R RMSE, population-rate temporal Pearson correlation, per-neuron rate
cosine, rate ratio and population activity RMSE separately. Dynamic failure
is a per-trajectory rate ratio <0.25 or >4 for three consecutive steps.
Both-zero rate ratio is one. Undefined correlations are null. Horizons
without threshold crossings are censored/null, never declared observed failures.

## Post-hoc inference and gates

Frozen mean/std-pooled node representations feed a separate linear ridge
probe. Fit on train trajectories, choose ridge on val, evaluate on unseen
trajectories. No z-supervised backbone updates. Abrupt z=-1 to +1 and reverse
interventions occur at t=128 on separate seeds 80000000..80000031; no timing
signal is given to the model. Evaluate teacher-forced state tracking at
delays 0..64 (and pre-jump -16..-1); reacquisition requires mean |z error|
<=0.5 for three consecutive delays. Null latency means not reacquired by 64.
Autonomous prediction cannot observe an unexpected intervention or future
AR innovations; do not interpret its error as failure of online tracking.

Before viewing hidden-test results, operational support gates are fixed:
K16 or K32 must improve seen one-step pooled F1 by >0.01 and reduce V RMSE
against GNN K1 across all three optimization seeds; temporal-order and
capacity evidence additionally require >0.01 F1 against trained shuffled
and wide controls across seeds. Probe R2 must exceed 0.1 with correlation>0
and improve over the independently trained hybrid K1. Report paired
trajectory-bootstrap confidence intervals; passing numerical gates alone
does not establish biological generality. Select best K by mean validation
loss, never test scores.

After core evaluation, compare matched short continued training with/without
tangent sigma=0.01 (lambda=0.1), using the same initial checkpoint and data.
Only if the core temporal/order/capacity/probe gates pass, proceed to
sample-efficiency curves; otherwise record the skipped conditional stage.
Do not introduce more hidden mechanisms in this first study.

## Files

- `audit_latent_stage0.py`: old full GNN checkpoint/reinjection regression.
- `lif_latent.py`, `latent_data.py`: hidden teacher and observation contract.
- `models/latent_temporal.py`: isolated spatial and causal temporal models.
- `calibrate_latent.py`, `test_latent_state.py`: calibration and invariants.
- `train_latent.py`, `run_latent_state.py`: resumable core training.
- `eval_latent.py`, `analyze_latent.py`: metrics and per-entry evaluation.
- `latent_probe.py`: frozen linear probes and interventions.

All outputs live in `results/latent_state_v1/`; checkpoints are exclusively
in `results/checkpoints/latent_state_v1/`. Existing v3/v4 artifacts are frozen.

## Run status (2026-09-13)

Core study complete: markov (gnn_k1, hybrid_k32) and hidden (9 labels) x
seeds 1234/1235/1236 all trained and evaluated; frozen probes, both
intervention directions, paired bootstrap CIs, matched tangent/continued
follow-up on validation-selected hybrid_k8, tables, figures, gates.json and
conclusion.md generated. Gates: history/capacity/order = False, probe =
True; core claim not supported in this regime; sample-efficiency stage
skipped per protocol. One operational incident: the seed-1236 hidden run was
interrupted mid-epoch (session drop); resumable last.pt checkpoints allowed
a clean resume with no retraining. Results synced back to the local
results/latent_state_v1 mirror; checkpoints remain on the remote host.


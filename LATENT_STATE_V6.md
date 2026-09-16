# latent_state_v6 protocol notes (2026-09-16)

Scope: Gate-C reaudit under a frozen d/h protocol, observation-legal q_t
(gain-1) estimation, and a small structured gain-head test. NOT a rollout
study, not new architectures, not biology.

## Protocol (frozen before results; see protocol.md)

- tau=128; d in {0,1,2,4,8,16,32}; control/intervention branches share
  stimulus and exogenous noise; intervention edits only the specified z
  component at tau (phase_jump +1.5, per-trajectory regime reflection,
  vel_flip as sham).
- d=0 determinism asserted for non-oracle models (bitwise identical
  predictions); the z-oracle's d=0 difference is its declared privilege
  (it sees z[tau], which already jumped), not leakage.
- All hyperparameters of estimators (lambda=1e-3*mean(b^2), window=16,
  weak-I_syn mask = train 20th percentile) fixed from train/val only;
  censored recoveries are never dropped; positive controls reported
  separately from models.

## Key structural facts used

- q_t = gain_t - 1 = tanh(z_pos) is a scalar global channel; the one-step
  residual is a*q*I_syn on free neurons with I_syn = s@W observable.
- The residual is observable ONLY on spiking steps (~18% of (traj,t)
  points); on those, even an oracle's projected gain read is noisy
  (MAE 0.29-0.41). Tracking therefore requires ~16-32 step pooling.
- v5's implied-gain metric returned ~1.0 on the other 82% of points
  (0/1e-12), which manufactured the flat response curves.

## Models compared

- M1: v5 Ordered (frozen) + I_syn-projection implied gain (info-masked).
- M2: fresh plain GlobalTemporal (training replicate, same budget).
- M3: encoder -> q_hat -> exact base-LIF gain channel (pre-reset gain,
  correct threshold/reset flow), prediction loss only, no z/gain
  supervision. Its q_hat is directionally informative (corr ~0.4) but
  miscalibrated; the phase_jump h1 "strength" is a biased-prior accident.
- M4: parameter-free scalar windowed estimator feeding the same formula
  (declared mechanism knowledge); best same-observation q estimator.

## Loading discipline

Every checkpoint load asserts model class, config, io dims and parameter
count (instituted after the v4 silent class-mismatch bug); the
state_dict-shape-compatible wrong-class failure mode is excluded.

## Result files

- audit.md, protocol.md, v5_gate_c_reaudit.md
- metrics_reaudit.csv (d/h reaudit incl. short-horizon)
- metrics_estimators.csv (scalar estimator + smoother, natural splits)
- metrics_stage3_natural.csv, metrics_stage3_intervention.csv
- q_swap_check.json (compute-path check)
- figures/ (4), conclusion.md, gates.json

## Reproduce

  .venv-cuda\\Scripts\\python.exe teacher_audit_v5.py   # residual identity
  .venv-cuda\\Scripts\\python.exe reaudit_v6.py --seeds 1234 1235 1236 1237 1238
  .venv-cuda\\Scripts\\python.exe estimators_v6.py
  .venv-cuda\\Scripts\\python.exe stage3_v6.py --part train
  .venv-cuda\\Scripts\\python.exe stage3_v6.py --part eval
  .venv-cuda\\Scripts\\python.exe stage3_intervention_v6.py

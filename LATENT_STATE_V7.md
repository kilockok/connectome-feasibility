# latent_state_v7 protocol notes (2026-09-16)

Scope: ONE new mechanism test - spike-triggered adaptation current replacing
the multiplicative gain. Same learning method, retrained; not zero-shot
transfer, not initialization transfer, not mechanism discovery, not biology.

## Teacher (lif_adapt_v7.py)

- Base LIF (verified graph, dt, contract) + per-neuron hidden adaptation a:
  V_pre -= c*a (non-refractory only), a <- rho*a + beta*fire, tau_a=20,
  rho=exp(-1/20), beta=0.3, c=0.5. Gain latent OFF.
- a0 sampled per (seed, neuron) from an approximate steady-state via a
  separate generator, so c=0 reproduces the original LIF data EXACTLY
  (verified) and trajectory identity does not leak the hidden init.
- Negative control: beta=0 teacher (a decays to 0; models should show ~0
  improvement over base - confirmed, mixed signs across seeds).
- Mechanistic reference (declared: exact equation + params): back out
  a[t] = -e[t]/c from completed non-reset transitions, recurse with rho/beta.
  a_hat MAE 0.0012-0.004 vs true a std 0.085 - near-exact, confirming the
  "simple solution" structure of this teacher.

## Models (models/residual_v7.py)

Shared SpatialEncoderV2; additive correction over the differentiable
base-LIF pre-reset update. k1 (current), k2 (current+previous),
set (permutation-invariant DeepSets, no time label), ordered (global causal
temporal), oracle (true a, privileged). No adaptation formula or labels in
any learned head. Budget 24 epochs x 48 x batch 16; pilot seed 1234 then
1235-1238 unchanged; last.pt resume; two process deaths resumed cleanly.

## Key numbers (5 paired seeds)

- improve over base V RMSE: k1/k2/set/ordered +0.0049..+0.0054, oracle
  +0.0077; negctrl ~+0.001 mixed sign.
- ordered - k1/k2 paired V-RMSE diffs change sign across seeds (no advantage);
  ordered < set marginally.
- intervention (a_scale x2/x0/sham at tau=128): all models' branch error
  decreases with d (0.021 -> 0.008 by d=32); ordered marginally lowest at
  long d; d=0 determinism asserted; oracle separate.
- short-horizon at tau+8: h=1 ~0.014 all learned models; h=4/h=8 0.22-0.27
  (worse than one-step for all; ordered worse than k2 there).
- probes: a_mean R2 0.96-0.997 (easy); corr-head vs true contribution
  R2 0.39-0.54, slope 0.41-1.70 (uncalibrated).

## Result files

audit.md, protocol.md, metrics_natural.csv, metrics_intervention.csv,
metrics_diagnostics.csv, figures/ (4), conclusion.md, gates.json,
LATENT_STATE_V7.md.

## Reproduce

  .venv-cuda\\Scripts\\python.exe run_v7.py --regime adapt --seeds 1234 1235 1236 1237 1238
  .venv-cuda\\Scripts\\python.exe run_v7.py --regime negctrl --seeds 1234 1235 --labels k1 k2 set ordered
  .venv-cuda\\Scripts\\python.exe eval_v7_natural.py
  .venv-cuda\\Scripts\\python.exe intervention_v7.py
  .venv-cuda\\Scripts\\python.exe diagnose_v7.py
  .venv-cuda\\Scripts\\python.exe report_v7_figs.py

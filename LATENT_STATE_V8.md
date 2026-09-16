# latent_state_v8 protocol notes (2026-09-17)

Scope: ONE question - does edge-latent STP restore an ordered-history
advantage where per-neuron adaptation (v7) destroyed it? Discrete
Tsodyks-Markram per-edge STP, 3 recorded clusters, norm = 1/U per edge,
STP-off == base LIF exactly (verified).

## Key engineering facts

- Teacher (lif_stp_v8.py): per-edge (u, x) state; current = w*u*x/U per edge
  at presynaptic spikes; discrete recovery between events; clusters
  depression (U=.5, tau_rec=16, tau_fac=2), facilitation (U=.08, tau_rec=4,
  tau_fac=24), mixed (U=.25, tau_rec=8, tau_fac=8). Rate 0.117, residual V
  RMS 0.436 vs base std 0.141.
- Models (models/residual_v8.py): shared SpatialEncoderV2 + additive
  correction over differentiable base-LIF update. EventSimple (age+count),
  EventRich (ISIs + multi-timescale exp traces) - both LEGAL (observed
  history only). Oracle gets the postsyn STP-modulated current (privileged).
- Negctrl models trained on the STP-off teacher in an ISOLATED namespace
  (negctrl/) after a namespace bug was caught (early negctrl eval
  accidentally measured transfer of STP models onto base data; fixed and
  rerun).
- Memory: STP paths (u/x/g_path) stay on CPU (15 GB GPU pressure was the
  cause of a multi-hour stall; only training tensors move to GPU).
- Controlled probe: burst pulses on presyn j at tau-24/20/16 vs sparse,
  identical probe at tau+d; teacher-level recovery curves verified per
  cluster (facilitation +0.166 -> +0.013, depression -0.026 -> -0.006).
- Two stalled-job recoveries used last.pt resume; no results lost.

## Result files

audit/ (teacher_audit.json, feature_audit.md, alias_pairs.csv), metrics/
(natural.csv, alias.csv, probe.csv, history_sweep.csv, decoder.csv),
figures/ (5), conclusion.md, gates.json, LATENT_STATE_V8.md.

## Reproduce

  .venv-cuda\\Scripts\\python.exe run_v8.py --regime stp --seeds 1234 1235 1236 1237 1238
  .venv-cuda\\Scripts\\python.exe eval_v8_natural.py
  .venv-cuda\\Scripts\\python.exe alias_v8.py
  .venv-cuda\\Scripts\\python.exe probe_v8.py
  .venv-cuda\\Scripts\\python.exe sweep_decoder_v8.py --part sweep
  .venv-cuda\\Scripts\\python.exe sweep_decoder_v8.py --part decoder
  .venv-cuda\\Scripts\\python.exe run_v8.py --regime negctrl --seeds 1234 1235 --labels k1 k2 set ordered event_simple event_rich

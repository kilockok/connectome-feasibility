# latent_state_v4 protocol notes (2026-09-16)

Goal: decompose the temporal advantage into permutation-invariant multi-frame
statistics vs genuine temporal order, and test the observability mechanism
("population observation sufficiency controls the need for temporal order").

## Models (models/history_set_encoder.py)

- set_k32: SetHistory DeepSets — per-step GNN -> population mean pool -> phi ->
  mean/std/max over time -> rho -> fused decoder. STRICTLY permutation
  invariant over time (verified max|dy|=3e-8); no position embedding, no
  attention, no recurrence. Parameter-matched to global_k32 within 0.7%.
- stats_k32: handcrafted population moments (per-step mean/std/rate + window
  mean/std/min/max + current) -> small MLP.
- deriv: explicit short-timescale derivative features (dV1/dV4/dRate/slope)
  -> small MLP.
- v2 labels delegated (gnn_k1, global_k32, gshuffle, wide, oracle).

## Design decisions worth recording

- v4 order_gain is DEFINED as Ordered - SetHistory (not Ordered - shuffled as
  in v3): the strict DeepSets floor is the clean order-free baseline. The
  shuffled Transformer is kept only as a secondary control; it is NOT a clean
  baseline (its attention computes cross-timestep statistics even on scrambled
  input; it exceeds DeepSets everywhere).
- N scaling reuses v3 entries where teacher/split/architecture/seed/budget are
  identical (N=100 from v3 replication, N=1000 from v3 scale), documented;
  N=250/500 generated in the v4 namespace with the same protocol.
- N_obs experiment: teacher fixed at N=1000; model observes a FIXED
  degree-stratified subset (Protocol A random subset as robustness screen);
  model input AND target restricted to observed neurons; model knows only the
  induced subgraph. No full-population statistics can leak (tokens are raw
  V/S/R/U; pooling happens inside the model over observed neurons only).
- K-sweep uses architecture-consistent inference masking (drop/zero older
  tokens), no retraining.
- Bug history: load_model_v4 double-popped 'cls' and silently rebuilt Global
  models as Local (state-dict shapes are identical, so load succeeded);
  detected as a 0.96->0.92 F1 mismatch, fixed, wrong eval entries re-evaluated;
  regression check added (class assertion per label).

## Headline results (3 paired seeds each)

N scaling: unordered_gain 0.0051 -> 0.020, order_gain 0.040 -> 0.030,
order_share 0.89 -> 0.61 as N grows 100 -> 1000.
N_obs (teacher N=1000): order_gain 0.083 (50) -> 0.030 (1000);
unordered ~ 0 at 50 observed neurons; deriv_gain explains most of the order
benefit at low observability (0.067/0.080 at obs50/100) and ~0 at full
(0.000 vs +0.030). Unordered z-observability rises with observability while
order_gain falls (r = -0.34, descriptive). Random-subset screen consistent.

# latent_state_v5 protocol notes (2026-09-16)

Goal: test whether the v4 order-specific predictive information corresponds to
the teacher's true hidden state and to the LIF residual omitted by the base
model. NOT a rollout study, NOT new architectures.

## Framework used to constrain claims

1. Delay-coordinate / partial observability: history can in principle carry
   state information missing from the current observation, but only under
   observability conditions - hence the explicit alias benchmark instead of
   dataset-average F1.
2. Identifiable nonlinear latent dynamics: predictive performance alone does
   not establish latent identifiability - hence Gates B (decoding) and C
   (intervention) as separate requirements from Gate A (prediction).
3. Interventional state-space models: observational fit can be association;
   interventions are the stronger causal test (Gate C).
4. Injective/readout identifiability: the model may invent a latent
   parametrization not equivalent to the teacher's z - hence
   mediation/subspace analyses instead of assuming z-alignment.

## Key engineering facts

- Residual identity (verified, R2 0.9915): Delta_true[V] = a*(gain-1)*I_syn
  on free neurons; residual prediction = scalar gain estimation projected on
  the observable I_syn direction. This is what makes the alias gain-assignment
  metric uncontaminated (earlier draft metrics were contaminated by
  base-continuation tracking; replaced and documented).
- Alias pairs: natural test_seen states, matched on standardized [V,S,R,U]
  (plus activity floor >=1 spike at t so I_syn != 0), |dz_pos| tiers
  strict/medium/loose + negative controls at |dz_pos| <= p25.
- Checkpoints reused: v3 replication (gnn_k1/global_k32/oracle, 5 seeds) and
  v4 n100 (set/stats/deriv, seeds 1234-1238 with 1237/1238 trained for v5).
  Every load asserts model class, config, io dims and param count (v4's
  silent class-mismatch bug class is now structurally excluded).
- Probe hygiene: trajectory-disjoint splits, frozen backbones, fixed capacity,
  early stopping, randomized-label control (all shuffled-label R2 ~ 0),
  CCA fitted on train and evaluated held-out (in-sample CCA shown to be
  inflated, discarded).

## Gates (see gates.json / gate_summary.csv)

0 PASS (teacher effect material), A PASS (alias disambiguation, 5/5 seeds vs
K1/Set/Deriv; oracle headroom), B FAIL (ordered ~= set latent decoding,
z_vel below set), C FAIL (no intervention gain tracking; obs-bump control
responds instead), D PASS (not an error compensator).

## Definitive statement

Alias-level: ordered history recovers hidden-state information unavailable
from current observation or unordered history, and it tracks the scalar
residual the base LIF omits (mediation M ~= 0.96). Intervention-level and
decoding-level: causal latent tracking is unsupported. The full F6 claim is
therefore NOT made; the honest upgrade of v4's order_gain is:
"order-specific predictive information corresponds to hidden-state-driven
residual dynamics at the level of natural state aliases."

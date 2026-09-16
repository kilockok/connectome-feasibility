# latent_state_v3 protocol notes (2026-09-16)

Goal: turn the v2 positive result into causal evidence. Same local host and
environment as v2 (Tesla P100, torch 2.7.1+cu126, .venv-cuda). v2 artifacts
untouched; v2 data caches are loaded read-only for exact replication.

## Design decisions worth recording

- Replication trains ONLY the informative subset (gnn_k1, global_k32,
  gshuffle, wide, oracle) at 5 paired seeds (1234..1238), identical data
  (v2 cache read-only), identical budget. Markov control: gnn_k1/global_k32
  at the same 5 seeds.
- Counterfactual matching: candidates require an ACTIVE recent window
  (>=8 population spikes in [t-31..t]); matched on current V/spikes/U/z_pos
  (v_rmse 0.021, spikes identical, U identical, |dz_pos| 0.23) with opposite
  z_vel sign (|z_vel| >= 0.5 std) AND raw history spike-pattern divergence
  (>p60 = 8.7%). Earlier draft without activity/divergence floors selected
  trivial resting-state pairs with indistinguishable histories - swap did
  not change the input. Redesigned and rerun.
- Deterministic teacher (S1): sigma=0 REQUIRES randomized initial z (else
  the damped oscillator sits at the fixed point 0 and alpha is moot - first
  det cache was exactly the markov data, caught and deleted). beta relaxed
  0.98 -> 0.995 so the noise-free oscillator stays meaningful over T=256
  (Jury stability 0.0036 < 0.005 holds). init stds match the stochastic
  teacher marginals (0.77 / 0.046).
- Full-future-z oracle (S2) is a diagnostic-only K=1 model receiving
  z[t..t+8]; threshold calibrated on val only; never a formal model.

## Gate outcomes

A replication PASS (order_gain +0.0187, 5/5 seeds, dz 1.5; 2/5 per-seed
bootstrap CIs touch zero). B markov PASS (-0.0041). C counterfactual PASS
only weakened (ordered directional, opposite>same>random; but shuffled
model responds MORE - not order-exclusive). D order-profile PASS (reverse
-0.019, preserve_last4 +0.0002, occlusion monotone; info mainly in last
4-8 ordered steps + older statistics). E sign probe FAIL (AUROC 0.643 vs
0.644). F stochasticity FAIL (sigma=0 no help; future-z oracle no help;
current-z oracle decays too) - rollout failure is autoregressive
neural-state error, not latent process noise. N=1000: temporal_gain
+0.0496, recovery_fraction 0.659 hold; order_gain attenuates to +0.0040
(population pooling strengthens order-free statistics at large N).

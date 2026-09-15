# latent_state_v2 protocol notes (2026-09-16)

Local CUDA host: Tesla P100-PCIE-16GB (sm_60), torch 2.7.1+cu126 (cu128 wheels
drop Pascal), Python 3.12.14 (.venv-cuda). No CUDA Toolkit required.

Teacher: ONE hidden mechanism, global gain = 1 + alpha*tanh(z_pos) driven by a
damped noisy 2-D oscillator. Selected (calibration seeds only):
alpha=1.0, beta=0.98, omega=0.06, sigma=0.008 (z_pos std 0.77, late activity
0.018, branch RMSE 0.133, vel-branch h32 0.202).

Stage 8 oracle ceiling (seed 1234, 16 epochs): gnn_k1 val F1 0.8981 vs oracle
0.9684 -> oracle_gain ~ +0.070 (target >= 0.03 passed).
Stage 9 observability gates passed: ordered GRU z_pos R2 0.191 > current 0.073,
> unordered 0.169, > shuffled 0.049 (thin margins; absolute R2 low).

Numerical identity note: exact trajectories reproduce only within the same
(GPU, batch-size) configuration (cuBLAS kernel choice changes ulps; spike
thresholds amplify them). All v2 caches use local B=64 chunk generation, so
every within-study comparison is self-consistent. v1-remote vs v2-local exact
equality is not expected and not required.

Stages: 0-12 complete (env, venv, install, smoke, repo tests, v1 smoke repro,
teacher, calibration, oracle ceiling, observability, architecture, unit tests,
smoke training). Stage 13 main training: hidden 9 labels x 3 seeds,
markov 4 labels x 3 seeds, 24 epochs, batch 16, AdamW 3e-4, patience 6.

## Final status (2026-09-16)

All stages 0-17 complete. Core result: ALL FOUR success criteria passed.
oracle_gain +0.0705+/-0.0024; temporal_gain (global_k32 - gnn_k1)
+0.0447+/-0.0052; recovery_fraction 0.633+/-0.052. ordered>shuffled
(+0.0184, one seed CI touches zero), global>local (+0.0264), temporal>wide
(+0.0448), probe ordered>shuffled (z_pos ridge R2 0.172 vs 0.156; z_vel
0.122 vs 0.107; global_k32 also highest z_vel). Markov control: temporal
advantage vanishes (+0.0024, CI crosses zero) - history matters only when
the hidden state exists. Interventions: vel_flip invisible by construction
(latency ~0), phase_jump recovery 17-19 steps for every model, regime 1-6.
Conditional unroll stage (entered, U=4/U=8): no rollout F1 improvement;
oracle rollout is equally low -> collapse is teacher stochasticity, not
model error. z observability is weak in absolute terms (R2<0.2 everywhere)
even though the causal effect is large - the regime is "hard to read, strong
to feel". conclusion.md answers all 15 questions.

# latent_state_v11 — Phase 0 design document (pre-implementation, frozen for review)

**Status: DESIGN ONLY. No v11 code has been written. Implementation starts
only after this plan is confirmed.**

## 0. Positioning: what changes after v10

v10 closed the active-identification question: interventions separate
candidate predictions (z ≈ 30) but fitted-estimator scoring cannot convert
that into mechanism identification (STP ≤ 0.43 at the oracle-design
bound). The bottleneck is the estimator/observation channel, not
information existence and not model capacity.

v11 therefore drops mechanism classification entirely and returns to the
predictive question:

> Learn the latent residual dynamics that the LIF/connectome model cannot
> explain, and use the latent state to improve neural dynamics prediction.

GAIN/ADAPT/STP are NOT targets or labels here — they are simply the
physics of the available teachers that make the residual nonzero. Success
is measured by prediction, ablation survival, generalization and rollout
stability — never by family accuracy.

## 1. Hypotheses (falsifiable)

- **H1 (residual learnability)**: a hybrid LIF + learned residual model
  predicts held-out one-step dynamics strictly better than the exact hard
  LIF transition, on every teacher family, 5/5 seeds.
- **H2 (latent, not input copy)**: the learned correction depends on
  history-integrated latent state — it collapses when the latent context
  is shuffled across trajectories, and is near zero when trained on the
  NULL teacher (whose dynamics the hard LIF explains exactly).
- **H3 (structure & history contribute)**: removing temporal history
  (k→0/1) and/or shuffling the connectome degrades held-out prediction —
  with the family-dependent profile v7/v8 established (adapt ≈
  current-frame sufficient; STP requires event history), reported
  honestly rather than assumed uniform.
- **H4 (z is a dynamical state variable)**: intervening on the latent
  (time-shift, freeze/persistence, cross-trajectory substitution) changes
  predictions coherently — error magnitude grows with the latent's own
  drift timescale, not arbitrarily.
- **H5 (rollout)**: the hybrid free-running rollout is more accurate and
  more stable than the pure-LIF rollout at 10/50/100 steps, including on
  OOD dynamics, without runaway/saturation.

## 2. Assets audit (reuse, all verified in v5–v10)

| asset | file | role in v11 |
|---|---|---|
| Hard LIF simulator | `lif.py` (LIFSimulator) | the exact base; v10 proved NULL == base bitwise |
| Unified mechanism teacher | `teachers_v9.py` (MechanismLIFSimulator: gain/adapt/stp/OU/null/mixtures) | all data generation; no new teacher code |
| Effect-matched dataset | `results/latent_state_v9/data/v9_data.pt` (train 510/family, val, testA/B/C, null/test) | core train/eval corpus |
| Connectome | `connectome.py` (N=100 ring graph, Dale, cached seed 1234) | fixed topology; shuffle control derives from it |
| GNN spatial encoder | `models/latent_temporal_v2.py` (SpatialEncoderV2) | per-step connectome message passing |
| Causal temporal encoder | `models/latent_temporal_v2.py` (GlobalTemporalPredictorV2) | transformer over per-step population tokens; reference architecture |
| Differentiable soft base | `models/residual_v7.py` (base_pre) | training-time spike surrogate ONLY (artifact documented in v10) |
| Residual corrector lineage | `models/residual_v8.py` + v9 checkpoints | baseline hybrid (re-encoding z); v11 extends it with an explicit per-step latent |
| Window/loss/eval utils | `latent_data.py`, `metrics.py`, `eval_latent.py`, `run_v9.py` | training pipeline (24 epochs × 48 steps × batch 16, AdamW 3e-4, 5 seeds) |
| Artifact-free eval path | v10 `common_v10.py` (hard transition, info-bearing/event masks) | all v11 evaluation |
| OOD stimulus library | v10 `results/latent_state_v10/protocol/intervention_library.json` | held-out stimulus patterns (bursts, paired, high-current, phase) |

Known pitfalls carried forward (from v5–v10, each with a planned control):
1. soft-base artifact (0.019–0.026) masquerading as residual signal →
   eval vs HARD base; NULL-teacher control quantifies artifact fitting.
2. one-step gain need not extend to rollout (v7: h=4/8 collapse) → Stage
   3 rollout is a first-class gate, not an afterthought.
3. representation ≠ mechanism identity (v9/v10) → irrelevant here by
   design; Stage 4 is correlation-only, no naming.
4. sparse residual dilution by all-step averaging → v10 masks
   (free / info-bearing / event) are the default reporting decomposition.

## 3. Model architecture (new code: `models/latent_hybrid_v11.py`)

```
per step t:   e_t = GNN(x_t; connectome)            # SpatialEncoderV2, [B,N,d]
              g_t = pool(e_t)                        # mean + max + rate features
              z_t = CausalTransformer(g_{t-K..t})    # [B,d_z], per-step output
decode:       V̂_{t+1} = HardLIF(x_t, u_t) + f_V([e_t, z_t])
              ŝ_{t+1} = logit_LIF(x_t, u_t) + f_S([e_t, z_t])
              r̂_{t+1} = r_LIF(x_t, u_t) + f_R([e_t, z_t])
```

- **Base**: exact hard LIF at eval; soft `base_pre` surrogate during
  training (spike BCE differentiability) — same pattern as v7–v9, with
  the artifact explicitly measured by the NULL control.
- **Latent z_t**: explicit, exposed per step (not just last-token) so it
  can be substituted/shifted/frozen — this is what Stage 2C intervenes
  on. d_z = 64, K = 32 context, causal blocks only (no future reads).
- **Decoder**: per-neuron head over [e_t (local), z_t (global context)];
  additive on top of the base computation (residual form), exactly the
  v7–v9 residual parameterization.
- **Reference baselines** (same training pipeline):
  - B0: hard LIF alone (zero parameters).
  - B1: soft base_pre alone (artifact floor reference).
  - B2: v8/v9 OrderedHistory corrector (re-encoding z; continuity check
    — v11 hybrid must be ≥ it).
  - B3: v11 hybrid with z ablated (k=0: decoder sees e_t only).
  - B4: v11 hybrid with GNN ablated (e_t = per-neuron MLP, no messages).
  - B5: v11 full.

## 4. Datasets

Core corpus: v9 effect-matched pool, mechanism-blind mixed training
(gain+adapt+stp, 510 train each; no labels anywhere), NULL test split as
the artifact control. Splits reused as designed: testA (seen params),
testB (held-out interpolated params — primary), testC (extrapolated).

New generations (existing teacher code only, cheap; cached in
`results/latent_state_v11/data/`):
1. **OU teacher** (additive colored current): structurally distinct OOD
   dynamics, never in training.
2. **Mixtures** (gain+stp, gain+adapt, adapt+stp, s=1/m=1): second OOD
   axis — unseen combinations of seen components.
3. **OOD stimulus patterns**: v10 intervention library entries (burst
   ISI structures, paired-pulse, high-current, phase) applied to
   mechanism teachers — the training stimulus protocol (sparse random
   pulses, amp 4–7, 1–5 neurons, dur 5–20) never contains these.
4. **Held-out neurons**: same trajectories; training loss masked to
   neurons 0–79, evaluation on neurons 80–99 (whose inputs include
   connectome messages from seen neurons — tests structure
   generalization, not trajectory memorization).

## 5. Training protocol

One unified mechanism-blind hybrid, 5 paired seeds (1234–1238), v9
hyperparameters (24 epochs × 48 steps × batch 16, AdamW 3e-4, wd 1e-4,
early stop on val, activity-biased window sampling). Loss = V MSE +
spike BCE (pos_weight 30) + R MSE (v9 `compute_loss`), masked to the
training neuron set. All ablation variants (B3/B4, history-k) share
capacity budgets within ~10% and identical data order (v4/v5 paired-seed
discipline). Checkpoints per seed with config/class assertions on load
(v5 lesson).

## 6. Stage plan, gates, failure criteria

### Stage 1 — residual prediction baseline (Gates G1, G2)

Eval: held-out one-step, testA/testB, masks {all, free, info-bearing,
event}; V RMSE + spike F1; 5 seeds paired.

- **G1**: B5 − B0 improvement > 0 on every family, 5/5 seeds, on testB.
  (v9 precedent: +0.003–0.004 V RMSE at matched effect.)
- **G2 (not input-copy)**: (i) B5 trained on NULL teacher: residual norm
  ≤ 1.5× the soft-base artifact floor, and improvement ≈ artifact-only;
  (ii) z-context shuffle across trajectories at eval removes ≥ 80% of
  the B5−B0 gain on mechanism families; (iii) residual correlates with
  history-integrated features beyond the current frame (partial R²
  controlling for x_t).
- **Fail F1**: G1 fails on ≥ 2 families → the hybrid learns nothing
  beyond LIF; stop, report, no Stage 2–4 claims.
- **Fail F2**: NULL residual ≫ artifact → the model fits base mismatch,
  not latent dynamics; fix base/loss before proceeding.

### Stage 2 — latent state validation (Gates G3, G4, G5)

- **G3 history ablation**: k ∈ {0, 1, 2, 8, 32} retrained (B3 = k0).
  Prediction vs k curve per family; H3 supported where v7/v8 predict
  (STP/event-driven families must need history; adapt may not — a flat
  adapt curve is a CONFIRMATION of mechanism structure, not a failure).
  Failure = flat curve for STP-type dynamics too → latent-history claim
  unsupported, report as v7 did.
- **G4 connectome ablation**: degree-preserving edge shuffle (same
  weight multiset, randomized topology) → retrain B5s; held-out +
  held-out-neuron evaluation. Failure = shuffle ≈ real → structure
  claim unsupported.
- **G5 latent interventions** (on B5, no retraining):
  - LI-1 time-shift: decode with z_{t+Δ} for Δ ∈ {1,2,4,8,16} → error
    growth must be coherent (monotone-ish in Δ, structured by family
    timescale: adapt fast, STP event-gated).
  - LI-2 freeze/persistence: hold z fixed for Δ steps while observations
    evolve → error growth compared against z's own autocorrelation time.
  - LI-3 cross-trajectory substitution: z from another trajectory →
    large error; z from the same trajectory at a matched activity regime
    → smaller error. (NOT a family test — v10 lesson: z need not encode
    family.)
  - LI-4 smoothness: z ← αz and z ← z+ε → response smooth/monotone.
  Failure = incoherent or null responses on all four → z is an ordinary
  embedding; downgrade claim to "predictive features".

### Stage 3 — generalization (Gate G6)

1. Held-out neurons (80–99): B5 vs B0 one-step, same masks.
2. Held-out stimulus patterns: v10 library entries as extra_stim;
   one-step on response windows.
3. OOD dynamics: OU + 3 mixtures; one-step + rollout.
4. Long rollout: branch at t0 = 96, free-run 100 steps with the TRUE
   continued stimulus (input known; dynamics challenged). Modes:
   (a) B0 pure LIF; (b) B5 free-running z (context from [t0−K, t0],
   latent re-encoded from the model's OWN predicted states
   autoregressively); (c) B5 assimilated z (teacher-forced observations
   into the encoder — upper bound). Horizons 10/50/100; V RMSE, spike
   rate error, saturation/runaway fraction vs teacher.
- **G6**: B5(b) ≤ B0 rollout RMSE at 10/50/100 on testB, 5/5 seeds, and
  no saturation events that B0 avoids; OOD rollout not worse than B0 by
  >20%.
- **Fail F6**: hybrid rollout diverges or exceeds B0 by 50%+ at h=50 →
  rollout claim fails explicitly (v7 precedent); one-step claims stand
  or fall separately.

### Stage 4 — biological interpretation (auxiliary, non-claim)

Correlate z_t (dims + PCA/CCA, train-fitted, held-out-evaluated — v5
hygiene) with observable summaries: firing rate, population V mean/std,
|I_syn|, event density, refractory fraction; plus the privileged
`hidden_summary` as a sanity reference (reported as privileged, never as
a claim). No naming of dimensions. If nothing robust: report
"predictive latent dynamics" and stop.

## 7. Metrics registry (frozen before results)

- One-step: V RMSE (all/free/info-bearing/event masks), spike F1
  (threshold calibrated on val, per seed), R RMSE.
- Rollout: V RMSE@10/50/100, spike-rate absolute error, saturation
  fraction (V at bounds or rate > 0.45), time-to-divergence.
- Latent: z-shuffle sensitivity Δ(B5−B0), ablation deltas (G3/G4),
  intervention response curves (G5), interpretation R² (Stage 4).
- Statistics: 5 paired seeds; per-seed values, mean, std, 95% CI, sign
  consistency; trajectory-disjoint everything; no metric edited after
  results (v9/v10 discipline).

## 8. Compute + file layout

P100 16GB. N=100, T=256, K=32: corrector-scale models train in ~minutes
per seed (v9: 24×48 steps ≈ 1 s/step); full v11 (B5 × 5 seeds + ~8
ablation variants × 5 seeds) ≈ a few GPU-hours; OOD data generation
minutes; rollout eval minutes. All within local budget.

```
results/latent_state_v11/
├── audit/ (artifact_control.md, leakage_audit.md, feature_audit.md)
├── protocol/ (protocol.md, configs/)
├── data/ (ou.pt, mixtures.pt, ood_stim.pt — caches)
├── checkpoints/ (b5_seed*, ablations)
├── metrics/ (onestep.csv, history_ablation.csv, connectome_ablation.csv,
│             latent_intervention.csv, heldout_neurons.csv, ood.csv,
│             rollout.csv, interpretation.csv)
├── figures/ (fig1 onestep, fig2 history-k, fig3 connectome, fig4 latent
│             intervention, fig5 rollout, fig6 interpretation)
├── gates.json
├── conclusion.md
└── LATENT_STATE_V11.md
```

New code (no v1–v10 file touched): `models/latent_hybrid_v11.py`,
`data_v11.py`, `train_v11.py`, `eval_v11.py`, `ablate_v11.py`,
`intervene_v11.py`, `rollout_v11.py`, `interpret_v11.py`,
`report_v11.py`. Root sync: `LATENT_STATE_V11.md` at the end.

## 9. Explicit non-goals (v10 lessons enforced)

- No mechanism classification, no family labels, no identification claim.
- No new teacher physics beyond existing MechanismLIFSimulator flags.
- No metric adjustment after seeing results; preregistered gates only.
- No claim that latent dims ARE biological variables (Stage 4 is
  correlation-level).
- Placeholder ring connectome as before: "connectome constraint" claims
  are within-simulator; FlyWire replacement is future work, same as v1–v10.

## 10. Stage order (strict)

Stage 0 this document → Stage 1 (G1/G2 gate) → Stage 2 (G3/G4/G5) →
Stage 3 (G6) → Stage 4 (auxiliary) → synthesis. Any gate failure stops
its dependent branch and is reported as a result, per the v5–v10
reporting convention.

# Phase 3: GNN + Temporal Transformer — Design & Runbook

Phase-1/2 result: connectome topology is the core structural prior for
one-step dynamics; the real bottleneck is autonomous rollout collapse
(closed-loop state drift, not teacher chaos — see `results/rollout_v2/`).
Phase 3 tests one hypothesis (user spec):

> Keeping the GNN's connectome-constrained spatial propagation, does adding
> per-neuron **temporal self-attention** over the multi-step history extend
> the stable autonomous-rollout horizon?

Architecture (models/gnn_temporal.py), strictly "GNN first, temporal second":

```
x [B,K,N,4]  -> per-timestep GNN (models.gnn MessageRound, unchanged)
             -> h_t [B,N,D] per step -> stack [B*N, k, D]
             -> temporal positional encoding (learned | sincos)
             -> L pre-norm Transformer blocks ALONG TIME ONLY
             -> last-position readout -> head {v, s_logits, r}
```

No N×K global attention: each neuron attends over its own history; space is
the GNN's job. `k_hist` truncates the window inside forward, so one pipeline
serves K=1 (last-state-only / Markov control) … K=cfg.K; K>cfg.K needs
`train.py --K`. Output contract is unchanged, so MechanisticWrapper
(ΔV + deterministic LIF projection = spec 八), rollout.py, dagger.py and the
threshold protocol all apply untouched.

## File map

| file | status | contents |
|---|---|---|
| `models/gnn_temporal.py` | new | `GNNTemporalTransformer`, `TemporalBlock` (optional attention-weight return), `sinusoidal_table`, `gnn_temporal_kwargs` |
| `models/__init__.py` | edited | `build_model(..., model_kwargs)`; `gnn_temporal`, `gnn_wide`; `match_gnn_wide` (param-matched control) |
| `train.py` | edited | `--model gnn_temporal/gnn_wide`, `--k-hist/--K/--t-layers/--t-heads/--pos/--causal/--dropout/--param-match`; checkpoint tags `_k{k}`/`_K{K}`/`_w{d}x{L}`; blobs carry `model_kwargs`+`K` |
| `losses.py` | edited | gated `threshold_loss` term (spec 十六; adds `thresh` to parts only when on — phase-2 behaviour unchanged) |
| `rollout_config.py` | edited | `stages_v3` (U=4,8,16,32,64 × teacher 0.9,0.75,0.5,0.25,0.0 — spec 十/十一), `EXPERIMENTS_V3` = D/E/F, `--matrix v3`, v3 path namespace (`results/rollout_v3/`, `checkpoints/rollout_v3/`) |
| `rollout_train.py` | new | the trainer pinned by PHASE2.md (curriculum, scheduled sampling + actual self-fed logging, DAgger mixing/refresh, per-epoch quick val + early stop, stage-end protection, history CSV + summary JSON) |
| `run_rollout_v2.py` | new | subprocess orchestrator for the matrix + unified eval hand-off |
| `temporal_eval.py` | new | spec 十七–二十四: per-entry rebuild from blob (mixed K OK), val-only thresholds, one-step + rollout + attractor + reinjection + history-shuffle + attention capture, FLOPs estimate, 3 tables, 10 figures |
| `test_phase3_temporal.py` | new | 9 unit tests (all pass, CPU) |

Phase-2 checkpoint rules still hold: v3 fine-tunes always start from the
one-step `ckpt_gnn_temporal_{scale}_seed*.pt` (the "C" entry), never
overwrite phase-1/phase-2 files, thresholds on val only, same cached splits.

## Experiment matrix (spec 二十一) → commands

Small scale (local A750/XPU):

```bash
# one-step feasibility (spec 九) — the C/D/E/F chain needs this first
python train.py --model gnn_temporal --scale small                    # C: K=16
python train.py --model gnn_temporal --scale small --k-hist 8         # B
python train.py --model gnn_temporal --scale small --k-hist 4         # ablation
python train.py --model gnn_temporal --scale small --k-hist 1         # A-control (Markov)
python train.py --model gnn_wide --scale small --param-match gnn_temporal   # fairness

# rollout training (each inits from the C checkpoint; ~hours total on A750)
python run_rollout_v2.py --scale small --model gnn --experiments G          # baseline B (best GNN+rollout)
python run_rollout_v2.py --scale small --model gnn_temporal --matrix v3 \
    --experiments D E F --skip-eval
```

Unified eval (tables + figures into `results/rollout_v3/`):

```bash
python temporal_eval.py --scale small \
  --entry A=results/checkpoints/ckpt_gnn_small_seed1234.pt \
  --entry G=results/checkpoints/ckpt_gnn_small_rollout_v2_G_seed1234.pt \
  --entry wide=results/checkpoints/ckpt_gnn_wide_w128x6_small_seed1234.pt \
  --entry k1=results/checkpoints/ckpt_gnn_temporal_k1_small_seed1234.pt \
  --entry k4=results/checkpoints/ckpt_gnn_temporal_k4_small_seed1234.pt \
  --entry k8=results/checkpoints/ckpt_gnn_temporal_k8_small_seed1234.pt \
  --entry C=results/checkpoints/ckpt_gnn_temporal_small_seed1234.pt \
  --entry D=results/checkpoints/ckpt_gnn_temporal_small_rollout_v3_D_seed1234.pt \
  --entry E=results/checkpoints/ckpt_gnn_temporal_small_rollout_v3_E_seed1234.pt \
  --entry F=results/checkpoints/ckpt_gnn_temporal_small_rollout_v3_F_seed1234.pt \
  --reinject-entries A C F --attention-entry F
```

Full scale (cloud 4090/5090, per CLOUD.md): same commands with
`--scale full` (K becomes 32; add `--k-hist 16` for the K=16 ablation and
`--K 64` runs for K=64). Notes:

* gnn_temporal costs ~K× a GNN step per forward — the per-timestep GNN is
  chunked (`gnn_chunk=32`) so peak memory stays bounded; late unroll stages
  always gradient-checkpoint (`force_ckpt`).
* `--buffer-capacity 20000` or lower at full scale (200k × K=32 × N=1000
  float32 contexts ≈ 100 GB RAM; 20k ≈ 10 GB).
* Run order: one-step models first (C must exist before D/E/F), then
  `run_rollout_v2.py --matrix v3`, then the big `temporal_eval.py` call.

## Success / failure criteria (spec 十九/二十)

Success = one-step parity (ΔF1 < 0.03 vs GNN) AND at least one of: later
collapse, higher h=20/50 F1, slower V-RMSE growth, better reinjection
K=20/50, seen≈OOD kept, stabler population dynamics at h=100–200.
Failure (valid result): one-step unchanged + rollout unchanged +
history-shuffle insensitive ⇒ temporal attention adds nothing in this
deterministic LIF setting; if K=1 ≈ K=32 the system is ~Markovian and the
lever is DAgger / state projection / precision / stability regularisation.
The 9 required answers (spec 二十五) go in the report next to
`results/rollout_v3/summary_tables.md`.

# Phase 2: Rollout-Stability Pipeline — Design & Interface Contract

Phase-1 result: graph models reach one-step OOD F1 0.93–0.99, but autonomous
rollout decorrelates within ~10 steps. Phase 2 asks: **can closed-loop /
on-policy training keep the strong one-step identification while greatly
improving autonomous rollout stability?**

Hard rules (from the project spec):

1. **Never overwrite phase-1 checkpoints** (`ckpt_*_{small,full}_seed*.pt`,
   `ckpt_*_ms_*`, `ckpt_*_ablate-*`). All phase-2 checkpoints go through the
   path helpers in `rollout_config.py`.
2. Every multi-step run **starts from the phase-1 best checkpoint** of the
   same model+scale (error out if missing — never train from scratch).
3. Thresholds are tuned on **val** only, then frozen for test_seen/test_ood.
   Never tune on test.
4. DAgger collection uses **train-split stimuli only** (the OOD protocol —
   test_ood neurons never directly stimulated in training — must survive
   phase 2; "OOD states" in the buffer means activity that propagated into
   the OOD region on train trajectories, never direct stimulation there).
5. Same test trajectories for every experiment (the cached splits).
6. Method search on **GNN only** (experiments A–G); the winning recipe is
   then transferred to the connectome transformer.

## File map (all new; phase-1 files untouched)

| file | owner | contents |
|---|---|---|
| `rollout_config.py` | (done) | `StageConfig`, `RolloutConfig`, `EXPERIMENTS`, stage curricula, path helpers |
| `surrogate.py` | agent A | surrogate-gradient spike + differentiable state composition |
| `models/mechanistic.py` | agent A | `MechanisticWrapper`: net predicts ΔV, LIF mechanics produce S/R/reset |
| `losses.py` | agent A | horizon-weighted multi-step loss (V, spike, R, rate, population, ΔV) |
| `dagger.py` | agent B | `ReplayBuffer` + on-policy collection with LIF teacher |
| `rollout_eval.py` | agent C | full rollout evaluation: metrics, attractor diagnostics, figures, tables |
| `rollout_train.py` | agent D | the trainer (curriculum, scheduled sampling, noise, DAgger mixing, guards) |
| `run_rollout_v2.py` | agent D | orchestrator CLI for the A–G matrix + transfer + eval |
| `test_phase2_mechanisms.py` | agent A | unit tests: surrogate/mechanistic/losses |
| `test_phase2_dagger.py` | agent B | unit tests: buffer + collection |

## Conventions shared by everything

- Device-agnostic (`device.py`), float32, no `.cuda()`, no torch_geometric;
  must run on torch 2.8 (cloud CUDA) and 2.14+xpu (local Arc A750), Win11 +
  Linux. `torch.load(..., weights_only=False)` as in the rest of the repo.
- State convention everywhere: features `[V, S, R_norm, U]`, `R_norm = R_remaining / refractory_period`.
- Time convention (matches `dataset.make_windows` / `rollout.py`): the model
  sees `(X, U)[t-K+1 .. t]` and predicts `X[t+1]`; the stimulus of the
  transition `t -> t+1` is `U[t+1]`, which the model never sees at prediction
  time. When appending a predicted state to the history window, pair it with
  the stimulus **at the same absolute time** as that state:
  `feat = (X̂[t+1], U[t+1])` — see `rollout.py` and `train.finetune_unroll`.
- Teacher one-step label for DAgger: `F_LIF(x̂_t, U[t+1])` — one simulator
  step from the model's state with the next true stimulus.
- Style: match the existing files (English docstrings, type hints,
  `from __future__ import annotations`, small focused functions).

## Module interfaces (pinned — implement exactly these signatures)

### `surrogate.py`

```python
def hard_spike(margin: torch.Tensor, mode: str = "ste",
               beta: float = 10.0) -> torch.Tensor:
    """Forward: (margin > 0).float(). Backward per mode:
    'ste'          straight-through on sigmoid(margin)
    'sigmoid'      beta * sigmoid'(beta * margin)
    'fast_sigmoid' beta-normalised 1 / (1 + beta*|margin|)^2
    Modes are switched by a plain string; safe under torch.no_grad too."""

def compose_step_learned(out: dict, cfg, threshold: float = 0.5,
                         mode: str = "ste", beta: float = 10.0):
    """Differentiable version of rollout.py's hard composition.
    out: model output dict (v, s_logits, r). margin = s_logits - logit(threshold).
    Returns (v, sp, r): clamps v to [v_min, 3*v_th], spike via hard_spike,
    fired -> v=v_reset & r=1, r>REFR_HOLD (import from rollout) & !fired ->
    v=v_reset. r clamped [0,1]."""

def compose_step_mechanistic(dv: torch.Tensor, v_t: torch.Tensor,
                             r_t: torch.Tensor, cfg, mode: str = "ste",
                             beta: float = 10.0):
    """Deterministic LIF mechanics given predicted membrane increment dv.
    refr = r_t > 0 (r_t is normalised); v_pre = clamp(v_t + dv, min=v_min);
    fire = hard_spike(v_pre - v_th) with surrogate grad into dv;
    refr  -> v=v_reset, sp=0, r_next = r_t - 1/period (clamped >=0)
    fire  -> v=v_reset, r_next=1
    else  -> v=v_pre,   r_next = r_t - 1/period (clamped >=0)
    Returns (v_next, sp, r_next, v_pre). All differentiable w.r.t. dv."""
```

### `models/mechanistic.py`

```python
class MechanisticWrapper(nn.Module):
    """Wraps a phase-1 model (GNN / ConnectomeTransformer) so that head
    channel 0 is reinterpreted as the membrane increment:
        dv = a * (base_v_out - v_t) + b        # a=1, b=0 at init (warm start)
    a, b are learnable per-neuron scalars (flag `dv_adapter`, default True).
    forward(x) -> dict with the SAME keys as any model
    (v, s_logits, r) computed by the deterministic LIF rule applied to the
    LAST input state (v_t, r_t from x[:, -1]), plus extra keys
    "dv", "v_pre", "s_logits_aux", "r_aux" (the base model's raw channels;
    s_logits_aux may be used as an auxiliary spike-BCE head in the loss).
    s_logits = (v_pre - v_th) * logit_scale  (default logit_scale=8.0), so
    sigmoid(.)>0.5 <=> v_pre>=v_th: all phase-1 eval/threshold code keeps
    working unchanged. Refractory neurons: s_logits very negative, sp=0.

def maybe_wrap(model, cfg, rc) -> nn.Module:
    """MechanisticWrapper(model, cfg) if rc.mechanistic else model.
    Sets attribute `is_mechanistic` on the returned module."""
```

Checkpoint blobs saved by the trainer include `"mechanistic": bool` so
`rollout_eval.py` rebuilds the right container. `MechanisticWrapper`
forward must work under `torch.utils.checkpoint` (no side effects).

### `losses.py`

```python
def population_groups(n_neurons: int, n_groups: int, device) -> torch.Tensor:
    """[G, N] 0/1 membership, contiguous chunks along the ring (ring-local
    populations; no biological labels available)."""

def focal_bce_with_logits(logits, targets, pos_weight, gamma: float = 2.0):
    """Focal loss with per-class alpha from pos_weight (alpha = pw/(1+pw))."""

def multistep_loss(step_outs: list[dict],        # len U, raw model outputs
                   step_states: list[tuple],     # len U, composed (v, sp, r)
                   targets: torch.Tensor,        # [B, U+1, N, 3]; index 0 =
                                                 # last TRUE context state,
                                                 # step u supervised by [:, u+1]
                   rc, pos_weight: torch.Tensor,
                   groups: torch.Tensor | None):
    """L = sum_h gamma^(h-1) * L_h / sum_h gamma^(h-1), per-step:
    L_V     = mse(composed v, target v)
    L_spike = BCE-with-logits(out s_logits, target s) [or focal if rc.focal];
              mechanistic mode: also 0.5 * BCE(s_logits_aux, target s)
    L_R     = mse(out r, target r)  (mechanistic: mse(composed r, target r))
    macro terms only if rc.macro_loss:
    L_rate  = mse(mean composed sp, mean target s)      * lambda_rate
    L_pop   = mse(groups @ composed sp / group_size,    * lambda_pop
                  groups @ target s / group_size)
    L_delta = mse(v_h - v_{h-1}, t_h - t_{h-1})         * lambda_delta
              (h-1 = 0 uses targets[:, 0] v and the FED state passed via
               step_states' predecessor — i.e. v_{-1} := targets[:, 0, :, 0])
    Returns (total, parts dict with every component + 'loss')."""
```

### `dagger.py`

```python
class ReplayBuffer:
    def __init__(self, capacity: int): ...
    def add(self, contexts: torch.Tensor,   # [B, K, N, 4] CPU
               targets: torch.Tensor,       # [B, N, 3] CPU
               priorities: torch.Tensor):   # [B] CPU
        """On overflow keep highest-priority entries."""
    def sample(self, n: int, generator: torch.Generator):
        """Uniform. Returns (contexts, targets) CPU tensors."""
    def __len__(self): ...

@torch.no_grad()
def collect_on_policy(model, sim, cfg, device, n_traj: int, horizon: int,
                      spike_threshold: float, mechanistic: bool,
                      seed: int, n_batches: int = 4):
    """Autonomous closed-loop rollout on TRAIN-split trajectories
    (fresh seeds via cfg.traj_seed('train', ...) drawn with `seed`).
    At every step: keep the model's own history window; teacher label =
    F_LIF(current model state, next true stimulus) via
    sim.simulate(u_next[:, None], state0=(v, s, r_unnorm), silence_mask=...).
    priority = mean|v err| + 5*spike mismatch + 0.5*near-threshold frac
               (|v_pre - v_th| < 0.1) + 0.5*(depth / horizon).
    Returns (contexts[B,K,N,4], targets[B,N,3], priorities[B]) on CPU."""
```

### `rollout_eval.py`

CLI:
```
python rollout_eval.py --scale full \
    --entry phase1=results/checkpoints/ckpt_gnn_full_seed1234.pt \
    --entry G=results/checkpoints/ckpt_gnn_full_rollout_v2_G_seed1234.pt ...
# or --model gnn --experiments A B C D E F G  (naming-convention lookup)
```
Per entry × split (test_seen, test_ood), horizons 1,2,5,10,20,25,50,100,200
(capped by T): spike F1/P/R, v_rmse, pop_rate_corr (per-neuron rate Pearson),
pop_rate_mae, active-neuron Jaccard, state cosine similarity, regional
activity corr (10 groups), rate-histogram L1 distance, plus time series
rate(t), active_count(t), state-variance(t). One-step eval (calibrated
threshold on val) per entry on test_seen/test_ood. Naive baseline entry
always included. Attractor diagnostics per entry+split over the last 50
steps: dead (rate < 1e-4), saturated (rate > 0.5), collapsed variance,
periodic (max lag-2..50 autocorr of rate(t) > 0.9), fixed-subset firing.
Threshold protocol: tune on val once per entry (reuse `evaluate.tune_threshold`
+ `tune_rollout_threshold`), freeze for both test splits; mechanistic entries
are rebuilt via `maybe_wrap` using blob["mechanistic"].
Writes `results/rollout_v2/metrics.json` + `metrics.csv` (long format),
`summary_tables.md` (the two spec tables), and figures:
`rollout_spike_f1.png`, `rollout_v_rmse.png`, `rollout_population_corr.png`,
`rollout_rate_mae.png`, `attractor_diagnostics.png`,
`one_step_vs_rollout_tradeoff.png`, `method_ablation.png`
(merges with, never deletes, existing lif_sensitivity/reinjection outputs).

### `rollout_train.py`

CLI: `python rollout_train.py --model gnn --experiment G --scale full
[--surrogate fast_sigmoid] [--stages s1_u8,s2_u16] [--smoke]`
(Args via `rollout_config.add_rollout_args` + `add_common_args`.)

Flow:
1. Build model via `build_model` + `maybe_wrap`; load phase-1 checkpoint
   (`rollout_config.phase1_ckpt_path`) — hard error if missing.
2. Reference: calibrated phase-1 val one-step F1 (`evaluate.tune_threshold`
   on `rc.n_val_windows`) → `f1_ref`; phase-1 val rollout score → `ref_score`.
3. If dagger: warm-up `collect_on_policy` with the phase-1 model.
4. Per stage (init from previous stage best; fresh
   `AdamW(lr=rc.base_lr*stage.lr_factor, weight_decay=cfg.weight_decay)`):
   - each epoch: iterate `stage.n_traj` trajectories in chunks
     (`generate_batch`, like `train.finetune_unroll`); per batch of size
     `rc.stage_batch_size(stage)`: with prob `stage.dagger_mix` draw a
     one-step batch from the buffer (plain `compute_loss` vs teacher label),
     else an unroll batch:
       * random t0, true window → hist; per step u in range(U):
         forward (under `torch.utils.checkpoint` when
         `rc.use_grad_ckpt(stage)` and model training),
         compose next state via `compose_step_learned` /
         `compose_step_mechanistic` (mode=rc.surrogate, beta),
         per-sample Bernoulli(stage.teacher_ratio) choice of fed state
         (true+noise vs composed), append `(state, U[t0+K+u])`;
         `rc.tbptt > 0`: detach hist every tbptt steps;
         collect step_outs/step_states/targets (targets[:, 0] = last true
         context state) → `multistep_loss`.
       * noise on teacher-fed states only: V += sigma*randn;
         per-neuron spike flip with rc.spike_flip_p (flip-on also sets
         v=v_reset, r=1); refractory jitter ±1/period with rc.refrac_jitter_p.
       * log the ACTUAL mean teacher-forced fraction per epoch.
     backward, `clip_grad_norm_(rc.grad_clip)`, log grad norm; warn once per
     epoch if clipped fraction > rc.clip_warn_frac.
   - after each epoch: quick val — one-step F1@0.5 on rc.n_val_windows fixed
     windows + val rollout (rc.val_rollout_traj trajectories, H =
     rc.val_rollout_horizon, closed-loop, threshold 0.5) giving
     score = f1@10 + pop_similarity@50 - firing_rate_err@50;
     early stop on score (patience rc.patience); save stage best
     (`stage_ckpt_path`, blob includes mechanistic flag, stage, experiment,
     val metrics, epoch).
   - if dagger and epoch % rc.dagger_collect_every == 0: refresh buffer with
     the CURRENT model (collect, add; buffer evicts by priority).
   - stage end: calibrated one-step F1 (tune_threshold). PROTECTION: if
     calibrated F1 < f1_ref - rc.protect_drop AND stage's best score <=
     previous stage's best score: stop the whole experiment here, keep the
     previous stage best as the experiment result, record the event.
5. Copy the last stage's best to `final_ckpt_path(rc)`; write
   `history_path(rc)` CSV (one row per epoch: stage, unroll, lr, teacher
   target/actual, every loss component, grad-norm stats, buffer size, val
   metrics) and `summary_path(rc)` JSON (config, f1_ref, per-stage bests,
   protection events).
Deterministic seeding: one `torch.Generator` seeded from
`cfg.seed + 7919 * experiment_index(exp)`, re-seeded per epoch (+epoch).

### `run_rollout_v2.py`

```
python run_rollout_v2.py --scale full --model gnn --experiments B C D E F G
python run_rollout_v2.py --scale full --model gnn --experiments G \
    --surrogate-sweep            # step 8: G x {ste, sigmoid, fast_sigmoid}
python run_rollout_v2.py --scale full --model gnn --eval-only   # unified eval
```
Runs experiments sequentially (each = one `rollout_train.py` in-process call
or subprocess — subprocess preferred for GPU memory hygiene), then one
`rollout_eval.py` pass over all produced checkpoints. `--smoke` propagates.

## Execution plan (cloud 5090, after local small-scale smoke passes)

1. `python sensitivity.py --scale full --n-traj 16` (step 1; h to 200)
2. `python reinjection.py --scale full --models gnn connectome` (step 2)
3. `run_rollout_v2.py --scale full --model gnn --experiments A B C D E F G`
4. Surrogate sweep on the best experiment (step 8)
5. Transfer best recipe to `--model connectome` (step 11)
6. Unified `rollout_eval.py` over everything (step 12)

## Success levels (spec section 十六)

L0 one-step stays high; L1 h=10–20 rollout ≫ phase-1; L2 h=50 keeps spike/V
advantage; L3 h=100–200 macro-dynamics (pop rate corr) clearly above naive
even if exact spikes decorrelate; L4 OOD ≈ seen; L5 no trivial attractor.
If exact-spike F1 stays collapsed AND the LIF sensitivity test shows the true
system itself decorrelates under tiny perturbations, the conclusion becomes
"local transition operator + macro-dynamics recovered; exact long-horizon
spike trajectories are not stably predictable in this system" — a valid
result, not a failure.

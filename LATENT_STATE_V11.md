# latent_state_v11 — Stage 1 report: hybrid LIF + latent residual vs the exact LIF prior

Core question (v11, set after v10 closed mechanism classification): learn
the latent residual dynamics the LIF/connectome model cannot explain and
use the latent state to improve neural dynamics prediction. Mechanism-blind
training (no GAIN/ADAPT/STP labels anywhere), frozen protocol
(protocol/protocol.md), 5 paired seeds. Not a biological claim.

## What Stage 1 found (headline)

**The exact hard LIF prior beats every learned residual model at one-step
prediction — on both V RMSE and spike F1 — at the v9 effect-matched
magnitudes. Gate G1 FAILS (0/5 seeds, every family, every mask). The
frozen failure rule F1 triggers: Stages 2-4 were not run.**

testB (held-out params), all-mask V RMSE / spike F1:

| model | gain | adapt | stp |
|---|---|---|---|
| B0 hard LIF (exact) | 0.0179 / 0.971 | 0.0121 / 0.987 | 0.0094 / 0.994 |
| B1 soft base_pre | 0.0246 / 0.971 | 0.0218 / (calib.) | 0.0279 / — |
| B2 v9 corrector | 0.0216 / 0.897 | 0.0187 / 0.878 | 0.0243 / 0.901 |
| B5 v11 hybrid | 0.0209 / 0.938 | 0.0182 / 0.927 | 0.0233 / 0.942 |

Per-mask V RMSE (b0 → b5): gain free 0.0131→0.0150, info 0.0952→0.1047,
event 0.1885→0.2182; adapt free 0.0076→0.0131, info 0.0243→0.0902; stp
free 0.0078→0.0163, info 0.0548→0.1038. On NULL the exact base is bitwise
(0.0) and b5 costs 0.0227.

## Why it fails (the diagnosis is the result)

1. **The base is exact where it matters.** With the artifact-free hard
   transition (v10), the LIF prior is the true operator on the ~99%
   free/non-event transitions; its only one-step errors are the mechanism
   residuals themselves (free-step RMS 0.008-0.013) and rare spike-timing
   boundary cases.
2. **The correction channel has an intrinsic noise floor ABOVE that
   residual.** A hybrid trained PURELY on the NULL teacher (nothing to
   learn) still emits correction RMS 0.012-0.018 and costs 0.0212 V RMSE
   on null/test where the base costs 0.0. The learned correction cannot
   express corrections finer than its own fitting noise.
3. **The v9-era "corrector beats base" claims were relative to the soft
   base_pre**: its V-RMSE deficit is the documented soft-reset artifact
   (0.019-0.026, v10), and its spike-F1 deficit was inflated by a
   shared-threshold artifact (base logits scored at the corrector's
   calibrated threshold; audit/artifact_control.md §3). Recomputed with
   per-model calibration, the soft base's F1 is 0.97, and v9's corrector
   F1 (0.88-0.90) is BELOW it. No v9 files were modified; the v9 V-RMSE
   improvements over the soft base replicate exactly (b2 rows) and remain
   valid as far as they go.
4. **The latent does carry real signal** — z-shuffle at eval degrades the
   hybrid consistently (info-mask +0.0047..+0.0057 V RMSE, 5/5 seeds) —
   but at these effect magnitudes the signal does not pay for the
   correction channel's own noise in one-step V prediction.

## Relationship to the frozen hypotheses

- H1 (hybrid > hard LIF one-step): REFUTED (5/5 seeds, all families).
- H2 (latent, not input copy): SUPPORTED DESCRIPTIVELY at small magnitude
  (z-shuffle sensitivity; NULL control quantifies the noise floor), but
  the preregistered G2 clause is vacuous without a G1 gain.
- H3-H5 (history/connectome ablations, latent intervention, rollout):
  NOT TESTED — F1 stop rule.

## What would change the verdict (deliberately NOT done this round)

- A correction channel with a sub-residual noise floor: sparse/gated
  correction (exact zero on free steps), event-weighted loss, or
  hard-consistent decoding (reset by the model's own spike decision).
- Effect magnitudes above the noise floor (the v9 band was deliberately
  LOW-effect matched); or metrics where base errors compound — rollout
  (the original v11 motivation: 1-3% per-step spike disagreement compounds
  through I_syn over 100 steps), OOD dynamics, held-out stimulus patterns.
  The plan gated those behind G1; whether to re-gate Stage 3 as a
  standalone rollout study is a user decision, not a unilateral metric
  edit.

## Files

protocol/protocol.md, audit/artifact_control.md, metrics/ (onestep.csv,
zshuffle.csv, null_control.csv, training/), checkpoints/ (full_seed*,
full_seed*_null), figures/ (fig1_onestep, fig2_controls), gates.json.
Code: models/latent_hybrid_v11.py, data_v11.py, train_v11.py, eval_v11.py.
Design doc: LATENT_STATE_V11_PLAN.md (root).

## Reproduce

  .venv-cuda\Scripts\python.exe data_v11.py            # NULL-teacher control data
  .venv-cuda\Scripts\python.exe train_v11.py           # 5 seeds, mixed pool
  .venv-cuda\Scripts\python.exe train_v11.py --null    # 5 seeds, NULL control
  .venv-cuda\Scripts\python.exe eval_v11.py            # Stage 1 eval + controls

# latent_state_v9 — candidate mechanism identification from residual dynamics

| Question | Answer | Evidence |
|---|---|---|
| Q1 effect-matched 后机制仍可区分吗？ | PARTIAL: amplitude unreadable; latent info exists; observation-only weak; intervention amplifies | effectmag-only = chance 0.34; oracle 0.87/0.88/0.92 (A/B/C); passive best 0.48-0.56; intervention 0.59-0.72 |
| Q2 passive history 能否识别 candidate？ | WEAK: above shortcut, far below oracle; order not exploited | raw_ordered 0.493/0.481/0.555 > shortcut 0.456/0.424/0.513; time-shuffle 0 drop; label-shuffle 0.35 ~ chance |
| Q3 correction representation 是否增加 mechanism information？ | NO | frozenZ 0.442-0.527 ~ raw; z clusters by effect size (sil +0.417, R2 0.960, LDA 0.119) not family (sil -0.059, LDA 0.0043); corrector fingerprint = base_pre fingerprint (0.566 vs 0.557; align 0.995 vs 0.996) |
| Q4 intervention 是否显著提升 identifiability？ | YES, teacher-side (observation-driven) | held-out params 0.589 vs passive best 0.481; extrapolated 0.719 vs 0.555; AUROC 0.87-0.91; base_pre matches |
| Q5 held-out parameters / unknown 上是否泛化？ | Params: degraded but nonzero; unknown: forced classification | testB hardest across channels; OU force-classified as gain 58% w/ higher confidence than known; rejection only via reversed entropy AUROC 0.69-0.85; NULL unrejectable 0.42-0.54 |
| Q6 是否已经足够称为 mechanism identification？ | Only ADAPT (+NULL): "intervention-validated candidate identification for ADAPT; closed-set candidate discrimination otherwise" | candidate fitting on UNSEEN intervention: adapt 63/64 (testB) 32/32 (testC); null 47/64; gain 11/64; stp 4/64 |

Terminology ladder enforced: functional residual correction < mechanism-relevant
representation < closed-set candidate discrimination < intervention-validated
candidate mechanism identification. v9 evidence reaches level 3 generally and
level 4 for ADAPT only. Never "biological mechanism discovery".

## Setup (frozen: protocol/protocol.md, configs/selection.json)

- Teachers (teachers_v9.py, bitwise-verified vs v2/v7/v8): GAIN global
  multiplicative gain (2-D oscillator), ADAPT per-neuron spike-triggered
  current, STP per-edge (u,x) plasticity (3 clusters, norm=1/U), NULL = base
  LIF, UNKNOWN = additive global OU current (never trained).
- Effect matching (Gate A): 100+ config pool; selected band resRMS
  [0.128, 0.153], spike disagreement [0.011, 0.017], |rate change|
  [0.001, 0.010] overlapping; trajectory-level stratified matching for
  identification subsets. Found + recorded: ADAPT c*beta exact degeneracy;
  ADAPT's structural firing suppression is the one non-matchable cue
  (rate-only shortcut ~0.36-0.47).
- Splits: A = held-out trajectories/seen params; B = held-out interpolated
  params (primary); C = extrapolated params. 3 train configs/family, 510
  train trajectories/family.
- Phase A: ONE unified OrderedHistory corrector (K=32), mixed pool, no
  mechanism label, no classification loss, 5 paired seeds.
- Resources: N=100, T=256; peak VRAM 10.8 GB (intervention model sweep);
  corrector epoch ~1 s; per-edge STP state never stored as a path.

## Results

### Gate B — mechanism-blind correction: PASS-TRIVIAL

Unified corrector improves over base_pre on every family, 5/5 seeds, on
seen AND held-out params (testA +0.0025..+0.0035, testB +0.0029..+0.0036 V
RMSE; spike F1 +0.26..+0.40). BUT the NULL reference improves +0.0035
(+0.338 F1) identically: at the matched effect level the one-step task is
dominated by the base_pre soft-reset artifact (artifact RMS 0.019 vs true
1-step mechanism residual: gain 0.0156 / adapt 0.0151 / stp 0.0083; MAE:
adapt 0.0044 tonic vs gain 0.0005 / stp 0.0002 sparse). Correction success
is real but NOT mechanism-attributable here.

### Gate C — passive identification: WEAK

5-seed balanced accuracy (testA/testB/testC): effectmag 0.34/0.35/0.34
(chance); shortcut 0.456/0.424/0.513; EventRich 0.342/0.338/0.351; RawK2
0.433/0.398/0.474; RawOrdered 0.493/0.481/0.555; FrozenZ-linear
0.457/0.442/0.527; FrozenZ-MLP 0.448/0.430/0.510; Oracle 0.869/0.879/0.922.
Ordering: effectmag < EventRich ~ shortcut < RawK2 < FrozenZ ~ RawOrdered
<< Oracle. Time-shuffle: no degradation (the ordered model does not use
temporal order for this task). Label-shuffle: chance (0.328-0.350).

### Gate D — representation enrichment: FAIL

Frozen corrector representation never beats raw-observation classifiers.
Variance decomposition: LDA ratio family 0.0043 vs effect-size 0.119;
silhouette family -0.059 vs effect +0.417; linear probe family acc 0.459 vs
effect R2 0.960. The corrector organizes windows by residual
amplitude/firing regime, not by mechanism family. On interventions, the
corrector's predicted fingerprint equals the base_pre formula's (transfer
0.566 vs 0.557 on held-out params; teacher-alignment cosine 0.995 vs 0.996).

### Gate E — intervention fingerprint: PASS (teacher-side, observation-driven)

16-d bounded shape fingerprint from 4 legal stimulus-only probe blocks
(delay recovery d=0..32, burst structure, edge-local vs neuron-global
preconditioning, history load). Teacher fingerprints classify family:
0.865 (seen) / 0.589 (held-out params, AUROC 0.909) / 0.719 (extrapolated,
AUROC 0.872) - clearly above every passive channel. Per-family held-out
confusion (testB): gain 0.95 / adapt 0.72 / stp 0.53 correct - the GAIN
fingerprint generalizes best across parameters; STP is hardest
(44% confused as gain). But the same
fingerprint computed from base_pre predictions (no learning, no mechanism)
matches the corrector's: the fingerprint's discriminative content is
carried by observable trajectory history, not by learned corrections.

### Gate F — candidate fitting + unseen-intervention validation: PARTIAL

Observation-legal recursive estimators (9 grid points each: GAIN windowed
scalar, ADAPT back-out, STP per-edge filter; NULL = base; a same-transition
leakage in the first ADAPT estimator draft was caught and fixed before
reporting - see audit/leakage_audit.md). Grid fit on
passive segment; family selected ONLY on the unseen intervention segment.
testB/testC winners: ADAPT 63/64 + 32/32 (intervention validation beats
passive-only selection 50/64 -> 63/64, as designed); NULL 47/64; GAIN
11/64; STP 4/64 (loses to null). Margins tiny (1e-4..3e-4) except ADAPT.
Consistent with the 1-step residual structure: ADAPT's tonic subtractive
current is uniquely trackable; GAIN/STP effects are sparse and below the
artifact floor at the matched level.

### Open set / unknown

OU (matched effect, never trained): every channel force-classifies it -
mostly as GAIN (58%; global additive drive ~ global multiplicative family)
- with HIGHER confidence than known-family data. Entropy-based rejection
works only in the reversed direction (AUROC: frozenZ 0.81, raw_ordered
0.85, shortcut 0.72, fingerprint 0.69). NULL cannot be rejected
(0.42-0.54). => "closed-set candidate discrimination", not general
open-set mechanism identification.

### Calibration

FrozenZ-linear probe is well calibrated: ECE 0.015/0.022/0.049, Brier
0.587/0.608/0.541, confidence ~ accuracy (0.42-0.47 vs 0.45-0.52) on
A/B/C. Honest uncertainty, no overconfidence on seen families.

### Mixture stage

SKIPPED per the pre-registered rule (mixture only after single-mechanism
gates basically succeed; B is trivial, C weak, D fails, F partial).

## 与 v5-v8 的统一科学链条

- v5/v6: history can recover missing latent information from partial
  observability (gain).
- v7: hidden state != history required; locally invertible => K1/K2 enough.
- v8: edge-level event memory restores a history requirement; compact
  event statistics explain most of it.
- v9: when candidate mechanisms are effect-matched, (i) one-step residual
  correction hits the artifact floor and is not mechanism-attributable;
  (ii) observable dynamics still carry weak candidate information
  (passive 0.48-0.56), amplified by legal interventions (0.59-0.72,
  AUROC ~0.9) - but readable WITHOUT any learning via the base formula;
  (iii) the learned corrector does NOT internalize mechanism identity
  (representation organized by effect size, not family); (iv) explicit
  candidate fitting + unseen-intervention validation identifies only the
  family with a tonic dense residual (ADAPT) and NULL.

Paper-safe statement: "Omitted neural dynamics can be functionally
recovered from observable activity when their latent effects are
identifiable from available history. Under effect-matched candidate
mechanisms, predictive improvement alone does not attribute a mechanism;
observable intervention responses discriminate candidates in closed set,
and explicit candidate fitting validates only mechanisms whose residual
structure is dense in time. Mechanistic claims require candidate-specific
intervention evidence, not learned representations."

## Next steps

1. Intervention sets that target the SPARSE mechanisms' fingerprint
   (edge-tagged repeated probing for STP; phase-locked perturbation for
   GAIN) with per-trajectory averaging to beat the artifact floor.
2. Open-set discipline: distance/energy-based rejection trained with
   held-out mechanism families as pseudo-unknowns.
3. Real-data regime: apply the effect-matching + intervention-fingerprint
   methodology where the candidate set is unknown a priori.

## Limitations

- Single network (N=100 ring graph), one stimulus protocol family.
- Matched effects are ~10-20x weaker than the v5-v8 default regimes;
  conclusions are specific to this artifact-dominated regime.
- Fingerprint blocks are hand-designed; coverage of mechanism space is
  incomplete (mixtures untested).
- OU unknown is one structurally simple family.

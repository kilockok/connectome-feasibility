# latent_state_v10 — artifact-free active intervention design for candidate mechanism discrimination

| Question | Answer | Evidence |
|---|---|---|
| Q1 numerical artifact 压到 mechanism signal 以下？ | YES (exact) | NULL-base = 0.0 bitwise (V RMS/maxabs 0, spike disagree 0) vs info-bearing residuals gain 0.070/adapt 0.084/stp 0.032; soft floor 0.019-0.026 quarantined to corrector secondary role |
| Q2 v9 manual intervention 在 clean pipeline 上仍有效？ | YES | v9hand testB 0.485 > random 0.442 > passive 0.429; STP 0.519 = best STP policy; signal was real, not artifact |
| Q3 optimal design 超过 random/manual？ | NO (GAIN-only yes) | optimized_adaptive - random = +0.006±0.051 (sign 0.6); vs v9hand -0.038 (sign 0.2); oracle-design bound 0.490; GAIN: optimized 0.537 best (+0.100 vs manual, sign 0.8) |
| Q4 GAIN/STP 真正分开？ | Predictively YES (z 0.3→30), empirically GAIN-only | pairwise landscape: 96-98% legal entries >2× passive; STP fitted-scoring 0.34, oracle-design bound 0.43; gain↔stp confusion 0.24/0.30 |
| Q5 held-out parameters 泛化？ | PARTIAL | opt_adaptive overall A/B/C 0.500/0.448/0.500; GAIN designed probe 0.537/0.537/0.406 (no design overfit); testB hardest; passive ADAPT 0.84-0.91 |
| Q6 sequential probing 提高 sample efficiency？ | NO | optimized ≈ random at every budget (1 probe 0.431 vs 0.458; 8 probes 0.392 vs 0.381); 70% never reach conf 0.8 in 8 probes; accuracy declines with budget |

## Current claim level: 2 — candidate discrimination

Ladder (protocol): 1 functional residual correction < 2 candidate
discrimination < 3 intervention-validated candidate identification < 4
active intervention-validated system identification.

v10 evidence supports level 2, with a family-resolved picture:

- GAIN: intervention-driven identification improves materially
  (0.244 passive → 0.537 designed probe, +0.294, sign 1.0 across seeds;
  the designed probe beats v9hand by +0.100, sign 0.8) — the strongest
  single-family result, but 0.54 is below the bar v9 set for level 3
  (ADAPT 0.98).
- ADAPT: identified PASSIVELY (0.856 testB tonic residual); every probe
  policy DEGRADES it (to ~0.45-0.52). v9's level-3 "intervention-validated"
  ADAPT claim is refined: the identification lives in the passive tonic
  residual, not in intervention responses.
- STP: not identified by any active policy (best = v9hand 0.519,
  optimized 0.338, oracle-design bound 0.431).

Not claimed: biological mechanism discovery.

## Setup (frozen: protocol/protocol.md; legacy_v9_audit.md)

- Teachers/estimators/splits: v9 reused unchanged (GAIN global
  multiplicative, ADAPT per-neuron current, STP per-edge (u,x), NULL=base;
  effect-matched; A seen / B held-out interpolated (primary) /
  C extrapolated; observation-legal recursive estimators, v9 grids).
- ARTIFACT-FREE path (new): HARD LIF transition only; NULL == base
  bitwise; info-bearing/event masks replace all-step averaging.
- INTERVENTION LIBRARY (new): 55 constrained entries (delay/burst/paired/
  edge/global/highcur/phase/passive); amp ≤7, neurons ≤40, charge ≤4000,
  induced rate <0.45 on the NULL teacher; cost(d) recorded;
  U − 0.05·cost.
- OBJECTIVES: U_pair (min-pair predictive z — found DEGENERATE, q10=0 on
  every entry; recorded, unused), U_JS (mean-pair symmetric KL, Gaussian
  predictive, theta-bank top-2 marginalized, sigma from passive
  residuals), U_robust = Q0.1(U_JS) − 0.05·cost over 8 design contexts
  per family from the TRAIN split only.
- SHORTLIST: top-12 by U_robust, ≤3 per intervention family, frozen
  before any test evaluation.
- POLICIES: passive / random / heuristic_stp (paired_i8) / heuristic_gain
  (highcur_a5.0) / optimized_global / optimized_adaptive / oracle
  (privileged: true family+params for CHOICE only; labelled upper bound) /
  v9hand (v9 16-branch package).
- SCORING (identification): masked Gaussian V NLL (per-candidate passive
  sigma) + spike BCE on response windows; winner = argmin NLL.
- Correctness: a shortlist order-permutation bug in the first Stage 8/9
  runs (argmax over library-ordered U mapped through rank-ordered ids)
  scrambled executed vs scored entries for the three optimized/oracle
  policies. Found via a Stage 9 crash, fixed, all affected rows purged
  and recomputed (audit/design_leakage_audit.md addendum). Unaffected
  policies were kept.

## Results

### Gate A — artifact removal: PASS (Stage 1-2)

NULL teacher vs HARD base predictor, same initial state and stimulus:
V max abs 0.0, V RMS 0.0, spike disagreement 0.0 — bitwise equality.
The v9 floor came from the SOFT base_pre path: V RMS 0.0263 overall,
decomposed as a 0.0191 everywhere bias on free steps (97.4% of
transitions) plus a 0.2265 RMS error on the 0.66% true-spike reset steps.
Both vanish under the hard path. Mechanism residuals vs the hard base
(testA): all-step gain 0.0156/adapt 0.0151/stp 0.0083, but info-bearing
(free, |I_syn| > median) gain 0.0705/adapt 0.0842/stp 0.0320 and
event-step gain 0.109/stp 0.064 — STP's all-step dilution (0.0083 <
old floor 0.019) is why all-step averaging had to go
(audit/artifact_audit.md, fig1).

### Gate B — candidate predictions diverge: PASS (Stage 5/7)

On design contexts (train split only), every legal library entry moves
the candidate predictive distributions apart far above passive:
U_JS mean passive 1096 vs library median 13444 / max 19579 (12-18×).
For the v9-unresolved pair specifically: D(GAIN,STP) predictive z
passive 0.31 (gain contexts) / 1.00 (stp contexts) rises to max 29.2 /
30.3 (~30-95×), with 96-98% of entries above 2× passive. Best GAIN-STP
probes: edge-local (edge_j2_r3, cost 0.36), short delays (d0-d2), long
bursts (burst_n8_i4_s16). The min-pair worst-case objective U_pair is
DEGENERATE (q10=0 everywhere: some candidate pair always overlaps at
some design context) — recorded; mean-pair U_JS used instead
(intervention_scores.csv, pairwise_gain_stp.csv, fig2, fig3).

### Stage 3 — v9 manual reference on the clean pipeline

The v9 16-branch hand protocol, rescored with artifact-free candidate
NLL: testB 0.485 overall (gain 0.438/adapt 0.500/stp 0.519) vs random
0.442 and passive 0.429; testC 0.458. v9's intervention signal survives
the artifact removal — it was real mechanism signal. It remains the best
STP policy anywhere in v10.

### Gates C/D/E — single-probe experiment (Stage 8, 5 paired seeds)

accuracy@1probe, testB (held-out params), per family + ALL:

| policy | gain | adapt | stp | ALL |
|---|---|---|---|---|
| passive | 0.244±0.061 | 0.856±0.081 | 0.188±0.040 | 0.429 |
| random | 0.487±0.083 | 0.450±0.081 | 0.388±0.067 | 0.442 |
| heuristic_stp | 0.450±0.025 | 0.500±0.052 | 0.494±0.078 | 0.481 |
| heuristic_gain | 0.506±0.031 | 0.450±0.064 | 0.225±0.031 | 0.394 |
| optimized_global | 0.419±0.032 | 0.431±0.064 | 0.388±0.076 | 0.413 |
| optimized_adaptive | 0.537±0.070 | 0.469±0.071 | 0.338±0.046 | 0.448 |
| oracle (bound) | 0.519±0.025 | 0.519±0.058 | 0.431±0.070 | 0.490 |
| v9hand | 0.438±0.059 | 0.500±0.052 | 0.519±0.070 | 0.485 |

Paired-seed differences (testB, 95% CI):
- Gate C (optimized_adaptive − random): ALL +0.006±0.051, sign 0.6 →
  FAIL. (gain +0.050, adapt +0.019, stp −0.050.)
- Gate D (optimized_adaptive − v9hand): ALL −0.038±0.033, sign 0.2 →
  FAIL overall; GAIN +0.100 sign 0.8 is the single positive family.
- vs passive: gain +0.294 (sign 1.0), stp +0.150 (sign 1.0),
  adapt −0.388 (sign 0.0 — probing destroys the tonic-residual
  identification).
- vs oracle: −0.042 sign 0.0 — the adaptive policy sits at/below the
  design upper bound, and that bound is only 0.490.

Confusion (testB, optimized_adaptive): gain→{gain .537, stp .237,
null .144}, adapt→{adapt .469, stp .512}, stp→{gain .300, stp .338,
null .344}. STP's flexible per-edge filter absorbs both GAIN's and
ADAPT's probe responses; ADAPT's probe responses get absorbed by STP.

testA (seen params; gain/adapt only per protocol coverage):
optimized_adaptive 0.500 ALL (gain 0.537/adapt 0.463), oracle 0.531,
heuristic_stp 0.566 (best), passive 0.562.
testC (extrapolated): optimized_adaptive 0.500 ALL (gain 0.406/adapt
0.438/stp 0.656, best together with optimized_global 0.479), oracle
0.406, v9hand 0.458, random 0.460, passive 0.406.

### Gate E detail — GAIN vs STP 专项 (Stage 7)

The 2×2×2 factorial (global current × edge history × ISI 4/32) teacher
response tensor shows the two mechanisms occupy the same qualitative
response corners and differ mainly in magnitude: e.g. edge-history-only
cells flip sign with ISI for both (gain −0.061→+0.029; stp −0.114→+0.027;
null −0.035→+0.026; adapt −0.030→+0.033), with STP ~1.9× GAIN at ISI 4.
This is the structural reason fitted scoring confuses them: the
fingerprint is a scaled copy, and the scale is parameter-dependent
(factorial.csv, fig7).

### Gate F — held-out/extrapolation robustness (Stage 10)

parameter_ood.csv (derived from single_probe): per-policy family accuracy
across cohorts. Design does not overfit the design parameters: GAIN
under optimized_adaptive 0.537 (A) / 0.537 (B) / 0.406 (C); STP under
optimized_adaptive —/0.338/0.656 (effect grows on extrapolation).
testB (interpolated) is the hardest cohort for every passive channel;
patterns otherwise replicate.

### Gate G — sequential active identification (Stage 9): FAIL

Posterior over 4 candidates, theta banks refit on ALL observations before
every probe, greedy posterior-weighted pairwise SKL choice vs random,
probes execute sequentially in time, budgets 0/1/2/4/8, testB, 5 paired
seeds × 32 contexts × 3 families (sequential.csv, fig56):

| budget | optimized acc (gain/adapt/stp/ALL) | random acc (ALL) | entropy opt/rand |
|---|---|---|---|
| 0* | 0.000/0.000/0.000/0.000 | 0.000 | 1.386/1.386 |
| 1 | 0.512/0.438/0.344/0.431 | 0.458 | 0.464/0.462 |
| 2 | 0.494/0.463/0.362/0.440 | 0.410 | 0.405/0.403 |
| 4 | 0.375/0.456/0.344/0.392 | 0.381 | 0.376/0.371 |
| 8 | 0.375/0.456/0.344/0.392 | 0.381 | 0.376/0.371 |

*budget 0 is the uniform-prior anchor (all winners null), not a
measurement.

- optimized − random: −0.027 (sign 0.2) at 1 probe; +0.010 (sign 0.6)
  at 8 probes → no separation at any budget.
- Sample efficiency: 70% of trajectories never reach conf(true) ≥ 0.8
  within 8 probes under EITHER policy; median first-correct budget 4
  (optimized) vs never (random), without persistence.
- Accuracy peaks at ONE probe and declines after: cumulative extra
  stimulus pushes the network into regimes the fitted candidate banks
  extrapolate worse — consistent with single-probe ADAPT degradation.

### Open-set sanity (Section 17 protocol)

OU unknowns (2 configs × 32): passive winners null 30/adapt 26 at conf
0.410; under optimized probes winners spread null 23/stp 16/gain 14/
adapt 11 at HIGHER conf 0.677. Force-classification persists; open-set
inference remains deferred per protocol (unknown_v10.csv).

### Stage 11 — corrector secondary analysis

(A) Under optimized-probe windows the frozen v9 corrector's one-step gain
over the hard base GROWS 2-3× (+0.0064..+0.0083 V RMSE vs +0.0031
passive): high-SNR observations help the corrector predictively.
(B) FrozenZ linear probe under the same high-SNR windows: 0.336-0.389
balanced accuracy vs 0.422 on passive windows (chance 0.333) — still no
mechanism-identity content. Third independent confirmation (v9 passive,
v9 intervention, v10 high-SNR) that predictive correction and mechanism
identity are distinct representations (corrector_secondary.csv).

## Outcome: C — predictive divergence exists, experimental identification fails (for STP) / partial (GAIN)

The chain of exclusions is the result:

1. NOT the artifact: removed exactly (Gate A, bitwise).
2. NOT the discriminability: legal interventions separate candidate
   predictions by z ≈ 30, 30-95× passive (Gate B).
3. NOT the design: oracle-design (true family+params for the choice)
   bounds the whole policy class at 0.490 overall and 0.431 for STP;
   manual ≈ optimized ≈ random within noise for STP/ADAPT.
4. NOT the budget: sequential probing is flat-to-declining from 1 probe
   on (Gate G).
5. The binding constraint is the ESTIMATOR / observation channel:
   fitted-candidate masked NLL cannot convert z≈30 predictive separation
   into identification for STP (per-edge filter absorbs other families'
   responses; GAIN/STP fingerprints are scaled copies, factorial Stage 7).

Consequence for v11+: improve the candidate scoring/estimation channel
(e.g. proper Bayesian parameter posteriors, richer per-edge observation,
or additional measurement/control channels) — NOT more intervention
design and NOT a bigger classifier. Outcome D (structural
non-identifiability) is NOT established: the z≈30 landscape says the
information exists; what is shown is practical non-conversion by the
current estimator class.

## 与 v5-v10 的统一故事

- v5/v6: history 可恢复部分隐藏动力学信息。
- v7: history value 取决于 observability，不取决于 hidden state 是否存在。
- v8: edge-event latent state 需要历史，但 compact event summary 可解释大部分
  temporal advantage。
- v9: predictive residual correction ≠ mechanism attribution；effect-matched
  passive observations 弱可辨，干预增强可辨识性，但 corrector 不编码
  mechanism identity。
- v10: 被动不足时主动选择 intervention 的命题被直接检验——artifact 被精确
  移除后，合法 intervention 确实让候选机制的可观测预测大幅分离（z≈30），
  且对 GAIN 首次给出 intervention-driven 的识别提升（0.24→0.54）；但
  fitted-estimator 的 observation channel 把 STP 的识别限制在 oracle-design
  bound 0.43 以下，active sequential design 相对 random 无 sample-efficiency
  收益。Identifiability 的瓶颈从 artifact（v9 诊断）转移到 estimator/
  observation channel（v10 证据）；分类器容量自始至终都不是瓶颈。

## Files

audit/ (artifact_audit, legacy_v9_audit, design_leakage_audit + bug
addendum, feature_audit), protocol/ (protocol.md,
intervention_library.json, configs/shortlist.json), metrics/ (artifact,
intervention_scores, single_probe, sequential, pairwise_gain_stp,
factorial, candidate_fitting, parameter_ood, corrector_secondary,
unknown_v10), figures/ (fig1_artifact, fig2_landscape,
fig3_landscape_detail, fig3_gain_stp, fig4_policies, fig56_sequential,
fig7_factorial, fig8_param_ood), gates.json, conclusion.md.

## Reproduce

  .venv-cuda\Scripts\python.exe artifact_v10.py        # Stage 1-2, Gate A
  .venv-cuda\Scripts\python.exe library_v10.py         # Stage 4 library
  .venv-cuda\Scripts\python.exe landscape_v10.py       # Stage 5 landscape
  .venv-cuda\Scripts\python.exe design_v10.py          # Stage 6 pilot
  .venv-cuda\Scripts\python.exe pairwise_v10.py        # Stage 7 GAIN-STP
  .venv-cuda\Scripts\python.exe design_v10b.py         # Stage 8 single-probe
  .venv-cuda\Scripts\python.exe sequential_v10.py      # Stage 9 sequential
  .venv-cuda\Scripts\python.exe unknown_v10.py         # open-set sanity
  .venv-cuda\Scripts\python.exe corrector_v10.py       # Stage 11 secondary
  .venv-cuda\Scripts\python.exe report_figs_v10.py     # aggregation + figs

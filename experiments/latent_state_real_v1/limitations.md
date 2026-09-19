# limitations.md — latent_state_real_v1

## Data limitations

1. Resting state only: no recorded stimulus or behavior. Unobserved inputs
   (neuromodulation, spontaneous behavior-correlated activity) are a
   declared confound for RQ5; we cannot separate "internal dynamics" from
   "unmeasured external drive".
2. Observation model: 1.2 Hz region-mean GCaMP fluorescence, high-pass
   filtered at 0.01 Hz by the source pipeline. All dynamics claims are at
   the fluorescence level; no spike-level or membrane-level inference is
   possible, and no LIF+calcium baseline was identifiable (recorded, not
   fitted).
3. Region granularity: 37 central-brain regions; single neurons are not
   resolved; connectome correspondence is region-to-region via atlas
   registration, not cell-to-cell. Cross-animal correspondence is at the
   atlas level only.
4. Animal independence proxy: splits use recording DATE as a conservative
   animal proxy (same-date runs treated as one animal). The true animal
   count may be lower than 10 (sessions per date could pool >1 fly or
   reuse one fly across dates - the dataset does not document this).
5. Effective sample sizes: 10 date groups; test = 2 dates (4 sessions).
   Exploratory-level evidence only.
6. Zenodo secondary datasets (A/B/C in the spec) were inaccessible from
   this network (HTTP 403 on all endpoints, 2026-09-19) - recorded as an
   access failure, not a data-availability judgment; no substitution was
   made.

## Model/evaluation limitations

7. All models were trained with MSE on z-scored fluorescence; a
   probabilistic (NLL) observation model was considered but not fitted -
   reported metrics are RMSE/MAE/R2 only.
8. Rollout statistics use 200-frame (~167 s) horizons from session-start
   contexts; longer-horizon behavior was not evaluated.
9. The structural matrix is near-complete at this granularity (99.9%
   nonzero off-diagonal): topology-shuffle controls degenerate, and the
   node-relabel control (P A P^T) can only detect region-identity-specific
   contributions. A sparser, cell-level connectome might show
   structure-specific value that region-level aggregation hides.
10. The small M3<M2 gap (latent hurts slightly) may reflect optimization
    difficulty in tiny-data regimes rather than a fundamental property;
    but under the frozen protocol it stands as the result.

## What this study does NOT claim

- No biological mechanism discovery; no latent-variable identification
  with any biological quantity.
- No claim that the connectome does or does not matter for neural
  dynamics in general - only that at THIS granularity, with THIS
  observation channel, region-level structure adds no measurable
  predictive increment over shuffled controls.

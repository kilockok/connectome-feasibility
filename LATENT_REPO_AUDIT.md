# Repository handoff audit for latent_state_v1

Local repository: `F:/Code/connectome/feasibility`; remote execution tree:
`/root/connectome-feasibility`. Remote Python is `/root/miniconda3/bin/python`
(the non-login SSH shell does not put `python` on PATH). CUDA reports torch
2.8.0+cu128 and an RTX 4080 SUPER with 32760 MiB. The remote tree has no .git.
The five checked core files (gnn_temporal, gnn, lif, dataset, rollout_eval_v3)
had identical SHA256 locally and remotely at handoff. No credentials belong
in experiment configs, source, or durable memory.

## Architecture findings

- `models/transformer.py:TemporalEncoder` already performs causal temporal
  attention on `[B*N,K,D]`. VanillaTransformer then performs spatial attention.
- `models/gnn.py:GNNBaseline` also uses that TemporalEncoder, before message
  passing. Both old GNN and wide-GNN names therefore include history.
- `models/gnn_temporal.py:GNNTemporalTransformer` genuinely performs GNN
  encoding at each time and then per-neuron temporal attention. Its default
  `causal=False` permits all tokens within the already observed context.
  That is not leakage of the prediction target, but differs from the new
  requirement that intermediate temporal representations be causal.
- `k_hist` alone just truncates context; it does not establish state recovery.
  The old K32-checkpoint-to-K1 evaluation changes the learned input regime.
  New K1 models are separately initialized/trained, and the new causal test
  verifies that perturbing later observations leaves earlier tokens unchanged.

## Scientific interpretation and protocol findings

- The graph is synthetic ring-geometry with Dale signs, not FlyWire/MaleCNS.
  Teacher dynamics are simulated LIF, not measured neural recordings.
- The legacy simulator stores its state AFTER applying `stimulus[:,t]`.
  Legacy windows pair that state with the same t input while targeting t+1;
  the last input therefore does not contain the upcoming transition's drive.
  Rollout similarly appends the stimulus after predicting the state. This
  permits past inputs to proxy unprovided upcoming input and complicates
  any claim that temporal gains in the old system refute Markov sufficiency.
  The new experiment explicitly stores pre-transition states and verifies
  exact-teacher rollout equality. Old results remain on their old protocol.
- Legacy randomly silenced neurons are another unobserved condition unless
  the mask is supplied. The new single-hidden-mechanism experiment disables
  silencing (alpha=0 numerical compatibility is tested separately).
- `rollout_eval_v3.horizon_metrics_v3` calls uncentred per-neuron rate cosine
  `pop_corr`; it is not Pearson correlation of population activity over time.
  New metrics name and calculate these quantities separately.
- Legacy per-trajectory and pooled F1 can differ greatly on sparse data;
  do not compare them numerically as identical metrics. Stage0's tiny first
  test had only six spikes (F1=.909); 16 full trajectories contained 54157
  spikes and reproduced pooled reinjection F1=.99638. Val F1=.99619.
- Old silent failure detection does not implement the new two-sided dynamic
  failure criterion. An absent silent collapse may mean rate explosion.
- `dagger.collect_on_policy` stores the history BEFORE a student step but
  labels it with a teacher step from the student's AFTER-step state, using
  the following input. This is a context/target time mismatch in that code.
  The old tests check shapes/state validity, not this transition identity.
  Thus negative v4 DAgger effects describe that implementation and protocol;
  they are not evidence that DAgger generally cannot work. DAgger remains off
  in the new stage, as requested; no new DAgger sweep or old-artifact rewrite.
- `losses.tangent_loss` labels the last context input as the upcoming input,
  although legacy windows use the stored-time input. New tangent teacher
  branches use explicitly aligned U[t] and the retained true z[t], without
  passing z to the student or a z supervision loss.

## Isolation and verification

The pre-existing local modification to `rollout_eval_v3.py` and untracked
merge/conclusion helpers were retained. `_fail.py`/`_pass.py` are legacy
drafts and are not part of the new pipeline. No existing model, simulator,
evaluator, cache or checkpoint was replaced by the new stage.

New implementation details and fixed research gates are documented in
`LATENT_STATE_V1.md`; measured outcomes belong to the generated conclusion.
The 9 original temporal tests passed remotely; the new scientific invariant
suite additionally verifies hidden-state rejection, exact alpha=0 teacher,
seed/split separation, time-order preservation, causal masks, K shapes,
exact rollout input alignment, two-sided failure timing, and independent
linear-probe recovery. The Stage0 numerical audit is saved under
`results/latent_state_v1/stage0/baseline_regression.json`.

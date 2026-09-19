# latent_state_real_v1 — README

Real-data-driven system identification of Drosophila neural dynamics with
connectome-constrained GNN and temporal latent states. v11b (synthetic)
closed with: the exact LIF baseline dominates and the correction channel
adds noise. This round moves to REAL resting-state region-level activity
(SC-FC dataset) and asks whether structure + history help predict it.

## Headline answer

We moved from learning a LIF teacher to a reproducible real-data pipeline
at the BRAIN-REGION level with strict hygiene (Gate D0/G5 pass). The
reproducible dynamics at this granularity are essentially
linear-autoregressive: no learned model beats AR(5) at one-step held-out
prediction; the real connectome adds no increment over shuffled controls;
the temporal latent adds no increment over the graph model. Autonomous
rollouts are stable but mean-reverting. See results_summary.md,
limitations.md, gates.json.

## Layout

- data_access_report.md / data_manifest.json / dataset_validation.json -
  Stage 1 (sources, hashes, QC, Gate D0)
- experiment_protocol.md - frozen Stage 2-7 protocol
- baseline_diagnostics.md - Stage 3
- results_summary.md - Stage 6-7
- limitations.md - declared confounds and non-claims
- gates.json - machine-readable gate verdicts
- prep_real.py - Stage 2 data prep (data_real.pt, splits.json)
- baselines_real.py / models_real.py / train_real.py / eval_real.py /
  rollout_real.py / test_real.py - Stage 3-6 code (test_real.py = hygiene
  tests, all passing)
- compat_pickle.py - pandas-2 unpickler for the legacy SC-FC pickles
- data/ - figshare downloads (hashes in data_manifest.json)
- results_real/ - checkpoints + metrics CSVs

## Reproduce

  python stage1_dl_small.py && python stage1_dl_main.py   # downloads
  python prep_real.py                                     # tensors+splits
  python baselines_real.py                                # B0/B2 baselines
  python train_real.py --kinds m1 m2 m3 b1 --hs 1         # main models
  python train_real.py --kinds m1 m2 m3 --hs 2 4 8        # horizons
  python train_real.py --kinds m3_nolatent m2_identity --hs 1
  python train_real.py --kinds m2 m3 --graphs edges weights --hs 1
  python eval_real.py && python rollout_real.py           # metrics
  python test_real.py                                     # hygiene (G5)

Environment: repo .venv-cuda (torch + numpy + pandas + scipy); no new
dependencies were added.

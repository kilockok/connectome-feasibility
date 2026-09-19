# latent_state_real_v1 — Stage 1: data access report

Date: 2026-09-19. Agent-local verification; all files hashed in
data_manifest.json; QC in dataset_validation.json.

## Primary dataset: SC-FC (figshare 13349282, DOI 10.6084/m9.figshare.13349282, v3)

Turner, Mann, Clandinin - "The connectome predicts resting state functional
connectivity across the Drosophila brain". Analysis code:
github.com/mhturner/SC-FC (master). License: MIT.

Downloaded: data_TurnerMannClandinin.tar.gz (245.7 MB, sha256 recorded),
body_ids.csv, StructuralMatrix_branson.csv, CorrelationMatrix_branson.csv.
Selectively extracted: ito_responses/, branson_responses/,
connectome_connectivity/, ito_68_atlas/Original_Index_panda_full.csv,
hemi_2_atlas/*.csv. Anatomy volumes (.nii.gz/.tif) NOT extracted (not needed
for region-level modeling).

### Verified content

| item | value | evidence |
|---|---|---|
| data type | REAL experimental (GCaMP functional imaging, resting state, no stimulus recorded) | paper + data inspection |
| species/sex | Drosophila melanogaster (female, per paper) | paper |
| individuals | 20 recording sessions (dates 2017-10-26 .. 2018-12-14, 1-2 runs per date); SAME sessions segmented at two atlas granularities (ito + branson row sets share date_run ids) | file inventory |
| granularity | brain REGIONS (not single neurons): ito = 72-75 rows/fly (subset of 86 atlas ROIs); branson = 605-705 rows/fly | stage1_inspect.py |
| sampling rate | fs = 1.2 Hz (trusted: hard-coded in the authors' own FC pipeline, getCmat: high-pass 0.01 Hz, fs=1.2) | scfc/functional_connectivity.py |
| continuous duration | 2000 frames (~27.8 min) most sessions; 4000 frames (~55.6 min) for 2017-11-08 runs | shapes |
| authors' preprocessing | high-pass Butterworth(1, 0.01 Hz) + per-session artifact trimming map (3 sessions have dropout/baseline-shift trims) + default drop of first 100 frames | functional_connectivity.py |
| raw values | mean voxel fluorescence per region per frame (NOT dF/F) | computeRegionResponses |
| structural matrix | JRC2018_ito_tbar_matrix.csv: 86x86 directed T-bar counts from the hemibrain onto ito ROIs (also cellcount; JRC2018 = hemibrain reference); branson granularity: JRC2018_branson_*_matrix.csv (999) + figshare StructuralMatrix_branson.csv (295 merged regions) | hemi_2_atlas/*.csv |
| region correspondence | functional row id = atlas ROI number = tbar matrix index (verified: union of response row ids = subset of 1..86; row 0 = background, dropped) | stage1_align.py |

### Modeling set chosen (Gate D0 evaluation)

- 37 central-brain regions present in ALL 20 sessions (intersection of the
  paper's 38 central-brain regions with per-fly present ROIs; MB_ML_L is
  absent in some sessions and excluded). Optic-lobe regions are excluded by
  the source paper's own convention (not covered functionally).
- Per-fly region traces with the authors' trim map; NaN fraction 0;
  flatline regions 0 (dataset_validation.json).
- Structural matrix on the 37-region common set: directed, 99.92% nonzero
  off-diagonal (T-bar counts), median nonzero 8000.

**Gate D0: PASS** — real activity data with a trusted sampling interval
(1.2 Hz), reliable observables (region mean fluorescence), 20 sessions,
~1900-3900 usable frames each, and a same-parcellation structural
connectome matrix.

## Secondary sources (evaluation)

| source | status | notes |
|---|---|---|
| A. Kenyon Cell olfactory (zenodo 8166598) | ACCESS FAILED: HTTP 403 on api and web endpoints (2026-09-19, this network) | recorded; no substitution made |
| B. Kenyon Cell calcium 2026 (zenodo 21821328) | ACCESS FAILED: HTTP 403 | recorded |
| C. MBON05 voltage imaging (zenodo 18675613) | ACCESS FAILED: HTTP 403 | recorded |
| D. MaleCNS v1.0 (male-cns.janelia.org) | reachable, but structure-only (EM connectome; no paired activity recordings) | not sufficient alone for the prediction task; potential future connectome upgrade |

If the Zenodo records become reachable from another network, A (Kenyon
olfactory responses) would be the first candidate for an independent
stimulus-driven validation set; the 403 block is a network-level
restriction, not a data-availability limitation of the record.

## Critical distinctions (per spec)

1. Real anatomy: hemibrain EM-derived region connectivity (T-bar counts) - YES.
2. Real activity: GCaMP region time series, resting state - YES.
3. Cross-individual region correspondence: via the shared ito_68 atlas
   (registration-based), NOT single-cell - recorded as a limitation.
4. Same-individual neuron correspondence: NONE (region granularity only) -
   no body-ID-to-ROI matching is claimed or needed at this granularity.

## What this data cannot support (recorded before modeling)

- Resting state only: no recorded external stimulus; unobserved-input
  confounds are a declared limitation for RQ5.
- 1.2 Hz region-average GCaMP: cannot recover millisecond membrane
  dynamics or LIF spike parameters; no LIF baseline is fitted at this
  granularity (baseline models are rate/AR class, Stage 3).
- Calcium observation filtering: all dynamics claims are at the
  fluorescence-observation level; calcium deconvolution is out of scope.

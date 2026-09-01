# IOPR Publication Release v1.3

Publication-grade analysis code for the revised manuscript:

**Computational Formalisation of Surgical Spatial Knowledge Using Inter-Organ Geometry**

## Scientific scope

This repository evaluates a patient-specific **inter-organ proximity representation (IOPR)**.

It does **not** claim that the geometric output is biological fascia or a true surgical dissection plane.

Two operators are intentionally separated:

- `physical_iopr`: spacing-aware Euclidean distance in physical millimetres.
- `legacy_iopr`: repeated six-connected voxel dilation retained only for transparent audit/sensitivity analysis.

## What this release supports

Given organ masks and expert annotations listed in `manifest.csv`, the pipeline can generate:

- HR and SR algorithm-to-expert metrics;
- expert-expert pairwise metrics;
- DSC, NSD, ASSD, HD95 and occupied-slice overlap;
- 20-mm translation controls;
- wrong-plane controls;
- deterministic compatible-grid wrong-case controls;
- physical-mm radius sweep;
- legacy voxel-radius sweep;
- construct-constrained Dice ceiling;
- case-level source data;
- explicit inclusion/exclusion and geometry-QC records;
- run metadata with manifest hash and configuration;
- publication-ready summary tables and figures.

## Key implementation safeguards

- **HD95 definition**: max of the two directed 95th percentiles.
- **ASSD definition**: pooled, surface-point-weighted symmetric mean.
- **No silent resampling** in the primary analysis.
- **Optional geometry reconciliation tool** is provided and must be run explicitly.
- **Nearest-neighbour physical-space resampling** only for labels/masks.
- **Streaming/groupwise evaluation** avoids caching the full study in memory.
- **No manuscript result is hard-coded**.
- **Negative-control legacy radius is explicit in configuration**.
- **No target-conditioned synthetic/degradation routine is present**.

## Install

Python 3.11 recommended.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

## Manifest

Copy `examples/manifest_template.csv` to `manifest.csv`.

Required columns:

- `case_id`
- `plane` (`HR` or `SR`)
- `annotator`
- `organ_a`
- `organ_b`
- `expert_mask`

Optional:

- `expert_segment_name`
- `reference_image`

Paths may be absolute or relative to the manifest location.

## Inspect Slicer segment names

For `.seg.nrrd` annotations, inspect exact segment metadata first:

```bash
python scripts/list_segments.py path/to/annotation.seg.nrrd
```

The pipeline uses pynrrd for Slicer `SegmentN_*` metadata and SimpleITK for image geometry. Segment names are matched exactly, including literal escape sequences such as `\\n` when present in the Slicer header.

## Step 1 — audit geometry

```bash
python scripts/audit_geometry.py --manifest manifest.csv --out geometry_audit.csv
```

## Step 2 — if needed, reconcile expert masks explicitly

```bash
python scripts/reconcile_geometry.py \
  --manifest manifest.csv \
  --out-manifest manifest_reconciled.csv \
  --out-dir reconciled_masks \
  --audit reconciled_geometry_audit.csv
```

This performs nearest-neighbour resampling into the organ-A reference grid using image origin, spacing and direction. It records pre/post geometry and voxel counts. No mask is silently altered.

## Step 3 — run the complete analysis

```bash
python scripts/run_all.py \
  --manifest manifest_reconciled.csv \
  --config config/analysis.json \
  --out results
```

`negative_control_legacy_radius_voxels` selects the single legacy radius used
for translation, wrong-plane and wrong-case controls. The default is 3 voxels.

## Step 4 — generate figures

```bash
python scripts/make_figures.py --results results
```

## Tests

```bash
pytest -q
```

A static audit is also included:

```bash
python scripts/verify_no_hardcoded_results.py
```

## Data policy

Study imaging and expert annotations are intentionally not bundled. AMOS data remain governed by the original dataset terms. Expert annotations and case-level source data should be deposited separately where permitted.

## License

MIT.


## Public release status

This software package is ready to publish as a GitHub release. After the GitHub release is public, archive the same release on Zenodo and add the assigned DOI to `CITATION.cff` and the manuscript Code Availability statement.

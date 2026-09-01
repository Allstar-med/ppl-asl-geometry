# Public release checklist

This repository is intended for direct public release.

## Code integrity
- No target-conditioned mask degradation or result synthesis.
- No manuscript result values are hard-coded.
- Physical-space and legacy operators are separated.
- The negative-control legacy radius is read from configuration.
- Geometry mismatch is never silently ignored.
- Label resampling is explicit, nearest-neighbour, and audit logged.
- Case-level QC failures are preserved with reasons.
- Study-scale analysis uses groupwise/streaming loading.

## Metric definitions
- DSC: standard volumetric Dice.
- NSD: symmetric surface tolerance score in physical millimetres.
- ASSD: pooled surface-point-weighted mean of both directed distances.
- HD95: maximum of the two directed 95th percentiles.
- Occupied-slice overlap: Dice over occupied slice-level indicators.

## Reproducibility
- Configuration is stored in `config/analysis.json`.
- Run metadata include manifest and configuration SHA256.
- Source-data CSVs are generated before summaries.
- `SHA256SUMS.txt` hashes every release file.
- Transient Python/pytest cache files are excluded from the release.

## Data boundary
AMOS imaging and expert annotations are not redistributed in this code repository.
The repository is complete software, not a substitute for controlled study data.

## Repository archiving
After publishing on GitHub, create an immutable Zenodo release and add the Zenodo DOI to the manuscript Code Availability statement and repository metadata.

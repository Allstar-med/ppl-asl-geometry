# Expert annotation sets

This directory contains three independently prepared annotation sets, identified as A, B and C. Personal names are not included in the public release.

Each annotation is a 3D Slicer `.seg.nrrd` file named with the corresponding de-identified AMOS case ID. CT images are not redistributed here.

- `A/`: 39 available cases; `amos_0083` was not supplied.
- `B/`: 39 available cases; `amos_0036` was supplied only as a detached `.nhdr` without its paired data file and is therefore not included.
- `C/`: 40 available cases.

For sets A and B, the intended segment names are `Hepatorenal Recess` and `Splenorenal Ligament`. Set C uses `HR_plane` and `SR_plane`, with the exact strings recorded in `manifest.csv`.

See `manifest.csv` for file-level availability and exact segment names. Known source-file exceptions are listed in `QC_NOTES.md`.

# Annotation QC notes

The public annotation package contains 40 de-identified AMOS case IDs.

- Set A: 39 files. `amos_0083` was not supplied.
- Set B: 39 files. The source for `amos_0036` was a detached `.nhdr` without its paired data file and was excluded. `amos_0297` contains two segments both named `Hepatorenal Recess`, with no segment named `Splenorenal Ligament`.
- Set C: 40 files. In `amos_0043`, the first segment is named `LR_plane` rather than `HR_plane`. Some set-C segment names contain a literal `\n`; the exact values are recorded in `manifest.csv`.

All 118 published `.seg.nrrd` files have readable NRRD headers. No names, email addresses, hospital identifiers or local user paths were found in those headers.

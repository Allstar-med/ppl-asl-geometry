# Data dictionary

`manifest.csv` contains one row per case × anatomical plane × annotator.

Required:
- `case_id`
- `plane`
- `annotator`
- `organ_a`
- `organ_b`
- `expert_mask`

Optional:
- `expert_segment_name`
- `reference_image`

Case-level outputs include:
- inclusion status;
- exclusion reason;
- image shape and spacing;
- method and radius;
- DSC;
- NSD at configured tolerances;
- ASSD;
- HD95;
- occupied-slice overlap;
- construct-constrained ceiling;
- negative-control type and pairing when applicable.


## Slicer segment names

For `.seg.nrrd`, `expert_segment_name` must match the stored Slicer segment name **exactly**.
Whitespace and newline characters are not silently removed.

Inspect a file before building the manifest:

```bash
python scripts/list_segments.py path/to/annotation.seg.nrrd
```

Examples that are intentionally treated as different names:

- `Hepatorenal Recess`
- `HR_plane`
- `HR_plane\n`

This exact-match rule prevents accidental selection of the wrong label.

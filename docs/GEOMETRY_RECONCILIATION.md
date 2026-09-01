# Geometry reconciliation

Primary analysis never silently resamples masks.

When expert annotations and organ masks do not share the same grid, use the explicit reconciliation script.

The script:
1. loads organ A as the reference grid;
2. loads the expert segmentation in physical space;
3. performs nearest-neighbour resampling into the organ-A reference grid;
4. writes the reconciled mask;
5. records shape, spacing, origin, direction, voxel count before/after, and output path.

Nearest-neighbour interpolation is mandatory for labels/masks.

A large voxel-count change or implausible physical bounding box should be reviewed before inclusion.

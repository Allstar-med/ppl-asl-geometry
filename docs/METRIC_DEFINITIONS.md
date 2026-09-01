# Metric definitions

## Dice similarity coefficient

DSC = 2|A∩B| / (|A|+|B|)

## NSD

The fraction of surface points from both masks whose nearest-neighbour distance to the opposite surface is within the configured physical tolerance.

## ASSD

The pooled surface-point-weighted mean of all directed nearest-neighbour surface distances:

ASSD = (sum(d(A→B)) + sum(d(B→A))) / (N_A + N_B)

This is not the unweighted average of the two directed means when the surface point counts differ.

## HD95

HD95 = max(P95(d(A→B)), P95(d(B→A)))

This is not the 95th percentile of a concatenated distance array.

## Occupied-slice overlap

Dice overlap of the binary vectors indicating whether each slice along the configured axis contains any foreground.

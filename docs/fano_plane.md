# Fano Plane Object

FanoSeq now treats the Fano plane as an explicit computational object, not only
as an implicit multiplication convention.

The fixed oriented line convention is:

```text
(1,2,3), (1,4,5), (1,7,6), (2,4,6), (2,5,7), (3,4,7), (3,6,5)
```

Each line contains three imaginary axes. The cyclic orientation defines the
positive imaginary products. For example, line `(1,2,3)` records:

```text
e1*e2=+e3; e2*e3=+e1; e3*e1=+e2
```

Reversing an ordered pair flips the sign.

## Commands

Export the Fano plane as point, line, incidence, and pair tables:

```bash
fanoseq fano-plane --axis-scheme dna-window-v1 --output-dir results/fano_plane
```

Draw a PNG diagram using axis-scheme labels:

```bash
fanoseq plot-fano-plane --axis-scheme dna-window-v1 --output results/fano_plane.png
```

Build downstream Fano-line feature summaries from an output directory:

```bash
fanoseq fano-features --input-dir results/example_dna_windows --output-dir results/example_dna_fano_features
```

Run a bootstrap stability audit of dominant Fano-line profiles:

```bash
fanoseq fano-stability --input-dir results/example_dna_windows --output-dir results/example_dna_fano_stability --n-bootstrap 100
```

## Exported Tables

`fano_plane_points` contains the seven imaginary-axis points, their labels, and
the three incident Fano lines for each point.

`fano_plane_lines` contains the seven oriented lines, line labels, axis labels,
and positive orientation rules.

`fano_plane_incidence` contains one row per point-line incidence. There are 21
rows because each of seven lines contains three points, and each point belongs
to three lines.

`fano_plane_pairs` contains ordered imaginary-axis products and unordered pair
membership. Ordered rows encode signs such as `e1*e2=+e3` and `e2*e1=-e3`.

## Feature Family

`fano_line_features` summarizes `fano_interactions` by sequence, mode,
sequence type, axis scheme, and frame when present.

The feature family includes:

- sum, mean, max, standard deviation, and share for each Fano line
- total Fano-line contribution norm
- normalized Fano-line entropy and profile L2 concentration
- dominant Fano line and dominant-line share
- incident-share load for each axis e1...e7

These are mathematical descriptors of how adjacent sequence units distribute
their imaginary-imaginary product contributions across the Fano plane. They are
not biological mechanisms unless validated against baselines.

## Stability

`fano-stability` bootstraps Fano attribution rows within each group and reports:

- dominant-line stability
- mean cosine similarity between bootstrap profiles and the full profile
- minimum cosine similarity across bootstrap profiles

This is a first stability check. A strong Fano-line signal should not depend on
a tiny number of rows or a single unstable event. More serious downstream
studies should also add sequence shuffles, axis-label shuffles, and task-level
benchmark ablations.

## Plot

`plot-fano-plane` draws the seven points and seven lines of the project
convention. The diagram uses a standard Fano-plane layout, while the exported
tables remain the source of truth for orientation and product signs.

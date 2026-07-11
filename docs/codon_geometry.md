# Codon Geometry Audit

`codon-product-v1` maps a codon `XYZ` to an ordered product:

```text
O(XYZ) = (B(X, position 1) * B(Y, position 2)) * B(Z, position 3)
```

The right-associated product is also audited:

```text
B1 * (B2 * B3)
```

The full associator vector is retained:

```text
((B1 * B2) * B3) - (B1 * (B2 * B3))
```

The audit writes these values in `codon_octonion_catalog` as `e0...e7`,
`right_e0...right_e7`, and `associator_e0...associator_e7`.

## Tables

The codon audit writes:

- `codon_octonion_catalog`
- `codon_collision_report`
- `codon_distance_matrix`
- `codon_synonymy_statistics`
- `codon_substitution_effects`
- `codon_geometry_rank_spectrum`

## Current Default Finding

Under the current implementation, standard genetic code, and default
non-normalized codon products, `codon-product-v1` is injective over the 64
standard DNA codons. The audit reports no exact or near collisions at the
default tolerance.

The component matrix has rank 8. This means the 64 codons occupy the full
available component space, but it does not mean that the geometry is
biologically meaningful by itself.

## Geometry Questions Measured By The Audit

The audit quantifies:

- exact and near collisions
- Euclidean distances
- octonion-commutator distances
- norm distribution
- component rank
- effective dimensionality
- synonymous-codon distances
- nonsynonymous-codon distances
- within-amino-acid versus between-amino-acid separation
- stop-codon geometry
- start-codon geometry
- position 1, 2, and 3 substitution effects
- transition versus transversion effects
- wobble-position sensitivity
- genetic-code table dependence through the selected `--codon-table`
- left-associated versus right-associated product differences

Synonymous clustering is not assumed. It must be read from
`codon_synonymy_statistics` and, for individual substitutions, from
`codon_substitution_effects`.

## Interpretation Boundary

If synonymous codons cluster under a chosen table and normalization setting,
that is a measured property of the current mapping. If they do not, that is also
a valid audit result. The codon octonion geometry should be compared with codon
one-hot features, RSCU, GC3, amino-acid identity, codon degeneracy, molecular
volume, polarity, charge, and learned embeddings before any biological utility
claim is made.


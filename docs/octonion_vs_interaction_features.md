# Octonions Versus Interaction Features

FanoSeq uses a fixed Fano-plane multiplication table for algebraic octonion
features. The audit also reports ordinary real-valued controls so that future
benchmarks can test whether the multiplication table adds information beyond
standard interaction expansions.

## Feature Families

For DNA windows, `feature_redundancy` compares:

- original component vectors
- adjacent component differences
- ordinary pairwise products
- antisymmetric exterior-product terms
- full octonion products
- commutators
- associators
- Fano-line norms
- real-valued antisymmetric controls
- ordinary polynomial interaction controls

The audit reports:

- numerical rank
- singular-value spectrum
- feature correlations
- mutual information with original descriptors when scikit-learn is available
- linear predictability
- small-tree predictability
- collision rates
- condition numbers
- PCA variance concentration

## Commutator Identity

For pure imaginary parts `u` and `v`, octonion multiplication can be written as:

```text
uv = -dot(u, v)e0 + cross_Fano(u, v)
```

Therefore:

```text
[u, v] = uv - vu = 2 * cross_Fano(u, v)
```

The commutator is thus a structured antisymmetric interaction expansion. It is
not automatically evidence of a biological mechanism.

Fano-line attribution groups terms of the form:

```text
x_i*y_j - x_j*y_i
```

by the seven Fano lines. This can be useful as a reproducible decomposition, but
the biological meaning of a dominant line depends on the chosen axis assignment.

## Axis Controls

`axis_permutation_stability` and `fano_automorphism_controls` distinguish:

- biological label changes
- coordinate permutations
- sign flips
- permutations preserving selected biological triads
- discrete Fano-plane automorphisms
- arbitrary coordinate permutations that are not automorphisms
- random antisymmetric multiplication tensors
- ordinary polynomial interaction features

These are different operations. They should not be conflated.

If dominant Fano lines, line shares, nearest neighbors, or transition rankings
change strongly under arbitrary axis reassignment, the interpretation is
axis-dependent. That is an audit finding, not a failure of the audit.


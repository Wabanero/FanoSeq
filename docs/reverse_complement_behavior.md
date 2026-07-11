# Reverse-Complement Behavior

This page documents the current reverse-complement behavior of
`dna-window-v1`.

## Implemented DNA Window Components

For a valid-only DNA window with counts `A`, `C`, `G`, `T` and length `L`,
FanoSeq uses:

| component | formula |
| --- | --- |
| `e0` | valid-base fraction |
| `e1` | `(A + G - C - T) / L` |
| `e2` | `(G + C - A - T) / L` |
| `e3` | `(A + C - G - T) / L` |
| `e4` | `(G - C) / (G + C + epsilon)` |
| `e5` | `(A - T) / (A + T + epsilon)` |
| `e6` | normalized k-mer entropy |
| `e7` | `2 * RC_similarity(window) - 1` |

Under reverse complement, counts transform as:

```text
A' = T
C' = G
G' = C
T' = A
```

Therefore the component transform is:

```python
T_RC = diag(1, -1, 1, -1, -1, -1, 1, 1)
```

FanoSeq exposes this as:

```python
from fanoseq import transform_octonion_rc

rc_octonion = transform_octonion_rc(octonion, scheme_id="dna-window-v1")
```

## What The Audit Tests

`reverse_complement_derivation` records the component-by-component formula
derivation.

`reverse_complement_transform_matrix` records the explicit matrix.

`reverse_complement_audit` compares:

```text
encode_dna_window(reverse_complement(window))
```

against:

```text
transform_octonion_rc(encode_dna_window(window))
```

The audit runs exhaustive checks over small unambiguous sequences and also tests
the sampled input windows.

## Exceptions And Boundary Cases

The transform applies to windows that both orientations can encode.

Important boundary cases:

- Empty windows are not encoded.
- Fully ambiguous windows are not encoded.
- Windows exceeding `max_ambiguous_fraction` are skipped.
- Ambiguous symbols are removed before descriptor calculation when the ambiguity
  threshold allows the window.
- `epsilon` denominators preserve the expected skew sign because the affected
  denominators are unchanged under complement.
- k-mer entropy is invariant because reverse complement is a bijection on valid
  k-mers.
- reverse-complement similarity is invariant by symmetry.
- Whole-sequence window tables can have finite-window phase effects. A window at
  start `s` maps to reverse-complement start `N - (s + window_size)`. If that
  mirrored start is not sampled by the chosen step, the table-level trajectory is
  not a simple row reversal.

For adjacent products, commutators, associators, and Fano-line contributions,
the audit reports measured behavior. It does not force a simple equivariance
claim when the chosen multiplication, window sampling, or association convention
does not support one.


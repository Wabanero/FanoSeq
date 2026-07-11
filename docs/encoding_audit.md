# Encoding Audit Framework

FanoSeq now includes an encoding-audit layer for checking what the current
representations preserve, discard, duplicate, or change. The audit is intended
to test the implemented encoders, not to add speculative biological claims.

Run it with:

```bash
fanoseq audit-encoding \
  --input examples/example_dna.fasta \
  --seq-type dna \
  --axis-scheme dna-window-v1 \
  --checks reverse-complement,permutation,collision,mutation,redundancy,codon \
  --output-dir results/encoding_audit
```

The command writes reproducible tables with:

- `software_version`
- `schema_version`
- `scheme_id`
- `fano_convention_id`
- `genetic_code_table`
- `normalization_settings`
- `random_seed`
- `input_hash`
- `tolerance`

## Contract Tables

`encoding_contracts` records a formal contract for every registered axis scheme
and every concrete encoding in `fanoseq.encodings`:

- input domain
- output dimension
- scalar-axis meaning
- imaginary-axis meanings
- normalization
- missing-data behavior
- ambiguity handling
- orientation convention
- association convention
- known invariances
- known non-invariances
- information lost
- recommended baselines

Eight-channel tensor encodings are explicitly labelled as tensors. They are not
described as algebraic octonions unless FanoSeq actually uses octonion
multiplication for that representation.

## Main Output Tables

The audit can write:

- `encoding_audit_summary`
- `encoding_contracts`
- `reverse_complement_transform_matrix`
- `reverse_complement_derivation`
- `reverse_complement_audit`
- `codon_octonion_catalog`
- `codon_collision_report`
- `codon_distance_matrix`
- `codon_synonymy_statistics`
- `codon_substitution_effects`
- `codon_geometry_rank_spectrum`
- `mutation_sensitivity`
- `feature_redundancy`
- `feature_rank_spectrum`
- `axis_permutation_stability`
- `fano_automorphism_controls`

The audit also writes lightweight PNG diagnostics when `--plots` is enabled:

- codon distance heatmap
- codon PCA
- synonymous-family geometry
- mutation sensitivity
- singular-value spectrum
- reverse-complement residuals
- axis-permutation stability
- Fano-line profile stability

## Current Findings

For `dna-window-v1`, reverse complementation has a simple component transform
for encodable windows:

```text
diag(1, -1, 1, -1, -1, -1, 1, 1)
```

This is derived from the implemented count formulas, not assumed from the audit
request. See `docs/reverse_complement_behavior.md`.

For `codon-product-v1`, the current ordered products are injective over the 64
standard DNA codons under the standard genetic code and default normalization
setting. This does not imply that codon geometry is biologically optimal. The
audit measures synonymous and nonsynonymous distances, stop/start geometry,
substitution effects, and left- versus right-associated product differences. See
`docs/codon_geometry.md`.

For commutators and Fano-line attributions, much of the signal is a structured
antisymmetric interaction expansion. The audit therefore writes a real-valued
antisymmetric control beside octonion-derived features. See
`docs/octonion_vs_interaction_features.md`.

Axis labels are modelling choices. The audit distinguishes biological label
changes, coordinate permutations, multiplication-table changes, and discrete
Fano-plane automorphisms. Interpretations that change drastically under arbitrary
axis reassignment should be treated as axis-dependent.


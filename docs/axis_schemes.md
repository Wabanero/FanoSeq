# Axis Schemes

FanoSeq separates octonion algebra from the biological meaning assigned to each
axis. The algebraic basis is fixed by the oriented Fano-plane convention, but
the interpretation of e1...e7 depends on an explicit, versioned axis scheme.
The concrete formulas and missing-data policies for each axis are documented in
`docs/axis_definitions.md` and exported by the CLI.

This avoids a common mistake: an eight-dimensional feature vector is not
automatically a biologically meaningful octonion. It becomes interpretable only
when the axis mapping, scalar role, and Fano-line semantics are declared.

## Commands

List registered schemes:

```bash
fanoseq list-axis-schemes
```

Describe one scheme:

```bash
fanoseq describe-axis-scheme dna-window-v1
```

List the concrete axis formulas:

```bash
fanoseq list-axis-definitions --scheme-id dna-window-v1
```

Validate all registered definitions:

```bash
fanoseq validate-axis-schemes
```

Write scheme metadata, axis definitions, and Fano-line definitions:

```bash
fanoseq describe-axis-scheme dna-window-v1 --output-dir results/axis_schemes/dna-window-v1
```

## Initial Registry

| scheme_id | status | purpose |
| --- | --- | --- |
| dna-window-v1 | stable | Current/default DNA window descriptor scheme |
| dna-coding-v1 | experimental | Defined but not runnable coding/CDS windows with GC3, frame periodicity, codon bias |
| dna-regulatory-v1 | experimental | Defined but not runnable regulatory windows with CpG, palindrome, motif-density, entropy |
| dna-shape-v1 | planned | Future DNA-shape and multi-track scheme |
| protein-sequence-v1 | stable | Current/default protein sequence descriptor scheme |
| codon-product-v1 | stable | Current/default ordered codon product scheme |

## dna-window-v1

This is the current reproducible DNA window scheme.

| axis | meaning |
| --- | --- |
| e0 | valid fraction / scalar reliability |
| e1 | purine/pyrimidine balance |
| e2 | GC/AT balance |
| e3 | amino/keto balance |
| e4 | GC skew |
| e5 | AT skew |
| e6 | k-mer entropy |
| e7 | reverse-complement symmetry |

The most interpretable Fano line is `(1,2,3)`, the base chemistry triad. Other
lines couple chemistry, skew, complexity, and reverse-complement symmetry. For
example, `(1,7,6)` is interpreted as the RY/symmetry/complexity triad.

## dna-coding-v1

This scheme has concrete axis definitions but is not yet wired into `fanoseq run`.

| axis | intended meaning |
| --- | --- |
| e0 | valid coding fraction |
| e1 | purine/pyrimidine balance |
| e2 | GC/AT balance |
| e3 | amino/keto balance |
| e4 | GC3 excess |
| e5 | period-3 frame signal |
| e6 | codon entropy or RSCU dispersion |
| e7 | wobble stability or ORF integrity |

This scheme is intended for CDS-oriented benchmarks. It must be compared against
standard codon usage, GC3, RSCU, CAI, amino-acid composition, and k-mer
baselines before any biological claim is made.

## dna-regulatory-v1

This scheme has concrete axis definitions but is not yet wired into `fanoseq run`.

| axis | intended meaning |
| --- | --- |
| e0 | valid fraction |
| e1 | purine/pyrimidine balance |
| e2 | GC/AT balance |
| e3 | amino/keto balance |
| e4 | CpG observed/expected |
| e5 | palindrome or inverted-repeat density |
| e6 | k-mer entropy |
| e7 | motif-density or AT-rich regulatory proxy |

This scheme should be benchmarked against ordinary motif, k-mer, CpG, GC, and
regulatory-sequence baselines.

## dna-shape-v1

This is a planned scheme for external tracks. It is not a FASTA-only encoder.
Potential axes include minor groove width, propeller twist, helix twist, roll,
accessibility or methylation, conservation, and track confidence.

## protein-sequence-v1

This is the current reproducible protein window scheme.

| axis | meaning |
| --- | --- |
| e0 | valid amino-acid fraction |
| e1 | hydrophobicity |
| e2 | net charge |
| e3 | polarity |
| e4 | aromaticity |
| e5 | residue volume |
| e6 | disorder/flexibility proxy |
| e7 | repeat/low-complexity score |

It is a sequence descriptor scheme, not a structure predictor.

## Policy

New schemes must be versioned. Do not mutate the meaning of an existing scheme
id once outputs have been produced. Add a new scheme id instead, such as
`dna-window-v2`.

Every scheme should define:

- scalar axis role
- imaginary axis labels and value ranges
- formula, input fields, normalization, and missing-data policy
- Fano-line labels
- recommended use
- limitations
- benchmark baselines

Runnable `fanoseq run --window-axis-scheme ...` and
`fanoseq run --codon-axis-scheme ...` support uses this registry as the source
of truth. Experimental schemes can be exported and validated, but `fanoseq run`
rejects them until their encoders are implemented.

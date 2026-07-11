# Axis Definitions

Axis schemes name the biological or computational meaning of e0...e7. Axis
definitions make that mapping operational by recording the formula, inputs,
normalization, missing-data policy, implementation status, and baseline
comparators for every component.

The goal is reproducibility. A FanoSeq output should not only say that a value
is e6. It should also state that e6 came from a specific scheme such as
dna-window-v1, where e6 is normalized k-mer entropy under the recorded
parameters.

## Commands

List all scheme-level entries:

```bash
fanoseq list-axis-schemes
```

List concrete axis definitions:

```bash
fanoseq list-axis-definitions --scheme-id dna-window-v1
```

Describe one scheme, including formulas and validation checks:

```bash
fanoseq describe-axis-scheme dna-window-v1
```

Validate the registry:

```bash
fanoseq validate-axis-schemes
```

Select an implemented scheme during a run:

```bash
fanoseq run --input examples/example_dna.fasta --seq-type dna --mode window --window-size 10 --output-dir results/example_dna --window-axis-scheme dna-window-v1
```

For single-mode runs, `--axis-scheme` is a shortcut:

```bash
fanoseq run --input examples/example_dna.fasta --seq-type dna --mode codon --output-dir results/example_codons --axis-scheme codon-product-v1
```

For `--mode both`, use `--window-axis-scheme` and `--codon-axis-scheme` so the
two encodings remain explicit.

## Current Runnable Definitions

### dna-window-v1

| component | definition |
| --- | --- |
| e0 | valid A/C/G/T count divided by cleaned window length |
| e1 | (A + G - C - T) / L |
| e2 | (G + C - A - T) / L |
| e3 | (A + C - G - T) / L |
| e4 | (G - C) / (G + C + epsilon) |
| e5 | (A - T) / (A + T + epsilon) |
| e6 | H_k(window) / log2(4^k) |
| e7 | 2 * RC_similarity(window) - 1 |

This scheme is runnable and is the default DNA window scheme.

### protein-sequence-v1

| component | definition |
| --- | --- |
| e0 | valid standard-residue count divided by cleaned window length |
| e1 | mean Kyte-Doolittle hydropathy divided by 4.5 |
| e2 | (K + R + 0.1H - D - E) / L |
| e3 | mean table-scaled polarity |
| e4 | (F + W + Y + H) / L |
| e5 | mean min/max scaled residue volume |
| e6 | disorder-promoting fraction minus order-promoting fraction |
| e7 | 1 - normalized amino-acid k-mer entropy |

This scheme is runnable and is the default protein window scheme.

### codon-product-v1

For codon XYZ, FanoSeq builds position-aware base octonions B1, B2, B3 and
uses the left-associated product (B1 B2) B3. Components e1...e3 begin from
base chemistry, e4...e6 from position gates, and e7 from the wobble marker.

This scheme is runnable and is the default DNA codon scheme.

## Defined But Not Runnable

`dna-coding-v1` records concrete definitions for GC3 excess, period-3 signal,
codon entropy, and ORF integrity, but it is not yet wired into `fanoseq run`.

`dna-regulatory-v1` records concrete definitions for CpG observed/expected,
palindrome density, k-mer entropy, and motif-density/AT-rich proxy, but it is
not yet wired into `fanoseq run`.

`dna-shape-v1` records the intended external-track layout for DNA-shape and
multi-track analyses. It is planned, not runnable, because it requires external
track readers and explicit scaling policies.

## Output Provenance

Runnable trajectory tables now include `axis_scheme_id`. Fano-line attribution
rows also include `axis_scheme_id` and `line_label`. Bundle manifests record the
window and codon axis schemes used in the run.

This matters downstream because the same component name can have different
meaning under different schemes. Comparing e4 across dna-window-v1 and
dna-regulatory-v1 without checking `axis_scheme_id` would mix GC skew with CpG
observed/expected.

## Validation Policy

Every registered scheme is checked for:

- scalar axis e0
- exactly one definition for each imaginary axis e1...e7
- Fano-line orientation matching the project octonion convention
- non-empty formula, normalization, value range, missing-data policy, and
  benchmark baseline fields
- stable schemes being runnable

Experimental and planned schemes may be defined without being runnable, but
they must not silently pass as production encoders.

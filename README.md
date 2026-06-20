<p align="center">
  <img src="assets/fanoseq-logo.png" alt="FanoSeq logo" width="420">
</p>

# FanoSeq

Sequence trajectories in Fano-structured octonion space

FanoSeq maps DNA or protein FASTA sequences into sliding-window octonion trajectories and maps DNA codons into ordered codon octonions. Each sequence unit becomes an octonion:

```text
O_i = x0e0 + x1e1 + ... + x7e7
```

Adjacent windows or codons are compared through octonion products and commutators. Consecutive triplets are compared through associators. The output is a set of mathematical descriptors for trajectory analysis, transition detection, codon-level algebraic profiling, sequence comparison, anomaly detection, and exploratory pattern discovery.

FanoSeq is a feature-engineering framework. It does not claim that DNA, proteins, or biological systems are intrinsically octonionic.

## What The Project Does

```text
FASTA
-> window mode or codon mode
-> sequence-derived descriptors
-> octonions in Fano-structured basis
-> products / commutators / associators
-> Fano-line attribution
-> transition, triplet and codon-level sequence descriptors
```

FanoSeq supports DNA FASTA input, protein FASTA input, window-level descriptors, codon-level descriptors for DNA, Fano-line attribution, and TSV outputs designed for downstream analysis.

## Mathematical Background

An octonion is:

```text
O = x0e0 + x1e1 + ... + x7e7
```

`e0` is the real scalar unit. `e1` through `e7` are imaginary units. The imaginary units multiply according to an oriented Fano plane. Octonions are non-commutative, so `xy != yx`, and non-associative, so `(xy)z != x(yz)`.

For adjacent sequence units:

```text
P_i = O_i O_{i+1}
[O_i, O_j] = O_i O_j - O_j O_i
T_i = ||[O_i, O_{i+1}]||
```

For triplets:

```text
[O_i, O_j, O_k] = (O_i O_j)O_k - O_i(O_j O_k)
A_i = ||[O_i, O_{i+1}, O_{i+2}]||
```

These are mathematical descriptors of local sequence transitions. They are not direct biological mechanisms.

## Fano Plane Convention

FanoSeq uses this oriented Fano-plane convention:

```text
(1, 2, 3)
(1, 4, 5)
(1, 6, 7)
(2, 4, 6)
(2, 5, 7)
(3, 4, 7)
(3, 5, 6)
```

For each oriented triple `(a,b,c)`, the cyclic products are positive:

```text
e_a e_b = e_c
e_b e_c = e_a
e_c e_a = e_b
```

Reversing the order changes the sign, and `e_i e_i = -e0` for imaginary units.

## DNA Window Encoding

DNA windows are cleaned to uppercase A/C/G/T plus possible ambiguous symbols. Strict mode skips windows containing ambiguity. The octonion components are:

| component | meaning |
| --- | --- |
| `e0` | scalar window mass / valid-base fraction |
| `e1` | purine/pyrimidine balance |
| `e2` | GC/AT balance |
| `e3` | amino/keto balance |
| `e4` | GC skew |
| `e5` | AT skew |
| `e6` | normalized k-mer entropy |
| `e7` | reverse-complement symmetry mapped to `[-1, +1]` |

Auxiliary columns include `mono_entropy`, `gc_content`, `valid_fraction`, and `ambiguous_fraction`.

FanoSeq intentionally keeps mononucleotide entropy outside the scalar component. The scalar component `e0` represents window mass or reliability. The seven imaginary components carry the descriptors that participate in Fano-structured octonion multiplication.

## Protein Window Encoding

Protein windows use the 20 standard amino acids. The protein scales are simplified sequence descriptors, not structure predictors.

| component | meaning |
| --- | --- |
| `e0` | scalar window mass / valid amino-acid fraction |
| `e1` | mean hydrophobicity using a Kyte-Doolittle-like scale |
| `e2` | net charge proxy |
| `e3` | polarity proxy |
| `e4` | aromatic fraction |
| `e5` | approximate residue volume |
| `e6` | disorder/flexibility proxy |
| `e7` | repeat / low-complexity score |

Auxiliary columns include `mono_entropy`, `valid_fraction`, and `ambiguous_fraction`. `gc_content` is reported as `NA` for proteins.

## Codon Mode

FanoSeq's default window mode maps sliding sequence windows to octonions. Codon mode instead maps each DNA codon to an ordered octonion product of three position-aware base octonions.

For codon `XYZ`:

```text
B1 = B(X, position 1)
B2 = B(Y, position 2)
B3 = B(Z, position 3)
O_XYZ = (B1 B2) B3
```

Base chemistry is encoded in `e1` through `e3`:

| base | RY | SW | MK |
| --- | ---: | ---: | ---: |
| A | +1 | -1 | +1 |
| C | -1 | +1 | +1 |
| G | +1 | +1 | -1 |
| T | -1 | -1 | -1 |

Position gates are encoded in `e4` through `e7`: position 1 uses `e4`, position 2 uses `e5`, position 3 uses `e6`, and `e7` marks the third or wobble position.

Codon mode is an exploratory codon-level descriptor layer inspired by the idea of treating codons as structured algebraic objects. It is not a replacement for standard codon-usage analysis and does not claim to reproduce or validate any existing hypercomplex genetic-code theory. It should be benchmarked against standard features such as GC3, codon frequency, RSCU, CAI, amino-acid composition, k-mers, and learned embeddings.

## Fano-Line Attribution

The octonion product is governed by the oriented Fano plane. Product components `p0...p7` show the product result, but they do not reveal which Fano-plane line generated the strongest imaginary-imaginary interaction.

Fano-line attribution decomposes each adjacent product into seven line-specific contributions. For a Fano line `(a,b,c)`:

```text
pair_ab_to_c = x_a*y_b - x_b*y_a
pair_bc_to_a = x_b*y_c - x_c*y_b
pair_ca_to_b = x_c*y_a - x_a*y_c
line_contribution_norm = sqrt(pair_bc_to_a^2 + pair_ca_to_b^2 + pair_ab_to_c^2)
```

This captures only imaginary-imaginary terms from the Fano line. Scalar-imaginary terms and same-axis terms are intentionally excluded.

The line labels depend on mode. DNA window mode labels include purine/pyrimidine balance, GC/AT balance, amino/keto balance, GC skew, AT skew, k-mer entropy, and reverse-complement symmetry. Protein mode labels include hydrophobicity, charge, polarity, aromaticity, volume, disorder/flexibility, and low-complexity. Codon mode labels include base RY, SW, MK, position gates, and wobble-position marker.

These line attributions are mathematical decompositions of the chosen octonion product. They are not direct biological causal mechanisms.

## Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate fanoseq
pip install -e .
pytest
```

The conda environment pins Python 3.11 for reproducible package resolution. The package itself supports Python 3.10 and newer. On Windows, `pyfastx` is installed through the `pip:` section of `environment.yml` because the conda package is not available for `win-64` on the standard channels used here.

Using mamba:

```bash
mamba env create -f environment.yml
conda activate fanoseq
pip install -e .
pytest
```

## Example Usage

DNA window mode:

```bash
fanoseq run --input examples/example_dna.fasta --seq-type dna --mode window --window-size 10 --step 1 --output-dir results/example_dna_windows
```

Protein window mode:

```bash
fanoseq run --input examples/example_protein.fasta --seq-type protein --mode window --window-size 15 --step 1 --output-dir results/example_protein_windows
```

DNA codon mode:

```bash
fanoseq run --input examples/example_dna.fasta --seq-type dna --mode codon --frame 0 --output-dir results/example_dna_codon
```

Full DNA mode:

```bash
fanoseq run --input examples/example_dna.fasta --seq-type dna --mode both --frame all --window-size 10 --step 1 --output-dir results/example_dna_full
```

Genome-scale or larger exploratory runs should prefer Parquet or bundle output:

```bash
fanoseq run --input examples/example_dna.fasta --seq-type dna --mode window --window-size 100 --step 10 --output-dir results/example_dna_parquet --output-format parquet
```

```bash
fanoseq run --input examples/example_dna.fasta --seq-type dna --mode both --frame all --window-size 100 --step 10 --output-dir results/example_dna.fanoseq --output-format bundle --summary-only
```

Sparse transition/event output can keep only strong local transitions:

```bash
fanoseq run --input examples/example_dna.fasta --seq-type dna --mode window --window-size 100 --step 10 --output-dir results/example_dna_events --output-format bundle --top-k-transitions 10000
```

## Output Files

FanoSeq supports three output formats:

```text
--output-format tsv        # small/debug runs; human-readable TSV files
--output-format parquet    # columnar Parquet files for serious analysis
--output-format bundle     # manifest.json plus partitioned Parquet datasets
--summary-only             # compact fingerprints/summaries, no row-heavy trajectory tables
--top-k-transitions 10000  # keep only the strongest transition products and Fano rows
--transition-threshold X   # keep only transition products with transition_score >= X
```

TSV remains the debug and example format. Parquet is better for large FASTA collections because it is compressed, columnar, and can be queried by pandas, Polars, DuckDB, Spark, and R tooling without loading every column.

Bundle mode writes a directory such as:

```text
sample.fanoseq/
  manifest.json
  window_sequence_summary.parquet/
  window_octonions.parquet/
  octonion_products.parquet/
  octonion_triplets.parquet/
  fano_interactions.parquet/
  codon_octonions.parquet/
  codon_transition_products.parquet/
  codon_usage_fano_features.parquet/
  codon_usage_sequence_summary.parquet/
```

The manifest stores schema version, input hash, run configuration, output table paths, row counts, and column names. Bundle tables are partitioned by `sequence_id` and, where appropriate, `frame`.

`window_sequence_summary.tsv` or `.parquet` contains compact per-sequence fingerprints for window-mode runs. This is the main output retained by `--summary-only` for window analysis.

`window_octonions.tsv` or `.parquet` contains one row per sequence window with auxiliary descriptors and `e0...e7` components.

```text
sequence_id  position  start  end  window      e0       e1
seq1         0         0      10   ACGTACGTAC  1.000000 0.000000
```

`octonion_products.tsv` contains adjacent-window product components `p0...p7`, product norm, commutator score, and transition score.

```text
sequence_id  position  window      next_window  product_norm  transition_score
seq1         0         ACGTACGTAC  CGTACGTACA   1.842311      0.912444
```

`octonion_triplets.tsv` contains consecutive triplet associator components `a0...a7` and `associator_score`.

`codon_octonions.tsv` contains codon coordinates, amino-acid annotation, start/stop flags, position-wise base properties, `codon_associator_score`, and `e0...e7`.

```text
sequence_id  frame  codon_index  codon  amino_acid  is_start  e0
seq1         0      0            ATG    M           True      2.000000
```

`codon_transition_products.tsv` contains adjacent-codon product components, product norm, commutator score, and transition score.

`codon_usage_fano_features.tsv` contains codon counts, frequencies, synonymous family size, RSCU, and codon-octonion summary features.

```text
sequence_id  frame  codon  amino_acid  count  frequency  rscu
seq1         0      ATG    M           2      0.200000   1.000000
```

`codon_usage_sequence_summary.tsv` contains per-sequence/per-frame summary features such as codon entropy, GC1/GC2/GC3 means, stop density, transition scores, and codon associator scores.

`fano_interactions.tsv` contains seven Fano-line attribution rows per adjacent product for window and codon modes.

```text
sequence_id  mode    position  fano_line  axis_a_label                 line_contribution_norm
seq1         window  0         (1,2,3)    purine/pyrimidine balance    0.250000
```

## Interpretation Guide

High `transition_score` means adjacent windows or codons have a strong directional, non-commutative change in octonion space.

High `associator_score` means three consecutive windows form a strong non-associative local pattern.

High `codon_associator_score` means the ordered base-position product of a codon differs strongly depending on grouping.

High `line_contribution_norm` means a specific Fano-plane line dominates the imaginary-imaginary part of a product.

These scores can suggest candidate transition zones, low-complexity boundaries, repeat transitions, compositional shifts, codon-usage shifts, or unusual local sequence organization. They are exploratory descriptors and must be benchmarked against standard sequence descriptors.

## Future Perspectives

### Generic Sequence Comparison

Whole-sequence fingerprints could be built from means, variances, maxima, and quantiles of window octonion components, product components, transition scores, associator scores, Fano-line contribution norms, and codon-mode descriptors. These fingerprints could support clustering, anomaly detection, or nearest-neighbour search across FASTA collections.

### Comparative Genomics

FanoSeq could be extended to compare homologous loci, gene families, synthetic constructs, or generic FASTA collections by clustering their octonion trajectory fingerprints. This should remain a complement to established sequence comparison methods.

### Cancer Evolution And Longitudinal Tumour Genomics

In settings where tumour-derived sequence panels, longitudinal amplicons, inferred haplotypes, or ctDNA-derived mutation-context sequences are available, FanoSeq-like octonion trajectory descriptors could be explored as additional mathematical summaries of local sequence-context change over time. This would require benchmarking against standard cancer genomics features, phylogenetic and evolutionary models, mutation-context analysis, and clinically grounded covariates.

### Single-Cell And Multi-Modal Extensions

The current implementation handles FASTA-derived sequence descriptors. Future versions could combine sequence-derived descriptors with metadata, clonotype labels, mutation burden, chromatin accessibility, or expression-derived features. This is outside the current implementation.

### Synthetic Biology And Sequence Design

Octonion trajectory smoothness, transition peaks, Fano-line profiles, and associator profiles could be explored as constraints or diagnostics when designing synthetic sequences, barcodes, or regulatory-like motifs.

### Protein Sequence Exploration

Protein FASTA mode can be used to identify mathematical transitions between hydrophobic, charged, aromatic, flexible, and low-complexity regions. It is not a replacement for structural prediction tools.

### Genetic-Code And Codon-Algebra Exploration

Codon mode represents all 64 codons as ordered octonion products of position-aware base descriptors. This enables comparisons between codon octonion geometry, amino-acid grouping, synonymous codon families, RSCU, GC3, codon transition profiles, and Fano-line codon interaction profiles. It provides a structured descriptor layer that can be compared with existing genetic-code and codon-usage frameworks.

## Limitations

FanoSeq is a feature-engineering and exploratory sequence-analysis project. Octonion scores are not direct biological mechanisms. The choice of axes affects interpretation. The Fano plane convention affects product components. Codon mode is not a replacement for standard codon-usage analysis. Benchmarks against standard k-mer, one-hot, physicochemical, codon-usage, and embedding-based methods are required. Protein scales are simplified. DNA shape, methylation, conservation, gene models, and experimental metadata are not included. The framework should be treated as hypothesis-generating.

## Selected Python Libraries

Biopython is used for sequence objects, translation, codon tables, and genetic-code handling when available.

pyfastx is selected for scalable FASTA/FASTQ iteration and indexed access in future high-throughput paths.

scikit-bio is selected for sequence validation utilities, k-mer utilities, and future sequence-distance comparisons.

NumPy is used for octonion component arrays and numerical operations.

PyArrow is used through pandas for Parquet and partitioned bundle output.

Pandas is used for all TSV outputs and tabular aggregation.

Numba is used for accelerated octonion multiplication, commutator scans, associator scans, and batch processing.

SciPy and scikit-learn are selected for downstream distance matrices, clustering, PCA, anomaly detection, and sequence-level fingerprints.

Typer and Rich provide the command-line interface and readable console output.

Pytest, Ruff, and MyPy support tests, linting, and type checking.

## Development Roadmap

- Add plotting
- Add PCA/UMAP of window octonions
- Add sequence-level fingerprints
- Add pairwise sequence distances
- Add benchmarks against k-mer features
- Add benchmarks against codon-usage features
- Add learned octonion filters
- Add configurable axis definitions
- Add JSON/YAML config files
- Add notebook examples
- Add HTML reports
- Add product-component attribution by Fano-plane interaction
- Add codon-space visualizations
- Add Fano-line heatmaps
- Add sequence-level Fano interaction fingerprints

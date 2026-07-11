# Do Fano-structured Octonion Interactions Improve Biological Sequence Representations Beyond Conventional Compositional and Alignment-Free Features?

Status: protocol report. No public benchmark evidence has been produced yet.

## Formal Hypothesis

FanoSeq's fixed octonion multiplication table is useful for biological sequence
representation only if Fano-structured features improve held-out performance or
perturbation-distance behavior beyond conventional sequence features and beyond
non-Fano nonlinear controls built from the same eight descriptors.

The decisive comparison is:

```text
raw descriptors
-> ordinary polynomial interactions
-> antisymmetric interactions
-> randomized Fano-like antisymmetric structure
-> fixed FanoSeq octonion/Fano-line interactions
```

A negative result is valid evidence. If fixed Fano interactions do not improve
over controls under leakage-safe validation, the biological claim should remain
unsupported for that task.

## Dataset Provenance

The dataset registry is in `datasets/registry.yaml`.

| dataset | source | primary holdout | current status |
| --- | --- | --- | --- |
| `coding-noncoding-v1` | GENCODE Human Release 50 | chromosome | source release selected |
| `taxonomy-v1` | SILVA SSU Ref NR 99 138.2 | species or genus | source release selected |
| `mutation-effect-v1` | GENCODE protein-coding CDS FASTA | parent CDS | protocol defined |

Prepared FASTA and metadata files are not vendored. Curators must run each
study's `prepare.py`, record `fasta_sha256` and `metadata_sha256`, and keep the
source release and filtering decisions fixed before benchmark execution.

## Leakage Controls

Required controls:

- group-held-out splits: chromosome, species/genus, or parent CDS;
- duplicate sequence ID rejection in FASTA and metadata;
- sequence-similarity audit with a conservative threshold for real studies;
- length and GC matching where those confounders define the task;
- homology filtering before fold assignment;
- nested cross-validation so scaling, imputation, feature selection, and model
  tuning use only training folds;
- no feature redesign, threshold tuning, or source rebalancing after external
  validation is declared.

## Feature Families

Conventional baselines:

- sequence length, GC/AT composition, skew, entropy, and ambiguity summaries;
- k-mer frequency matrices;
- FCGR cell frequencies;
- codon usage, RSCU, GC1/GC2/GC3, codon entropy, and amino-acid composition.

FanoSeq families:

- raw window components `e0...e7`;
- adjacent octonion product components `p0...p7`;
- commutator and transition summaries;
- associator summaries;
- Fano-line attribution summaries;
- combined sequence fingerprints.

Control families:

- ordinary real polynomial expansion from the same eight descriptors;
- ordinary antisymmetric adjacent-window terms `x_i*y_j - x_j*y_i`;
- randomized Fano-like projection of antisymmetric terms;
- random projections of conventional features matched to FanoSeq dimensionality.

## Null Models

Sequence nulls:

- mononucleotide shuffle;
- dinucleotide-preserving shuffle;
- codon-order shuffle;
- synonymous-codon shuffle.

Representation nulls:

- scalar component removal;
- imaginary-axis permutation;
- sign flips;
- random orthogonal transforms;
- Fano-line relabeling;
- random antisymmetric interaction tensors.

Label nulls:

- repeated label permutations under the same fold design.

## Baseline Models

The first complete report should use simple, auditable models before moving to
larger learned representations:

- regularized logistic regression or linear classifier;
- random forest as a nonlinear tabular baseline;
- optional calibrated support-vector or gradient-boosting model if added with
  nested tuning and fixed model cards.

Deep sequence models, genomic foundation models, and protein language models are
external baselines for later studies, not substitutes for the classical controls.

## Nested Cross-Validation

Development benchmarks use repeated outer folds and inner folds:

```text
outer split: held-out chromosome/species/genus/parent
  inner split: tune model hyperparameters on training groups only
  final model: refit on outer training groups
  evaluation: score untouched outer test groups
```

Every output report must include fold assignments, group leakage checks,
benchmark configuration, input hashes, feature definitions, random seeds, and
software versions.

## Ablations

The benchmark engine reports incremental Fano ablations:

```text
base descriptors
+ octonion products
+ commutators
+ associators
+ Fano-line summaries
```

For the methods report, this must be read beside the stronger control ladder:

```text
base descriptors
+ ordinary polynomial interactions
+ ordinary antisymmetric interactions
+ randomized Fano-like structure
+ fixed FanoSeq interactions
```

Only the second ladder can distinguish a meaningful Fano-table contribution from
a generic nonlinear expansion.

## External Validation

After development on one source release, freeze:

- source filters;
- preprocessing thresholds;
- feature extraction settings;
- model families and tuning grids;
- random seeds and fold policy;
- primary metric and reporting tables.

Then evaluate on another species, assembly, source database, or sequence-length
distribution without redesign. The report table should include at least:

| feature family | development | external |
| --- | ---: | ---: |
| k-mer | TBD | TBD |
| FCGR | TBD | TBD |
| raw descriptors | TBD | TBD |
| antisymmetric control | TBD | TBD |
| randomized Fano structure | TBD | TBD |
| FanoSeq | TBD | TBD |

## Runtime And Memory

Each benchmark report should record:

- number of sequences and total bases;
- feature dimensionality by feature family;
- feature extraction wall time;
- model training wall time;
- peak memory if available from the runner;
- output table sizes.

## Negative Results

Negative results must remain in the report. Examples:

- FanoSeq matches but does not exceed k-mer or FCGR features;
- randomized Fano controls match fixed Fano features;
- apparent gains disappear under chromosome/species-held-out splits;
- gains disappear after length, GC, or homology matching;
- perturbation distances do not order synonymous, conservative, radical, stop,
  and frameshift edits as expected.

## Limitations

FanoSeq is a feature-engineering framework. It does not show that DNA, proteins,
or biological systems are intrinsically octonionic. The current DNA axes are
hand-designed descriptors, the protein axes remain simplified, and all biological
claims are pending public benchmark evidence.


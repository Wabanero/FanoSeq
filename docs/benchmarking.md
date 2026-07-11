# Benchmarking FanoSeq Features

FanoSeq now includes a manifest-driven benchmark command for testing whether
octonion/Fano-plane feature families add held-out predictive information beyond
ordinary biological sequence features.

```bash
fanoseq benchmark --config examples/benchmark.yaml --output-dir results/benchmark
```

The benchmark is a feature-engineering validation framework. It does not claim
that DNA, proteins, or biological systems are intrinsically octonionic. A
well-designed benchmark may falsify the usefulness of the Fano structure for a
given task.

## Configuration

Benchmark manifests may be YAML or JSON. The minimal structure is:

```yaml
dataset:
  table: metadata.tsv
  fasta: sequences.fasta
  id_column: sequence_id
  target_column: label
  group_column: species_or_subject
  task: classification
  seq_type: dna

features:
  - fanoseq_components
  - fanoseq_products
  - fanoseq_commutators
  - fanoseq_associators
  - fanoseq_fano_lines
  - fanoseq_fingerprints
  - nucleotide_composition
  - kmer
  - fcgr
  - codon_usage
  - real_polynomial_control
  - antisymmetric_control
  - randomized_fano_structure
  - random_projection_control

evaluation:
  outer_folds: 5
  inner_folds: 3
  repeats: 3
  random_seed: 42
  split_strategy: stratified_group
  primary_metric: balanced_accuracy
```

Paths are resolved relative to the configuration file. Supported task labels are
`classification`, `binary_classification`, `multiclass_classification`,
`regression`, and `clustering`.

## Feature Families

The benchmark compares:

- raw FanoSeq window components `e0...e7`;
- octonion product components `p0...p7`;
- commutator and transition summaries;
- associator summaries;
- Fano-line attribution summaries;
- combined FanoSeq fingerprints;
- nucleotide composition, k-mer, FCGR, and codon-usage features;
- real-valued polynomial controls built from the same eight descriptors;
- ordinary antisymmetric controls built as `x_i*y_j - x_j*y_i` from the same
  imaginary descriptor axes;
- randomized Fano-like antisymmetric controls matched to Fano-line
  dimensionality;
- random-projection controls matched to the FanoSeq dimensionality.

The real-valued interaction control is important: it contains pairwise products,
absolute differences, squares, and cubes without Cayley multiplication. This
distinguishes fixed octonion interactions from a generic nonlinear feature
expansion.

The decisive Fano comparison is:

```text
raw descriptors
-> ordinary polynomial interactions
-> ordinary antisymmetric interactions
-> randomized Fano-like structure
-> fixed FanoSeq octonion/Fano-line interactions
```

## Dataset Registry

The first curated study protocols live under `datasets/`:

- `coding-noncoding-v1`: GENCODE coding versus noncoding windows with
  chromosome-held-out splits.
- `taxonomy-v1`: SILVA SSU Ref NR 99 related-species classification with
  species/genus-held-out splits.
- `mutation-effect-v1`: controlled CDS perturbation sensitivity with parent-CDS
  held-out splits.

The manifests define source release, citation, license notes, preprocessing,
hash fields, split policy, confounder controls, and expected prepared outputs.
They are not benchmark evidence until source FASTA/metadata are prepared,
hashed, and run through `fanoseq benchmark`.

## Leakage Prevention

The benchmark evaluates sequence-level matrices, so overlapping windows from the
same parent sequence are never split independently. Scaling, imputation, optional
feature selection, and hyperparameter tuning are inside the cross-validation
pipeline. Nested CV uses inner folds only on the training portion of each outer
fold.

Use `group_column` for species, patient, chromosome, gene family, subject, or
other dependence structure. The output includes exact fold assignments and a
leakage-check table. A simple optional train/test sequence-similarity audit can
flag highly similar sequences across folds.

## Outputs

The command writes TSV or Parquet tables:

- `benchmark_runs`
- `benchmark_folds`
- `benchmark_metrics`
- `benchmark_predictions`
- `benchmark_feature_sets`
- `benchmark_ablation_results`
- `benchmark_null_results`
- `benchmark_permutation_tests`
- `benchmark_config_resolved`

It also writes `benchmark_manifest.json` with FanoSeq version, schema version,
input hashes, resolved config, random seeds, split assignment path, feature
definitions, software versions, timestamp, and output paths. A compact
`benchmark_report.md` summarizes task design, leakage checks, baseline ranking,
FanoSeq versus the strongest conventional baseline, null models, ablations, and
limitations.

## Null Models And Ablations

The benchmark package provides sequence null-model generators for
mononucleotide shuffling, dinucleotide-preserving shuffling, codon-order
shuffling, synonymous-codon replacement, and label permutation. It also provides
representation perturbations for imaginary-axis permutation, sign flips, random
orthogonal transforms, Fano-line relabeling, scalar removal, and random
antisymmetric interaction tensors.

Axis permutations are not automatically called octonion automorphisms. The
utility `is_oriented_fano_automorphism` checks whether a coordinate permutation
preserves the project oriented Fano products; arbitrary biological axis-label
permutations are reported separately.

The incremental ablation table reports:

```text
base descriptors
+ octonion products
+ commutators
+ associators
+ Fano-line summaries
```

## Interpretation

Predictive superiority, not visual complexity, is the relevant test. FanoSeq
features should be considered useful for a task only when they improve
held-out performance over ordinary k-mer, codon, FCGR, composition, and
real-valued interaction controls, and when the signal survives appropriate null
models.

If the report says FanoSeq does not outperform ordinary features, that is a
valid scientific result. It should guide axis-scheme revision, dataset choice,
or discontinuation of the feature family for that task.

## Suggested Public Follow-Up

A useful next benchmark is a coding-versus-noncoding DNA task built from a public
genome annotation source, with chromosomes or gene families as groups. Another
strong taxonomy benchmark is species or genus prediction from marker-gene
sequences, grouped by study or assembly source to reduce leakage.

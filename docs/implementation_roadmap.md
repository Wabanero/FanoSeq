# FanoSeq Implementation Roadmap

This roadmap turns the deep-research report into repo-level work packages. The
order matters: later claims depend on earlier validation and baselines.

## Phase 1: Algebra Validation

Status: partially implemented.

Implemented:

- `fanoseq validate-basis`
- basis multiplication table
- structure constants tensor A[i,j,k]
- left and right multiplication matrices
- checks for basis squares, oriented Fano products, reversed signs, scalar
  identity, associators, alternativity, norm multiplicativity, and operator
  consistency

Next improvements:

- Add a short math appendix with the exact sign convention.
- Add performance comparisons between direct multiplication, structure tensor,
  and left/right operator forms.
- Add optional JSON output for validation metadata.

## Phase 2: Mandatory Baselines

Status: first baseline command implemented.

Implemented:

- `fanoseq baselines`
- DNA sequence composition features
- DNA k-mer frequency tables
- FCGR-like integer coordinates for DNA k-mers
- wide k-mer feature matrices for classical ML
- codon usage, GC3, synonymous-family size, and RSCU
- protein sequence composition and residue/k-mer features

Next improvements:

- Add true FCGR image/tensor export.
- Add iCGR-style compact integer signatures.
- Add seq2vec-style embedding baselines or adapters to external embeddings.
- Add benchmark manifests that combine FanoSeq outputs and baseline outputs.

## Phase 3: Reproducible Benchmark Harness

Status: planned.

Target command:

```bash
fanoseq benchmark --manifest benchmarks/example.tsv --features fanoseq,kmer,gc,codon --task classify
```

Minimum outputs:

- `feature_table.parquet`
- `splits.tsv`
- `metrics.tsv`
- `predictions.tsv`
- `run_manifest.json`

First candidate tasks:

- coding versus noncoding toy benchmark
- small viral or bacterial taxonomy benchmark
- CDS codon-bias clustering benchmark

Required metrics:

- accuracy and macro-F1 for multiclass classification
- AUROC/AUPRC for binary tasks where labels support them
- ARI/NMI for clustering tasks
- runtime and memory summaries

## Phase 4: Codon And Matrix-Genetics Rigor

Status: partially implemented.

Implemented:

- canonical 64-codon order
- 8x8 codon matrix entries
- first-two-base root degeneracy
- Walsh-Hadamard spectrum over codon signals
- dyadic-shift summaries
- GF(8)-style labels

Next improvements:

- Implement actual GF(8) addition and multiplication tables.
- Support alternative codon matrix order conventions.
- Compare Fano/codon matrix features against codon usage, GC3, RSCU, and CAI.
- Add wobble-equivalence and synonymous-family perturbation tests.

## Phase 5: Fano Plane, Triads, And Hypergraphs

Status: explicit Fano-plane object and first triad-count command implemented.

Implemented:

- `fanoseq fano-plane`
- `fanoseq plot-fano-plane`
- `fanoseq fano-features`
- `fanoseq fano-stability`
- Fano-plane point, line, incidence, and pair-product tables
- Fano-line feature summaries for downstream fingerprints
- bootstrap stability summaries for dominant Fano-line profiles
- `fanoseq fano-triads`
- DNA and protein symbolic axis maps
- Fano-line triad counts

Next improvements:

- Add shuffled null models.
- Add enrichment scores and empirical p-values.
- Export NetworkX/hypergraph-ready edge tables.
- Add axis-label shuffle ablations for Fano-line feature families.
- Prototype regulatory or chromatin triads only with a curated dataset.

## Phase 6: Protein Geometry

Status: planned.

Planned:

- PDB/mmCIF readers.
- Backbone angle features: phi, psi, omega.
- Side-chain chi angle features where available.
- sin/cos angle pairs packed into 8D residue-level representations.
- Baselines using one-hot residues and physicochemical features.

## Phase 7: Hypercomplex ML

Status: intentionally deferred.

Planned only after benchmark baselines are stable:

- PyTorch dataset class for `[N, 8, L]` tensors.
- Real-valued Conv1D/MLP baselines.
- Fano-triad regularizers.
- Fixed octonion linear/convolution blocks using structure constants.
- Learned parameterized hypercomplex multiplication ablation.

Success criterion:

Hypercomplex layers must improve task metrics after controlling for parameter
count, seed variance, throughput, and equivalent real-valued baselines.

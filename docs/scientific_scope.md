# FanoSeq Scientific Scope

FanoSeq is a feature-engineering and representation toolkit. It uses the
Fano-plane convention for octonion multiplication to build sequence descriptors,
but it does not claim that DNA, proteins, regulatory systems, or evolution are
intrinsically octonionic.

The project is organized by evidence maturity.

| Evidence layer | External maturity | FanoSeq role | Repo policy |
| --- | --- | --- | --- |
| Octonion and Fano-plane mathematics | Very strong | Core algebra, kernels, invariants, validation | Must be validated by exact basis tests and documented sign conventions |
| Hypercomplex computation and ML | Moderate to strong | Optional representations, multiplication tensors, future parameter-efficient layers | Experimental modules must be compared with plain real-valued baselines |
| Direct octonion bioinformatics | Sparse and exploratory | Hypothesis-generating descriptors and candidate applications | No biological usefulness claims without public benchmark results |

## Representation Types

FanoSeq distinguishes strict octonion algebra from 8-channel tensor encodings.

| Representation type | Meaning | Safe use |
| --- | --- | --- |
| algebraic-octonion | e0 is the scalar axis and e1...e7 are imaginary axes under the Fano convention | Products, commutators, associators, Fano-line attribution |
| eight-channel-tensor | Eight real channels arranged for ML or feature extraction | Classical ML, tensor export, neural baselines; octonion products require explicit justification |
| matrix-genetics | Codon-table and genetic-code summaries | Codon degeneracy, dyadic shifts, Hadamard/Walsh summaries, coding-theory comparisons |

This distinction prevents a common error: an 8-dimensional vector is not
automatically a biologically meaningful octonion. It becomes an octonion only
when the component semantics and multiplication convention are explicit.

## Claim Policy

Allowed claims:

- FanoSeq implements a fixed oriented Fano-plane octonion convention.
- FanoSeq computes deterministic products, commutators, associators, and
  line-attribution descriptors from chosen encodings.
- FanoSeq can export features and tensors for downstream benchmark studies.

Claims that require evidence:

- FanoSeq features improve a biological classifier.
- Octonion descriptors capture nonredundant codon, genome, protein, or regulatory
  signal beyond standard baselines.
- Hypercomplex layers outperform real-valued layers for a biological task.

Claims to avoid:

- Biology is intrinsically octonionic.
- The genetic code is explained by octonions.
- Fano-line patterns are biological mechanisms without experimental support.

## Minimum Evidence Standard

Any application-facing result should include:

1. A public or reproducible dataset.
2. A non-leaky train/test split or unsupervised validation protocol.
3. Standard baselines such as k-mers, GC content, codon usage, RSCU, FCGR-like
   features, one-hot encodings, or physicochemical features.
4. FanoSeq ablations: descriptors without products, without associators, without
   Fano-line attribution, and with shuffled labels where appropriate.
5. Runtime and memory reporting for large-sequence workflows.

Context-aware claims additionally require explicit genome coordinates, assembly,
feature type, assay/sample metadata, synchronized-track provenance, and holdout
groups for chromosome, donor, tissue/cell type, or source study as appropriate.
See `biological_context.md`. Higher-order dependence claims must report the
estimator, discretization, null distribution, multiplicity correction, and
external validation; see `information_theory.md`.

Negative results are scientifically useful and should be kept. If octonion
features do not improve a task, that result still clarifies where the
representation is or is not helpful.

# Changelog

All notable changes to FanoSeq will be recorded here.

## 0.2.0 - 2026-07-18

### Added

- Fail-closed grouped and stratified-grouped splitting with an explicitly
  recorded unsafe override.
- Fold-local feature-quality filtering and machine-readable diagnostics.
- Executed sequence and representation null models with configurable repeats.
- A single resolved extraction plan and provenance-rich complete-analysis
  manifest with hashes, versions, dimensions, timing, memory, and warnings.
- Explicit stop-codon policy for RSCU calculations.

### Changed

- Removed the exact `transition_score` alias from the commutator benchmark
  feature family while retaining the raw pipeline column for compatibility.
- Renamed the lightweight leakage metric to positional identity so it cannot be
  mistaken for an alignment or homology estimate.
- Invalid requested genetic-code tables now fail instead of silently falling
  back to the standard code.

### Fixed

- Feature-matrix provenance prefixes when an intermediate feature set is empty.
- Repository URL and release metadata in `CITATION.cff`.

## 0.1.0 - 2026-07-12

### Added

- Source-available FanoSeq feature-engineering package.
- DNA, protein, and codon encoders.
- Octonion products, commutators, associators, and Fano-line attribution.
- Baseline feature extraction for composition, k-mers, FCGR-like coordinates,
  codon usage, and protein residue features.
- Leakage-controlled benchmark harness with nested cross-validation, ablations,
  null-model reporting, manifests, and markdown reports.
- Dataset registry protocols for coding/noncoding, taxonomy, and controlled
  mutation-effect studies.
- Antisymmetric and randomized-Fano benchmark controls.
- Scientific evidence-status table, methods report scaffold, and reference list.

### Changed

- License metadata is aligned to the repository's source-available license file.

### Pending

- Public benchmark preparation from pinned source releases.
- External validation after the first development benchmark is frozen.

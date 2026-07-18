# Versioning and Reproducibility

FanoSeq uses separate version namespaces. They must not be inferred from one
another.

| Namespace | Current example | Meaning |
| --- | --- | --- |
| Software release | `0.2.0` | Python package and CLI behavior |
| Complete-analysis schema | `1.0.0` | `analysis_manifest.json` contract |
| Benchmark schema | `1.0.0` | Benchmark tables and manifest contract |
| Pipeline bundle schema | `0.3.0` | Core pipeline table/bundle contract |
| Encoding-audit schema | `0.8.0` | Audit table metadata contract |
| Axis scheme | `dna-window-v1` | Scientific meaning assigned to axes |
| Fano convention | `fanoseq-fano-lines-v1` | Oriented multiplication convention |

A software release may preserve an output schema, and a schema may change
independently when its contract changes. Axis-scheme identifiers are immutable:
changing a formula, normalization, missing-data policy, or semantic meaning
requires a new identifier.

The complete-analysis manifest records the software version, Git commit, schema
version, input and output SHA-256 hashes, resolved extraction settings, axis and
Fano convention IDs, seeds, split provenance, dependency versions, table
dimensions, runtimes, traced peak memory, warnings, and evidence status.

Release metadata is maintained in `pyproject.toml`, `src/fanoseq/__init__.py`,
`CITATION.cff`, and `CHANGELOG.md`. `CITATION.cff` points to the canonical
repository at <https://github.com/Wabanero/FanoSeq>.

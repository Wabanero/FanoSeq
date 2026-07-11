# Scientific Evidence Status

FanoSeq separates software readiness, mathematical validation, and biological
validation. A component can be stable software while still having no biological
evidence.

| component | software status | mathematical validation | biological validation |
| --- | --- | --- | --- |
| Octonion kernel | stable | validated | not applicable |
| DNA window encoder | stable | audited | pending |
| Codon encoder | experimental | audited | pending |
| Fano-line features | experimental | algebraically validated | pending |
| Matrix genetics | exploratory | partially validated | not predictive |
| Benchmark engine | beta | tested | datasets pending |
| Protein encoder | experimental | partial | unvalidated |
| Dataset registry | initial | schema-smoke-tested | data preparation pending |
| Methods report | protocol | not applicable | results pending |

## Interpretation Policy

- Algebra validation means the implementation follows the stated Fano-plane and
  octonion convention.
- Encoding audit means the repository reports what an encoding preserves,
  changes, duplicates, or discards.
- Biological validation requires public datasets, leakage controls, classical
  baselines, nonlinear controls, external validation, and negative results.
- Benchmark success is task-specific. It is not evidence for an intrinsic
  octonionic mechanism in biology.


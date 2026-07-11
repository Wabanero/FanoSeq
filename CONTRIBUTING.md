# Contributing

FanoSeq is publicly visible for evaluation and portfolio-review purposes, but it
is not an open-source project. See `LICENSE` before copying, modifying,
redistributing, or using any part of the repository.

## Issues And Discussions

Bug reports, reproducibility notes, benchmark-design critiques, and citation
corrections are welcome. Good reports include:

- operating system and Python version;
- the command that failed;
- a minimal FASTA/config example when possible;
- the observed output and expected behavior;
- whether the issue affects software behavior, mathematical validation, or
  biological evidence.

## Pull Requests

Pull requests may be reviewed at the maintainer's discretion. By submitting a
pull request, you confirm that you have the right to contribute the work and you
grant the project owner permission to use, modify, sublicense, and redistribute
the contribution under the repository's current or future license terms.

Do not submit confidential data, unpublished biological datasets, patient data,
private credentials, or source files whose licenses do not permit inclusion.

## Development Checks

Recommended local checks:

```bash
python -m pip install -e ".[dev,downstream]"
python -m ruff check .
python -m mypy src/fanoseq
python -m pytest
fanoseq validate-basis
fanoseq benchmark --config examples/benchmark.yaml --output-dir results/benchmark_smoke
```

Benchmark-result claims should include dataset manifests, input hashes, fold
assignments, baseline/control results, and negative results.


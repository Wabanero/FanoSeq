# Information-theory audits

`fanoseq.information_theory` provides transparent empirical diagnostics beyond
pairwise correlation:

- Shannon entropy and pairwise mutual information;
- conditional mutual information;
- signed three-way interaction information;
- total correlation (multi-information);
- a table builder that records the estimator and numeric discretization.

The implementation uses a plug-in discrete estimator in bits. Numeric columns
are quantile-binned with the bin count recorded in every row. These estimates
are biased on small samples and sensitive to discretization; they are audit
statistics, not automatic evidence of biological synergy.

The XOR regression test demonstrates why pairwise analysis is insufficient:
each input is pairwise independent of the other, while conditioning on the XOR
output reveals one bit and the three-way interaction information is negative
under the documented `I(X;Y)-I(X;Y|Z)` convention.

For a biological claim, calculate these statistics inside resampling folds,
compare with label and representation null distributions, correct the family of
hypotheses, and validate the selected interaction on a frozen external dataset.

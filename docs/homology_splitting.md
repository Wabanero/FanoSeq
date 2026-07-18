# Homology-aware splitting

The benchmark's optional positional-identity audit is not a homology detector.
For biological studies, cluster sequences with a domain-appropriate external
tool (for example MMseqs2, CD-HIT, or an alignment/graph workflow), then attach
those cluster assignments to metadata:

```bash
fanoseq prepare-homology-groups \
  --metadata prepared/metadata.tsv \
  --clusters prepared/external_clusters.tsv \
  --member-column sequence_id \
  --cluster-column homology_cluster \
  --tool mmseqs2 \
  --tool-version <recorded-version> \
  --minimum-identity 0.8 \
  --minimum-coverage 0.8 \
  --output prepared/metadata_homology.tsv
```

The cluster table must have a header and one member-to-cluster assignment per
row. Missing or conflicting assignments fail closed by default. The command
writes grouped metadata plus a `.homology.json` manifest containing input and
output hashes, tool/version, thresholds, row/group counts, and the group-column
instruction. Set `dataset.group_column: homology_cluster` in the benchmark
manifest.

Threshold choice is dataset- and molecule-specific and must be frozen before
external validation. FanoSeq records the choice but does not claim that one
universal identity or coverage threshold defines biological homology.

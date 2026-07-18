# Biological context and synchronized tracks

Sequence-only descriptors cannot answer whether a pattern is coding,
regulatory, structural, or epigenomic. FanoSeq therefore defines context as
explicit input data rather than inferring it from a Fano-plane visualization.

`validate_genomic_context` expects one row per sequence with:

`sequence_id, genome_build, chromosome, start, end, strand, feature_type`

Optional columns may include gene/transcript IDs, biotype, exon/intron status,
reading frame, regulatory class, cell type, assay, tissue, and source accession.
Coordinates use zero-based half-open intervals and are validated.

`validate_multitrack_table` expects synchronized long-format observations:

`sequence_id, start, end, track_name, value`

`align_tracks_to_windows` pivots those tracks onto exact FanoSeq windows and
fails if any window lacks a value unless the caller explicitly selects a
missing-data policy. Track manifests should additionally record assembly,
assay, sample, normalization, strand handling, coordinate transformation, and
source hashes.

Minimum comparisons for a context-aware study are:

1. sequence-only conventional features;
2. context/tracks alone;
3. sequence plus context using ordinary real-valued concatenation/interactions;
4. randomized or permuted track controls;
5. the fixed FanoSeq interaction design.

Cell type, tissue, chromosome, donor, batch, and source study are candidate
grouping/confounding variables. External validation must hold out the relevant
context rather than randomly splitting neighboring windows. The registered
`dna-regulatory-v1` and `dna-shape-v1` schemes remain experimental/planned until
the corresponding inputs and comparators are included in a frozen study.

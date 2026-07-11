# References

This list separates mathematical support, related biological representation
work, and external benchmark context. None of these references alone validates
FanoSeq's biological usefulness; that requires the benchmark studies in
`datasets/`.

## Octonion And Fano-Plane Mathematics

### Baez 2002

- Full citation: Baez JC. The octonions. Bulletin of the American Mathematical
  Society. 2002;39(2):145-205.
- DOI: 10.1090/S0273-0979-01-00934-X.
- Publication type: review article.
- Status: peer-reviewed.
- Direct relevance to FanoSeq: clear mathematical background for octonions,
  non-associativity, normed division algebras, and Fano-plane multiplication
  conventions.
- What it does not support: no claim about DNA, proteins, sequence encoding, or
  predictive bioinformatics.

### Springer And Veldkamp 2000

- Full citation: Springer TA, Veldkamp FD. Octonions, Jordan Algebras and
  Exceptional Groups. Springer Monographs in Mathematics. Springer, 2000.
- DOI: 10.1007/978-3-662-12622-6.
- Publication type: mathematical monograph.
- Status: scholarly book.
- Direct relevance to FanoSeq: rigorous reference for octonion algebra and
  exceptional algebraic structures.
- What it does not support: no validation of hand-designed biological axes.

## Matrix Genetics

### Petoukhov 2008

- Full citation: Petoukhov SV. Matrix Genetics, Algebras of the Genetic Code,
  Noise Immunity. RCD, 2008.
- DOI: not assigned.
- Publication type: monograph.
- Status: book; not a benchmark study.
- Direct relevance to FanoSeq: motivates caution around 8x8 genetic-code
  matrices, dyadic patterns, and algebraic codon arrangements.
- What it does not support: does not validate Cayley octonion multiplication as a
  biological sequence model.

### Petoukhov And He 2010

- Full citation: Petoukhov SV, He M. Symmetrical Analysis Techniques for Genetic
  Systems and Bioinformatics: Advanced Patterns and Applications. IGI Global,
  2010.
- DOI: 10.4018/978-1-60566-124-7.
- Publication type: edited scholarly book.
- Status: book.
- Direct relevance to FanoSeq: background for symmetry-based genetic-code
  analyses and matrix-oriented encodings.
- What it does not support: no direct evidence that FanoSeq features outperform
  k-mer, codon-usage, or FCGR baselines.

## Quaternionic And Hypercomplex DNA Representations

### Shu And Li 2014

- Full citation: Shu JJ, Li Y. Hypercomplex cross-correlation of DNA sequences.
  arXiv:1402.5341, 2014.
- DOI: not assigned.
- Publication type: preprint.
- Status: preprint.
- Direct relevance to FanoSeq: example of hypercomplex representations for DNA
  sequence comparison.
- What it does not support: not an octonion/Fano-plane benchmark and not
  evidence for FanoSeq's fixed multiplication table.

### Carlevaro, Irastorza, And Vericat 2015

- Full citation: Carlevaro CM, Irastorza RM, Vericat F. Quaternionic
  representation of the genetic code. arXiv:1505.04656, 2015.
- DOI: not assigned.
- Publication type: preprint.
- Status: preprint.
- Direct relevance to FanoSeq: related example of assigning genetic-code objects
  to quaternionic structures.
- What it does not support: not a validation of FanoSeq DNA-window descriptors,
  codon products, or octonion interactions.

## Algebraic Genetic-Code Models

### Hornos And Hornos 1993

- Full citation: Hornos JEM, Hornos YMM. Algebraic model for the evolution of
  the genetic code. Physical Review Letters. 1993;71(26):4401-4404.
- DOI: 10.1103/PhysRevLett.71.4401.
- Publication type: research article.
- Status: peer-reviewed.
- Direct relevance to FanoSeq: example of an algebraic model tied to genetic-code
  degeneracy and evolution.
- What it does not support: no claim about Fano-plane sequence features or
  alignment-free prediction.

### Bashford, Tsohantjis, And Jarvis 1998

- Full citation: Bashford JD, Tsohantjis I, Jarvis PD. A supersymmetric model for
  the evolution of the genetic code. Proceedings of the National Academy of
  Sciences. 1998;95(3):987-992.
- DOI: 10.1073/pnas.95.3.987.
- Publication type: research article.
- Status: peer-reviewed.
- Direct relevance to FanoSeq: another algebraic genetic-code model useful for
  positioning codon-level claims.
- What it does not support: no evidence that FanoSeq descriptors are predictive
  biological features.

## CGR And Alignment-Free Methods

### Jeffrey 1990

- Full citation: Jeffrey HJ. Chaos game representation of gene structure.
  Nucleic Acids Research. 1990;18(8):2163-2170.
- DOI: 10.1093/nar/18.8.2163.
- Publication type: research article.
- Status: peer-reviewed.
- Direct relevance to FanoSeq: foundational visual/alignment-free representation
  of DNA sequences and a comparator family for FanoSeq benchmarks.
- What it does not support: no octonion or Fano-plane mechanism.

### Goldman 1993

- Full citation: Goldman N. Nucleotide, dinucleotide and trinucleotide
  frequencies explain patterns observed in chaos game representations of DNA
  sequences. Nucleic Acids Research. 1993;21(10):2487-2491.
- DOI: 10.1093/nar/21.10.2487.
- Publication type: research article.
- Status: peer-reviewed.
- Direct relevance to FanoSeq: cautionary control showing that visually rich
  sequence representations may reduce to ordinary composition statistics.
- What it does not support: no validation of FanoSeq, but it strongly motivates
  k-mer and FCGR controls.

### Vinga And Almeida 2003

- Full citation: Vinga S, Almeida J. Alignment-free sequence comparison-a
  review. Bioinformatics. 2003;19(4):513-523.
- DOI: 10.1093/bioinformatics/btg005.
- Publication type: review article.
- Status: peer-reviewed.
- Direct relevance to FanoSeq: baseline context for alignment-free sequence
  comparison and feature families.
- What it does not support: no evidence for octonion-specific improvements.

## Hypercomplex Neural Networks

### Trabelsi et al. 2018

- Full citation: Trabelsi C, Bilaniuk O, Zhang Y, Serdyuk D, Subramanian S,
  Santos JF, Mehri S, Rostamzadeh N, Bengio Y, Pal C. Deep Complex Networks.
  International Conference on Learning Representations, 2018.
- DOI: not assigned; arXiv:1705.09792.
- Publication type: conference paper.
- Status: peer-reviewed conference publication.
- Direct relevance to FanoSeq: demonstrates that complex-valued neural modules
  can be engineered and benchmarked against real-valued networks.
- What it does not support: not about genomics and not about octonions.

### Gaudet And Maida 2018

- Full citation: Gaudet CJ, Maida AS. Deep Quaternion Networks. International
  Joint Conference on Neural Networks, 2018.
- DOI: 10.1109/IJCNN.2018.8489651.
- Publication type: conference paper.
- Status: peer-reviewed conference publication.
- Direct relevance to FanoSeq: related hypercomplex neural-network engineering
  with quaternion-valued operations.
- What it does not support: no evidence for FanoSeq sequence descriptors or
  octonion-valued biological models.

### Parcollet et al. 2019

- Full citation: Parcollet T, Ravanelli M, Morchid M, Linares G, Trabelsi C,
  De Mori R, Bengio Y. Quaternion Recurrent Neural Networks. International
  Conference on Learning Representations, 2019.
- DOI: not assigned; arXiv:1806.04418.
- Publication type: conference paper.
- Status: peer-reviewed conference publication.
- Direct relevance to FanoSeq: relevant to future quaternion or octonion-aware
  neural layers and parameter-count controls.
- What it does not support: no DNA/protein benchmark evidence.

## Genomic Foundation Models

### DNABERT 2021

- Full citation: Ji Y, Zhou Z, Liu H, Davuluri RV. DNABERT: pre-trained
  Bidirectional Encoder Representations from Transformers model for DNA-language
  in genome. Bioinformatics. 2021;37(15):2112-2120.
- DOI: 10.1093/bioinformatics/btab083.
- Publication type: research article.
- Status: peer-reviewed.
- Direct relevance to FanoSeq: external learned baseline for sequence
  representation tasks once classical benchmarks are stable.
- What it does not support: no evidence for fixed octonion interactions.

### Enformer 2021

- Full citation: Avsec Z et al. Effective gene expression prediction from
  sequence by integrating long-range interactions. Nature Methods.
  2021;18:1196-1203.
- DOI: 10.1038/s41592-021-01252-x.
- Publication type: research article.
- Status: peer-reviewed.
- Direct relevance to FanoSeq: strong learned genomic sequence baseline for
  regulatory prediction contexts.
- What it does not support: not an alignment-free FanoSeq comparator for small
  tabular benchmark studies.

### Nucleotide Transformer 2023

- Full citation: Dalla-Torre H et al. The Nucleotide Transformer: building and
  evaluating robust foundation models for human genomics. bioRxiv, 2023.
- DOI: 10.1101/2023.01.11.523679.
- Publication type: preprint.
- Status: preprint.
- Direct relevance to FanoSeq: candidate external embedding baseline for later
  held-out biological tasks.
- What it does not support: no validation of hand-designed FanoSeq axes or
  octonion multiplication.


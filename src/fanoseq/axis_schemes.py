"""Versioned axis-scheme and axis-definition registry for FanoSeq encodings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from fanoseq.octonion import FANO_LINES

SchemeStatus = Literal["stable", "experimental", "planned"]


@dataclass(frozen=True)
class AxisDefinition:
    """Definition of one axis in a FanoSeq scheme."""

    index: int
    symbol: str
    label: str
    description: str
    formula: str
    inputs: tuple[str, ...]
    normalization: str
    value_hint: str
    role: str
    missing_policy: str
    implemented: bool
    benchmark_baseline: str


@dataclass(frozen=True)
class FanoLineDefinition:
    """Semantic description of one oriented Fano-plane line."""

    axes: tuple[int, int, int]
    label: str
    interpretation: str


@dataclass(frozen=True)
class AxisScheme:
    """A versioned mapping from sequence-derived features to octonion axes."""

    scheme_id: str
    name: str
    seq_type: str
    mode: str
    version: str
    status: SchemeStatus
    representation: str
    implemented: bool
    scalar_axis: AxisDefinition
    imaginary_axes: tuple[AxisDefinition, ...]
    fano_lines: tuple[FanoLineDefinition, ...]
    recommended_use: str
    limitations: str

    @property
    def axes(self) -> tuple[AxisDefinition, ...]:
        """Return scalar and imaginary axes in component order."""
        return (self.scalar_axis, *self.imaginary_axes)

    @property
    def is_runnable(self) -> bool:
        """Return True when the scheme can be used by ``fanoseq run``."""
        return self.implemented and all(axis.implemented for axis in self.axes)

    def axis_by_index(self, index: int) -> AxisDefinition:
        """Return one axis definition by component index."""
        for axis in self.axes:
            if axis.index == index:
                return axis
        raise ValueError(f"Axis e{index} is not defined for scheme {self.scheme_id}.")

    def axis_labels(self) -> dict[int, str]:
        """Return labels for imaginary axes e1...e7."""
        return {axis.index: axis.label for axis in self.imaginary_axes}

    def line_label(self, axes: tuple[int, int, int]) -> str:
        """Return the semantic label for a Fano line."""
        for line in self.fano_lines:
            if line.axes == axes:
                return line.label
        return " x ".join(self.axis_labels()[axis] for axis in axes)


def list_axis_schemes() -> pd.DataFrame:
    """Return registered axis schemes as a compact table."""
    rows = []
    for scheme in AXIS_SCHEME_REGISTRY.values():
        rows.append(
            {
                "scheme_id": scheme.scheme_id,
                "name": scheme.name,
                "seq_type": scheme.seq_type,
                "mode": scheme.mode,
                "version": scheme.version,
                "status": scheme.status,
                "representation": scheme.representation,
                "implemented": scheme.implemented,
                "runnable": scheme.is_runnable,
                "recommended_use": scheme.recommended_use,
            }
        )
    return pd.DataFrame(rows)


def list_axis_definitions(scheme_id: str | None = None) -> pd.DataFrame:
    """Return axis definitions for one scheme or the whole registry."""
    schemes = (
        [get_axis_scheme(scheme_id)]
        if scheme_id is not None
        else AXIS_SCHEME_REGISTRY.values()
    )
    rows: list[dict[str, object]] = []
    for scheme in schemes:
        for axis in scheme.axes:
            row = _axis_row(axis)
            rows.append(
                {
                    "scheme_id": scheme.scheme_id,
                    "seq_type": scheme.seq_type,
                    "mode": scheme.mode,
                    **row,
                }
            )
    return pd.DataFrame(rows)


def get_axis_scheme(scheme_id: str) -> AxisScheme:
    """Return one registered axis scheme."""
    try:
        return AXIS_SCHEME_REGISTRY[scheme_id]
    except KeyError as exc:
        choices = ", ".join(sorted(AXIS_SCHEME_REGISTRY))
        raise ValueError(
            f"Unknown axis scheme {scheme_id!r}. Available schemes: {choices}."
        ) from exc


def default_axis_scheme_id(seq_type: str, mode: str) -> str:
    """Return the default scheme id for an existing FanoSeq context."""
    if mode == "codon":
        return "codon-product-v1"
    if seq_type == "dna" and mode == "window":
        return "dna-window-v1"
    if seq_type == "protein" and mode == "window":
        return "protein-sequence-v1"
    raise ValueError(
        f"Unsupported axis-scheme context: seq_type={seq_type!r}, mode={mode!r}."
    )


def resolve_axis_scheme(
    seq_type: str,
    mode: str,
    scheme_id: str | None = None,
    *,
    require_runnable: bool = False,
) -> AxisScheme:
    """Return a context-compatible axis scheme, optionally requiring run support."""
    scheme = get_axis_scheme(scheme_id or default_axis_scheme_id(seq_type, mode))
    if scheme.seq_type != seq_type:
        raise ValueError(
            f"Axis scheme {scheme.scheme_id!r} is for seq_type={scheme.seq_type!r}, "
            f"not seq_type={seq_type!r}."
        )
    if scheme.mode != mode:
        raise ValueError(
            f"Axis scheme {scheme.scheme_id!r} is for mode={scheme.mode!r}, not mode={mode!r}."
        )
    if require_runnable and not scheme.is_runnable:
        raise ValueError(
            f"Axis scheme {scheme.scheme_id!r} is registered but not implemented by fanoseq run. "
            "Use a runnable scheme or export its definitions with describe-axis-scheme."
        )
    return scheme


def axis_labels_for_context(
    seq_type: str,
    mode: str,
    scheme_id: str | None = None,
) -> dict[int, str]:
    """Return imaginary-axis labels for a mode/sequence context."""
    return resolve_axis_scheme(seq_type=seq_type, mode=mode, scheme_id=scheme_id).axis_labels()


def axis_scheme_tables(scheme_id: str) -> dict[str, pd.DataFrame]:
    """Return metadata, axis, Fano-line, and validation tables for one scheme."""
    scheme = get_axis_scheme(scheme_id)
    metadata = pd.DataFrame(
        [
            {
                "scheme_id": scheme.scheme_id,
                "name": scheme.name,
                "seq_type": scheme.seq_type,
                "mode": scheme.mode,
                "version": scheme.version,
                "status": scheme.status,
                "representation": scheme.representation,
                "implemented": scheme.implemented,
                "runnable": scheme.is_runnable,
                "recommended_use": scheme.recommended_use,
                "limitations": scheme.limitations,
            }
        ]
    )
    axes = list_axis_definitions(scheme_id)
    labels = scheme.axis_labels()
    lines = pd.DataFrame(
        [
            {
                "scheme_id": scheme.scheme_id,
                "fano_line": f"({line.axes[0]},{line.axes[1]},{line.axes[2]})",
                "axis_a": line.axes[0],
                "axis_b": line.axes[1],
                "axis_c": line.axes[2],
                "axis_a_label": labels[line.axes[0]],
                "axis_b_label": labels[line.axes[1]],
                "axis_c_label": labels[line.axes[2]],
                "line_label": line.label,
                "interpretation": line.interpretation,
            }
            for line in scheme.fano_lines
        ]
    )
    return {
        "axis_scheme_metadata": metadata,
        "axis_scheme_axes": axes,
        "axis_scheme_fano_lines": lines,
        "axis_scheme_validation": validate_axis_scheme_definitions(scheme_id),
    }


def validate_axis_scheme_definitions(scheme_id: str | None = None) -> pd.DataFrame:
    """Validate structural completeness of registered axis definitions."""
    schemes = (
        [get_axis_scheme(scheme_id)]
        if scheme_id is not None
        else AXIS_SCHEME_REGISTRY.values()
    )
    rows: list[dict[str, object]] = []
    for scheme in schemes:
        _append_check(
            rows,
            scheme,
            "scalar_axis_is_e0",
            scheme.scalar_axis.index == 0,
            f"Scalar axis is {scheme.scalar_axis.symbol}.",
        )
        imaginary_indices = [axis.index for axis in scheme.imaginary_axes]
        _append_check(
            rows,
            scheme,
            "imaginary_axes_are_e1_to_e7",
            sorted(imaginary_indices) == list(range(1, 8)) and len(set(imaginary_indices)) == 7,
            f"Imaginary axis indices are {imaginary_indices}.",
        )
        _append_check(
            rows,
            scheme,
            "fano_lines_match_basis_convention",
            tuple(line.axes for line in scheme.fano_lines) == FANO_LINES,
            "Fano-line orientation matches fanoseq.octonion.FANO_LINES.",
        )
        missing_definition_fields = [
            axis.symbol
            for axis in scheme.axes
            if not axis.label
            or not axis.description
            or not axis.formula
            or not axis.normalization
            or not axis.value_hint
            or not axis.missing_policy
            or not axis.benchmark_baseline
        ]
        _append_check(
            rows,
            scheme,
            "axis_definitions_are_complete",
            not missing_definition_fields,
            "Missing definition fields: "
            + (", ".join(missing_definition_fields) if missing_definition_fields else "none."),
        )
        _append_check(
            rows,
            scheme,
            "stable_schemes_are_runnable",
            scheme.status != "stable" or scheme.is_runnable,
            f"Runnable={scheme.is_runnable} for status={scheme.status}.",
        )
    return pd.DataFrame(rows)


def _append_check(
    rows: list[dict[str, object]],
    scheme: AxisScheme,
    check_id: str,
    passed: bool,
    message: str,
) -> None:
    rows.append(
        {
            "scheme_id": scheme.scheme_id,
            "check_id": check_id,
            "passed": bool(passed),
            "message": message,
        }
    )


def _axis_row(axis: AxisDefinition) -> dict[str, object]:
    return {
        "axis": axis.index,
        "symbol": axis.symbol,
        "component_column": axis.symbol,
        "label": axis.label,
        "description": axis.description,
        "formula": axis.formula,
        "inputs": ", ".join(axis.inputs),
        "normalization": axis.normalization,
        "value_hint": axis.value_hint,
        "role": axis.role,
        "missing_policy": axis.missing_policy,
        "implemented": axis.implemented,
        "benchmark_baseline": axis.benchmark_baseline,
    }


def _axis(
    index: int,
    label: str,
    description: str,
    formula: str,
    inputs: tuple[str, ...],
    normalization: str,
    value_hint: str,
    role: str,
    missing_policy: str,
    benchmark_baseline: str,
    *,
    implemented: bool = True,
) -> AxisDefinition:
    return AxisDefinition(
        index=index,
        symbol=f"e{index}",
        label=label,
        description=description,
        formula=formula,
        inputs=inputs,
        normalization=normalization,
        value_hint=value_hint,
        role=role,
        missing_policy=missing_policy,
        implemented=implemented,
        benchmark_baseline=benchmark_baseline,
    )


def _line(axes: tuple[int, int, int], label: str, interpretation: str) -> FanoLineDefinition:
    if axes not in FANO_LINES:
        raise ValueError(f"Line {axes} is not in the project Fano convention.")
    return FanoLineDefinition(axes=axes, label=label, interpretation=interpretation)


def _generic_lines(axis_labels: dict[int, str], prefix: str) -> tuple[FanoLineDefinition, ...]:
    return tuple(
        _line(
            line,
            f"{prefix}: {axis_labels[line[0]]} / {axis_labels[line[1]]} / {axis_labels[line[2]]}",
            "Exploratory Fano triad; interpretation depends on benchmark and null-model stability.",
        )
        for line in FANO_LINES
    )


DNA_WINDOW_SCALAR = _axis(
    0,
    "valid fraction",
    "Window mass/reliability scalar.",
    "valid A/C/G/T count divided by cleaned window length",
    ("window", "valid_base_count", "cleaned_window_length"),
    "fraction in [0,1]",
    "[0,1]",
    "scalar reliability",
    "Window is skipped when ambiguous_fraction exceeds max_ambiguous_fraction.",
    "GC content, valid-base fraction, ambiguous-base fraction",
)

DNA_WINDOW_AXES = (
    _axis(
        1,
        "purine/pyrimidine balance",
        "Signed R/Y chemistry balance.",
        "(A + G - C - T) / L",
        ("A_count", "C_count", "G_count", "T_count", "valid_length"),
        "divide by valid A/C/G/T length L",
        "[-1,+1]",
        "base chemistry",
        "Computed after removing ambiguous symbols allowed by the run policy.",
        "Mononucleotide composition and purine fraction",
    ),
    _axis(
        2,
        "GC/AT balance",
        "Signed strong/weak base-pair balance.",
        "(G + C - A - T) / L",
        ("A_count", "C_count", "G_count", "T_count", "valid_length"),
        "divide by valid A/C/G/T length L",
        "[-1,+1]",
        "base chemistry",
        "Computed after removing ambiguous symbols allowed by the run policy.",
        "GC content",
    ),
    _axis(
        3,
        "amino/keto balance",
        "Signed M/K chemistry balance.",
        "(A + C - G - T) / L",
        ("A_count", "C_count", "G_count", "T_count", "valid_length"),
        "divide by valid A/C/G/T length L",
        "[-1,+1]",
        "base chemistry",
        "Computed after removing ambiguous symbols allowed by the run policy.",
        "Mononucleotide composition",
    ),
    _axis(
        4,
        "GC skew",
        "Strand/composition asymmetry between G and C.",
        "(G - C) / (G + C + epsilon)",
        ("G_count", "C_count", "epsilon"),
        "epsilon-stabilized signed ratio",
        "[-1,+1]",
        "strand/composition asymmetry",
        "Returns near zero when G+C is zero because epsilon stabilizes the denominator.",
        "GC skew",
    ),
    _axis(
        5,
        "AT skew",
        "Strand/composition asymmetry between A and T.",
        "(A - T) / (A + T + epsilon)",
        ("A_count", "T_count", "epsilon"),
        "epsilon-stabilized signed ratio",
        "[-1,+1]",
        "strand/composition asymmetry",
        "Returns near zero when A+T is zero because epsilon stabilizes the denominator.",
        "AT skew",
    ),
    _axis(
        6,
        "k-mer entropy",
        "Local sequence complexity at the configured k.",
        "H_k(window) / log2(4^k)",
        ("valid_window", "kmer_k"),
        "Shannon entropy normalized by the full DNA k-mer alphabet size",
        "[0,1]",
        "local complexity",
        "Invalid k-mers containing ambiguity are excluded; empty sets return 0.",
        "k-mer entropy and low-complexity filters",
    ),
    _axis(
        7,
        "reverse-complement symmetry",
        "Signed reverse-complement positional similarity.",
        "2 * RC_similarity(window) - 1",
        ("valid_window", "reverse_complement"),
        "linear map from identity fraction [0,1] to [-1,+1]",
        "[-1,+1]",
        "sequence symmetry",
        "Computed on the valid-only window after ambiguity handling.",
        "Palindrome and inverted-repeat features",
    ),
)

DNA_WINDOW_LINES = (
    _line((1, 2, 3), "base chemistry triad", "Joint RY, GC/AT, and MK chemistry balance."),
    _line(
        (1, 4, 5),
        "RY-skew asymmetry triad",
        "Purine/pyrimidine balance coupled to GC and AT skew.",
    ),
    _line(
        (1, 7, 6),
        "RY-symmetry-complexity triad",
        "Purine/pyrimidine balance coupled to reverse-complement symmetry and k-mer entropy.",
    ),
    _line(
        (2, 4, 6),
        "GC-skew-complexity triad",
        "GC/AT balance coupled to GC skew and local sequence complexity.",
    ),
    _line(
        (2, 5, 7),
        "GC-ATskew-symmetry triad",
        "GC/AT balance coupled to AT skew and reverse-complement symmetry.",
    ),
    _line(
        (3, 4, 7),
        "MK-GCskew-symmetry triad",
        "Amino/keto balance coupled to GC skew and reverse-complement symmetry.",
    ),
    _line(
        (3, 6, 5),
        "MK-complexity-ATskew triad",
        "Amino/keto balance coupled to k-mer entropy and AT skew.",
    ),
)

CODING_IMPLEMENTED = False
DNA_CODING_SCALAR = _axis(
    0,
    "valid coding fraction",
    "Coding-window reliability scalar.",
    "valid in-frame A/C/G/T bases divided by coding-window length",
    ("coding_window", "valid_base_count"),
    "fraction in [0,1]",
    "[0,1]",
    "scalar reliability",
    "Not emitted by fanoseq run until the coding-window encoder is implemented.",
    "Valid coding fraction and missing-base rate",
    implemented=CODING_IMPLEMENTED,
)

DNA_CODING_AXES = (
    _axis(
        1,
        "purine/pyrimidine balance",
        "Coding-window RY balance.",
        "(A + G - C - T) / L",
        ("coding_window_counts",),
        "divide by valid coding length L",
        "[-1,+1]",
        "base chemistry",
        "Defined for valid coding bases only.",
        "Mononucleotide composition",
        implemented=CODING_IMPLEMENTED,
    ),
    _axis(
        2,
        "GC/AT balance",
        "Coding-window GC/AT balance.",
        "(G + C - A - T) / L",
        ("coding_window_counts",),
        "divide by valid coding length L",
        "[-1,+1]",
        "base chemistry",
        "Defined for valid coding bases only.",
        "GC content",
        implemented=CODING_IMPLEMENTED,
    ),
    _axis(
        3,
        "amino/keto balance",
        "Coding-window MK balance.",
        "(A + C - G - T) / L",
        ("coding_window_counts",),
        "divide by valid coding length L",
        "[-1,+1]",
        "base chemistry",
        "Defined for valid coding bases only.",
        "Mononucleotide composition",
        implemented=CODING_IMPLEMENTED,
    ),
    _axis(
        4,
        "GC3 excess",
        "Third-position GC signal relative to local GC.",
        "mean(GC at codon position 3) - mean(GC across all codon positions)",
        ("codon_frame", "gc1", "gc2", "gc3"),
        "difference of fractions",
        "[-1,+1]",
        "codon bias",
        "Requires a selected reading frame and complete codons.",
        "GC3 and codon-position GC",
        implemented=CODING_IMPLEMENTED,
    ),
    _axis(
        5,
        "period-3 frame signal",
        "Strength of frame-periodic base composition.",
        "|sum_i s_i * exp(-2*pi*j*i/3)| / L, with s_i = +1 for GC and -1 for AT",
        ("coding_window", "selected_frame"),
        "DFT amplitude normalized by valid length",
        "[0,1]",
        "coding periodicity",
        "Requires enough valid bases to estimate a period-3 signal.",
        "Frame-periodicity and three-base periodicity features",
        implemented=CODING_IMPLEMENTED,
    ),
    _axis(
        6,
        "codon entropy",
        "Observed codon diversity within the coding window.",
        "H(codons) / log2(64)",
        ("in_frame_codons",),
        "Shannon entropy normalized by 64 possible codons",
        "[0,1]",
        "codon usage",
        "Complete valid codons are counted; ambiguous codons are skipped.",
        "Codon frequency, RSCU, CAI",
        implemented=CODING_IMPLEMENTED,
    ),
    _axis(
        7,
        "ORF integrity",
        "Open-reading-frame continuity proxy.",
        "1 - stop_codon_count / max(valid_codon_count, 1)",
        ("in_frame_codons", "genetic_code"),
        "stop-density complement",
        "[0,1]",
        "coding integrity",
        "Requires a genetic code; ambiguous codons are skipped.",
        "Stop density and ORF annotation",
        implemented=CODING_IMPLEMENTED,
    ),
)

REGULATORY_IMPLEMENTED = False
DNA_REGULATORY_SCALAR = _axis(
    0,
    "valid fraction",
    "Regulatory-window reliability scalar.",
    "valid A/C/G/T count divided by cleaned window length",
    ("window", "valid_base_count", "cleaned_window_length"),
    "fraction in [0,1]",
    "[0,1]",
    "scalar reliability",
    "Not emitted by fanoseq run until the regulatory-window encoder is implemented.",
    "Valid-base fraction and ambiguous-base fraction",
    implemented=REGULATORY_IMPLEMENTED,
)

DNA_REGULATORY_AXES = (
    _axis(
        1,
        "purine/pyrimidine balance",
        "Regulatory-window RY balance.",
        "(A + G - C - T) / L",
        ("window_counts",),
        "divide by valid length L",
        "[-1,+1]",
        "base chemistry",
        "Defined for valid bases only.",
        "Mononucleotide composition",
        implemented=REGULATORY_IMPLEMENTED,
    ),
    _axis(
        2,
        "GC/AT balance",
        "Regulatory-window GC/AT balance.",
        "(G + C - A - T) / L",
        ("window_counts",),
        "divide by valid length L",
        "[-1,+1]",
        "base chemistry",
        "Defined for valid bases only.",
        "GC content",
        implemented=REGULATORY_IMPLEMENTED,
    ),
    _axis(
        3,
        "amino/keto balance",
        "Regulatory-window MK balance.",
        "(A + C - G - T) / L",
        ("window_counts",),
        "divide by valid length L",
        "[-1,+1]",
        "base chemistry",
        "Defined for valid bases only.",
        "Mononucleotide composition",
        implemented=REGULATORY_IMPLEMENTED,
    ),
    _axis(
        4,
        "CpG observed/expected",
        "Observed CpG density relative to mononucleotide expectation.",
        "(CpG_count / max(L - 1, 1)) / ((C/L) * (G/L) + epsilon)",
        ("window", "C_count", "G_count", "CpG_count", "epsilon"),
        "epsilon-stabilized observed/expected ratio",
        "[0,+inf)",
        "regulatory composition",
        "Returns 0 when no valid dinucleotide is available.",
        "CpG observed/expected and CpG island features",
        implemented=REGULATORY_IMPLEMENTED,
    ),
    _axis(
        5,
        "palindrome/inverted-repeat density",
        "Reverse-complement motif symmetry proxy.",
        "palindromic_kmer_count / max(L - palindrome_k + 1, 1)",
        ("window", "palindrome_k"),
        "fraction of valid k-mer positions",
        "[0,1]",
        "motif symmetry",
        "Requires a configured palindrome length; invalid k-mers are skipped.",
        "Palindrome and inverted-repeat density",
        implemented=REGULATORY_IMPLEMENTED,
    ),
    _axis(
        6,
        "k-mer entropy",
        "Regulatory-window sequence complexity.",
        "H_k(window) / log2(4^k)",
        ("window", "kmer_k"),
        "Shannon entropy normalized by DNA k-mer alphabet size",
        "[0,1]",
        "local complexity",
        "Invalid k-mers containing ambiguity are excluded.",
        "k-mer entropy",
        implemented=REGULATORY_IMPLEMENTED,
    ),
    _axis(
        7,
        "motif-density/AT-rich proxy",
        "Motif-density when motifs are supplied, otherwise AT-richness proxy.",
        "motif_hit_count / searchable_positions, or 1 - GC_content without motifs",
        ("window", "motif_set", "gc_content"),
        "fraction or GC-complement proxy",
        "[0,1]",
        "regulatory signal",
        "Requires explicit motif-set provenance when used as motif density.",
        "PWM/k-mer motif density and AT content",
        implemented=REGULATORY_IMPLEMENTED,
    ),
)

SHAPE_IMPLEMENTED = False
DNA_SHAPE_SCALAR = _axis(
    0,
    "track coverage/reliability",
    "Coverage or confidence scalar for external tracks.",
    "covered_positions / requested_positions",
    ("external_tracks", "coverage"),
    "fraction in [0,1]",
    "[0,1]",
    "scalar reliability",
    "Missing external-track values must be imputed or masked before use.",
    "Track coverage and quality metrics",
    implemented=SHAPE_IMPLEMENTED,
)

DNA_SHAPE_AXES = (
    _axis(
        1,
        "minor groove width",
        "DNA-shape track: minor groove width.",
        "scaled mean minor-groove-width track",
        ("DNA_shape_track",),
        "dataset-level z-score or documented min/max scaling",
        "external track",
        "DNA shape",
        "Requires external DNA-shape prediction or measurement.",
        "DNA-shape MGW tracks",
        implemented=SHAPE_IMPLEMENTED,
    ),
    _axis(
        2,
        "propeller twist",
        "DNA-shape track: propeller twist.",
        "scaled mean propeller-twist track",
        ("DNA_shape_track",),
        "dataset-level z-score or documented min/max scaling",
        "external track",
        "DNA shape",
        "Requires external DNA-shape prediction or measurement.",
        "DNA-shape ProT tracks",
        implemented=SHAPE_IMPLEMENTED,
    ),
    _axis(
        3,
        "helix twist",
        "DNA-shape track: helix twist.",
        "scaled mean helix-twist track",
        ("DNA_shape_track",),
        "dataset-level z-score or documented min/max scaling",
        "external track",
        "DNA shape",
        "Requires external DNA-shape prediction or measurement.",
        "DNA-shape HelT tracks",
        implemented=SHAPE_IMPLEMENTED,
    ),
    _axis(
        4,
        "roll",
        "DNA-shape track: roll.",
        "scaled mean roll track",
        ("DNA_shape_track",),
        "dataset-level z-score or documented min/max scaling",
        "external track",
        "DNA shape",
        "Requires external DNA-shape prediction or measurement.",
        "DNA-shape Roll tracks",
        implemented=SHAPE_IMPLEMENTED,
    ),
    _axis(
        5,
        "methylation/accessibility",
        "Optional epigenomic or accessibility track.",
        "scaled mean methylation or accessibility signal",
        ("external_epigenomic_track",),
        "track-specific scaling recorded in manifest",
        "external track",
        "functional track",
        "Requires explicit track provenance.",
        "Methylation, ATAC-seq, DNase-seq",
        implemented=SHAPE_IMPLEMENTED,
    ),
    _axis(
        6,
        "conservation",
        "Optional conservation or phylogenetic track.",
        "scaled mean conservation signal",
        ("external_conservation_track",),
        "track-specific scaling recorded in manifest",
        "external track",
        "evolutionary track",
        "Requires explicit track provenance.",
        "phyloP, phastCons, alignment conservation",
        implemented=SHAPE_IMPLEMENTED,
    ),
    _axis(
        7,
        "track confidence",
        "Confidence, coverage, or replicate support.",
        "mean confidence or support score",
        ("external_track_quality",),
        "track-specific scaling recorded in manifest",
        "external track",
        "reliability",
        "Requires explicit confidence definition.",
        "Coverage and replicate support",
        implemented=SHAPE_IMPLEMENTED,
    ),
)

PROTEIN_SCALAR = _axis(
    0,
    "valid amino-acid fraction",
    "Window mass/reliability scalar.",
    "valid standard-residue count divided by cleaned window length",
    ("window", "valid_residue_count", "cleaned_window_length"),
    "fraction in [0,1]",
    "[0,1]",
    "scalar reliability",
    "Window is skipped when ambiguous_fraction exceeds max_ambiguous_fraction.",
    "Valid-residue fraction and ambiguous-residue fraction",
)

PROTEIN_SEQUENCE_AXES = (
    _axis(
        1,
        "hydrophobicity",
        "Mean Kyte-Doolittle hydropathy proxy.",
        "mean(KD(residue) / 4.5)",
        ("residue_sequence", "HYDROPATHY"),
        "divide each residue score by 4.5 before averaging",
        "approximately [-1,+1]",
        "physicochemical",
        "Non-standard residues are excluded according to the run ambiguity policy.",
        "Hydrophobicity scales",
    ),
    _axis(
        2,
        "net charge",
        "Net charge proxy at a simple residue-class level.",
        "(K + R + 0.1*H - D - E) / L",
        ("residue_counts",),
        "divide by valid residue length L",
        "[-1,+1]",
        "physicochemical",
        "Non-standard residues are excluded according to the run ambiguity policy.",
        "Charge and isoelectric-point features",
    ),
    _axis(
        3,
        "polarity",
        "Mean polarity proxy.",
        "mean(POLARITY(residue))",
        ("residue_sequence", "POLARITY"),
        "average table-scaled residue values",
        "approximately [-1,+1]",
        "physicochemical",
        "Non-standard residues are excluded according to the run ambiguity policy.",
        "Polarity and physicochemical scales",
    ),
    _axis(
        4,
        "aromaticity",
        "Aromatic residue fraction.",
        "(F + W + Y + H) / L",
        ("residue_counts",),
        "fraction of valid residues",
        "[0,1]",
        "composition",
        "Non-standard residues are excluded according to the run ambiguity policy.",
        "Aromatic residue fraction",
    ),
    _axis(
        5,
        "residue volume",
        "Mean min/max scaled residue volume.",
        "mean(2 * (volume(residue) - min_volume) / (max_volume - min_volume) - 1)",
        ("residue_sequence", "RESIDUE_VOLUME"),
        "min/max scale residue volumes to [-1,+1] before averaging",
        "[-1,+1]",
        "physicochemical",
        "Non-standard residues are excluded according to the run ambiguity policy.",
        "Residue volume scales",
    ),
    _axis(
        6,
        "disorder/flexibility proxy",
        "Difference between disorder-promoting and order-promoting residue fractions.",
        "(P+E+S+K+Q+G)/L - (W+F+Y+I+L+V+C)/L",
        ("residue_counts",),
        "difference of residue-class fractions",
        "[-1,+1]",
        "structural proxy",
        "This is a crude sequence proxy, not a calibrated disorder predictor.",
        "IUPred/ESM disorder baselines where available",
    ),
    _axis(
        7,
        "repeat/low-complexity score",
        "One minus normalized amino-acid k-mer entropy.",
        "1 - H_k(window) / log2(20^k)",
        ("residue_sequence", "kmer_k"),
        "entropy complement using the full 20-residue k-mer alphabet",
        "[0,1]",
        "local complexity",
        "Invalid k-mers containing non-standard residues are excluded.",
        "SEG-style low-complexity and k-mer entropy",
    ),
)

CODON_PRODUCT_SCALAR = _axis(
    0,
    "base validity/product scalar",
    "Scalar part from ordered products of position-aware base octonions.",
    "scalar(((B1 * B2) * B3))",
    ("codon", "base_position_octonion", "Fano multiplication"),
    "raw product component; optional codon-level norm normalization can be requested",
    "open",
    "scalar product",
    "Ambiguous bases use zero chemistry and zero scalar; codon may be skipped by ambiguity policy.",
    "Codon validity and codon frequency",
)

CODON_PRODUCT_AXES = (
    _axis(
        1,
        "base RY property",
        "Codon base purine/pyrimidine property after ordered product.",
        "e1 component of ((B1 * B2) * B3)",
        ("codon", "BASE_PROPERTIES", "position_gates"),
        "raw product component unless --codon-normalize is used",
        "open",
        "base chemistry",
        "Ambiguous bases use zero chemistry.",
        "Codon composition and R/Y position features",
    ),
    _axis(
        2,
        "base SW property",
        "Codon base strong/weak property after ordered product.",
        "e2 component of ((B1 * B2) * B3)",
        ("codon", "BASE_PROPERTIES", "position_gates"),
        "raw product component unless --codon-normalize is used",
        "open",
        "base chemistry",
        "Ambiguous bases use zero chemistry.",
        "GC1/GC2/GC3 and codon composition",
    ),
    _axis(
        3,
        "base MK property",
        "Codon base amino/keto property after ordered product.",
        "e3 component of ((B1 * B2) * B3)",
        ("codon", "BASE_PROPERTIES", "position_gates"),
        "raw product component unless --codon-normalize is used",
        "open",
        "base chemistry",
        "Ambiguous bases use zero chemistry.",
        "Codon composition",
    ),
    _axis(
        4,
        "position-1 gate",
        "Position-aware first-base gate after ordered product.",
        "e4 component of ((B1 * B2) * B3)",
        ("codon", "position_gates"),
        "raw product component unless --codon-normalize is used",
        "open",
        "position",
        "Ambiguous bases still preserve position gates unless the codon is skipped.",
        "Position-specific base features",
    ),
    _axis(
        5,
        "position-2 gate",
        "Position-aware second-base gate after ordered product.",
        "e5 component of ((B1 * B2) * B3)",
        ("codon", "position_gates"),
        "raw product component unless --codon-normalize is used",
        "open",
        "position",
        "Ambiguous bases still preserve position gates unless the codon is skipped.",
        "Position-specific base features",
    ),
    _axis(
        6,
        "position-3 gate",
        "Position-aware third-base gate after ordered product.",
        "e6 component of ((B1 * B2) * B3)",
        ("codon", "position_gates"),
        "raw product component unless --codon-normalize is used",
        "open",
        "position",
        "Ambiguous bases still preserve position gates unless the codon is skipped.",
        "GC3 and third-position features",
    ),
    _axis(
        7,
        "wobble-position marker",
        "Third/wobble-position marker after ordered product.",
        "e7 component of ((B1 * B2) * B3)",
        ("codon", "position_gates"),
        "raw product component unless --codon-normalize is used",
        "open",
        "wobble",
        "Ambiguous bases still preserve position gates unless the codon is skipped.",
        "Wobble, RSCU, synonymous-family features",
    ),
)


AXIS_SCHEME_REGISTRY: dict[str, AxisScheme] = {
    "dna-window-v1": AxisScheme(
        scheme_id="dna-window-v1",
        name="DNA window descriptors v1",
        seq_type="dna",
        mode="window",
        version="1",
        status="stable",
        representation="algebraic-octonion",
        implemented=True,
        scalar_axis=DNA_WINDOW_SCALAR,
        imaginary_axes=DNA_WINDOW_AXES,
        fano_lines=DNA_WINDOW_LINES,
        recommended_use=(
            "Default reproducible DNA window mode for composition, complexity, "
            "and symmetry boundaries."
        ),
        limitations=(
            "Exploratory descriptor scheme; axis placement affects Fano-line "
            "interpretation."
        ),
    ),
    "dna-coding-v1": AxisScheme(
        scheme_id="dna-coding-v1",
        name="DNA coding descriptors v1",
        seq_type="dna",
        mode="window",
        version="1",
        status="experimental",
        representation="algebraic-octonion",
        implemented=False,
        scalar_axis=DNA_CODING_SCALAR,
        imaginary_axes=DNA_CODING_AXES,
        fano_lines=_generic_lines(
            {axis.index: axis.label for axis in DNA_CODING_AXES},
            "coding",
        ),
        recommended_use=(
            "Future CDS-oriented windows with GC3, frame periodicity, codon entropy, "
            "and ORF integrity."
        ),
        limitations="Definition only; extraction is not wired into fanoseq run yet.",
    ),
    "dna-regulatory-v1": AxisScheme(
        scheme_id="dna-regulatory-v1",
        name="DNA regulatory descriptors v1",
        seq_type="dna",
        mode="window",
        version="1",
        status="experimental",
        representation="algebraic-octonion",
        implemented=False,
        scalar_axis=DNA_REGULATORY_SCALAR,
        imaginary_axes=DNA_REGULATORY_AXES,
        fano_lines=_generic_lines(
            {axis.index: axis.label for axis in DNA_REGULATORY_AXES},
            "regulatory",
        ),
        recommended_use=(
            "Future regulatory-window analyses with CpG, palindrome, motif-density, "
            "and complexity proxies."
        ),
        limitations=(
            "Definition only; motif and palindrome extraction require validation "
            "and baselines."
        ),
    ),
    "dna-shape-v1": AxisScheme(
        scheme_id="dna-shape-v1",
        name="DNA shape and track descriptors v1",
        seq_type="dna",
        mode="track",
        version="1",
        status="planned",
        representation="eight-channel-tensor",
        implemented=False,
        scalar_axis=DNA_SHAPE_SCALAR,
        imaginary_axes=DNA_SHAPE_AXES,
        fano_lines=_generic_lines(
            {axis.index: axis.label for axis in DNA_SHAPE_AXES},
            "shape",
        ),
        recommended_use=(
            "Future DNA-shape or multi-track analyses using external shape/omics "
            "tracks."
        ),
        limitations="Planned definition; requires external track readers and scaling policies.",
    ),
    "protein-sequence-v1": AxisScheme(
        scheme_id="protein-sequence-v1",
        name="Protein sequence descriptors v1",
        seq_type="protein",
        mode="window",
        version="1",
        status="stable",
        representation="algebraic-octonion",
        implemented=True,
        scalar_axis=PROTEIN_SCALAR,
        imaginary_axes=PROTEIN_SEQUENCE_AXES,
        fano_lines=_generic_lines(
            {axis.index: axis.label for axis in PROTEIN_SEQUENCE_AXES},
            "protein",
        ),
        recommended_use=(
            "Default reproducible protein window mode for physicochemical "
            "sequence boundaries."
        ),
        limitations="Sequence-only descriptor; not a calibrated structure predictor.",
    ),
    "codon-product-v1": AxisScheme(
        scheme_id="codon-product-v1",
        name="Codon product descriptors v1",
        seq_type="dna",
        mode="codon",
        version="1",
        status="stable",
        representation="algebraic-octonion",
        implemented=True,
        scalar_axis=CODON_PRODUCT_SCALAR,
        imaginary_axes=CODON_PRODUCT_AXES,
        fano_lines=_generic_lines(
            {axis.index: axis.label for axis in CODON_PRODUCT_AXES},
            "codon",
        ),
        recommended_use="Default codon mode for ordered position-aware codon octonions.",
        limitations=(
            "Exploratory codon descriptor; not a replacement for standard "
            "codon-usage statistics."
        ),
    ),
}

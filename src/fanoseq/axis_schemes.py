"""Versioned axis-scheme registry for FanoSeq encodings."""

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
    value_hint: str
    role: str


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
    scalar_axis: AxisDefinition
    imaginary_axes: tuple[AxisDefinition, ...]
    fano_lines: tuple[FanoLineDefinition, ...]
    recommended_use: str
    limitations: str

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
                "recommended_use": scheme.recommended_use,
            }
        )
    return pd.DataFrame(rows)


def get_axis_scheme(scheme_id: str) -> AxisScheme:
    """Return one registered axis scheme."""
    try:
        return AXIS_SCHEME_REGISTRY[scheme_id]
    except KeyError as exc:
        choices = ", ".join(sorted(AXIS_SCHEME_REGISTRY))
        raise ValueError(f"Unknown axis scheme {scheme_id!r}. Available schemes: {choices}.") from exc


def default_axis_scheme_id(seq_type: str, mode: str) -> str:
    """Return the default scheme id for an existing FanoSeq context."""
    if mode == "codon":
        return "codon-product-v1"
    if seq_type == "dna":
        return "dna-window-v1"
    if seq_type == "protein":
        return "protein-sequence-v1"
    raise ValueError(f"Unsupported axis-scheme context: seq_type={seq_type!r}, mode={mode!r}.")


def axis_labels_for_context(seq_type: str, mode: str, scheme_id: str | None = None) -> dict[int, str]:
    """Return imaginary-axis labels for a mode/sequence context."""
    scheme = get_axis_scheme(scheme_id or default_axis_scheme_id(seq_type, mode))
    return scheme.axis_labels()


def axis_scheme_tables(scheme_id: str) -> dict[str, pd.DataFrame]:
    """Return metadata, axis, and Fano-line tables for one scheme."""
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
                "recommended_use": scheme.recommended_use,
                "limitations": scheme.limitations,
            }
        ]
    )
    axes = pd.DataFrame([_axis_row(scheme.scalar_axis)] + [_axis_row(axis) for axis in scheme.imaginary_axes])
    lines = pd.DataFrame(
        [
            {
                "scheme_id": scheme.scheme_id,
                "fano_line": f"({line.axes[0]},{line.axes[1]},{line.axes[2]})",
                "axis_a": line.axes[0],
                "axis_b": line.axes[1],
                "axis_c": line.axes[2],
                "axis_a_label": scheme.axis_labels()[line.axes[0]],
                "axis_b_label": scheme.axis_labels()[line.axes[1]],
                "axis_c_label": scheme.axis_labels()[line.axes[2]],
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
    }


def _axis_row(axis: AxisDefinition) -> dict[str, object]:
    return {
        "axis": axis.index,
        "symbol": axis.symbol,
        "label": axis.label,
        "description": axis.description,
        "value_hint": axis.value_hint,
        "role": axis.role,
    }


def _axis(index: int, label: str, description: str, value_hint: str, role: str) -> AxisDefinition:
    return AxisDefinition(
        index=index,
        symbol=f"e{index}",
        label=label,
        description=description,
        value_hint=value_hint,
        role=role,
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


DNA_WINDOW_AXES = (
    _axis(1, "purine/pyrimidine balance", "(A + G - C - T) / L.", "[-1,+1]", "base chemistry"),
    _axis(2, "GC/AT balance", "(G + C - A - T) / L.", "[-1,+1]", "base chemistry"),
    _axis(3, "amino/keto balance", "(A + C - G - T) / L.", "[-1,+1]", "base chemistry"),
    _axis(4, "GC skew", "(G - C) / (G + C + epsilon).", "[-1,+1]", "strand/composition asymmetry"),
    _axis(5, "AT skew", "(A - T) / (A + T + epsilon).", "[-1,+1]", "strand/composition asymmetry"),
    _axis(6, "k-mer entropy", "Normalized k-mer entropy for the window.", "[0,1]", "local complexity"),
    _axis(7, "reverse-complement symmetry", "2*RC_similarity - 1.", "[-1,+1]", "sequence symmetry"),
)

DNA_WINDOW_LINES = (
    _line((1, 2, 3), "base chemistry triad", "Joint RY, GC/AT, and MK chemistry balance."),
    _line((1, 4, 5), "RY-skew asymmetry triad", "Purine/pyrimidine balance coupled to GC and AT skew."),
    _line((1, 7, 6), "RY-symmetry-complexity triad", "Purine/pyrimidine balance coupled to reverse-complement symmetry and k-mer entropy."),
    _line((2, 4, 6), "GC-skew-complexity triad", "GC/AT balance coupled to GC skew and local sequence complexity."),
    _line((2, 5, 7), "GC-ATskew-symmetry triad", "GC/AT balance coupled to AT skew and reverse-complement symmetry."),
    _line((3, 4, 7), "MK-GCskew-symmetry triad", "Amino/keto balance coupled to GC skew and reverse-complement symmetry."),
    _line((3, 6, 5), "MK-complexity-ATskew triad", "Amino/keto balance coupled to k-mer entropy and AT skew."),
)

DNA_CODING_AXES = (
    _axis(1, "purine/pyrimidine balance", "Coding-window RY balance.", "[-1,+1]", "base chemistry"),
    _axis(2, "GC/AT balance", "Coding-window GC/AT balance.", "[-1,+1]", "base chemistry"),
    _axis(3, "amino/keto balance", "Coding-window MK balance.", "[-1,+1]", "base chemistry"),
    _axis(4, "GC3 excess", "Third-position GC signal relative to local GC.", "open", "codon bias"),
    _axis(5, "period-3 frame signal", "Strength of frame-periodic composition.", "[0,+inf)", "coding periodicity"),
    _axis(6, "codon entropy/RSCU dispersion", "Codon usage complexity or synonymous dispersion.", "[0,+inf)", "codon usage"),
    _axis(7, "wobble/ORF integrity", "Wobble stability, stop/start, or ORF-integrity proxy.", "scheme-dependent", "coding integrity"),
)

DNA_REGULATORY_AXES = (
    _axis(1, "purine/pyrimidine balance", "Regulatory-window RY balance.", "[-1,+1]", "base chemistry"),
    _axis(2, "GC/AT balance", "Regulatory-window GC/AT balance.", "[-1,+1]", "base chemistry"),
    _axis(3, "amino/keto balance", "Regulatory-window MK balance.", "[-1,+1]", "base chemistry"),
    _axis(4, "CpG observed/expected", "Observed CpG density relative to expectation.", "[0,+inf)", "regulatory composition"),
    _axis(5, "palindrome/inverted-repeat density", "Local inverted-repeat or palindrome proxy.", "[0,1]", "motif symmetry"),
    _axis(6, "k-mer entropy", "Regulatory-window sequence complexity.", "[0,1]", "local complexity"),
    _axis(7, "motif-density/AT-rich proxy", "Motif-density or AT-rich regulatory proxy.", "scheme-dependent", "regulatory signal"),
)

DNA_SHAPE_AXES = (
    _axis(1, "minor groove width", "DNA-shape track: minor groove width.", "external track", "DNA shape"),
    _axis(2, "propeller twist", "DNA-shape track: propeller twist.", "external track", "DNA shape"),
    _axis(3, "helix twist", "DNA-shape track: helix twist.", "external track", "DNA shape"),
    _axis(4, "roll", "DNA-shape track: roll.", "external track", "DNA shape"),
    _axis(5, "methylation/accessibility", "Optional epigenomic or accessibility track.", "external track", "functional track"),
    _axis(6, "conservation", "Optional conservation or phylogenetic track.", "external track", "evolutionary track"),
    _axis(7, "track confidence", "Confidence, coverage, or replicate support.", "external track", "reliability"),
)

PROTEIN_SEQUENCE_AXES = (
    _axis(1, "hydrophobicity", "Mean hydrophobicity scale.", "scaled", "physicochemical"),
    _axis(2, "net charge", "Net charge proxy.", "scaled", "physicochemical"),
    _axis(3, "polarity", "Mean polarity proxy.", "scaled", "physicochemical"),
    _axis(4, "aromaticity", "Aromatic residue fraction.", "[0,1]", "composition"),
    _axis(5, "residue volume", "Approximate residue volume.", "scaled", "physicochemical"),
    _axis(6, "disorder/flexibility proxy", "Disorder-promoting minus order-promoting residue signal.", "scaled", "structural proxy"),
    _axis(7, "repeat/low-complexity score", "One minus normalized k-mer entropy.", "[0,1]", "local complexity"),
)

CODON_PRODUCT_AXES = (
    _axis(1, "base RY property", "Codon base purine/pyrimidine property.", "{-1,+1}", "base chemistry"),
    _axis(2, "base SW property", "Codon base strong/weak property.", "{-1,+1}", "base chemistry"),
    _axis(3, "base MK property", "Codon base amino/keto property.", "{-1,+1}", "base chemistry"),
    _axis(4, "position-1 gate", "Position-aware gate for the first codon base.", "{0,1}", "position"),
    _axis(5, "position-2 gate", "Position-aware gate for the second codon base.", "{0,1}", "position"),
    _axis(6, "position-3 gate", "Position-aware gate for the third codon base.", "{0,1}", "position"),
    _axis(7, "wobble-position marker", "Marker for the third/wobble codon position.", "{0,1}", "wobble"),
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
        scalar_axis=_axis(0, "valid fraction", "Window mass/reliability scalar.", "[0,1]", "scalar reliability"),
        imaginary_axes=DNA_WINDOW_AXES,
        fano_lines=DNA_WINDOW_LINES,
        recommended_use="Default reproducible DNA window mode for composition, complexity, and symmetry boundaries.",
        limitations="Exploratory descriptor scheme; axis placement affects Fano-line interpretation.",
    ),
    "dna-coding-v1": AxisScheme(
        scheme_id="dna-coding-v1",
        name="DNA coding descriptors v1",
        seq_type="dna",
        mode="window",
        version="1",
        status="experimental",
        representation="algebraic-octonion",
        scalar_axis=_axis(0, "valid coding fraction", "Valid coding-window reliability scalar.", "[0,1]", "scalar reliability"),
        imaginary_axes=DNA_CODING_AXES,
        fano_lines=_generic_lines({axis.index: axis.label for axis in DNA_CODING_AXES}, "coding"),
        recommended_use="Future CDS-oriented windows with GC3, frame periodicity, codon bias, and ORF integrity.",
        limitations="Registry definition only; extraction is not wired into fanoseq run yet.",
    ),
    "dna-regulatory-v1": AxisScheme(
        scheme_id="dna-regulatory-v1",
        name="DNA regulatory descriptors v1",
        seq_type="dna",
        mode="window",
        version="1",
        status="experimental",
        representation="algebraic-octonion",
        scalar_axis=_axis(0, "valid fraction", "Window mass/reliability scalar.", "[0,1]", "scalar reliability"),
        imaginary_axes=DNA_REGULATORY_AXES,
        fano_lines=_generic_lines({axis.index: axis.label for axis in DNA_REGULATORY_AXES}, "regulatory"),
        recommended_use="Future regulatory-window analyses with CpG, palindrome, motif-density, and complexity proxies.",
        limitations="Registry definition only; motif and palindrome extraction require validation and baselines.",
    ),
    "dna-shape-v1": AxisScheme(
        scheme_id="dna-shape-v1",
        name="DNA shape and track descriptors v1",
        seq_type="dna",
        mode="track",
        version="1",
        status="planned",
        representation="eight-channel-tensor",
        scalar_axis=_axis(0, "track coverage/reliability", "Coverage or confidence scalar for external tracks.", "[0,1]", "scalar reliability"),
        imaginary_axes=DNA_SHAPE_AXES,
        fano_lines=_generic_lines({axis.index: axis.label for axis in DNA_SHAPE_AXES}, "shape"),
        recommended_use="Future DNA-shape or multi-track analyses using external shape/omics tracks.",
        limitations="Planned registry entry; requires external track readers and scaling policies.",
    ),
    "protein-sequence-v1": AxisScheme(
        scheme_id="protein-sequence-v1",
        name="Protein sequence descriptors v1",
        seq_type="protein",
        mode="window",
        version="1",
        status="stable",
        representation="algebraic-octonion",
        scalar_axis=_axis(0, "valid amino-acid fraction", "Window mass/reliability scalar.", "[0,1]", "scalar reliability"),
        imaginary_axes=PROTEIN_SEQUENCE_AXES,
        fano_lines=_generic_lines({axis.index: axis.label for axis in PROTEIN_SEQUENCE_AXES}, "protein"),
        recommended_use="Default reproducible protein window mode for physicochemical sequence boundaries.",
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
        scalar_axis=_axis(0, "base validity/product scalar", "Scalar part from position-aware base products.", "open", "scalar product"),
        imaginary_axes=CODON_PRODUCT_AXES,
        fano_lines=_generic_lines({axis.index: axis.label for axis in CODON_PRODUCT_AXES}, "codon"),
        recommended_use="Default codon mode for ordered position-aware codon octonions.",
        limitations="Exploratory codon descriptor; not a replacement for standard codon-usage statistics.",
    ),
}

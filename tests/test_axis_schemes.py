import pytest

from fanoseq.axis_schemes import (
    axis_labels_for_context,
    axis_scheme_tables,
    default_axis_scheme_id,
    get_axis_scheme,
    list_axis_schemes,
)
from fanoseq.fano_attribution import axis_labels


def test_axis_scheme_registry_contains_requested_schemes() -> None:
    table = list_axis_schemes()
    expected = {
        "dna-window-v1",
        "dna-coding-v1",
        "dna-regulatory-v1",
        "dna-shape-v1",
        "protein-sequence-v1",
    }
    assert expected.issubset(set(table["scheme_id"]))


def test_dna_window_scheme_defines_axes_and_fano_lines() -> None:
    scheme = get_axis_scheme("dna-window-v1")
    assert scheme.status == "stable"
    assert scheme.scalar_axis.label == "valid fraction"
    assert scheme.axis_labels()[6] == "k-mer entropy"
    assert len(scheme.fano_lines) == 7
    assert scheme.line_label((1, 7, 6)) == "RY-symmetry-complexity triad"


def test_axis_scheme_tables_are_exportable() -> None:
    tables = axis_scheme_tables("dna-window-v1")
    assert {"axis_scheme_metadata", "axis_scheme_axes", "axis_scheme_fano_lines"} == set(tables)
    assert len(tables["axis_scheme_axes"]) == 8
    assert len(tables["axis_scheme_fano_lines"]) == 7
    assert "line_label" in tables["axis_scheme_fano_lines"].columns


def test_default_axis_scheme_contexts() -> None:
    assert default_axis_scheme_id("dna", "window") == "dna-window-v1"
    assert default_axis_scheme_id("protein", "window") == "protein-sequence-v1"
    assert default_axis_scheme_id("dna", "codon") == "codon-product-v1"


def test_fano_attribution_uses_axis_scheme_registry() -> None:
    assert axis_labels_for_context("dna", "window")[7] == "reverse-complement symmetry"
    assert axis_labels("codon", "dna")[7] == "wobble-position marker"


def test_unknown_axis_scheme_raises_useful_error() -> None:
    with pytest.raises(ValueError, match="Unknown axis scheme"):
        get_axis_scheme("not-a-scheme")

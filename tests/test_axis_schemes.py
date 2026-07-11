import pytest

from fanoseq.axis_schemes import (
    axis_labels_for_context,
    axis_scheme_tables,
    default_axis_scheme_id,
    get_axis_scheme,
    list_axis_definitions,
    list_axis_schemes,
    resolve_axis_scheme,
    validate_axis_scheme_definitions,
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
    assert scheme.scalar_axis.formula == "valid A/C/G/T count divided by cleaned window length"
    assert scheme.axis_labels()[6] == "k-mer entropy"
    assert len(scheme.fano_lines) == 7
    assert scheme.line_label((1, 7, 6)) == "RY-symmetry-complexity triad"


def test_axis_scheme_tables_are_exportable() -> None:
    tables = axis_scheme_tables("dna-window-v1")
    assert {
        "axis_scheme_metadata",
        "axis_scheme_axes",
        "axis_scheme_fano_lines",
        "axis_scheme_validation",
    } == set(tables)
    assert len(tables["axis_scheme_axes"]) == 8
    assert len(tables["axis_scheme_fano_lines"]) == 7
    assert "line_label" in tables["axis_scheme_fano_lines"].columns
    assert "formula" in tables["axis_scheme_axes"].columns
    assert tables["axis_scheme_validation"]["passed"].all()


def test_axis_definitions_are_complete_and_validated() -> None:
    definitions = list_axis_definitions("dna-coding-v1")
    assert len(definitions) == 8
    assert definitions.loc[definitions["symbol"] == "e4", "formula"].item().startswith("mean(GC")
    assert definitions["implemented"].eq(False).all()

    validation = validate_axis_scheme_definitions()
    assert validation.groupby("scheme_id")["passed"].all().all()


def test_default_axis_scheme_contexts() -> None:
    assert default_axis_scheme_id("dna", "window") == "dna-window-v1"
    assert default_axis_scheme_id("protein", "window") == "protein-sequence-v1"
    assert default_axis_scheme_id("dna", "codon") == "codon-product-v1"
    assert resolve_axis_scheme("dna", "window").scheme_id == "dna-window-v1"


def test_non_runnable_axis_scheme_is_rejected_when_required() -> None:
    with pytest.raises(ValueError, match="not implemented by fanoseq run"):
        resolve_axis_scheme("dna", "window", "dna-coding-v1", require_runnable=True)


def test_fano_attribution_uses_axis_scheme_registry() -> None:
    assert axis_labels_for_context("dna", "window")[7] == "reverse-complement symmetry"
    assert axis_labels("codon", "dna")[7] == "wobble-position marker"


def test_unknown_axis_scheme_raises_useful_error() -> None:
    with pytest.raises(ValueError, match="Unknown axis scheme"):
        get_axis_scheme("not-a-scheme")

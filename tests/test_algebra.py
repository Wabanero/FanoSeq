import numpy as np

from fanoseq.algebra import (
    basis_octonion,
    basis_product,
    left_multiplication_matrix,
    multiplication_table,
    multiply_with_structure_constants,
    right_multiplication_matrix,
    structure_constants,
    validation_passed,
    validation_report,
)
from fanoseq.octonion import FANO_LINES, Octonion


def test_basis_product_table_matches_fano_convention() -> None:
    product = basis_product(1, 2)
    assert product.sign == 1
    assert product.basis == 3
    reversed_product = basis_product(2, 1)
    assert reversed_product.sign == -1
    assert reversed_product.basis == 3
    square = basis_product(4, 4)
    assert square.sign == -1
    assert square.basis == 0


def test_structure_constants_reproduce_octonion_multiplication() -> None:
    constants = structure_constants()
    assert constants.shape == (8, 8, 8)
    x = Octonion([0.2, 1.0, -0.5, 0.3, 0.1, -0.2, 0.7, 0.4])
    y = Octonion([1.0, -0.3, 0.8, 0.2, -0.4, 0.5, 0.6, -0.1])
    assert np.allclose(multiply_with_structure_constants(x.components, y.components), (x * y).components)


def test_left_and_right_multiplication_matrices_match_direct_product() -> None:
    x = Octonion([0.2, 1.0, -0.5, 0.3, 0.1, -0.2, 0.7, 0.4])
    y = Octonion([1.0, -0.3, 0.8, 0.2, -0.4, 0.5, 0.6, -0.1])
    direct = (x * y).components
    assert np.allclose(left_multiplication_matrix(x.components) @ y.components, direct)
    assert np.allclose(right_multiplication_matrix(y.components) @ x.components, direct)


def test_validation_report_passes_and_checks_same_line_associators() -> None:
    report = validation_report()
    assert validation_passed(report)
    assert report["passed"].all()
    for a, b, c in FANO_LINES:
        associator = basis_octonion(a).associator(basis_octonion(b), basis_octonion(c))
        assert np.isclose(associator.norm(), 0.0)


def test_multiplication_table_has_full_basis_grid() -> None:
    table = multiplication_table()
    assert len(table) == 64
    assert {"left", "right", "sign", "basis", "product"}.issubset(table.columns)

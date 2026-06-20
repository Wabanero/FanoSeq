"""Validated octonion algebra representations for FanoSeq."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from fanoseq.octonion import FANO_LINES, Octonion, multiply_components


@dataclass(frozen=True)
class BasisProduct:
    """Product of two basis units: e_left * e_right = sign * e_basis."""

    left: int
    right: int
    sign: int
    basis: int


def basis_octonion(index: int) -> Octonion:
    """Return the basis octonion e_index."""
    if index < 0 or index > 7:
        raise ValueError("basis index must be between 0 and 7.")
    components = np.zeros(8, dtype=float)
    components[index] = 1.0
    return Octonion(components)


def basis_product(left: int, right: int) -> BasisProduct:
    """Return the signed basis product for e_left * e_right."""
    if left < 0 or left > 7 or right < 0 or right > 7:
        raise ValueError("basis indices must be between 0 and 7.")
    if left == 0:
        return BasisProduct(left=left, right=right, sign=1, basis=right)
    if right == 0:
        return BasisProduct(left=left, right=right, sign=1, basis=left)
    if left == right:
        return BasisProduct(left=left, right=right, sign=-1, basis=0)
    for a, b, c in FANO_LINES:
        for x, y, z in ((a, b, c), (b, c, a), (c, a, b)):
            if left == x and right == y:
                return BasisProduct(left=left, right=right, sign=1, basis=z)
            if left == y and right == x:
                return BasisProduct(left=left, right=right, sign=-1, basis=z)
    raise ValueError(f"No Fano product found for e{left} * e{right}.")


def multiplication_table() -> pd.DataFrame:
    """Return the full 8x8 signed basis multiplication table."""
    rows = []
    for left in range(8):
        for right in range(8):
            product = basis_product(left, right)
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "sign": product.sign,
                    "basis": product.basis,
                    "product": f"{'-' if product.sign < 0 else ''}e{product.basis}",
                }
            )
    return pd.DataFrame(rows)


def structure_constants() -> NDArray[np.float64]:
    """Return A[i,j,k] such that (xy)_k = sum_ij A[i,j,k] x_i y_j."""
    constants = np.zeros((8, 8, 8), dtype=float)
    for left in range(8):
        for right in range(8):
            product = basis_product(left, right)
            constants[left, right, product.basis] = float(product.sign)
    return constants


def multiply_with_structure_constants(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Multiply two component vectors using the explicit structure tensor."""
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.shape != (8,) or y_values.shape != (8,):
        raise ValueError("Octonion component arrays must have shape (8,).")
    constants = structure_constants()
    result = np.zeros(8, dtype=float)
    for i in range(8):
        for j in range(8):
            if x_values[i] == 0.0 or y_values[j] == 0.0:
                continue
            result += constants[i, j, :] * x_values[i] * y_values[j]
    return result


def left_multiplication_matrix(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return L_x where L_x @ y equals x*y."""
    x_values = np.asarray(x, dtype=float)
    if x_values.shape != (8,):
        raise ValueError("Octonion component arrays must have shape (8,).")
    matrix = np.zeros((8, 8), dtype=float)
    constants = structure_constants()
    for row in range(8):
        for column in range(8):
            matrix[row, column] = float(np.sum(x_values * constants[:, column, row]))
    return matrix


def right_multiplication_matrix(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return R_y where R_y @ x equals x*y."""
    y_values = np.asarray(y, dtype=float)
    if y_values.shape != (8,):
        raise ValueError("Octonion component arrays must have shape (8,).")
    matrix = np.zeros((8, 8), dtype=float)
    constants = structure_constants()
    for row in range(8):
        for column in range(8):
            matrix[row, column] = float(np.sum(y_values * constants[column, :, row]))
    return matrix


def validation_report(tolerance: float = 1e-12) -> pd.DataFrame:
    """Return basis-level validation checks for the project Fano convention."""
    rows: list[dict[str, object]] = []
    rows.extend(_validate_basis_squares(tolerance))
    rows.extend(_validate_oriented_lines(tolerance))
    rows.extend(_validate_scalar_identity(tolerance))
    rows.extend(_validate_structure_tensor(tolerance))
    rows.extend(_validate_operator_matrices(tolerance))
    rows.extend(_validate_associators(tolerance))
    rows.extend(_validate_norm_multiplicativity(tolerance))
    return pd.DataFrame(rows)


def validation_passed(report: pd.DataFrame) -> bool:
    """Return True if all validation report rows passed."""
    return bool(not report.empty and report["passed"].all())


def _check(name: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": name, "passed": bool(passed), "detail": detail}


def _validate_basis_squares(tolerance: float) -> list[dict[str, object]]:
    rows = []
    expected = -basis_octonion(0).components
    for index in range(1, 8):
        actual = (basis_octonion(index) * basis_octonion(index)).components
        rows.append(
            _check(
                f"basis_square_e{index}",
                bool(np.allclose(actual, expected, atol=tolerance)),
                f"e{index}*e{index}=-e0",
            )
        )
    return rows


def _validate_oriented_lines(tolerance: float) -> list[dict[str, object]]:
    rows = []
    for a, b, c in FANO_LINES:
        for left, right, out in ((a, b, c), (b, c, a), (c, a, b)):
            actual = (basis_octonion(left) * basis_octonion(right)).components
            expected = basis_octonion(out).components
            rows.append(
                _check(
                    f"oriented_e{left}_e{right}",
                    bool(np.allclose(actual, expected, atol=tolerance)),
                    f"e{left}*e{right}=e{out}",
                )
            )
            reversed_actual = (basis_octonion(right) * basis_octonion(left)).components
            rows.append(
                _check(
                    f"reversed_e{right}_e{left}",
                    bool(np.allclose(reversed_actual, -expected, atol=tolerance)),
                    f"e{right}*e{left}=-e{out}",
                )
            )
    return rows


def _validate_scalar_identity(tolerance: float) -> list[dict[str, object]]:
    rows = []
    scalar = basis_octonion(0)
    for index in range(8):
        basis = basis_octonion(index)
        rows.append(
            _check(
                f"left_identity_e{index}",
                bool(np.allclose((scalar * basis).components, basis.components, atol=tolerance)),
                f"e0*e{index}=e{index}",
            )
        )
        rows.append(
            _check(
                f"right_identity_e{index}",
                bool(np.allclose((basis * scalar).components, basis.components, atol=tolerance)),
                f"e{index}*e0=e{index}",
            )
        )
    return rows


def _validate_structure_tensor(tolerance: float) -> list[dict[str, object]]:
    x = np.array([0.5, 1.0, -0.2, 0.3, -0.7, 0.9, 0.4, -0.1], dtype=float)
    y = np.array([-0.3, 0.8, 0.1, -0.5, 0.2, -0.4, 0.6, 0.7], dtype=float)
    direct = multiply_components(x, y)
    tensor = multiply_with_structure_constants(x, y)
    return [
        _check(
            "structure_constants_match_direct_multiply",
            bool(np.allclose(direct, tensor, atol=tolerance)),
            "A[i,j,k] reproduces multiply_components",
        )
    ]


def _validate_operator_matrices(tolerance: float) -> list[dict[str, object]]:
    x = np.array([0.5, 1.0, -0.2, 0.3, -0.7, 0.9, 0.4, -0.1], dtype=float)
    y = np.array([-0.3, 0.8, 0.1, -0.5, 0.2, -0.4, 0.6, 0.7], dtype=float)
    direct = multiply_components(x, y)
    left = left_multiplication_matrix(x) @ y
    right = right_multiplication_matrix(y) @ x
    return [
        _check(
            "left_multiplication_matrix_matches_direct",
            bool(np.allclose(direct, left, atol=tolerance)),
            "L_x @ y equals x*y",
        ),
        _check(
            "right_multiplication_matrix_matches_direct",
            bool(np.allclose(direct, right, atol=tolerance)),
            "R_y @ x equals x*y",
        ),
    ]


def _validate_associators(tolerance: float) -> list[dict[str, object]]:
    rows = []
    for a, b, c in FANO_LINES:
        associator = basis_octonion(a).associator(basis_octonion(b), basis_octonion(c))
        rows.append(
            _check(
                f"same_line_associator_{a}_{b}_{c}",
                bool(associator.norm() <= tolerance),
                f"[e{a},e{b},e{c}]=0 on one Fano line",
            )
        )
    cross = basis_octonion(1).associator(basis_octonion(2), basis_octonion(5))
    rows.append(
        _check(
            "cross_line_associator_nonzero",
            bool(cross.norm() > tolerance),
            "[e1,e2,e5] is nonzero",
        )
    )
    x = Octonion([0.5, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    y = Octonion([1.0, -0.3, 0.2, -0.5, 0.6, 0.1, -0.8, 0.4])
    rows.append(
        _check(
            "left_alternativity",
            bool(x.associator(x, y).norm() <= tolerance),
            "[x,x,y]=0 for validation vector",
        )
    )
    rows.append(
        _check(
            "right_alternativity",
            bool(x.associator(y, y).norm() <= tolerance),
            "[x,y,y]=0 for validation vector",
        )
    )
    return rows


def _validate_norm_multiplicativity(tolerance: float) -> list[dict[str, object]]:
    x = Octonion([0.5, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])
    y = Octonion([1.0, -0.3, 0.2, -0.5, 0.6, 0.1, -0.8, 0.4])
    product_norm = (x * y).norm()
    expected_norm = x.norm() * y.norm()
    return [
        _check(
            "norm_multiplicativity",
            bool(abs(product_norm - expected_norm) <= tolerance),
            "||xy|| = ||x|| ||y|| for validation vectors",
        )
    ]

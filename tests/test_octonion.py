import numpy as np

from fanoseq.octonion import Octonion
from fanoseq.octonion_numba import (
    octonion_associator,
    octonion_commutator,
    octonion_multiply,
)


def basis(index: int) -> Octonion:
    values = np.zeros(8)
    values[index] = 1.0
    return Octonion(values)


def test_basis_multiplication_convention() -> None:
    assert np.allclose((basis(1) * basis(2)).components, basis(3).components)
    assert np.allclose((basis(2) * basis(1)).components, -basis(3).components)
    assert np.allclose((basis(1) * basis(1)).components, -basis(0).components)


def test_norm_conjugate_and_inverse() -> None:
    x = Octonion([1, 2, 0, 0, 0, 0, 0, 0])
    assert np.isclose(x.norm(), np.sqrt(5))
    assert np.allclose(x.conjugate().components, [1, -2, 0, 0, 0, 0, 0, 0])
    identity = x * x.inverse()
    assert np.allclose(identity.components, basis(0).components)


def test_non_commutativity_and_non_associativity() -> None:
    assert not np.allclose((basis(1) * basis(2)).components, (basis(2) * basis(1)).components)
    associator = basis(1).associator(basis(2), basis(5))
    assert associator.norm() > 0


def test_numba_kernels_match_pure_octonions() -> None:
    x = Octonion([0.2, 1.0, -0.5, 0.3, 0.1, -0.2, 0.7, 0.4])
    y = Octonion([1.0, -0.3, 0.8, 0.2, -0.4, 0.5, 0.6, -0.1])
    z = Octonion([0.7, 0.2, 0.4, -0.9, 0.3, 0.1, -0.2, 0.5])

    assert np.allclose(octonion_multiply(x.components, y.components), (x * y).components)
    assert np.allclose(octonion_commutator(x.components, y.components), x.commutator(y).components)
    assert np.allclose(octonion_associator(x.components, y.components, z.components), x.associator(y, z).components)


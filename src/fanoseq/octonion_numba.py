"""Numba-accelerated octonion kernels with pure-Python fallback."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

_numba: Any = None
try:  # pragma: no cover - behavior is exercised through the public functions.
    import numba as _imported_numba
except Exception:  # pragma: no cover
    pass
else:  # pragma: no cover
    _numba = _imported_numba


def _optional_njit(*args: object, **kwargs: object) -> Any:
    """Use Numba's decorator when available, otherwise return the function unchanged."""
    if _numba is not None:
        return _numba.njit(*args, **kwargs)

    def decorator(func: object) -> object:
        return func

    if args and callable(args[0]) and not kwargs:
        return args[0]
    return decorator


@_optional_njit(cache=True)
def _basis_product(left: int, right: int) -> tuple[int, int]:
    if left == right:
        return -1, 0
    triples = (
        (1, 2, 3),
        (1, 4, 5),
        (1, 7, 6),
        (2, 4, 6),
        (2, 5, 7),
        (3, 4, 7),
        (3, 6, 5),
    )
    for a, b, c in triples:
        if (left == a and right == b) or (left == b and right == c) or (
            left == c and right == a
        ):
            if left == a and right == b:
                return 1, c
            if left == b and right == c:
                return 1, a
            return 1, b
        if (left == b and right == a) or (left == c and right == b) or (
            left == a and right == c
        ):
            if left == b and right == a:
                return -1, c
            if left == c and right == b:
                return -1, a
            return -1, b
    return 0, 0


@_optional_njit(cache=True)
def octonion_multiply(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Multiply two length-8 octonion component arrays."""
    out = np.zeros(8, dtype=np.float64)
    out[0] = x[0] * y[0]
    for i in range(1, 8):
        out[0] -= x[i] * y[i]
        out[i] += x[0] * y[i] + x[i] * y[0]

    for i in range(1, 8):
        for j in range(1, 8):
            if i != j:
                sign, basis = _basis_product(i, j)
                out[basis] += sign * x[i] * y[j]
    return out


@_optional_njit(cache=True)
def octonion_commutator(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return x*y - y*x."""
    return octonion_multiply(x, y) - octonion_multiply(y, x)


@_optional_njit(cache=True)
def octonion_associator(
    x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return (x*y)*z - x*(y*z)."""
    return octonion_multiply(octonion_multiply(x, y), z) - octonion_multiply(
        x, octonion_multiply(y, z)
    )


@_optional_njit(cache=True)
def batch_adjacent_products(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return products for consecutive rows in a component matrix."""
    n = values.shape[0]
    out = np.zeros((max(n - 1, 0), 8), dtype=np.float64)
    for i in range(n - 1):
        out[i, :] = octonion_multiply(values[i, :], values[i + 1, :])
    return out


@_optional_njit(cache=True)
def batch_adjacent_commutator_scores(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return commutator norms for consecutive rows in a component matrix."""
    n = values.shape[0]
    out = np.zeros(max(n - 1, 0), dtype=np.float64)
    for i in range(n - 1):
        comm = octonion_commutator(values[i, :], values[i + 1, :])
        total = 0.0
        for j in range(8):
            total += comm[j] * comm[j]
        out[i] = np.sqrt(total)
    return out


@_optional_njit(cache=True)
def batch_triplet_associator_scores(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return associator norms for consecutive triplets in a component matrix."""
    n = values.shape[0]
    out = np.zeros(max(n - 2, 0), dtype=np.float64)
    for i in range(n - 2):
        assoc = octonion_associator(values[i, :], values[i + 1, :], values[i + 2, :])
        total = 0.0
        for j in range(8):
            total += assoc[j] * assoc[j]
        out[i] = np.sqrt(total)
    return out

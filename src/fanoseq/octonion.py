"""Pure NumPy octonion algebra using a fixed oriented Fano-plane convention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

FANO_LINES: tuple[tuple[int, int, int], ...] = (
    (1, 2, 3),
    (1, 4, 5),
    (1, 7, 6),
    (2, 4, 6),
    (2, 5, 7),
    (3, 4, 7),
    (3, 6, 5),
)


def _multiplication_table() -> dict[tuple[int, int], tuple[int, int]]:
    table: dict[tuple[int, int], tuple[int, int]] = {}
    for a, b, c in FANO_LINES:
        cyclic = ((a, b, c), (b, c, a), (c, a, b))
        for left, right, out in cyclic:
            table[(left, right)] = (1, out)
            table[(right, left)] = (-1, out)
    return table


_IMAGINARY_TABLE = _multiplication_table()


def multiply_components(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Multiply two component arrays under the project Fano convention."""
    if x.shape != (8,) or y.shape != (8,):
        raise ValueError("Octonion component arrays must have shape (8,).")

    result = np.zeros(8, dtype=float)
    result[0] = x[0] * y[0] - float(np.dot(x[1:], y[1:]))
    result[1:] = x[0] * y[1:] + y[0] * x[1:]

    for i in range(1, 8):
        for j in range(1, 8):
            if i == j:
                continue
            sign, basis = _IMAGINARY_TABLE[(i, j)]
            result[basis] += sign * x[i] * y[j]
    return result


@dataclass(frozen=True)
class Octonion:
    """An octonion represented by eight real components in basis e0...e7."""

    components: NDArray[np.float64]

    def __init__(self, components: Iterable[float] | NDArray[np.float64]) -> None:
        values = np.asarray(list(components), dtype=float)
        if values.shape != (8,):
            raise ValueError("Octonion requires exactly 8 components.")
        object.__setattr__(self, "components", values)

    def __add__(self, other: Self) -> Self:
        return cast(Self, Octonion(self.components + other.components))

    def __sub__(self, other: Self) -> Self:
        return cast(Self, Octonion(self.components - other.components))

    def __mul__(self, other: Self | float) -> Self:
        if isinstance(other, Octonion):
            return cast(Self, Octonion(multiply_components(self.components, other.components)))
        return cast(Self, Octonion(self.components * float(other)))

    def __rmul__(self, other: float) -> Self:
        return cast(Self, Octonion(float(other) * self.components))

    def __truediv__(self, scalar: float) -> Self:
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide an octonion by zero.")
        return cast(Self, Octonion(self.components / scalar))

    def add(self, other: Self) -> Self:
        """Return self + other."""
        return self + other

    def sub(self, other: Self) -> Self:
        """Return self - other."""
        return self - other

    def mul(self, other: Self) -> Self:
        """Return the octonion product self * other."""
        return self * other

    def conjugate(self) -> Self:
        """Return the octonion conjugate."""
        values = self.components.copy()
        values[1:] *= -1.0
        return cast(Self, Octonion(values))

    def norm(self) -> float:
        """Return the Euclidean norm of the component vector."""
        return float(np.linalg.norm(self.components))

    def inverse(self) -> Self:
        """Return the multiplicative inverse, if the norm is non-zero."""
        norm_sq = float(np.dot(self.components, self.components))
        if norm_sq == 0:
            raise ZeroDivisionError("Cannot invert a zero-norm octonion.")
        return self.conjugate() / norm_sq

    def to_list(self) -> list[float]:
        """Return components as a plain Python list."""
        return [float(x) for x in self.components]

    @classmethod
    def from_list(cls, components: Iterable[float]) -> Self:
        """Build an octonion from a list-like object."""
        return cls(components)

    def commutator(self, other: Self) -> Self:
        """Return [self, other] = self * other - other * self."""
        return (self * other) - (other * self)

    def associator(self, y: Self, z: Self) -> Self:
        """Return [self, y, z] = (self * y) * z - self * (y * z)."""
        return (self * y) * z - self * (y * z)

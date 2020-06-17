"""Mathematical Analysis library"""

from __future__ import annotations
from typing import *
from dataclasses import dataclass

Number = Union[int, float]


@dataclass
class Polynomial:
    """A class representing a polynomial."""

    def __init__(self, *args: Sequence[Number]):
        if not all(type(e) in get_args(Number) for e in args):
            raise ValueError("Polynomial coefficients have to be numeric.")

        self.coefficients = [] if len(args) == 0 else list(args)

    def __str__(self):
        """A string representation -- 4x^2 + 2x^3 - ..."""
        result = ""

        # to not print the tailing zeroes
        last_non_zero_index = 0
        for i in range(len(self)):
            if self[i] != 0:
                last_non_zero_index = i + 1

        for i in range(last_non_zero_index):
            current = str(abs(self[i]))

            if i >= 1:
                current += "x"
            if i > 1:
                current += f"^{i}"

            if i != 0:
                current = f" {'+' if self[i] >= 0 else '-'} " + current

            result += current

        return "0" if result == "" else result

    __repr__ = __str__

    def __add__(self, other):
        """Polynomial addition."""
        if type(other) in get_args(Number):
            other = Polynomial(other)

        result = Polynomial()

        for i in range(max(len(self), len(other))):
            result[i] = self[i] + other[i]

        return result

    def __sub__(self, other):
        """Polynomial subtraction."""
        return self + (-1 * other)

    def __len__(self):
        """The degree of the polynomial."""
        return len(self.coefficients)

    def __getitem__(self, i: int) -> Number:
        """Return the i-th coefficient of the polynomial. Return 0 if outside the range
        of values."""
        # TODO wrap around on negatives

        return self.coefficients[i] if i < len(self) else 0

    def __setitem__(self, i: int, coefficient: Number):
        """Set the i-th polynomial coefficient to the given value. Raises an exception
        if i is negative."""
        # TODO wrap around on negatives

        # make room (if there isn't enough)
        self.coefficients += [0] * (i - len(self) + 1)

        self.coefficients[i] = coefficient

    degree = __len__

    def __mul__(self, other):
        """Polynomial multiplication."""
        # if we're multiplying a number, make it a polynomial -- removes duplicity
        if type(other) in get_args(Number):
            other = Polynomial(other)

        result = Polynomial()

        # n^2 multiplication -- each term in the bracket with one another
        for i in range(len(self)):
            for j in range(len(other)):
                result += Polynomial(*(([0] * (i + j)) + [self[i] * other[j]]))

        return result

    __rmul__ = __mul__

    def __div__(self, other: Polynomial) -> Polynomial:
        """Divide a polynomial by another polynomial (only if it's a root!)"""
        # TODO

    def at(self, x: Number) -> Number:
        """Evaluate the polynomial at the given point (using Horner's scheme)."""
        value = self[len(self) - 1]

        for i in reversed(range(len(self) - 1)):
            value = value * x + self[i]

        return value

    def __eq__(self, other: Polynomial) -> bool:
        """Polynomial equivalence -- same coefficients."""
        for i in range(max(len(self), len(other))):
            if self[i] != other[i]:
                return False
        return True

    def __pow__(self, exponent: int) -> Polynomial:
        """Return the polynomial to the n-th power."""
        # I do realize that this means that 0^0 = 1, but this is quite useful in things
        # like the Taylor expansion, so I'll keep it this way
        if exponent == 0:
            return Polynomial(1)

        result = self
        for _ in range(exponent - 1):
            result *= self

        return result

    def __derivate(self) -> Polynomial:
        """Perform a single derivation."""
        result = Polynomial()

        for i in range(len(self)):
            result[i] = self[i + 1] * (i + 1)

        return result

    def derivative(self, n: int = 1) -> Polynomial:
        """Return the n-th derivative of the polynomial."""
        result = self

        for _ in range(n):
            result = result.__derivate()

        return result

    def __integrate(self) -> Polynomial:
        """Perform a single integration."""
        result = Polynomial()

        for i in range(len(self)):
            result[i + 1] = self[i] / (i + 1)

        return result

    def integral(self, n: int = 1) -> Polynomial:
        """Return the n-th integral of the polynomial."""
        result = self

        for _ in range(n):
            result = result.__integrate()

        return result

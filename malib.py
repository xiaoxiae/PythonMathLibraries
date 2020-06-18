from __future__ import annotations
from typing import *
from dataclasses import dataclass
from random import random
from abc import ABC
from math import factorial

Number = Union[int, float, complex]


class Differentiable(ABC):
    """A class inherited by objects that are differentiable."""

    def derivative(self, n: int = 1):
        """Return the n-th derivative of the function."""


class Integrable(ABC):
    """A class inherited by objects that are integrable."""

    def integral(self, n: int = 1):
        """Return the n-th integral of the function."""


@dataclass
class Polynomial(Integrable, Differentiable):
    """A class representing a polynomial."""

    def strip_tailing_zeroes(function):
        """A decorator for calling __strip_tailing_zeroes on methods returning
        polynomials."""

        def wrapper(self, *args, **kwargs):
            result = function(self, *args, **kwargs)

            # if the function didn't return anything, use self instead
            returned_none = result is None

            if returned_none:
                result = self

            while len(result.coefficients) != 1 and result.coefficients[-1] == 0:
                result.coefficients.pop()

            # if the function didn't return anything, don't return anything either
            if not returned_none:
                return result

        return wrapper

    @strip_tailing_zeroes
    def __init__(self, *args: Sequence[Number]):
        if not all(type(e) in get_args(Number) for e in args):
            raise ValueError("Polynomial coefficients have to be numeric.")

        self.coefficients = [0] if len(args) == 0 else list(args)

    def __str__(self):
        """A string representation of the polynomial."""
        result = ""

        # special case for zero polynomial
        if self == Polynomial():
            return "0"

        non_zero_found = False
        for i in range(len(self)):
            # skip zeroes
            if self[i] == 0:
                continue

            # if coefficients are complex or we're at the first one, simply str it
            # else abs it to later put space between -/+
            if isinstance(self[i], complex) or not non_zero_found:
                current = str(self[i])
            else:
                current = str(abs(self[i]))

            if i >= 1:
                current += "x"
            if i > 1:
                current += f"^{i}"

            if non_zero_found:
                current = (
                    f" {'+' if isinstance(self[i], complex) or self[i] >= 0 else '-'} "
                    + current
                )

            result += current

            non_zero_found = True

        return result

    __repr__ = __str__

    @strip_tailing_zeroes
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
        """Length is the number of coefficients of the polynomial."""
        return len(self.coefficients)

    def degree(self):
        """The degree of the polynomial."""
        return len(self) - 1

    def __getitem__(self, i: int) -> Number:
        """Return the i-th coefficient of the polynomial. Return 0 if outside the range
        of values."""
        return self.coefficients[i] if i < len(self) else 0

    @strip_tailing_zeroes
    def __setitem__(self, i: int, coefficient: Number):
        """Set the i-th polynomial coefficient to the given value. Raises an exception
        if i is negative."""
        # make room (if there isn't enough)
        self.coefficients += [0] * (i - len(self) + 1)
        self.coefficients[i] = coefficient

    @strip_tailing_zeroes
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

    def at(self, x: Number) -> Number:
        """Evaluate the polynomial at the given point (using Horner's scheme)."""
        value = self[len(self) - 1]

        for i in reversed(range(len(self) - 1)):
            value = value * x + self[i]

        return value

    def __eq__(self, other: Polynomial) -> bool:
        """Polynomial equivalence -- same coefficients."""
        for i in range(max(len(self) + 1, len(other) + 1)):
            if self[i] != other[i]:
                return False
        return True

    @strip_tailing_zeroes
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

        for i in range(len(self) + 1):
            result[i] = self[i + 1] * (i + 1)

        return result

    @strip_tailing_zeroes
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

    @strip_tailing_zeroes
    def integral(self, n: int = 1) -> Polynomial:
        """Return the n-th integral of the polynomial."""
        result = self

        for _ in range(n):
            result = result.__integrate()

        return result

    def __beter_zero(self, a: Number, b: Number):
        """Return a or b, depending on whether it is closer to a zero of the
        polynomial."""
        return a if abs(self.at(a)) <= abs(self.at(b)) else b

    def roots(self, tolerance=(10 ** (-14))) -> List[Number]:
        """Return the roots of the polynomial using Aberth's method
        See https://en.wikipedia.org/wiki/Aberth_method."""
        p = self
        n = p.degree()

        # special case for a polynomial of degree 0
        if n == 0:
            return []

        # first, find the lower/upper bound for the roots
        # https://en.wikipedia.org/wiki/Properties_of_polynomial_roots#Lagrange's_and_Cauchy's_bounds
        cauchy = 1 + sum([abs(p[n] / p[-1]) for n in range(len(p) - 1)])

        # pick degree() random complex numbers
        z = [complex(random(), random()) for _ in range(n)]

        # set the magnitudes to be somewhere from (0, cauchy)
        for i in range(len(z)):
            z[i] = (random() * cauchy) * (z[i] / abs(z[i]))

        # iterate the roots
        pd = self.derivative()

        while True:
            z_new = []  # new roots
            converged = 0  # number of roots that converged
            for k in range(len(z)):
                # if a root converged (well enough), simply add it
                if abs(p.at(z[k])) <= tolerance:
                    z_new.append(z[k])
                    converged += 1
                else:
                    z_new.append(
                        z[k]
                        - 1
                        / (
                            pd.at(z[k]) / p.at(z[k])
                            - sum([1 / (z[k] - z[j]) for j in range(n) if j != k])
                        )
                    )

            # if all of them converged, return them
            if converged == len(z_new):
                z_modified = []

                # try to convert roots from complex to real and see if it improves the
                # approximation
                for e in z_new:
                    places = 4

                    # attempt to round the real and the complex part
                    e = self.__beter_zero(complex(round(e.real, places), e.imag), e)
                    e = self.__beter_zero(complex(e.real, round(e.imag, places)), e)

                    # attempt to make a real
                    e = self.__beter_zero(e.real, e)

                    z_modified.append(e)

                return z_modified

            z = z_new


def taylor(f, a: Number, n: int = 1):
    """Return the n-th Taylor series of the given differentiable function at a."""
    return sum(
        [
            f.derivative(i).at(a) / factorial(i) * (Polynomial(0, 1) - a) ** i
            for i in range(n)
        ],
        type(f)(),
    )

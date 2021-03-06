from __future__ import annotations
from typing import *
from itertools import permutations, product
from functools import reduce
from math import gcd, sqrt, sin, cos

from malib import Polynomial


Number = Union[complex, int, float, Polynomial]


class Utilities:
    @classmethod
    def sign(cls, p: Sequence[int]) -> int:
        """Return the sign of the given permutation."""
        p = list(p)  # preserve p
        sgn = 1

        for i in range(len(p)):
            if i != p[i]:
                j = p.index(i)
                sgn *= -1

                p[i], p[j] = p[j], p[i]

        return sgn


class Matrix:
    """A Python implementation of a matrix class."""

    matrix = None

    def __init__(self, *args: Sequence[Number]):
        """Creates a matrix from a list of values."""
        self.matrix = [list(arg) for arg in args]

        for i, row in enumerate(self.matrix):
            if len(row) != len(self.matrix[i - 1]):
                raise ValueError("Length of rows not uniform!")

        # a little hack -- since we can't do int + Polynomial (but we do want to make
        # it possible for matrices to have polynomials as values), we will treat floats
        # and ints as polynomials of degree 0, since that is pretty much what they are
        for i, j in self.__yield_indexes():
            if type(self[i, j]) in (int, float):
                self[i, j] = Polynomial(self[i, j])

        if not all(type(e) in get_args(Number) for e in self.__yield_values()):
            raise ValueError("Matrix elements have to be numeric.")

    def assert_square(function):
        """A decorator for asserting a square matrix."""

        def wrapper(self, *args, **kwargs):
            if self.rows() != self.columns():
                raise ValueError("Matrix not square!")

            return function(self, *args, **kwargs)

        return wrapper

    def assert_symmetric(function):
        """A decorator for asserting a symmetric matrix."""

        def wrapper(self, *args, **kwargs):
            if not self.is_symmetric():
                raise ValueError("Matrix not symmetric!")

            return function(self, *args, **kwargs)

        return wrapper

    def __yield_indexes(self) -> Iterator[Tuple[int, int]]:
        """Yield all indexes of the matrix."""
        for i in range(self.rows()):
            for j in range(self.columns()):
                yield (i, j)

    def __yield_values(self) -> Iterator[Number]:
        """Yield all elements of the matrix."""
        for index in self.__yield_indexes():
            yield self[index]

    def __eq__(self, other):
        """Matrix equality."""
        if self.rows() != other.rows() or self.columns() != other.columns():
            raise ValueError("Mismatched matrix dimensions!")

        return all([self[i, j] == other[i, j] for i, j in self.__yield_indexes()])

    def __str__(self) -> str:
        """Defines the string representation of the matrix:
        ⌈3 1 2⌉                 ⌈x     ⌉
        |2 5 1|    [1 2 3 4]    |x + 2 |    ...
        ⌊2 6 1⌋                 ⌊2x - 3⌋
        """

        column_widths = [0] * self.columns()
        for i, j in self.__yield_indexes():
            column_widths[j] = max(column_widths[j], len(str(self[i, j])))

        result = ""

        for i in range(self.rows()):
            for j in range(self.columns()):
                if j == 0:
                    result += (
                        "["
                        if self.rows() == 1
                        else "⌈"
                        if i == 0
                        else "⌊"
                        if i == self.rows() - 1
                        else "|"
                    )

                result += str(self[i, j]).rjust(column_widths[j] + (1 if j != 0 else 0))

                if j == self.columns() - 1:
                    result += (
                        "]"
                        if self.rows() == 1
                        else "⌉"
                        if i == 0
                        else "⌋"
                        if i == self.rows() - 1
                        else "|"
                    )

            result += "\n"

        return result.strip()

    __repr__ = __str__

    def __getitem__(self, pos: Tuple[int, int]) -> Number:
        """Return the value of the matrix at the given position."""
        return self.matrix[pos[0]][pos[1]]

    def __setitem__(self, pos: Tuple[int, int], val: Number):
        """Set the value of the matrix at a given position to a certain value."""
        self.matrix[pos[0]][pos[1]] = val

    def rows(self) -> int:
        """Return the number of rows of the matrix."""
        return len(self.matrix)

    def columns(self) -> int:
        """Return the number of columns of the matrix."""
        return len(self.matrix[0])

    def __swap(self, i: int, j: int):
        """Swap two rows of the matrix."""
        self.matrix[i], self.matrix[j] = self.matrix[j], self.matrix[i]

    def __end_condition(self, i: int, j: int) -> bool:
        """Returns True if the end condition of Gauss-Jordan is satisfied; else False."""
        for k in range(i, self.rows()):
            for l in range(j, self.columns()):
                if self[k, l] != 0:
                    return False
        return True

    def __multiply_row(self, row: int, constant: Number):
        """Multiply a row by a constant."""
        for column in range(self.columns()):
            self[row, column] *= constant

    def __add_to_row(self, i: int, j: int, multiple: Number):
        """Add a multiple of j to i."""
        for column in range(self.columns()):
            self[i, column] += multiple * self[j, column]

    def __empty_column(self, column: int, starting_index: int) -> bool:
        """Checks if a column contains only zeroes after a certain index."""
        for row in range(starting_index, self.rows()):
            if self[row, column] != 0:
                return False
        return True

    def rref(self) -> Matrix:
        """Returns the RREF-version of the matrix (using Gauss-Jordan)."""
        res = self.copy()  # the resulting matrix
        rows, columns = res.rows(), res.columns()

        i, j = 0, 0
        while not res.__end_condition(i, j):
            j = min([l for l in range(j, columns) if not res.__empty_column(l, i)])
            res.__swap(i, min([l for l in range(i, rows) if res[l, j] != 0]))
            res.__multiply_row(i, 1 / res[i, j])

            for k in range(rows):
                if k != i:
                    res.__add_to_row(k, i, -1 * res[k, j])

            i, j = i + 1, j + 1

        return res

    @assert_square
    def inverse(self) -> Matrix:
        """Return the inverse of the matrix."""
        result = Matrix.null(self.rows(), self.columns() * 2)

        # copy the matrix to the first part
        for i, j in self.__yield_indexes():
            result[i, j] = self[i, j]

        # copy the unit matrix to the second
        for i, j in self.__yield_indexes():
            result[i, j + self.columns()] = 1 if i == j else 0

        rref = result.rref()

        # check, if the left part is unit
        for i in range(self.rows()):
            if rref[i, i] != 1:
                raise ValueError("Matrix not invertible (rows are linearly dependent)!")

        return Matrix(*[row[self.columns() :] for row in rref.matrix])

    def transposed(self) -> Matrix:
        """Return this matrix, transposed."""
        result = Matrix.null(self.columns(), self.rows())

        for i, j in self.__yield_indexes():
            result[j, i] = self[i, j]

        return result

    @classmethod
    def null(
        cls, rows: Union[int, Matrix] = None, columns: Optional[int] = None
    ) -> Matrix:
        """Return a null matrix of the given size."""
        # if the parameter is a matrix, copy its size
        if isinstance(rows, cls):
            columns = rows.columns()
            rows = rows.rows()

        return cls(*([0] * (columns or rows) for row in range(rows)))

    @classmethod
    def unit(cls, n: int) -> Matrix:
        """Return a unit square matrix of the given size."""
        return cls(*([(0 if i != j else 1) for j in range(n)] for i in range(n)))

    def copy(self) -> Matrix:
        """Return a copy of this matrix. Note for future self: do not define this in
        terms of other functions (like self * 1), because they use this function."""
        result = Matrix.null(self)

        for index in self.__yield_indexes():
            result[index] = self[index]

        return result

    def __add__(self, other: Matrix) -> Matrix:
        """Defines matrix addition."""
        if not isinstance(other, Matrix):
            raise TypeError(f"Addition undefined for <type(other).__name__)>!")

        if self.rows() != other.rows() or self.columns() != other.columns():
            raise ValueError("Mismatched matrix dimensions!")

        result = Matrix.copy(self)

        for i, j in self.__yield_indexes():
            result[i, j] += other[i, j]

        return result

    def __sub__(self, other: Matrix) -> Matrix:
        """Defines matrix subtraction (in terms of addition)."""
        return self + (-1 * other)

    def __mul__(self, other: Union[Number, Matrix]) -> Matrix:
        """Defines scalar and matrix multiplication for a matrix object."""
        if type(other) in get_args(Number):
            return self.__scalar_mul(other)

        if isinstance(other, Matrix):
            return self.__matrix_mul(other)

        raise TypeError(f"Multiplication not defined for <{type(other).__name__}>!")

    __rmul__ = __mul__

    def __neg__(self) -> Matrix:
        """Defines matrix negation (in terms of multiplication)."""
        return self * -1

    def __scalar_mul(self, scalar: Number) -> Number:
        """Returns a result of scalar multiplication."""
        result = Matrix.copy(self)

        for i, j in self.__yield_indexes():
            result[i, j] *= scalar

        return result

    def __matrix_mul(self, other) -> Matrix:
        """Returns a result of matrix multiplication."""
        if self.columns() != other.rows():
            raise ValueError("Mismatched matrix dimensions!")

        result = Matrix.null(self.rows(), other.columns())

        for i, j in result.__yield_indexes():
            for k in range(other.rows()):
                result[i, j] += self[i, k] * other[k, j]

        return result

    def det(self) -> Number:
        """Calculate the determinant of a square matrix."""
        n = self.rows()

        return sum(
            [
                Utilities.sign(p)
                * reduce(lambda x, y: x * y, [self[i, p.index(i)] for i in range(n)])
                for p in permutations({i for i in range(n)})
            ],
            Polynomial(),
        )

    def magnitude(self) -> Number:
        """Return the magnitude of a vector (a 1 by n or n by 1 matrix)."""
        if not (self.rows() == 1 or self.columns() == 1):
            raise ValueError("Matrix is not a vector!")

        return sqrt(sum([v ** 2 for v in self.__yield_values()]))

    def normalized(self) -> Matrix:
        """Return the vector (a 1 by n or n by 1 matrix) normalized."""
        return self * (1 / self.magnitude())

    def __get_row(self, i) -> Matrix:
        """Return a vector from the given matrix row."""
        return Matrix(self.matrix[i])

    def __set_row(self, i, row: Matrix):
        """Set the i-th row of the matrix to the given value."""
        self.matrix[i] = row.matrix[0]

    def orthogonalized(self) -> Matrix:
        """Return this matrix, orthogonalized (by rows), using Gram-Schmidt."""
        res = self.copy()  # the resulting matrix

        for k in range(self.rows()):
            y_k = self.__get_row(k)

            # subtract the projections
            for i in range(k):
                y_k -= (
                    self.__get_row(k) * res.__get_row(i).transposed()
                ) * res.__get_row(i)

            # check if they aren't linearly dependent
            if y_k.magnitude() == 0:
                raise ValueError(
                    "Matrix not orthogonalizable (rows are linearly dependent)!"
                )

            # if not, normalize and set
            res.__set_row(k, y_k.normalized())

        return res

    @assert_square
    def is_symmetric(self) -> bool:
        """Return True if the matrix is symmetric, else False."""
        for i in range(self.rows()):
            for j in range(i, self.rows()):
                if self[i, j] != self[j, i]:
                    return False
        return True

    def submatrix(self, x1: int, y1: int, x2: int, y2: int):
        """Return a submatrix (inclusive)."""
        if (
            x1 > x2
            or y1 > y2
            or min(x1, x2) < 0
            or max(x1, x2) >= self.columns()
            or min(y1, y2) < 0
            or max(y1, y2) >= self.rows()
        ):
            raise ValueError("Invalid coordinates!")

        result = Matrix.null(x2 - x1 + 1, y2 - y1 + 1)

        for x in range(x1, x2 + 1):
            for y in range(y1, y2 + 1):
                result[x - x1, y - y1] = self[x, y]

        return result

    def __definitiveness(self, matrix: Matrix = None) -> int:
        """Recursively calculate the definitiveness of a given matrix, returning 0 for
        not, 1 for semi-definite and 2 for definite."""
        if matrix is None:
            matrix = self

        alpha = matrix[0, 0]  # top left element

        # look at the diagonal
        res = 2 if alpha > 0 else 1 if alpha == 0 else 0

        # if we're not at a 1 by 1 matrix, recurse
        if matrix.rows() != 1:
            a = Matrix(matrix.matrix[0][1:])  # row vector (without alpha)
            A = matrix.submatrix(1, 1, matrix.rows() - 1, matrix.rows() - 1)

            res = min(res, self.__definitiveness(A - 1 / alpha * a.transposed() * a))

        return verdict

    @assert_square
    @assert_symmetric
    def is_definite(self) -> bool:
        """Whether the matrix is definite (using the recursive formula)."""
        return self.__definitiveness() == 2

    @assert_square
    @assert_symmetric
    def is_semidefinite(self) -> bool:
        """Whether the matrix is definite (using the recursive formula)."""
        return self.__definitiveness() == 1

    @assert_square
    def characteristic_polynomial(self) -> Polynomial:
        """Return the characteristic polynomial of the matrix."""
        return (self - Matrix.unit(self.rows()) * Polynomial(0, 1)).det()

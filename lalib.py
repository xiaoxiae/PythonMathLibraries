"""Linear Algebra library"""
# TODO: positively definite, gram schmidt, eigenvalues and eigenvectors, spectral decomposition
# - add vectors
#   - probably a vector class that plays nice with the matrix
#   - getting row and column vectors
#   - make a class hierarchy?

from __future__ import annotations
from typing import *
from itertools import permutations, product
from functools import reduce
from math import gcd, sqrt, sin, cos


Number = Union[int, float, complex]


class Matrix:
    """A Python implementation a matrix class."""

    matrix = None

    def __init__(self, *args: Sequence[Number]):
        """Creates a matrix from a list of values."""
        self.matrix = [list(arg) for arg in args]

        if not all(
            len(self.matrix[i]) == len(self.matrix[i + 1])
            for i in range(len(self.matrix) - 1)
        ):
            raise ValueError("One of matrix rows has an incorrect dimension.")

        if not all(type(e) in get_args(Number) for e in self.__yield_values()):
            raise ValueError("Matrix elements have to be numeric.")

    def __yield_indexes(self):
        """Yield all indexes of the matrix."""
        for i in range(self.rows()):
            for j in range(self.columns()):
                yield (i, j)

    def __yield_values(self):
        """Yield all elements of the matrix."""
        for index in self.__yield_indexes():
            yield self[index]

    def __str__(self):
        """Defines the string representation of the matrix."""
        return str(self.matrix)

    __repr__ = __str__

    def __getitem__(self, pos: Tuple[int, int]):
        """Return the value of the matrix at the given position."""
        return self.matrix[pos[0]][pos[1]]

    def __setitem__(self, pos: Tuple[int, int], val: Number):
        """Set the value of the matrix at a given position to a certain value."""
        self.matrix[pos[0]][pos[1]] = val

    def rows(self):
        """Return the number of rows of the matrix."""
        return len(self.matrix)

    def columns(self):
        """Return the number of columns of the atrix."""
        return len(self.matrix[0])

    def __swap(self, i: int, j: int):
        """Swap two rows of the matrix."""
        self.matrix[i], self.matrix[j] = self.matrix[j], self.matrix[i]

    def __end_condition(self, i: int, j: int):
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

    def __empty_column(self, column: int, starting_index: int):
        """Checks if a column contains only zeroes after a certain index."""
        for row in range(starting_index, self.rows()):
            if self[row, column] != 0:
                return False
        return True

    def rref(self):
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

    def inverse(self):
        """Return the inverse of the matrix."""
        # TODO: what to do about this?
        # if self.rows() != self.columns():
        #     # return a more abstract inverse, if the dimensions don't match
        #     return (self.transposed() * self).inverse() * self.transposed()

        result = Matrix.null(self.rows(), self.columns() * 2)

        # copy the matrix to the first part
        for i, j in self.__yield_indexes():
            result[i, j] = self[i, j]

        # copy the unit matrix to the second
        for i, j in self.__yield_indexes():
            result[i, j + self.columns()] = 1 if i == j else 0

        # return the second part
        return Matrix(*[row[self.columns() :] for row in result.rref().matrix])

    def transposed(self):
        """Return this matrix, transposed."""
        result = Matrix.null(self.columns(), self.rows())

        for i, j in self.__yield_indexes():
            result[j, i] = self[i, j]

        return result

    @classmethod
    def null(cls, rows: Union[int, Matrix] = None, columns: Optional[int] = None):
        """Return a null matrix of the given size."""
        # if the parameter is a matrix, copy its size
        if type(rows) is cls:
            columns = rows.columns()
            rows = rows.rows()

        return cls(*([0] * (columns or rows) for row in range(rows)))

    @classmethod
    def unit(cls, n: int):
        """Return a unit matrix of the given size."""
        return cls(*([(0 if i != j else 1) for j in range(n)] for i in range(n)))

    def copy(self):
        """Return a copy of this matrix."""
        result = Matrix.null(self)

        for index in self.__yield_indexes():
            result[index] = self[index]

        return result

    def __add__(self, other: Matrix):
        """Defines matrix addition."""
        if type(other) is not Matrix:
            raise TypeError(f"Addition undefined for <type(other).__name__)>!")

        if self.rows() != other.rows() or self.columns() != other.columns():
            raise ValueError("Mismatched matrix dimensions!")

        result = Matrix.copy(self)

        for i, j in self.__yield_indexes():
            result[i, j] += other[i, j]

        return result

    def __sub__(self, other: Matrix):
        """Defines matrix subtraction (in terms of addition)."""
        return self + (-1 * other)

    def __mul__(self, other: Union[Number, Matrix]):
        """Defines scalar and matrix multiplication for a matrix object."""
        if type(other) in get_args(Number):
            return self.__scalar_mul(other)

        elif type(other) is Matrix:
            return self.__matrix_mul(other)

        else:
            raise TypeError(f"Multiplication not defined for <{type(other).__name__}>!")

    __rmul__ = __mul__

    def __scalar_mul(self, scalar: Number):
        """Returns a result of scalar multiplication."""
        result = Matrix.copy(self)

        for i, j in self.__yield_indexes():
            result[i, j] *= scalar

        return result

    def __matrix_mul(self, other):
        """Returns a result of matrix multiplication."""
        if self.columns() != other.rows():
            raise ValueError("Mismatched matrix dimensions!")

        result = Matrix.null(self.rows(), other.columns())

        for i, j in self.__yield_indexes():
            for k in range(self.rows()):
                result[i, j] += self[i, k] * other[k, j]

        return result

    def det(self):
        """Calculate the determinant of a square matrix."""
        if self.rows() != self.columns():
            raise ValueError("Matrix not square!")

        n = self.rows()

        return sum(
            [
                Utilities.sign(p)
                * reduce(lambda x, y: x * y, [self[i, p.index(i)] for i in range(n)])
                for p in permutations({i for i in range(n)})
            ]
        )


class Utilities:
    @classmethod
    def sign(self, p: Sequence[int]):
        """Return the sign of the given permutation."""
        p = list(p)  # perserve P
        transpositions = 0

        for i in range(len(p)):
            if i != p[i]:
                j = p.index(i)
                transpositions += 1

                p[i], p[j] = p[j], p[i]

        return 1 if transpositions % 2 == 0 else -1


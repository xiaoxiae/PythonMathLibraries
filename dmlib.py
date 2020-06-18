import itertools
from typing import *
from math import factorial as f

def yield_subsets(iterable):
    """Yield all subsets of a given iterable."""
    for k in range(len(iterable) + 1):
        for combination in itertools.combinations(iterable, k):
            yield set(combination)


def yield_relations(s):
    """Yield all relations of the given iterable."""
    for subset in yield_subsets(tuple(itertools.product(s, s))):
        yield subset


def is_reflexive(r, s):
    """Returns True if a relation is reflexive."""
    for x in s:
        if (x, x) not in r:
            return False

    return True


def is_symmetric(r, s):
    """Returns True if a relation is symmetric."""
    for x in s:
        for y in s:
            if (x, y) in r and not (y, x) in r:
                return False
    return True


def is_antisymmetric(r, s):
    """Returns True if a relation is antisymmetric."""
    for x in s:
        for y in s:
            if (x, y) in r and (y, x) in r and x is not y:
                return False
    return True


def is_transitive(r, s):
    """Returns True if a relation is transitive."""
    for x in s:
        for y in s:
            for z in s:
                if (x, y) in r and (y, z) in r and (x, z) not in r:
                    return False
    return True


def is_equivalence(r, s):
    """Returns True if a given relation is an equivalence."""
    return is_transitive(r, s) and is_reflexive(r, s) and is_symmetric(r, s)


def is_ordering(r, s):
    """Returns True if a given relation is an ordering."""
    return is_transitive(r, s) and is_reflexive(r, s) and is_antisymmetric(r, s)


def yield_anagrams(word: str):
    """Yield all anagrams of the given word."""
    for anagram in itertools.permutations(word):
        yield "".join(anagram)


def contains_word(word: str, string: str):
    """Check, whether a string contains the characters of the word in the right order."""
    while len(string) != 0:
        if word[0] == string[0]:
            word = word[1:]

        string = string[1:]

        if len(word) == 0:
            return True

    return False


def binom(n, k):
    """Return the binomial coefficient, given the values."""
    return f(n) // (f(k) * f(n - k))


def binomial_values(a, b, k):
    """Return the values of (a + b)^k."""
    return [binom(k, n) * a ** (k - n) * b ** n for n in range(k + 1)]

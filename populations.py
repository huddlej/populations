import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import random


class Population(object):
    """
    Genotypes are biallelic in the form of (0, 1) where 0 is a recessive allele
    and 1 is a dominant allele. Each genotype is associated with a character of
    the organism (e.g., seed color) and recessive and dominant phenotypes (e.g.,
    "green" and "yellow").
    """

    @classmethod
    def create_parental(cls, n, allele):
        """
        Create a pure-breeding parental population of the given size, n, consisting
        entirely of the given allele.
        """
        return cls(np.array([allele] * 2*n).reshape(n, 2))

    def __init__(self, population):
        """
        Instantiate a population with an array of alleles.
        """
        self.population = population

    def __repr__(self):
        """
        Return representation of the array underlying this population.
        """
        return repr(self.population)

    def get_alleles(self):
        """
        Return a random array of alleles from the given population.
        """
        return np.apply_along_axis(np.random.choice, 1, self.population)

    def cross(self, other_population):
        """
        Create a new population from alleles sampled from this and the given
        population.
        """
        return self.__class__(np.array([self.get_alleles(), other_population.get_alleles()]).T)

    def get_allele_frequencies(self):
        """
        Return the frequency of each allele in this population.
        """
        unique, inverse = np.unique(self.population, return_inverse=True)
        count = np.zeros(len(unique), np.int)
        np.add.at(count, inverse, 1)
        return count / np.float(count.sum())

    def get_expected_genotype_frequencies(self):
        """
        Return an array of genotype frequencies expected for this population
        based on Hardy-Weinberg equilibrium.
        """
        allele_frequencies = self.get_allele_frequencies()
        return np.array([
            allele_frequencies[0] ** 2,
            2 * np.product(allele_frequencies),
            allele_frequencies[1] ** 2
        ])

    def get_observed_genotype_frequencies(self):
        """
        Return the observed genotype frequencies of this population.
        """
        genotypes = np.apply_along_axis(sum, 1, self.population)
        return np.bincount(genotypes) / np.sum(genotypes, dtype=np.float64)

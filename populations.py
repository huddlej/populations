import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import random


def create_parental_population(n, allele):
    """
    Create a pure-breeding parental population of the given size, n, consisting
    entirely of the given allele.
    """
    return np.array([allele] * 2*n).reshape(n, 2)


def get_alleles(population):
    """
    Return a random array of alleles from the given population.
    """
    return np.apply_along_axis(np.random.choice, 1, population)


def cross_populations(population1, population2):
    """
    Sample alleles from two given populations and create a new population of the
    same size.
    """
    return np.array([get_alleles(population1), get_alleles(population2)]).T


def get_allele_frequencies(population):
    """
    Return the frequency of each allele in the given population.
    """
    unique, inverse = np.unique(population, return_inverse=True)
    count = np.zeros(len(unique), np.int)
    np.add.at(count, inverse, 1)
    return count / np.float(count.sum())


def get_expected_genotype_frequencies(population):
    """
    Return an array of genotype frequencies expected for the given population
    based on Hardy-Weinberg equilibrium.
    """
    allele_frequencies = get_allele_frequencies(population)
    return np.array([
        allele_frequencies[0] ** 2,
        2 * np.product(allele_frequencies),
        allele_frequencies[1] ** 2
    ])


def get_observed_genotype_frequencies(population):
    """
    Return the observed genotype frequencies of the given population.
    """
    genotypes = np.apply_along_axis(sum, 1, population)
    return np.bincount(genotypes) / np.sum(genotypes, dtype=np.float64)

# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import random

# <markdowncell>

# Genotypes and phenotypes
# ========================
# 
# Genotypes are biallelic in the form of (0, 1) where 0 is a recessive allele and 1 is a dominant allele. Each genotype is associated with a character of the organism (e.g., seed color) and recessive and dominant phenotypes (e.g., "green" and "yellow").

# <codecell>

def create_parental_population(n, is_dominant):
    if is_dominant:
        return np.ones((n, 2), dtype=np.integer)
    else:
        return np.zeros((n, 2), dtype=np.integer)

# <codecell>

n = 1000

# <codecell>

p1_dominant = create_parental_population(n, True)

# <codecell>

p1_recessive = create_parental_population(n, False)

# <codecell>

def cross_populations(population1, population2):
    return np.array(
        zip(
            np.apply_along_axis(random.choice, 1, population1),
            np.apply_along_axis(random.choice, 1, population2)
        )
    )

# <codecell>

f1 = cross_populations(p1_dominant, p1_recessive)

# <codecell>

f2 = cross_populations(f1, f1)

# <codecell>

def get_genotype_frequencies(population):
    df = DataFrame(Series(np.apply_along_axis(sum, 1, population)), columns=["genotypes"])
    counts = df.groupby("genotypes").count()
    return counts.apply(lambda x: x / float(counts.sum()))

# <codecell>

get_genotype_frequencies(p1_recessive)

# <codecell>

get_genotype_frequencies(p1_dominant)

# <codecell>

get_genotype_frequencies(f1)

# <codecell>

get_genotype_frequencies(f2)

# <codecell>

f3 = cross_populations(f2, f2)

# <codecell>

get_genotype_frequencies(f3)

# <codecell>



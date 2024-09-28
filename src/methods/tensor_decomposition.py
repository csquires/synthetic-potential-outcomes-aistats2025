# === IMPORTS: BUILT-IN ===
from dataclasses import dataclass
from typing import List, Dict

# === IMPORTS: THIRD-PARTY ===
import numpy as np
from tensorly.decomposition import parafac

# === IMPORTS: LOCAL ===
from moments.empirical_moments import EmpiricalMoments
from src.problem_dims import ProblemDimensions


class TensorDecomposition:
    def __init__(self, config: ProblemDimensions):
        self.config = config

    def fit(self, moments: EmpiricalMoments):
        expectations = moments.expectations
        conditional_second_moments = moments.conditional_second_moments
        conditional_third_moments = moments.conditional_third_moments

        # if x and x' are of length d and d'
        # then should this be the 2 x d x d' tensor where the first index is treatment
        breakpoint()        

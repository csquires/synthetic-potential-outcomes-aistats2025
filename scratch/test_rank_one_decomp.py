# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.problem_dims import ProblemDimensions
from src.data_generation.binary_generator import DiscreteGenerator

from src.moments.population_moments_binary import PopulationMomentsBinary
from src.moments.empirical_moments import EmpiricalMoments
from src.methods.tensor_decomposition import TensorDecomposition
from src.calculate_marginals import Marginal

from tensorly.decomposition import parafac


np.random.seed(123)
# ==== DEFINE THE PROBLEM CONFIGURATION ====
nz = 2
nx = 3
ngroups = 2
ntreatments = 2
problem_dims = ProblemDimensions(nz, nx, ngroups, ntreatments)

# ==== DEFINE DATA GENERATOR ====
generator = DiscreteGenerator(problem_dims, matching_coef=0.25, treatment_coef=0.25)
marginal = generator.true_marginal()
oracle_moments = PopulationMomentsBinary(problem_dims, marginal)

# marginal_obj = Marginal(marginal)
oracle_third_moments = oracle_moments.third_moments

# # ==== GENERATE SAMPLES AND COMPUTE MOMENTS ====
# nsamples = 1000
# full_samples, obs_samples = generator.generate(nsamples=nsamples)
# moments = EmpiricalMoments(problem_dims, obs_samples)
# emp_third_moments = moments.third_moments

rank = 2
res = parafac(oracle_third_moments, rank)
weights = res.weights
factors = res.factors
Z_factor = factors[0]
X_factor = factors[1]
YT_factor = factors[2]


def rescale_cols(weights, factors):
    last_factor = factors[-1]
    scale = np.sum(weights)
    new_weights = weights / scale

    new_factors = []
    for factor in factors[:-1]:
        col_scales = factor.sum(axis=0)
        new_factor = factor / col_scales[None, :]
        new_factors.append(new_factor)
        last_factor = last_factor * col_scales[None, :]
    new_factors.append(last_factor)

    return new_weights, new_factors


weights2, (Z_factor2, X_factor2, YT_factor2) = rescale_cols(weights, factors)

# y_ix = problem_dims.y_ix
# t_ix = problem_dims.t_ix
# M2 = np.zeros([2, 2, 2])
# for i, z_ix in enumerate(problem_dims.z_ixs):
#     for j, x_ix in enumerate(problem_dims.x_ixs):
#         M2[i, j, 0] = marginal_obj.get_marginal([z_ix, x_ix, y_ix, t_ix])[1, 1, 1, 0]
#         M2[i, j, 1] = marginal_obj.get_marginal([z_ix, x_ix, y_ix, t_ix])[1, 1, 1, 1]

# components = []
# for r in range(rank):
#     components.append(np.einsum('i,j,k', factors[0][:, r], factors[1][:, r], factors[2][:, r]))
# total = sum(components)
# print(np.max(np.abs(total - third_moments)))
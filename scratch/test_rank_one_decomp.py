# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.problem_dims import ProblemDimensions
from src.data_generation.discrete_generator import DiscreteFixedGenerator

from src.moments.population_moments_binary import PopulationMomentsBinary
from src.moments.empirical_moments import EmpiricalMoments
from src.methods.tensor_decomposition import TensorDecomposition

from tensorly.decomposition import parafac


np.random.seed(123)
# ==== DEFINE THE PROBLEM CONFIGURATION ====
nz = 2
nx = 2
ngroups = 2
ntreatments = 2
problem_dims = ProblemDimensions(nz, nx, ngroups, ntreatments)

# ==== DEFINE DATA GENERATOR ====
generator = DiscreteFixedGenerator(problem_dims, matching_coef=0.25, treatment_coef=0.25)
marginal = generator.true_marginal()
oracle_moments = PopulationMomentsBinary(problem_dims, marginal)
true_mean_y0 = oracle_moments.moments_Y0(1)[1]  # 0th moment is 1, 1st moment is mean
true_mean_y1 = oracle_moments.moments_Y1(1)[1]  # 0th moment is 1, 1st moment is mean

# ==== RUN METHOD ====
nsamples = int(1e5)

# ==== GENERATE SAMPLES AND COMPUTE MOMENTS ====
full_samples, obs_samples = generator.generate(nsamples=nsamples)
moments = EmpiricalMoments(problem_dims, obs_samples)
third_moments = moments.third_moments

rank = 8
res = parafac(third_moments, rank)
weights = res.weights
factors = res.factors
print(factors[0].shape)  # 8 x rank
print(factors[1].shape)  # 8 x rank
print(factors[2].shape)  # 8 X rank

components = []
for r in range(rank):
    components.append(np.einsum('i,j,k', factors[0][:, r], factors[1][:, r], factors[2][:, r]))
total = sum(components)
print(np.max(np.abs(total - third_moments)))
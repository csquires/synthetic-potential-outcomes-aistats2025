# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.problem_dims import ProblemDimensions
from src.data_generation.binary_generator import BinaryGenerator

from src.moments.population_moments_binary import PopulationMomentsBinary
from src.moments.empirical_moments import EmpiricalMoments
from src.methods.tensor_decomposition import TensorDecomposition

from tensorly.decomposition import parafac


np.random.seed(123)
# ==== DEFINE THE PROBLEM CONFIGURATION ====
nz = 2
nx = 3
ngroups = 2
ntreatments = 2
problem_dims = ProblemDimensions(nz, nx, ngroups, ntreatments)

# ==== DEFINE DATA GENERATOR ====
generator = BinaryGenerator(problem_dims, matching_coef=0.25, treatment_coef=0.25)
marginal = generator.true_marginal()
oracle_moments = PopulationMomentsBinary(problem_dims, marginal)
oracle_third_moments = oracle_moments.third_moments
oracle_expectations = oracle_moments.expectations
EZ = oracle_expectations[problem_dims.z_ixs]
EX = oracle_expectations[problem_dims.x_ixs]
EYT = np.array([
    oracle_moments.Pytu[1, 0].sum(), 
    oracle_moments.Pytu[1, 1].sum()
])


rank = 2
res = parafac(oracle_third_moments, rank)
weights = res.weights
factors = res.factors
Z_factor = factors[0]
X_factor = factors[1]
YT_factor = factors[2]


scale_Z = None  # EZ, Z_factor
scale_X = None  # EX, X_factor
scale_YT = None  # EYT, YT_factor


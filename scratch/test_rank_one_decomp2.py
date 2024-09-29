# === IMPORTS: THIRD-PARTY ===
import numpy as np
from numpy.linalg import lstsq

# === IMPORTS: LOCAL ===
from src.problem_dims import ProblemDimensions
from src.data_generation.binary_generator import BinaryGenerator

from src.moments.population_moments_discrete import PopulationMomentsDiscrete
from src.moments.empirical_moments import EmpiricalMoments
from src.methods.tensor_decomposition import TensorDecomposition

from tensorly.decomposition import parafac, parafac_power_iteration


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
marginal = marginal.reshape(2**nz, 2**nx, 2, ntreatments, ngroups)


oracle_moments = PopulationMomentsDiscrete(marginal)
M_ZXtY = oracle_moments.M_ZXtY
E_Z = oracle_moments.E_Z
E_X = oracle_moments.E_X
E_tY = oracle_moments.E_tY

M_ZX = oracle_moments.M_ZX
M_ZtY = oracle_moments.M_ZtY
M_XtY = oracle_moments.M_XtY

dz = 2 ** nz
dx = 2 ** nx
dt = 2
M_aug = np.zeros((dz+1, dx+1, dt+1))
M_aug[:dz, :dx, :dt] = M_ZXtY
M_aug[:dz, :dx, -1] = M_ZX
M_aug[:dz, -1, :dt] = M_ZtY
M_aug[-1, :dx, :dt] = M_XtY
M_aug[:dz, -1, -1] = E_Z
M_aug[-1, :dx, -1] = E_X
M_aug[-1, -1, :dt] = E_tY
M_aug[-1, -1, -1] = 1

rank = 2

PARAFAC = True
if PARAFAC:
    res = parafac(M_aug, rank, n_iter_max=10000)
    weights = res.weights
    factors = res.factors
    Z_factor = factors[0]
    X_factor = factors[1]
    tY_factor = factors[2]
else:
    res = parafac_power_iteration(M_aug, rank)
    weights, factors = res[0], res[1]
    Z_factor = factors[0]
    X_factor = factors[1]
    tY_factor = factors[2]

P = np.einsum("u,zu,xu,tu->zxt", weights, Z_factor, X_factor, tY_factor)
diff = np.max(np.abs(P - M_aug))
print(diff)



D_Z = Z_factor[-1, :]
D_X = X_factor[-1, :]
D_tY = tY_factor[-1, :]

Z_factor_new = Z_factor / D_Z[None, :]
X_factor_new = X_factor / D_X[None, :]
tY_factor_new = tY_factor / D_tY[None, :]
W_scale = np.sum(weights * D_Z * D_X * D_tY)
weights_new = weights * D_Z * D_X * D_tY / W_scale

P_new = np.einsum("u,zu,xu,tu->zxt", weights_new, Z_factor_new, X_factor_new, tY_factor_new)
diff_new = np.max(np.abs(P_new - M_aug))
print(diff_new)


true_weights = oracle_moments.Pu
true_Z_factor = np.einsum("zu,u->zu", oracle_moments.Pzu, oracle_moments.Pu ** -1)
true_X_factor = np.einsum("xu,u->xu", oracle_moments.Pxu, oracle_moments.Pu ** -1)
true_tY_factor = np.einsum("tu,u->tu", oracle_moments.Pytu[1, :, :], oracle_moments.Pu ** -1)
true_tY_factor = np.concatenate((true_tY_factor, np.ones((1, 2))))

T = np.einsum("u,zu,xu,tu->zxt", true_weights, true_Z_factor, true_X_factor, true_tY_factor)
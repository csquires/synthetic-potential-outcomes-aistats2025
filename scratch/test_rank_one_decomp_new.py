# === IMPORTS: THIRD-PARTY ===
import numpy as np
from tensorly.decomposition import parafac, parafac_power_iteration

# === IMPORTS: LOCAL ===
from src.data_generation.generator_main import BinaryGeneratorMain
from src.observable_moments.population_moments_discrete import compute_observable_moments_discrete
from src.mixture_moments.mixture_moments_discrete import compute_mixture_moments_discrete


np.random.seed(123)
# ==== DEFINE DATA GENERATOR ====
generator = BinaryGeneratorMain(zt_strength=0, xy_strength=0)
marginal = generator.true_marginal()


oracle_moments = compute_observable_moments_discrete(marginal)
mixture_moments = compute_mixture_moments_discrete(marginal)
M_ZXS = oracle_moments.M_ZXS
E_Z = oracle_moments.E_Z
E_X = oracle_moments.E_X
E_S = oracle_moments.E_S
print(np.linalg.matrix_rank(mixture_moments.EZ_U))
print(np.linalg.matrix_rank(mixture_moments.EX_U))
print(np.linalg.matrix_rank(mixture_moments.ES_U))

rank = 2

PARAFAC = True
if PARAFAC:
    res = parafac(M_ZXS, rank, n_iter_max=10000, init="random")
    weights = res.weights
    factors = res.factors
    Z_factor = factors[0]
    X_factor = factors[1]
    S_factor = factors[2]
else:
    res = parafac_power_iteration(M_ZXS, rank, n_repeat=30, n_iteration=30)
    weights, factors = res[0], res[1]
    Z_factor = factors[0]
    X_factor = factors[1]
    S_factor = factors[2]

P = np.einsum("u,zu,xu,tu->zxt", weights, Z_factor, X_factor, S_factor)
P2 = np.einsum("u,zu,xu,tu->zxt", mixture_moments.Pu, mixture_moments.EX_U, mixture_moments.EZ_U, mixture_moments.ES_U)
diff = np.max(np.abs(P - M_ZXS))
diff2 = np.max(np.abs(P2 - M_ZXS))
print(diff)
print(diff2)



# D_Z = Z_factor[-1, :]
# D_X = X_factor[-1, :]
# D_tY = tY_factor[-1, :]

# Z_factor_new = Z_factor / D_Z[None, :]
# X_factor_new = X_factor / D_X[None, :]
# tY_factor_new = tY_factor / D_tY[None, :]
# W_scale = np.sum(weights * D_Z * D_X * D_tY)
# weights_new = weights * D_Z * D_X * D_tY / W_scale

# P_new = np.einsum("u,zu,xu,tu->zxt", weights_new, Z_factor_new, X_factor_new, tY_factor_new)
# diff_new = np.max(np.abs(P_new - M_aug))
# print(diff_new)


# true_weights = oracle_moments.Pu
# true_Z_factor = np.einsum("zu,u->zu", oracle_moments.Pzu, oracle_moments.Pu ** -1)
# true_X_factor = np.einsum("xu,u->xu", oracle_moments.Pxu, oracle_moments.Pu ** -1)
# true_tY_factor = np.einsum("tu,u->tu", oracle_moments.Pytu[1, :, :], oracle_moments.Pu ** -1)
# true_tY_factor = np.concatenate((true_tY_factor, np.ones((1, 2))))

# T = np.einsum("u,zu,xu,tu->zxt", true_weights, true_Z_factor, true_X_factor, true_tY_factor)
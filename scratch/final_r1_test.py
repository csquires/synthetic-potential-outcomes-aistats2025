from pprint import pprint
from dataclasses import asdict

# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.data_generation.generator_main import BinaryGeneratorMain
from src.observable_moments.empirical_moments import compute_empirical_moments, binary_feature_map_single
from src.observable_moments.population_moments_discrete import compute_observable_moments_discrete
from src.mixture_moments.mixture_moments_discrete import compute_mixture_moments_discrete

from src.methods.tensor_decomposition import TensorDecompositionBinary

np.random.seed(123)


# ==== DEFINE GENERATOR ====
zt_strength = 0
generator = BinaryGeneratorMain(zt_strength=zt_strength, xy_strength=0)
marginal = generator.true_marginal()
obs_moments = compute_observable_moments_discrete(marginal)
true_mixture_moments = compute_mixture_moments_discrete(marginal)

# ==== GENERATE SAMPLES AND COMPUTE MOMENTS ====
# nsamples = int(1e3)
# full_samples, obs_samples = generator.generate(nsamples=nsamples)
# new_samples = binary_feature_map_single(obs_samples)
# moments = compute_empirical_moments(generator.problem_dims, new_samples)

# ==== RUN METHOD ====
td = TensorDecompositionBinary(generator.problem_dims, decomposition_method="parafac")
res = td.fit(obs_moments, check_recovery=True)


print("==== RES ====")
pprint(asdict(res))
print("==== TRUE ====")
pprint(asdict(true_mixture_moments))
# print("==== OBS MOMENTS ====")
# pprint(asdict(obs_moments))


M_ZXS_2 = np.einsum("u,zu,xu,su->zxs", 
    true_mixture_moments.Pu,
    true_mixture_moments.EZ_U,
    true_mixture_moments.EX_U,
    true_mixture_moments.ES_U,
)

print(np.linalg.matrix_rank(true_mixture_moments.EZ_U))
print(np.linalg.matrix_rank(true_mixture_moments.EX_U))
print(np.linalg.matrix_rank(true_mixture_moments.ES_U))
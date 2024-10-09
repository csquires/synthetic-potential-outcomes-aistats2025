from dataclasses import asdict

# === IMPORTS: THIRD-PARTY ===
import numpy as np
from tqdm import trange

# === IMPORTS: LOCAL ===
from src.data_generation.generator_main import BinaryGeneratorMain

from src.causal_moments.causal_moments_discrete import compute_potential_outcome_moments_discrete

from src.observable_moments import ObservableMoments
from src.observable_moments.population_moments_discrete import compute_observable_moments_discrete
from src.observable_moments.empirical_moments import compute_empirical_moments, binary_feature_map_single
from src.methods.synthetic_potential_outcomes import SyntheticPotentialOutcomes


np.random.seed(123)
# ==== DEFINE DATA GENERATOR ====
a, b = 0, 0
generator = BinaryGeneratorMain(lambda_treatment=a, lambda_outcome=b)
marginal = generator.true_marginal()
obs_moments = compute_observable_moments_discrete(marginal)
causal_moments = compute_potential_outcome_moments_discrete(marginal, 1)
true_means = causal_moments.E_R_U


# ==== GENERATE SAMPLES AND COMPUTE MOMENTS ====
nsamples = int(10e5)
full_samples, obs_samples = generator.generate(nsamples=nsamples)
new_samples = binary_feature_map_single(obs_samples)
obs_moments_empirical = compute_empirical_moments(generator.problem_dims, new_samples)

# obs_moments_dict = asdict(obs_moments)
# obs_moments_empirical_dict = asdict(obs_moments_empirical)
# for key in obs_moments_dict:
#     print(key)
#     print(obs_moments_dict[key])
#     print(obs_moments_empirical_dict[key])
#     print("===========")


# ==== RUN METHOD ====
spo = SyntheticPotentialOutcomes(generator.problem_dims, decomposition_method="matrix_pencil")
res = spo.fit(obs_moments_empirical)
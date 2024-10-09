# === IMPORTS: BUILT-IN ===
import pickle

# === IMPORTS: THIRD-PARTY ===
import numpy as np
from tqdm import trange

# === IMPORTS: LOCAL ===
from src.data_generation.generator_main import BinaryGeneratorMain

from src.causal_moments.causal_moments_discrete import compute_potential_outcome_moments_discrete
from src.observable_moments.empirical_moments import compute_empirical_moments, binary_feature_map_single
from src.methods.synthetic_potential_outcomes import SyntheticPotentialOutcomes


np.random.seed(123)
# ==== DEFINE DATA GENERATOR ====
a, b = 0, 0
generator = BinaryGeneratorMain(lambda_treatment=a, lambda_outcome=b)
marginal = generator.true_marginal()
causal_moments = compute_potential_outcome_moments_discrete(marginal, 1)
true_means = causal_moments.E_R_U

# ==== RUN METHOD ====
nsamples = int(5e5)
nruns = 100
all_estimated_source_probs = np.zeros((nruns, 2))
all_estimated_means = np.zeros((nruns, 2))
for r in trange(nruns):
    # ==== GENERATE SAMPLES AND COMPUTE MOMENTS ====
    full_samples, obs_samples = generator.generate(nsamples=nsamples)
    new_samples = binary_feature_map_single(obs_samples)
    moments = compute_empirical_moments(generator.problem_dims, new_samples)

    # ==== RUN METHOD ====
    spo = SyntheticPotentialOutcomes(generator.problem_dims, decomposition_method="matrix_pencil")
    res = spo.fit(moments)

    # === SAVE RESULTS ===
    recovered_source_probs = res["source_probs"]
    recovered_means = res["means"]
    all_estimated_source_probs[r] = recovered_source_probs
    all_estimated_means[r] = recovered_means


results = dict(
    all_estimated_source_probs=all_estimated_source_probs,
    all_estimated_means=all_estimated_means,
    true_source_probs=causal_moments.Pu,
    true_means=true_means,
)
pickle.dump(results, open("experiments/ate_experiment_main/results.pkl", "wb"))
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
# ==== RUN METHOD ====
nsamples = int(1e3)
xy_strengths = np.linspace(0, 1, 11)
nruns = 100

true_dists = dict()
estimated_source_probs = {xy_strength: np.zeros((nruns, 2)) for xy_strength in xy_strengths}
estimated_mtes = {xy_strength: np.zeros((nruns, 2)) for xy_strength in xy_strengths}
estimated_ates = {xy_strength: np.zeros(nruns) for xy_strength in xy_strengths}

for xy_strength in xy_strengths:
    generator = BinaryGeneratorMain(zt_strength=1, xy_strength=xy_strength)
    marginal = generator.true_marginal()
    causal_moments = compute_potential_outcome_moments_discrete(marginal, 1)
    true_dists[xy_strength] = marginal

    for r_ix in trange(nruns):
        # ==== GENERATE SAMPLES AND COMPUTE MOMENTS ====
        full_samples, obs_samples = generator.generate(nsamples=nsamples)
        new_samples = binary_feature_map_single(obs_samples)
        moments = compute_empirical_moments(generator.problem_dims, new_samples)

        # ==== RUN METHOD ====
        spo = SyntheticPotentialOutcomes(generator.problem_dims, decomposition_method="matrix_pencil")
        res = spo.fit(moments)

        # === SAVE RESULTS ===
        estimated_source_probs[xy_strength][r_ix] = res["source_probs"]
        estimated_mtes[xy_strength][r_ix] = res["means"]
        estimated_ates[xy_strength][r_ix] = res["causal_moments"][1]


results = dict(
    xy_strengths=xy_strengths,
    true_dists=true_dists,
    estimated_source_probs=estimated_source_probs,
    estimated_mtes=estimated_mtes,
    estimated_ates=estimated_ates
)
pickle.dump(results, open("experiments/level3_vs_level4/results.pkl", "wb"))
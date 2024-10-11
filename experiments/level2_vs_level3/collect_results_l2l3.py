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
from src.methods.tensor_decomposition import TensorDecompositionBinary


np.random.seed(123)
# ==== RUN METHOD ====
nsamples = int(1e3)
zt_strengths = np.linspace(0, 1, 11)
nruns = 100

true_dists = dict()
estimated_source_probs = {zt_strength: np.zeros((nruns, 2)) for zt_strength in zt_strengths}
estimated_mtes = {zt_strength: np.zeros((nruns, 2)) for zt_strength in zt_strengths}
estimated_mixtures = {zt_strength: dict() for zt_strength in zt_strengths}

for zt_strength in zt_strengths:
    generator = BinaryGeneratorMain(zt_strength=zt_strength, xy_strength=0)
    marginal = generator.true_marginal()
    causal_moments = compute_potential_outcome_moments_discrete(marginal, 1)
    true_dists[zt_strength] = marginal

    for r_ix in trange(nruns):
        # ==== GENERATE SAMPLES AND COMPUTE MOMENTS ====
        full_samples, obs_samples = generator.generate(nsamples=nsamples)
        new_samples = binary_feature_map_single(obs_samples)
        moments = compute_empirical_moments(generator.problem_dims, new_samples)

        # ==== RUN METHODS ====
        spo = SyntheticPotentialOutcomes(generator.problem_dims, decomposition_method="matrix_pencil")
        spo_res = spo.fit(moments)

        td = TensorDecompositionBinary(generator.problem_dims, decomposition_method="nn_parafac")
        td_res = td.fit(moments)

        # === SAVE RESULTS ===
        estimated_source_probs[zt_strength][r_ix] = spo_res["source_probs"]
        estimated_mtes[zt_strength][r_ix] = spo_res["means"]
        estimated_mixtures[zt_strength][r_ix] = td_res["mixture_moments"]


results = dict(
    zt_strengths=zt_strengths,
    true_dists=true_dists,
    estimated_source_probs=estimated_source_probs,
    estimated_mtes=estimated_mtes,
    estimated_mixtures=estimated_mixtures,
)
pickle.dump(results, open("experiments/level2_vs_level3/results.pkl", "wb"))
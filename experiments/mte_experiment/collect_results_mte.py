# === IMPORTS: BUILT-IN ===
import pickle

# === IMPORTS: THIRD-PARTY ===
import numpy as np
from tqdm import trange

# === IMPORTS: LOCAL ===
from src.problem_config import ProblemConfig
from src.data_generation.discrete_generator import DiscreteFixedGenerator

from src.population_moments_binary import PopulationMomentsBinary
from src.empirical_moments import EmpiricalMoments
from src.population_moments_binary import compute_source_probs_and_means
from src.methods.synthetic_potential_outcomes import SyntheticPotentialOutcomes


np.random.seed(123)
# ==== DEFINE THE PROBLEM CONFIGURATION ====
nproxies = 6
nmodifiers = 0
ngroups = 2
ntreatments = 2
config = ProblemConfig(nproxies, nmodifiers, ngroups, ntreatments)
xref = [0, 1]
xsyn1 = [2, 3]
xsyn2 = [4, 5]

# ==== DEFINE DATA GENERATOR ====
generator = DiscreteFixedGenerator(config, matching_coef=0.25, treatment_coef=0.25)
population_moments = PopulationMomentsBinary(config, generator.true_marginal())
true_source_probs, true_means = compute_source_probs_and_means(population_moments.p_ytu)

# ==== RUN METHOD ====
nsamples = int(5e5)
nruns = 100
all_estimated_source_probs = np.zeros((nruns, config.ngroups))
all_estimated_means = np.zeros((nruns, config.ngroups))
for r in trange(nruns):
    # ==== GENERATE SAMPLES AND COMPUTE MOMENTS ====
    full_samples, obs_samples = generator.generate(nsamples=nsamples)
    moments = EmpiricalMoments(config, obs_samples)
    expectations = moments.expectations
    conditional_second_moments = moments.conditional_second_moments
    conditional_third_moments = moments.conditional_third_moments

    # ==== RUN METHOD ====
    spo = SyntheticPotentialOutcomes(config, decomposition_method="matrix_pencil")
    res = spo.fit_fixed_partition(
        expectations, 
        conditional_second_moments, 
        conditional_third_moments,
        xref, 
        xsyn1,
        xsyn2
    )

    # === SAVE RESULTS ===
    recovered_source_probs = res["source_probs"]
    recovered_means = res["means"]
    all_estimated_source_probs[r] = recovered_source_probs
    all_estimated_means[r] = recovered_means


results = dict(
    all_estimated_source_probs=all_estimated_source_probs,
    all_estimated_means=all_estimated_means,
    true_source_probs=true_source_probs,
    true_means=true_means,
)
pickle.dump(results, open("experiments/mte_experiment/results.pkl", "wb"))
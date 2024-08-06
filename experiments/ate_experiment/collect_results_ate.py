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

# ==== DEFINE DATA GENERATOR ====
generator = DiscreteFixedGenerator(config, matching_coef=0.25, treatment_coef=0.25)
marginal = generator.true_marginal()
oracle_moments = PopulationMomentsBinary(config, marginal)
true_mean_y0 = oracle_moments.moments_Y0(1)[1]  # 0th moment is 1, 1st moment is mean
true_mean_y1 = oracle_moments.moments_Y1(1)[1]  # 0th moment is 1, 1st moment is mean

# ==== RUN METHOD ====
nsamples = int(1e5)
nruns = 100
y0_ests = np.zeros(nruns)
y1_ests = np.zeros(nruns)
for r in trange(nruns):
    # ==== GENERATE SAMPLES AND COMPUTE MOMENTS ====
    full_samples, obs_samples = generator.generate(nsamples=nsamples)
    moments = EmpiricalMoments(config, obs_samples)
    expectations = moments.expectations
    conditional_second_moments = moments.conditional_second_moments

    # ==== RUN METHOD ====
    spo = SyntheticPotentialOutcomes(config)
    y0_est, y1_est, _ = spo.only_first_step(expectations, conditional_second_moments, xref, xsyn1)
    y0_ests[r] = y0_est
    y1_ests[r] = y1_est


results = dict(
    true_mean_y0=true_mean_y0,
    true_mean_y1=true_mean_y1,
    y0_ests=y0_ests,
    y1_ests=y1_ests
)
pickle.dump(results, open("experiments/ate_experiment/results.pkl", "wb"))
# === IMPORTS: BUILT-IN ===
import pickle

# === IMPORTS: THIRD-PARTY ===
import numpy as np
from tqdm import trange

# === IMPORTS: LOCAL ===
from src.problem_dims import ProblemDimensions
from src.data_generation.discrete_generator import DiscreteFixedGenerator

from src.moments.population_moments_binary import PopulationMomentsBinary
from src.moments.empirical_moments import EmpiricalMoments
from src.methods.synthetic_potential_outcomes import SyntheticPotentialOutcomes


np.random.seed(123)
# ==== DEFINE THE PROBLEM CONFIGURATION ====
nz = 2
nx = 2
ngroups = 2
ntreatments = 2
problem_dims = ProblemDimensions(nz, nx, ngroups, ntreatments)

# ==== DEFINE DATA GENERATOR ====
generator = DiscreteFixedGenerator(problem_dims, matching_coef=0.25, treatment_coef=0.25)
marginal = generator.true_marginal()
oracle_moments = PopulationMomentsBinary(problem_dims, marginal)
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
    moments = EmpiricalMoments(problem_dims, obs_samples)
    expectations = moments.expectations
    conditional_second_moments = moments.conditional_second_moments

    # ==== RUN METHOD ====
    spo = SyntheticPotentialOutcomes(problem_dims)
    y0_est, y1_est = spo.only_first_step(expectations, conditional_second_moments)
    y0_ests[r] = y0_est
    y1_ests[r] = y1_est


results = dict(
    true_mean_y0=true_mean_y0,
    true_mean_y1=true_mean_y1,
    y0_ests=y0_ests,
    y1_ests=y1_ests
)
pickle.dump(results, open("experiments/ate_experiment/results.pkl", "wb"))
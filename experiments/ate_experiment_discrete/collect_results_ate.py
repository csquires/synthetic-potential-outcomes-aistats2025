# === IMPORTS: BUILT-IN ===
import pickle

# === IMPORTS: THIRD-PARTY ===
import numpy as np
from tqdm import trange

# === IMPORTS: LOCAL ===
from src.problem_dims import ProblemDimensions
from src.data_generation.binary_generator import BinaryGenerator

from src.causal_moments.causal_moments_discrete import compute_potential_outcome_moments_discrete
from src.observable_moments.empirical_moments import compute_empirical_moments
from src.methods.synthetic_potential_outcomes import SyntheticPotentialOutcomes


np.random.seed(123)
# ==== DEFINE THE PROBLEM CONFIGURATION ====
nz = 2
nx = 2
ngroups = 2
ntreatments = 2
problem_dims = ProblemDimensions(nz, nx, ngroups, ntreatments)

# ==== DEFINE DATA GENERATOR ====
generator = BinaryGenerator(problem_dims, matching_coef=0.25, treatment_coef=0.25)
marginal = generator.true_marginal()
y0_moments, y1_moments, r_moments, Pu = compute_potential_outcome_moments_discrete(marginal, 1)
true_mean_y0, true_mean_y1 = y0_moments[1], y1_moments[1]


# ==== RUN METHOD ====
nsamples = int(1e5)
nruns = 100
y0_ests = np.zeros(nruns)
y1_ests = np.zeros(nruns)
for r in trange(nruns):
    # ==== GENERATE SAMPLES AND COMPUTE MOMENTS ====
    full_samples, obs_samples = generator.generate(nsamples=nsamples)
    obs_moments = compute_empirical_moments(problem_dims, obs_samples)

    # ==== RUN METHOD ====
    spo = SyntheticPotentialOutcomes(problem_dims)
    y0_est, y1_est = spo.only_first_step(obs_moments)
    y0_ests[r] = y0_est
    y1_ests[r] = y1_est


results = dict(
    true_mean_y0=true_mean_y0,
    true_mean_y1=true_mean_y1,
    y0_ests=y0_ests,
    y1_ests=y1_ests
)
pickle.dump(results, open("experiments/ate_experiment/results.pkl", "wb"))
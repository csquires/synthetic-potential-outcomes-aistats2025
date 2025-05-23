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
causal_moments = compute_potential_outcome_moments_discrete(marginal, 1)
true_means = causal_moments.E_R_U

# ==== RUN METHOD ====
nsamples = int(5e5)
nruns = 100
all_estimated_source_probs = np.zeros((nruns, problem_dims.ngroups))
all_estimated_means = np.zeros((nruns, problem_dims.ngroups))
for r in trange(nruns):
    # ==== GENERATE SAMPLES AND COMPUTE MOMENTS ====
    full_samples, obs_samples = generator.generate(nsamples=nsamples)
    moments = compute_empirical_moments(problem_dims, obs_samples)

    # ==== RUN METHOD ====
    spo = SyntheticPotentialOutcomes(problem_dims, decomposition_method="matrix_pencil")
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
pickle.dump(results, open("experiments/mte_experiment_appendix/results.pkl", "wb"))
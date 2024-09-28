# === IMPORTS: THIRD-PARTY ===
import numpy as np

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

spo = SyntheticPotentialOutcomes(problem_dims)
res = spo.fit(oracle_moments)
print(res["source_probs"])
print(res["means"])

p_u = np.array([0.5, 0.5])
p_y_given_tu = generator.p_y_given_tu()
p_y0 = np.einsum("yu,u", p_y_given_tu[:, 0, :], p_u)[1]  # causal effect T=0
p_y1 = np.einsum("yu,u", p_y_given_tu[:, 1, :], p_u)[1]  # causal effect T=1
ate = p_y1 - p_y0

p_y0_u0 = p_y_given_tu[1, 0, 0]  # causal effect T=0, U=0
p_y0_u1 = p_y_given_tu[1, 0, 1]  # causal effect T=0, U=1
p_y1_u0 = p_y_given_tu[1, 1, 0]  # causal effect T=1, U=0
p_y1_u1 = p_y_given_tu[1, 1, 1]  # causal effect T=1, U=1
mte_u0 = p_y1_u0 - p_y0_u0
mte_u1 = p_y1_u1 - p_y0_u1
print(mte_u0)
print(mte_u1)
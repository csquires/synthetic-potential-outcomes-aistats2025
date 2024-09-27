# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.problem_dims import ProblemDimensions


class ContinuousFixedGenerator:
    def __init__(
            self, 
            problem_dims: ProblemDimensions,
        ):
        self.problem_dims = problem_dims
        self.matching_coef = 0.25
        self.treatment_coef = 0.25
        self.subgroup_coef = 0

    def generate(self, nsamples: int):
        nproxies = self.problem_dims.nz + self.problem_dims.nx
        u_ix = self.problem_dims.u_ix
        t_ix = self.problem_dims.t_ix
        y_ix = self.problem_dims.y_ix
        
        full_samples = np.ndarray((nsamples, nproxies + 3))
        # generate U
        u_vals = np.random.binomial(n=1, p=0.5, size=nsamples)
        # generate T
        t_vals = np.random.uniform(size=nsamples) < (3/4 - u_vals/2)
        # generate Y
        probs_y = 1/4 + self.matching_coef * (u_vals == t_vals) + self.treatment_coef * t_vals + self.subgroup_coef * u_vals
        y_vals = np.random.uniform(size=nsamples) < probs_y
        full_samples[:, u_ix] = u_vals
        full_samples[:, t_ix] = t_vals
        full_samples[:, y_ix] = y_vals

        # generate proxies
        for i in range(nproxies):
            if i % 2 == 0:
                ps = self.proxy_biases[i] + self.proxy_shift * u_vals
            else:
                ps = self.proxy_biases[i] + self.proxy_shift * (1 - u_vals)
            full_samples[:, i] = np.random.uniform(size=nsamples) < ps

        obs_samples = full_samples[:, :-1]
        return full_samples, obs_samples
    

if __name__ == "__main__":
    nproxies = 4
    config = ProblemDimensions(nproxies=nproxies, ngroups=2, ntreatments=2)
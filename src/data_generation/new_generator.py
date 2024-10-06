# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.problem_dims import ProblemDimensions


class ContinuousFixedGenerator:
    def __init__(
            self, 
            problem_dims: ProblemDimensions,
            xrange_u0: tuple[int] = (0, 1),
            zrange_u0: tuple[int] = (0, 1),
            xrange_u1: tuple[int] = (0.25, 1.25),
            zrange_u1: tuple[int] = (0.25, 1.25),
        ):
        self.problem_dims = problem_dims
        self.matching_coef = 0.25
        self.treatment_coef = 0.25
        self.subgroup_coef = 0

        self.xwidth_u0, self.xshift_u0 = xrange_u0[1] - xrange_u0[0], xrange_u0[0]
        self.xwidth_u1, self.xshift_u1 = xrange_u1[1] - xrange_u1[0], xrange_u1[0]
        self.zwidth_u0, self.zshift_u0 = zrange_u0[1] - zrange_u0[0], zrange_u0[0]
        self.zwidth_u1, self.zshift_u1 = zrange_u1[1] - zrange_u1[0], zrange_u1[0]

    def generate(self, nsamples: int):
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

        for z_ix in self.problem_dims.z_ixs:
            scales = self.zwidth_u0 * (1 - u_vals) + self.zwidth_u1 * u_vals
            shifts = self.zshift_u0 * (1 - u_vals) + self.zshift_u1 * u_vals
            z_vals = np.random.uniform(size=nsamples) * scales + shifts
            full_samples[:, z_ix] = z_vals

        for x_ix in self.problem_dims.x_ixs:
            scales = self.xwidth_u0 * (1 - u_vals) + self.xwidth_u1 * u_vals
            shifts = self.xshift_u0 * (1 - u_vals) + self.xshift_u1 * u_vals
            x_vals = np.random.uniform(size=nsamples) * scales + shifts
            full_samples[:, x_ix] = x_vals

        obs_samples = full_samples[:, :-1]
        return full_samples, obs_samples
    

if __name__ == "__main__":
    nproxies = 4
    config = ProblemDimensions(nproxies=nproxies, ngroups=2, ntreatments=2)
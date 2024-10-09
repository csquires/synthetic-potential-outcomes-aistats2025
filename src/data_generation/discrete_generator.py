# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.problem_dims import ProblemDimensions


class DiscreteGenerator:
    def __init__(
            self, 
            problem_dims: ProblemDimensions, 
            matching_coef: float = 0.25,
            treatment_coef: float = 0.25,
            subgroup_coef: float = 0
        ):
        self.problem_dims = problem_dims
        self.matching_coef = matching_coef
        self.treatment_coef = treatment_coef
        self.subgroup_coef = subgroup_coef
        nz, nx = problem_dims.nz, problem_dims.nx

        # P(U)
        self.Pu = np.ndarray([0.5, 0.5])

        # P(T | U)
        self.Pt_u = np.zeros((2, 2))
        self.Pt_u[:, 0] = np.array([1/4, 3/4])  # given U = 0
        self.Pt_u[:, 1] = np.array([3/4, 1/4])  # given U = 1

        # P(Y | T, U)
        a = self.matching_coef
        b = self.treatment_coef
        c = self.subgroup_coef
        self.Py_tu = np.zeros((2, 2, 2))
        self.Py_tu[:, 0, 0] = np.array([3/4 - a, 1/4 + a])  # given T=0, U=0
        self.Py_tu[:, 0, 1] = np.array([3/4 - c, 1/4 + c])  # given T=0, U=1
        self.Py_tu[:, 1, 0] = np.array([3/4 - b, 1/4 + b])  # given T=1, U=0
        self.Py_tu[:, 1, 1] = np.array([3/4 - a - b - c, 1/4 + a + b + c])  # given T=1, U=1

        # P(Z | U)
        self.Pz_u = np.zeros((nz, 2))
        self.Pz_u[:, 0] = np.arange(1, nz+1)
        self.Pz_u[:, 1] = np.arange(nz+1, 1, -1)
        self.Pz_u = np.einsum("zu,u->zu", self.Pz_u, self.Pz_u.sum(axis=0))

        # P(X | U)
        self.Px_u = np.zeros((nx, 2))
        self.Px_u[:, 0] = np.arange(1, nx+1)
        self.Px_u[:, 1] = np.arange(nx+1, 1, -1)
        self.Px_u = np.einsum("zu,u->zu", self.Px_u, self.Px_u.sum(axis=0))

    def generate(self, nsamples: int):
        z_ix = 0
        x_ix = 1
        y_ix = 2
        t_ix = 3
        u_ix = 4
        full_samples = np.ndarray((nsamples, 5))

        u_vals = np.random.binomial(n=1, p=0.5, size=nsamples)
        # P(T=1|U=0) = 3/4, P(T=1|U=1) = 1/4
        t_vals = np.random.uniform(size=nsamples) < (3/4 - u_vals/2)
        # P(Y=1|U=u,T=t) = 1/4 + a * 1(U=T) + b * T + c * U
        ps = 1/4 + self.matching_coef * (u_vals == t_vals) + self.treatment_coef * t_vals + self.subgroup_coef * u_vals
        y_vals = np.random.uniform(size=nsamples) < ps
        full_samples[:, u_ix] = u_vals
        full_samples[:, t_ix] = t_vals
        full_samples[:, y_ix] = y_vals

        full_samples[:, z_ix] = None
        full_samples[:, x_ix] = None

        obs_samples = full_samples[:, :-1]
        return full_samples, obs_samples
    
    def true_marginal(self):
        marginal = np.einsum(
            "zu,xu,ytu,tu,u->zxytu",
            self.Pz_u,
            self.Px_u,
            self.Py_tu, 
            self.Pt_u, 
            self.Pu
        )

        return marginal
    

if __name__ == "__main__":
    nproxies = 4
    config = ProblemDimensions(nproxies=nproxies, ngroups=2, ntreatments=2)
    generator = DiscreteGenerator(config)
    full_samples, obs_samples = generator.generate(1000)
    u0_samples = full_samples[full_samples[:, config.u_ix] == 0]
    u1_samples = full_samples[full_samples[:, config.u_ix] == 1]

    print(np.mean(u0_samples[:, :nproxies], axis=0))
    print(np.mean(u1_samples[:, :nproxies], axis=0))

    marginal = generator.true_marginal()
    print(marginal.sum(axis=(1, 2, 3, 4, 5, 6))[1])
    print(marginal.sum(axis=(0, 2, 3, 4, 5, 6))[1])
    print(marginal.sum(axis=(0, 1, 3, 4, 5, 6))[1])
    print(marginal.sum(axis=(0, 1, 2, 4, 5, 6))[1])

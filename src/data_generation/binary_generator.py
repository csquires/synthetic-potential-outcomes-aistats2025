# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.problem_dims import BinaryProblemDimensions


class BinaryGenerator:
    def __init__(
            self, 
            problem_dims: BinaryProblemDimensions, 
            matching_coef: float = 0.25,
            treatment_coef: float = 0.25,
            subgroup_coef: float = 0,
            proxy_biases: np.ndarray = None,
            proxy_shift: float = 0.3
        ):
        self.problem_dims = problem_dims
        self.matching_coef = matching_coef
        self.treatment_coef = treatment_coef
        self.subgroup_coef = subgroup_coef

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

        if proxy_biases is None:
            proxy_biases = np.linspace(0.2, 0.6, problem_dims.nz + problem_dims.nx)
        self.proxy_biases = proxy_biases
        self.proxy_shift = proxy_shift

    def generate(self, nsamples: int):
        nproxies = self.problem_dims.nz + self.problem_dims.nx
        y_ix = self.problem_dims.y_ix
        t_ix = self.problem_dims.t_ix
        u_ix = self.problem_dims.u_ix
        full_samples = np.ndarray((nsamples, nproxies + 3))

        u_vals = np.random.binomial(n=1, p=0.5, size=nsamples)
        # P(T=1|U=0) = 3/4, P(T=1|U=1) = 1/4
        t_vals = np.random.uniform(size=nsamples) < (3/4 - u_vals/2)
        # P(Y=1|U=u,T=t) = 1/4 + a * 1(U=T) + b * T + c * U
        ps = 1/4 + self.matching_coef * (u_vals == t_vals) + self.treatment_coef * t_vals + self.subgroup_coef * u_vals
        y_vals = np.random.uniform(size=nsamples) < ps
        full_samples[:, u_ix] = u_vals
        full_samples[:, t_ix] = t_vals
        full_samples[:, y_ix] = y_vals

        for i in range(nproxies):
            if i % 2 == 0:
                ps = self.proxy_biases[i] + self.proxy_shift * u_vals
            else:
                ps = self.proxy_biases[i] + self.proxy_shift * (1 - u_vals)
            full_samples[:, i] = np.random.uniform(size=nsamples) < ps

        obs_samples = full_samples[:, :-1]
        return full_samples, obs_samples
    
    def proxy_conditional(self, i):
        proxy_bias = self.proxy_biases[i]
        proxy_shift = self.proxy_shift
        P = np.zeros((2, 2))
        if i % 2 == 0:
            P[:, 0] = np.array([1 - proxy_bias, proxy_bias])  # given U=0
            P[:, 1] = np.array([1 - proxy_bias - proxy_shift, proxy_bias + proxy_shift])  # given U=1
        else:
            P[:, 0] = np.array([1 - proxy_bias - proxy_shift, proxy_bias + proxy_shift])  # given U=0
            P[:, 1] = np.array([1 - proxy_bias, proxy_bias])  # given U=1
        return P
    
    def true_marginal(self):
        proxy_conditionals = [self.proxy_conditional(i) for i in range(self.problem_dims.nx + self.problem_dims.nz)]
        
        current_marginal = np.einsum("ytu,tu,u->ytu", self.Py_tu, self.Pt_u, self.Pu)
        for proxy_conditional in reversed(proxy_conditionals):
            current_marginal = np.einsum("vu,...u->v...u", proxy_conditional, current_marginal)

        dz, dx = self.problem_dims.dz, self.problem_dims.dx
        dy, dt, du = 2, self.problem_dims.ntreatments, self.problem_dims.ngroups
        final_marginal = current_marginal.reshape(dz, dx, dy, dt, du)
        return final_marginal
    


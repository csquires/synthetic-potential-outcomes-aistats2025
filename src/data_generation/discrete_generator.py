# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.problem_dims import ProblemDimensions


class DiscreteFixedGenerator:
    def __init__(
            self, 
            problem_dims: ProblemDimensions, 
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
    
    def true_marginal(self):
        p_u = np.array([0.5, 0.5])
        p_t_given_u = np.array([
            [1/4, 3/4],  # given U=0
            [3/4, 1/4]   # given U=1
        ])
        a = self.matching_coef
        b = self.treatment_coef
        c = self.subgroup_coef
        p_y_given_tu = np.array([
            [[3/4 - a, 1/4 + a],  # given T=0, U=0
             [3/4 - c, 1/4 + c]], # given T=0, U=1  
            [[3/4 - b, 1/4 + b],  # given T=1, U=0
             [3/4 - a - b -c, 1/4 + a + b + c]]  # given T=1, U=1
        ])
        proxy_conditionals = []
        for i, proxy_bias in enumerate(self.proxy_biases):
            if i % 2 == 0:
                conditional = np.array([
                    [1-proxy_bias, proxy_bias],  # given U=0
                    [1-proxy_bias-self.proxy_shift, proxy_bias+self.proxy_shift]   # given U=1
                ])
            else:
                conditional = np.array([
                    [1-proxy_bias-self.proxy_shift, proxy_bias+self.proxy_shift],  # given U=0
                    [1-proxy_bias, proxy_bias]   # given U=1
                ])
            proxy_conditionals.append(conditional)
        
        current_marginal = np.einsum("tuy,ut,u->ytu", p_y_given_tu, p_t_given_u, p_u)
        for proxy_conditional in reversed(proxy_conditionals):
            current_marginal = np.einsum("uv,...u->v...u", proxy_conditional, current_marginal)

        return current_marginal
    

if __name__ == "__main__":
    nproxies = 4
    config = ProblemDimensions(nproxies=nproxies, ngroups=2, ntreatments=2)
    generator = DiscreteFixedGenerator(config)
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

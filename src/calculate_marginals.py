# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.problem_config import ProblemConfig



class MarginalsCalculator:
    def __init__(self, config: ProblemConfig, full_samples: np.ndarray):
        self.config = config
        self.full_samples = full_samples

        y_ix = self.config.y_ix
        t_ix = self.config.t_ix
        u_ix = self.config.u_ix


        # === DIVIDE SAMPLES BY U
        u0_samples = full_samples[full_samples[:, u_ix] == 0]
        u1_samples = full_samples[full_samples[:, u_ix] == 1]
        # === DIVIDE SAMPLES BY T
        t0_samples = full_samples[full_samples[:, t_ix] == 0]
        t1_samples = full_samples[full_samples[:, t_ix] == 1]
        # === DIVIDE SAMPLES BY T AND U
        u0_t0_samples = full_samples[(full_samples[:, u_ix] == 0) & (full_samples[:, t_ix] == 0)]
        u1_t0_samples = full_samples[(full_samples[:, u_ix] == 1) & (full_samples[:, t_ix] == 0)]
        u0_t1_samples = full_samples[(full_samples[:, u_ix] == 0) & (full_samples[:, t_ix] == 1)]
        u1_t1_samples = full_samples[(full_samples[:, u_ix] == 1) & (full_samples[:, t_ix] == 1)]

        # === MARGINALS OF U
        self.p_u1 = full_samples[:, u_ix].mean()
        self.p_u0 = 1 - self.p_u1

        # === MARGINALS OF T
        self.p_t1 = full_samples[:, t_ix].mean()
        self.p_t0 = 1 - self.p_t1

        # === CONDITIONAL OF U ON T
        self.p_u1_given_t0 = t0_samples[:, u_ix].mean()
        self.p_u1_given_t1 = t1_samples[:, u_ix].mean()

        # === CONDITIONAL OF T ON U
        self.p_t1_given_u0 = u0_samples[:, t_ix].mean()
        self.p_t0_given_u0 = 1 - self.p_t1_given_u0
        self.p_t1_given_u1 = u1_samples[:, t_ix].mean()
        self.p_t0_given_u1 = 1 - self.p_t1_given_u1

        # === CONDITIONAL OF Y ON U
        self.p_y1_given_u0 = u0_samples[:, y_ix].mean()
        self.p_y1_given_u1 = u1_samples[:, y_ix].mean()

        # === CONDITIONAL OF U AND T
        self.p_y1_given_u0_t0 = u0_t0_samples[:, y_ix].mean()
        self.p_y1_given_u1_t0 = u1_t0_samples[:, y_ix].mean()
        self.p_y1_given_u0_t1 = u0_t1_samples[:, y_ix].mean()
        self.p_y1_given_u1_t1 = u1_t1_samples[:, y_ix].mean()

        # === INTERVENTIONAL DISTRIBUTIONS
        self.p_y_do_t0 = self.p_u0 * self.p_y1_given_u0_t0 + self.p_u1 * self.p_y1_given_u1_t0
        self.p_y_do_t1 = self.p_u0 * self.p_y1_given_u0_t1 + self.p_u1 * self.p_y1_given_u1_t1


    def print_dists(self):
        print("P(U=0) = ", self.p_u1)
        print("P(U=1) = ", self.p_u1)
    
        print('===========================')
        print("P(T=0) = ", self.p_t1)
        print("P(T=1) = ", self.p_t0)

        print('===========================')
        print("P(U=1 | T=0) = ", self.p_u1_given_t0)
        print("P(U=1 | T=1) = ", self.p_u1_given_t1)

        print('===========================')
        print("P(T=1|U=0) = ", self.p_t1_given_u0)
        print("P(T=1|U=1) = ", self.p_t1_given_u1)
    
        print('===========================')
        print("P(Y=1 | U=0) = ", self.p_y1_given_u0)
        print("P(Y=1 | U=1) = ", self.p_y1_given_u1)
    
        print('===========================')
        print("P(Y=1|U=0, T=0) = ", self.p_y1_given_u0_t0)
        print("P(Y=1|U=1, T=0) = ", self.p_y1_given_u1_t0)
        print("P(Y=1|U=0, T=1) = ", self.p_y1_given_u0_t1)
        print("P(Y=1|U=1, T=1) = ", self.p_y1_given_u1_t1)

        print('===========================')
        print("P(Y=1|DO(T=0)) = ", self.p_y_do_t0)
        print("P(Y=1|DO(T=1)) = ", self.p_y_do_t1)
    

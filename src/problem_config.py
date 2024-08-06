# === IMPORTS: BUILT-IN ===
from dataclasses import dataclass


@dataclass
class ProblemConfig:
    nproxies: int  # number of proxies
    nmodifiers: int  # number of effect modifiers
    ngroups: int = 2  # number of values for the latent confounder
    ntreatments: int = 2

    def __post_init__(self):
        # X: proxies
        self.x_ixs = list(range(self.nproxies))
        # M: effect modifiers
        self.m_ixs = list(range(self.nproxies, self.nproxies + self.nmodifiers))
        # Y: outcome
        self.y_ix = self.nproxies + self.nmodifiers
        # T: treatment
        self.t_ix = self.nproxies + self.nmodifiers + 1
        # U: unobserved confounder
        self.u_ix = self.nproxies + self.nmodifiers + 2

        self.xmy_ixs = tuple(range(self.nproxies + self.nmodifiers + 1))
# === IMPORTS: BUILT-IN ===
from dataclasses import dataclass


@dataclass
class ProblemDimensions:
    nz: int  # number of Z variables
    nx: int  # number of X variables
    ngroups: int = 2  # number of values for the latent confounder
    ntreatments: int = 2

    def __post_init__(self):
        # Z: proxies that can cause T
        self.z_ixs = list(range(self.nx))
        # X: proxies that are independent
        self.x_ixs = list(range(self.nz, self.nz + self.nx))
        # Y: outcome
        self.y_ix = self.nz + self.nx
        # T: treatment
        self.t_ix = self.nx + self.nx + 1
        # U: unobserved confounder
        self.u_ix = self.nx + self.nx + 2

        self.zxy_ixs = self.z_ixs + self.x_ixs + [self.y_ix]
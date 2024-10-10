# === IMPORTS: BUILT-IN ===
from dataclasses import dataclass


@dataclass
class BinaryProblemDimensions:
    nz: int  # number of Z variables
    nx: int  # number of X variables
    ngroups: int = 2  # number of values for the latent confounder
    ntreatments: int = 2

    def __post_init__(self):
        self.dz = 2 ** self.nz
        self.dx = 2 ** self.nz

        # Z: proxies that can cause T
        self.z_ixs = list(range(self.dz))
        # X: proxies that are independent
        self.x_ixs = list(range(self.dz, self.dz + self.dx))
        # Y: outcome
        self.y_ix = self.dz + self.dx
        # T: treatment
        self.t_ix = self.dz + self.dx + 1
        # U: unobserved confounder
        self.u_ix = self.dz + self.dx + 2

        self.zxy_ixs = self.z_ixs + self.x_ixs + [self.y_ix]


@dataclass
class GeneralProblemDimensions:
    dz: int  # dimensionality of Z (after mapping)
    dx: int  # dimensionality of X (after mapping)
    ngroups: int = 2  # number of values for the latent confounder
    ntreatments: int = 2

    def __post_init__(self):
        # Z: proxies that can cause T
        self.z_ixs = list(range(self.dz))
        # X: proxies that are independent
        self.x_ixs = list(range(self.dz, self.dz + self.dx))
        # Y: outcome
        self.y_ix = self.dz + self.dx
        # T: treatment
        self.t_ix = self.dz + self.dx + 1
        # U: unobserved confounder
        self.u_ix = self.dz + self.dx + 2

        self.zxy_ixs = self.z_ixs + self.x_ixs + [self.y_ix]

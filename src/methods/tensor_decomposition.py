# === IMPORTS: THIRD-PARTY ===
import numpy as np
from tensorly.decomposition import parafac, parafac_power_iteration

# === IMPORTS: LOCAL ===
from src.problem_dims import GeneralProblemDimensions
from src.observable_moments import ObservableMoments
from src.mixture_moments import MixtureMoments


class TensorDecomposition:
    def __init__(
            self, 
            problem_dims: GeneralProblemDimensions,
            decomposition_method: str = "parafac"
    ):
        self.problem_dims = problem_dims
        self.decomposition_method = decomposition_method

    def fit(self, obs_moments: ObservableMoments):
        dz, dx, dt = self.problem_dims.dz, self.problem_dims.dx, 2
        M_aug = np.zeros((dz+1, dx+1, dt+1))

        M_aug[:dz, :dx, :dt] = obs_moments.M_ZXtY
        # second moments
        M_aug[:dz, :dx, -1] = obs_moments.M_ZX
        M_aug[:dz, -1, :dt] = obs_moments.M_ZtY
        M_aug[-1, :dx, :dt] = obs_moments.M_XtY
        # first moments
        M_aug[:dz, -1, -1] = obs_moments.E_Z
        M_aug[-1, :dx, -1] = obs_moments.E_X
        M_aug[-1, -1, :dt] = obs_moments.E_tY
        # zero-th moment
        M_aug[-1, -1, -1] = 1

        rank = 2

        if self.decomposition_method == "parafac":
            res = parafac(M_aug, rank, n_iter_max=10000)
            weights = res.weights
            factors = res.factors
            Z_factor = factors[0]
            X_factor = factors[1]
            tY_factor = factors[2]
        elif self.decomposition_method == "parafac_power":
            res = parafac_power_iteration(M_aug, rank)
            weights, factors = res[0], res[1]
            Z_factor = factors[0]
            X_factor = factors[1]
            tY_factor = factors[2]
        else:
            raise ValueError

        return MixtureMoments(
            Pu=weights,
            E_Z_U={u: Z_factor[:, u] for u in [0, 1]},
            E_X_U={u: X_factor[:, u] for u in [0, 1]},
            E_tY_U={u: tY_factor[:, u] for u in [0, 1]},
        ) 

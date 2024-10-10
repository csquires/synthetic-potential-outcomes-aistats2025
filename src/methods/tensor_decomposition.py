# === IMPORTS: THIRD-PARTY ===
import numpy as np
from tensorly.decomposition import parafac, parafac_power_iteration
from tensorly.cp_tensor import cp_to_tensor

# === IMPORTS: LOCAL ===
from src.problem_dims import GeneralProblemDimensions
from src.observable_moments import ObservableMoments
from src.mixture_moments import MixtureMoments


class TensorDecompositionBinary:
    def __init__(
            self, 
            problem_dims: GeneralProblemDimensions,
            decomposition_method: str = "parafac"
    ):
        self.problem_dims = problem_dims
        self.decomposition_method = decomposition_method

    def fit(self, obs_moments: ObservableMoments, check_recovery=False):
        rank = 2
        if self.decomposition_method == "parafac":
            res = parafac(obs_moments.M_ZXS, rank, n_iter_max=10000, init="random")
            weights = res.weights
            factors = res.factors
            Z_factor = factors[0]
            X_factor = factors[1]
            S_factor = factors[2]
        elif self.decomposition_method == "parafac_power":
            res = parafac_power_iteration(obs_moments.M_ZXS, rank)
            weights, factors = res[0], res[1]
            Z_factor = factors[0]
            X_factor = factors[1]
            S_factor = factors[2]
        else:
            raise ValueError
    
        recovery_error = np.max(np.abs(cp_to_tensor(res) - obs_moments.M_ZXS))
        if check_recovery:
            print("difference between M_ZXS and recovered:", recovery_error)
            if recovery_error > 0.0001:
                breakpoint()

        Zscale = Z_factor.sum(axis=0)
        Xscale = X_factor.sum(axis=0)
        Sscale = S_factor.sum(axis=0)
        EZ_U = np.einsum("zu,u->zu", Z_factor, Zscale ** -1)
        EX_U = np.einsum("xu,u->xu", X_factor, Xscale ** -1)
        ES_U = np.einsum("su,u->su", S_factor, Sscale ** -1)
        Pu = np.einsum("u,u,u,u->u", weights, Zscale, Xscale, Sscale)
        
        mixture_moments = MixtureMoments(
            Pu=Pu,
            EZ_U=EZ_U,
            EX_U=EX_U,
            ES_U=ES_U,
            E_Z_U={u: EZ_U[:, u] for u in [0, 1]},
            E_X_U={u: EX_U[:, u] for u in [0, 1]},
            E_S_U={u: ES_U[:, u] for u in [0, 1]},
        ) 
        return dict(
            mixture_moments=mixture_moments,
            recovery_error=recovery_error
        )


class TensorDecomposition:
    def __init__(
            self, 
            problem_dims: GeneralProblemDimensions,
            decomposition_method: str = "parafac"
    ):
        self.problem_dims = problem_dims
        self.decomposition_method = decomposition_method

    def fit(self, obs_moments: ObservableMoments):
        dz, dx, ds = self.problem_dims.dz, self.problem_dims.dx, 4
        M_aug = np.zeros((dz+1, dx+1, ds+1))

        M_aug[:dz, :dx, :ds] = obs_moments.M_ZXS
        # second moments
        M_aug[:dz, :dx, -1] = obs_moments.M_ZX
        M_aug[:dz, -1, :ds] = obs_moments.M_ZS
        M_aug[-1, :dx, :ds] = obs_moments.M_XS
        # first moments
        M_aug[:dz, -1, -1] = obs_moments.E_Z
        M_aug[-1, :dx, -1] = obs_moments.E_X
        M_aug[-1, -1, :ds] = obs_moments.E_S
        # zero-th moment
        M_aug[-1, -1, -1] = 1

        rank = 2

        if self.decomposition_method == "parafac":
            res = parafac(M_aug, rank, n_iter_max=10000)
            weights = res.weights
            factors = res.factors
            Z_factor = factors[0]
            X_factor = factors[1]
            S_factor = factors[2]
        elif self.decomposition_method == "parafac_power":
            res = parafac_power_iteration(M_aug, rank)
            weights, factors = res[0], res[1]
            Z_factor = factors[0]
            X_factor = factors[1]
            S_factor = factors[2]
        else:
            raise ValueError

        Zscale = Z_factor[-1, :]
        Xscale = X_factor[-1, :]
        Sscale = S_factor[-1, :]
        EZ_U = np.einsum("zu,u->zu", Z_factor, Zscale ** -1)
        EX_U = np.einsum("xu,u->xu", X_factor, Xscale ** -1)
        ES_U = np.einsum("su,u->su", S_factor, Sscale ** -1)
        Pu = np.einsum("u,u,u,u->u", weights, Zscale, Xscale, Sscale)
        return MixtureMoments(
            Pu=Pu,
            EZ_U=EZ_U,
            EX_U=EX_U,
            ES_U=ES_U,
            E_Z_U={u: EZ_U[:, u] for u in [0, 1]},
            E_X_U={u: EX_U[:, u] for u in [0, 1]},
            E_S_U={u: ES_U[:, u] for u in [0, 1]},
        ) 

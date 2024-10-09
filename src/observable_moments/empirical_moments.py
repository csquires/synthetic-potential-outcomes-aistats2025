# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.problem_dims import ProblemDimensions
from src.observable_moments import ObservableMoments


def binary_feature_map_single(obs_samples):
    assert obs_samples.shape[1] == 4

    new_samples = np.zeros((obs_samples.shape[0], 6))
    # Z
    new_samples[:, 0] = 1 - obs_samples[:, 0]
    new_samples[:, 1] = obs_samples[:, 0]
    # X
    new_samples[:, 2] = 1 - obs_samples[:, 1]
    new_samples[:, 3] = obs_samples[:, 1]
    # Y
    new_samples[:, 4] = obs_samples[:, 2]
    # U
    new_samples[:, 5] = obs_samples[:, 3]

    return new_samples


def compute_empirical_moments(
    problem_dims: ProblemDimensions,
    obs_samples: np.ndarray
):
    # === UNCONDITIONAL ===
    Zsamples = obs_samples[:, problem_dims.z_ixs]
    Xsamples = obs_samples[:, problem_dims.x_ixs]
    Ysamples = obs_samples[:, problem_dims.y_ix]
    Tsamples = (obs_samples[:, problem_dims.t_ix] == np.array([[0], [1]])).T
    tYsamples = np.einsum("i,it->it", Ysamples, Tsamples)
    nsamples = obs_samples.shape[0]

    # first moments
    E_Z = Zsamples.mean(axis=0)
    E_X = Xsamples.mean(axis=0)
    E_tY = tYsamples.mean(axis=0)
    # second moments
    M_ZX = np.einsum("iz,ix->zx", Zsamples, Xsamples) / nsamples
    M_ZtY = np.einsum("iz,it->zt", Zsamples, tYsamples) / nsamples
    M_XtY = np.einsum("ix,it->xt", Xsamples, tYsamples) / nsamples
    # third moments
    M_ZXtY = np.einsum("iz,ix,it->zxt", Zsamples, Xsamples, tYsamples) / nsamples

    # === CONDITIONAL ===
    E_Z_T = dict()
    E_X_T = dict()
    M_ZX_T = dict()
    M_ZY_T = dict()
    M_ZXY_T = dict()
    for t in range(problem_dims.ntreatments):
        t_ixs = obs_samples[:, problem_dims.t_ix] == t
        nsamples_t = sum(t_ixs)
        E_Z_T[t] = Zsamples[t_ixs].mean(axis=0)
        E_X_T[t] = Xsamples[t_ixs].mean(axis=0)
        M_ZX_T[t] = np.einsum("iz,ix->zx", Zsamples[t_ixs], Xsamples[t_ixs]) / nsamples_t
        M_ZY_T[t] = np.einsum("iz,i->z", Zsamples[t_ixs], Ysamples[t_ixs]) / nsamples_t
        M_ZXY_T[t] = np.einsum("iz,ix,i->zx", Zsamples[t_ixs], Xsamples[t_ixs], Ysamples[t_ixs]) / nsamples_t

    return ObservableMoments(
        # first moments
        E_Z, 
        E_X, 
        E_tY,
        # second moments
        M_ZX, 
        M_ZtY, 
        M_XtY,
        # third moments
        M_ZXtY,
        # conditional first moments
        E_Z_T, 
        E_X_T,
        # conditional second moments
        M_ZX_T,
        M_ZY_T,
        # conditional third moments
        M_ZXY_T
    )

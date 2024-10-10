# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.observable_moments import ObservableMoments


def compute_observable_moments_discrete(full_marginal: np.ndarray):
    ntreatments = full_marginal.shape[3]

    Pzxyt = np.einsum("zxytu->zxyt", full_marginal)
    dz, dx, _, _ = Pzxyt.shape
    
    # === OBSERVED ===
    # 3-way marginals
    Pzxy = np.einsum("zxyt->zxy", Pzxyt)
    Pzxt = np.einsum("zxyt->zxt", Pzxyt)
    Pzyt = np.einsum("zxyt->zyt", Pzxyt)
    Pxyt = np.einsum("zxyt->xyt", Pzxyt)
    Pzxs = Pzxyt.reshape((dz, dx, 4))

    # 2-way marginals
    Pzx = np.einsum("zxy->zx", Pzxy)
    Pzt = np.einsum("zxt->zt", Pzxt)
    Pxt = np.einsum("zxt->xt", Pzxt)
    Pyt = np.einsum("xyt->yt", Pxyt)
    Pzs = np.einsum("zxs->zs", Pzxs)
    Pxs = np.einsum("zxs->xs", Pzxs)

    # univariate marginals
    Pz = np.einsum("zx->z", Pzx)
    Px = np.einsum("zx->x", Pzx)
    Pt = np.einsum("yt->t", Pyt)
    Ps = np.einsum("zs->s", Pzs)
    
    Pz_t = np.einsum("zt,t->zt", Pzt, Pt ** -1)
    Px_t = np.einsum("xt,t->xt", Pxt, Pt ** -1)
    Pzx_t = np.einsum("zxt,t->zxt", Pzxt, Pt ** -1)
    Pzy_t = np.einsum("zyt,t->zyt", Pzyt, Pt ** -1)
    Pzxy_t = np.einsum("zxyt,t->zxyt", Pzxyt, Pt ** -1)

    return ObservableMoments(
        # first moments
        E_Z=Pz, 
        E_X=Px, 
        E_S=Ps, 
        # second moments
        M_ZX=Pzx,
        M_ZS=Pzs,
        M_XS=Pxs,
        # third moments
        M_ZXS=Pzxs,
        # conditional first moments
        E_Z_T={t: Pz_t[:, t] for t in range(ntreatments)},
        E_X_T={t: Px_t[:, t] for t in range(ntreatments)},
        # conditional second moments
        M_ZX_T={t: Pzx_t[:, :, t] for t in range(ntreatments)},
        M_ZY_T={t: Pzy_t[:, 1, t] for t in range(ntreatments)},
        # conditional third moments
        M_ZXY_T={t: Pzxy_t[:, :, 1, t] for t in range(ntreatments)}
    )
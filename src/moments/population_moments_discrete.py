# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.moments.moments import Moments


def get_population_moments(full_marginal: np.ndarray):
    ntreatments = full_marginal.shape[3]

    Pzxyt = np.einsum("zxytu->zxyt", full_marginal)
    
    # === OBSERVED ===
    # 3-way marginals
    Pzxy = np.einsum("zxyt->zxy", Pzxyt)
    Pzxt = np.einsum("zxyt->zxt", Pzxyt)
    Pzyt = np.einsum("zxyt->zyt", Pzxyt)
    Pxyt = np.einsum("zxyt->xyt", Pzxyt)

    # 2-way marginals
    Pzx = np.einsum("zxy->zx", Pzxy)
    Pzt = np.einsum("zxt->zt", Pzxt)
    Pxt = np.einsum("zxt->xt", Pzxt)
    Pyt = np.einsum("xyt->yt", Pxyt)

    # univariate marginals
    Pz = np.einsum("zx->z", Pzx)
    Px = np.einsum("zx->x", Pzx)
    Pt = np.einsum("yt->t", Pyt)
    
    Pz_t = np.einsum("zt,t->zt", Pzt, Pt ** -1)
    Px_t = np.einsum("xt,t->xt", Pxt, Pt ** -1)
    Pzx_t = np.einsum("zxt,t->zxt", Pzxt, Pt ** -1)
    Pzxy_t = np.einsum("zxyt,t->zxyt", Pzxyt, Pt ** -1)

    return Moments(
        # first moments
        E_Z=Pz, 
        E_X=Px, 
        E_tY=Pyt[1, :], 
        # second moments
        M_ZX=Pzx,
        M_ZtY=Pzyt[:, 1, :],
        M_XtY=Pxyt[:, 1, :],
        # third moments
        M_ZXtY=Pzxyt[:, :, 1, :],
        # conditional first moments
        E_Z_T={t: Pz_t[:, t] for t in range(ntreatments)},
        E_X_T={t: Px_t[:, t] for t in range(ntreatments)},
        # conditional second moments
        M_ZX_T={t: Pzx_t[:, :, t] for t in range(ntreatments)},
        # conditional third moments
        MZXY_T={t: Pzxy_t[:, :, 1, t] for t in range(ntreatments)}
    )



    # def moments_Y1(self, max_order: int):
    #     moments = [1]
    #     for order in range(1, max_order+1):
    #         mean_y1_given_u = np.einsum("u,u->u", self.Pytu[1, 1, :], self.Ptu[1, :]**-1)
    #         moments_y1_given_u = mean_y1_given_u ** order
    #         moment = np.einsum("u,u", self.Pu, moments_y1_given_u)
    #         moments.append(moment)
    #     return moments

    # def moments_Y0(self, max_order: int):
    #     moments = [1]
    #     for order in range(1, max_order+1):
    #         mean_y0_given_u = np.einsum("u,u->u", self.Pytu[1, 0, :], self.Ptu[0, :]**-1)
    #         moments_y0_given_u = mean_y0_given_u ** order
    #         moment = np.einsum("u,u", self.Pu, moments_y0_given_u)
    #         moments.append(moment)
    #     return moments
    
    # def moments_R(self, max_order: int):
    #     moments = [1]
    #     for order in range(1, max_order+1):
    #         mean_r_given_u = np.einsum("u,u->u", self.Pytu[1, 1, :] - self.Pytu[1, 0, :], self.Ptu[1, :]**-1)
    #         moments_r_given_u = mean_r_given_u ** order
    #         moment = np.einsum("u,u", self.Pu, moments_r_given_u)
    #         moments.append(moment)
    #     return moments
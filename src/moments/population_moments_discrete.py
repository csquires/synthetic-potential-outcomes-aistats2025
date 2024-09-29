# === IMPORTS: BUILT-IN ===
from typing import Dict

# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.moments.moments import Moments



class PopulationMomentsDiscrete(Moments):
    def __init__(self, full_marginal: np.ndarray):
        self.dz = full_marginal.shape[0]
        self.dx = full_marginal.shape[1]
        self.ntreatments = full_marginal.shape[3]
        self.ngroups = full_marginal.shape[4]

        self.full_marginal = full_marginal  # (z, x, y, t, u)
        self.Pzxyt = np.einsum("zxytu->zxyt", full_marginal)
        
        # === OBSERVED ===
        # 3-way marginals
        self.Pzxy = np.einsum("zxyt->zxy", self.Pzxyt)
        self.Pzxt = np.einsum("zxyt->zxt", self.Pzxyt)
        self.Pzyt = np.einsum("zxyt->zyt", self.Pzxyt)
        self.Pxyt = np.einsum("zxyt->xyt", self.Pzxyt)

        # 2-way marginals
        self.Pzx = np.einsum("zxy->zx", self.Pzxy)
        self.Pzy = np.einsum("zxy->zy", self.Pzxy)
        self.Pzt = np.einsum("zxt->zt", self.Pzxt)
        self.Pxt = np.einsum("zxt->xt", self.Pzxt)
        self.Pyt = np.einsum("xyt->yt", self.Pxyt)

        # univariate marginals
        self.Pz = np.einsum("zx->z", self.Pzx)
        self.Px = np.einsum("zx->x", self.Pzx)
        self.Py = np.einsum("zy->y", self.Pzy)
        self.Pt = np.einsum("yt->t", self.Pyt)
        
        # === UNOBSERVED ===
        # 3-way marginals
        self.Pzxu = np.einsum("zxytu->zxu", full_marginal)
        self.Pytu = np.einsum("zxytu->ytu", full_marginal)
        # 2-way marginals
        self.Pzu = np.einsum("zxu->zu", self.Pzxu)
        self.Pxu = np.einsum("zxu->xu", self.Pzxu)
        self.Pyu = np.einsum("ytu->yu", self.Pytu)
        self.Ptu = np.einsum("ytu->tu", self.Pytu)
        # univariate marginals
        self.Pu = np.einsum("tu->u", self.Ptu)

        self.Pz_t = np.einsum("zt,t->zt", self.Pzt, self.Pt ** -1)
        self.Px_t = np.einsum("xt,t->xt", self.Pxt, self.Pt ** -1)
        self.Pzx_t = np.einsum("zxt,t->zxt", self.Pzxt, self.Pt ** -1)
        self.Pzxy_t = np.einsum("zxyt,t->zxyt", self.Pzxyt, self.Pt ** -1)

    def moments_Y1(self, max_order: int):
        moments = [1]
        for order in range(1, max_order+1):
            mean_y1_given_u = np.einsum("u,u->u", self.Pytu[1, 1, :], self.Ptu[1, :]**-1)
            moments_y1_given_u = mean_y1_given_u ** order
            moment = np.einsum("u,u", self.Pu, moments_y1_given_u)
            moments.append(moment)
        return moments

    def moments_Y0(self, max_order: int):
        moments = [1]
        for order in range(1, max_order+1):
            mean_y0_given_u = np.einsum("u,u->u", self.Pytu[1, 0, :], self.Ptu[0, :]**-1)
            moments_y0_given_u = mean_y0_given_u ** order
            moment = np.einsum("u,u", self.Pu, moments_y0_given_u)
            moments.append(moment)
        return moments
    
    def moments_R(self, max_order: int):
        moments = [1]
        for order in range(1, max_order+1):
            mean_r_given_u = np.einsum("u,u->u", self.Pytu[1, 1, :] - self.Pytu[1, 0, :], self.Ptu[1, :]**-1)
            moments_r_given_u = mean_r_given_u ** order
            moment = np.einsum("u,u", self.Pu, moments_r_given_u)
            moments.append(moment)
        return moments
    
    @property
    def E_Z(self) -> np.ndarray:
        """
        Return [P(Z=1), P(Z=2), ..., P(Z=dz)]
        """
        return self.Pz
    
    @property
    def E_X(self) -> np.ndarray:
        """
        Return [P(X=1), P(X=2), ..., P(X=dx)]
        """
        return self.Px
    
    @property
    def E_tY(self) -> np.ndarray:
        """
        Return [P(Y=1, T=1), P(Y=1, T=2), ...]
        """
        return self.Pyt[1, :]
    
    @property
    def E_Z_T(self) -> Dict[int, np.ndarray]:
        """
        Return 
        {t: 
            [P(Z=1 | Z=t), P(Z=2 | T=t), ..., P(Z=dz | T=t)]
        }
        """
        return {t: self.Pz_t[:, t] for t in range(self.ntreatments)}
    
    @property
    def E_X_T(self) -> Dict[int, np.ndarray]:
        """
        Return 
        {t: 
            [P(X=1 | T=t), P(X=2 | T=t), ..., P(X=dx | T=t)]
        }
        """
        return {t: self.Px_t[:, t] for t in range(self.ntreatments)}
    
    @property
    def M_ZX(self) -> np.ndarray:
        """
        Return 
        [[P(Z=1, X=1), P(Z=1, X=2), ..., P(Z=1, X=dx)],
            [P(Z=2, X=1), P(Z=2, X=2), ..., P(Z=2, X=dx)],
            ...
            [P(Z=dz, X=1), P(Z=dz, X=2), ..., P(Z=dz, X=dx)]]
        """
        return self.Pzx
    
    @property
    def M_ZtY(self) -> np.ndarray:
        """
        Return 
        [[P(Z=1, Y=1, T=1), P(Z=1, Y=1, T=2), ..., P(Z=1, Y=1, T=dt)],
         [P(Z=2, Y=1, T=2), P(Z=2, Y=1, T=2), ..., P(Z=2, Y=1, T=dt)],
         ...
         [P(Z=dz, Y=1, T=2), P(Z=dz, Y=1, T=2), ..., P(Z=dz, Y=1, T=dt)],]
        """
        return self.Pzyt[:, 1, :]
    
    @property
    def M_XtY(self) -> np.ndarray:
        """
        Return 
        [[P(X=1, Y=1, T=1), P(X=1, Y=1, T=2), ..., P(X=1, Y=1, T=dt)],
         [P(X=2, Y=1, T=2), P(X=2, Y=1, T=2), ..., P(X=2, Y=1, T=dt)],
         ...
         [P(X=dx, Y=1, T=2), P(X=dx, Y=1, T=2), ..., P(X=dx, Y=1, T=dt)],]
        """
        return self.Pxyt[:, 1, :]
    
    @property
    def M_ZX_T(self) -> Dict[int, np.ndarray]:
        """
        Return 
        {t: 
            [[P(Z=1, X=1 | T=t), P(Z=1, X=2 | T=t), ..., P(Z=1, X=dx | T=t)],
             [P(Z=2, X=1 | T=t), P(Z=2, X=2 | T=t), ..., P(Z=2, X=dx | T=t)],
             ...
             [P(Z=dz, X=1 | T=t), P(Z=dz, X=2 | T=t), ..., P(Z=dz, X=dx | T=t)]]
        }
        """
        return {t: self.Pzx_t[:, :, t] for t in range(self.ntreatments)}
    
    @property
    def M_ZXY_T(self) -> Dict[int, np.ndarray]:
        """
        Return 
        {t: 
            [[P(Z=1, X=1, Y=1 | T=t), P(Z=1, X=2, Y=1 | T=t), ..., P(Z=1, X=dx, Y=1 | T=t)],
             [P(Z=2, X=1, Y=1 | T=t), P(Z=2, X=2, Y=1 | T=t), ..., P(Z=2, X=dx, Y=1 | T=t)],
             ...
             [P(Z=dz, X=1, Y=1 | T=t), P(Z=dz, X=2, Y=1 | T=t), ..., P(Z=dz, X=dx, Y=1 | T=t)]]
        }
        """
        return {t: self.Pzxy_t[:, :, 1, t] for t in range(self.ntreatments)}
    
    @property
    def M_ZXtY(self) -> np.ndarray:
        """
        Return
        [
            [[P(Z=1, X=1, Y=1, T=1), P(Z=1, X=2, Y=1, T=1), ..., P(Z=1, X=dx, Y=1, T=1)],
             [P(Z=2, X=1, Y=1, T=1), P(Z=2, X=2, Y=1, T=1), ..., P(Z=2, X=dx, Y=1, T=1)],
             ...
             [P(Z=dz, X=1, Y=1, T=1), P(Z=dz, X=2, Y=1, T=1), ..., P(Z=dz, X=dx, Y=1, T=1)]],
            [[P(Z=1, X=1, Y=1, T=2), P(Z=1, X=2, Y=1, T=2), ..., P(Z=1, X=dx, Y=1, T=2)],
             [P(Z=2, X=1, Y=1, T=2), P(Z=2, X=2, Y=1, T=2), ..., P(Z=2, X=dx, Y=1, T=2)],
             ...
             [P(Z=dz, X=1, Y=1, T=2), P(Z=dz, X=2, Y=1, T=2), ..., P(Z=dz, X=dx, Y=1, T=2)]],
            ...
        ]
        Size: dz * dx * dt
        """
        return self.Pzxyt[:, :, 1, :]
    

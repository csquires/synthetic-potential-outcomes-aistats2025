# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.mixture_moments import MixtureMoments


def compute_mixture_moments_discrete(full_marginal: np.ndarray):
    Pzu = np.einsum("zxytu->zu", full_marginal)
    Pxu = np.einsum("zxytu->xu", full_marginal)
    Pytu = np.einsum("zxytu->ytu", full_marginal)
    Psu = Pytu[1, :, :] - Pytu[0, :, :]
    Pu = np.einsum("zu->u", Pzu)
    
    Pz_u = np.einsum("zu,u->zu", Pzu, Pu ** -1)
    Px_u = np.einsum("xu,u->xu", Pxu, Pu ** -1)
    Ps_u = np.einsum("su,u->su", Psu, Pu ** -1)
    
    return MixtureMoments(
        Pu,
        EZ_U=Pz_u,
        EX_U=Px_u,
        ES_U=Ps_u,
        E_Z_U={u: Pz_u[:, u] for u in range(2)},
        E_X_U={u: Px_u[:, u] for u in range(2)},
        E_S_U={u: (Ps_u[:, u]) for u in range(2)},
    )
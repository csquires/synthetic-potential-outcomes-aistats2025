# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.mixture_moments import MixtureMoments


def compute_mixture_moments_discrete(full_marginal: np.ndarray):
    Pzu = np.einsum("zxytu->zu", full_marginal)
    Pxu = np.einsum("zxytu->xu", full_marginal)
    Pytu = np.einsum("zxytu->ytu", full_marginal)
    Pu = np.einsum("zu->u", Pzu)
    
    Pz_u = np.einsum("zu,u->zu", Pzu, Pu ** -1)
    Px_u = np.einsum("xu,u->xu", Pxu, Pu ** -1)
    PtY_u = np.einsum("tu,u->tu", Pytu[1], Pu ** -1)
    
    return MixtureMoments(
        Pu,
        E_Z_U={u: Pz_u[:, u] for u in range(2)},
        E_X_U={u: Px_u[:, u] for u in range(2)},
        E_tY_U={u: (PtY_u[:, u]) for u in range(2)},
    )
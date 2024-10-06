# === IMPORTS: THIRD-PARTY ===
import numpy as np


def compute_potential_outcome_moments_discrete(full_marginal: np.ndarray, max_order: int):
    Pytu = np.einsum("zxytu->ytu", full_marginal)
    Pu = np.einsum("ytu->u", Pytu)
    E_y0_u = np.einsum("tu,u->u", Pytu[:, 0, :], Pu ** -1)
    E_y1_u = np.einsum("tu,u->u", Pytu[:, 1, :], Pu ** -1)

    y0_moments = []
    y1_moments = []
    r_moments = []
    for order in range(1, max_order+1):
        y0_moment = np.einsum("u,u", Pu, E_y0_u ** order)
        y1_moment = np.einsum("u,u", Pu, E_y1_u ** order)
        r_moment = np.einsum("u,u", Pu, (E_y1_u - E_y0_u) ** order)
        y0_moments.append(y0_moment)
        y1_moments.append(y1_moment)
        r_moments.append(r_moment)
    
    return y0_moments, y1_moments, r_moments, Pu
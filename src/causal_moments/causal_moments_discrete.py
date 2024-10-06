# === IMPORTS: THIRD-PARTY ===
import numpy as np


def compute_potential_outcome_moments_discrete(full_marginal: np.ndarray, max_order: int):
    Pytu = np.einsum("zxytu->ytu", full_marginal)
    Ptu = np.einsum("ytu->tu", Pytu)
    Pu = np.einsum("tu->u", Ptu)
    E_y0_u = np.einsum("u,u->u", Pytu[1, 0, :], Ptu[0, :] ** -1)
    E_y1_u = np.einsum("u,u->u", Pytu[1, 1, :], Ptu[1, :] ** -1)

    y0_moments = [1]
    y1_moments = [1]
    r_moments = [1]
    for order in range(1, max_order+1):
        y0_moment = np.einsum("u,u", Pu, E_y0_u ** order)
        y1_moment = np.einsum("u,u", Pu, E_y1_u ** order)
        r_moment = np.einsum("u,u", Pu, (E_y1_u - E_y0_u) ** order)
        y0_moments.append(y0_moment)
        y1_moments.append(y1_moment)
        r_moments.append(r_moment)
    
    return y0_moments, y1_moments, r_moments, Pu
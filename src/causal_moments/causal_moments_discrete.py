# === IMPORTS: BUILT-IN ===
from dataclasses import dataclass
from typing import Dict

# === IMPORTS: THIRD-PARTY ===
import numpy as np


@dataclass
class CausalMoments:
    # distributions
    Pu: np.ndarray

    # moments
    moments_Y0: list[np.ndarray]
    moments_Y1: list[np.ndarray]
    moments_R: list[np.ndarray]

    # conditional first moments
    E_Y0_U: Dict[int, float]
    E_Y1_U: Dict[int, float]
    E_R_U: Dict[int, float]


def compute_potential_outcome_moments_discrete(full_marginal: np.ndarray, max_order: int):
    Pytu = np.einsum("zxytu->ytu", full_marginal)
    Ptu = np.einsum("ytu->tu", Pytu)
    Pu = np.einsum("tu->u", Ptu)
    E_y0_u = np.einsum("u,u->u", Pytu[1, 0, :], Ptu[0, :] ** -1)
    E_y1_u = np.einsum("u,u->u", Pytu[1, 1, :], Ptu[1, :] ** -1)

    moments_Y0 = [1]
    moments_Y1 = [1]
    moments_R = [1]
    for order in range(1, max_order+1):
        y0_moment = np.einsum("u,u", Pu, E_y0_u ** order)
        y1_moment = np.einsum("u,u", Pu, E_y1_u ** order)
        r_moment = np.einsum("u,u", Pu, (E_y1_u - E_y0_u) ** order)
        moments_Y0.append(y0_moment)
        moments_Y1.append(y1_moment)
        moments_R.append(r_moment)
    
    return CausalMoments(
        Pu,
        moments_Y0,
        moments_Y1,
        moments_R,
        E_Y0_U={u: E_y0_u[u] for u in range(2)},
        E_Y1_U={u: E_y1_u[u] for u in range(2)},
        E_R_U={u: (E_y1_u[u] - E_y0_u[u]) for u in range(2)},
    )
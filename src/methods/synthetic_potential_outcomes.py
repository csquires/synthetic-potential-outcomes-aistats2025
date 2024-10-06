# === IMPORTS: BUILT-IN ===
from dataclasses import dataclass
from typing import List, Dict

# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.problem_dims import ProblemDimensions
from src.observable_moments import ObservableMoments
from src.mixture_utils import prony, matrix_pencil


@dataclass
class HistoryItem:
    zx_moments: Dict[int, np.ndarray]
    cross_moments: Dict[int, np.ndarray]
    alpha_1: np.ndarray
    alpha_0: np.ndarray
    cond0: float
    cond1: float



class SyntheticPotentialOutcomes:
    def __init__(
            self, 
            problem_dims: ProblemDimensions, 
            decomposition_method: str = "matrix_pencil"
        ):
        self.problem_dims = problem_dims

        self.z_ixs = self.problem_dims.z_ixs
        self.x_ixs = self.problem_dims.x_ixs
        self.y_ix = self.problem_dims.y_ix

        self.ngroups = self.problem_dims.ngroups
        self.decomposition_method = decomposition_method

        self.history = []

    def _add_history(self, M_ZX_0, M_ZX_1, M_ZY_0, M_ZY_1, alpha_1, alpha_0):
        history_item = HistoryItem(
            zx_moments={0: M_ZX_0, 1: M_ZX_1},
            cross_moments={0: M_ZY_0, 1: M_ZY_1},
            alpha_1=alpha_1,
            alpha_0=alpha_0,
            cond0=np.linalg.cond(M_ZX_0),
            cond1=np.linalg.cond(M_ZX_1)
        )
        self.history.append(history_item)

    def _first_moment_coefs(self, obs_moments: ObservableMoments):
        # === CALCULATE ALPHA
        M_ZX_1, M_ZY_1 = obs_moments.M_ZX_T[1], obs_moments.M_ZY_T[1]
        alpha_1 = np.linalg.lstsq(M_ZX_1, M_ZY_1, rcond=None)[0]

        # === CALCULATE BETA
        M_ZX_0, M_ZY_0 = obs_moments.M_ZX_T[0], obs_moments.M_ZY_T[0]
        alpha_0 = np.linalg.lstsq(M_ZX_0, M_ZY_0, rcond=None)[0]

        self._add_history(M_ZX_0, M_ZX_1, M_ZY_0, M_ZY_1, alpha_1, alpha_0)
        gamma = alpha_1 - alpha_0
        return alpha_1, alpha_0, gamma

    def _next_moment_coefs(self, obs_moments: ObservableMoments, gamma_previous: np.ndarray):
        # === CALCULATE ALPHA
        M_ZX_1, M_ZXY_1 = obs_moments.M_ZX_T[1], obs_moments.M_ZXY_T[1]
        alpha_1_pre = np.linalg.lstsq(M_ZX_1, M_ZXY_1, rcond=None)[0]
        alpha_1 = alpha_1_pre @ gamma_previous

        # === CALCULATE BETA
        M_ZX_0, M_ZXY_0 = obs_moments.M_ZX_T[0], obs_moments.M_ZXY_T[0]
        alpha_0_pre = np.linalg.lstsq(M_ZX_0, M_ZXY_0, rcond=None)[0]
        alpha_0 = alpha_0_pre @ gamma_previous

        self._add_history(M_ZX_0, M_ZX_1, M_ZXY_0, M_ZXY_1, alpha_1, alpha_0)
        gamma = alpha_1 - alpha_0
        return alpha_1, alpha_0, gamma
    
    def only_first_step(self, obs_moments: ObservableMoments):
        alpha_1, alpha_0, gamma = self._first_moment_coefs(obs_moments)

        y0_mean = np.sum(alpha_0 * obs_moments.E_X)
        y1_mean = np.sum(alpha_1 * obs_moments.E_X)

        return y0_mean, y1_mean
    
    def fit(
        self,
        obs_moments: ObservableMoments
    ):  
        # === LIST OF GAMMAS AND MOMENTS
        gammas = [None]
        causal_moments = [1]

        # === RUN FIRST STEP AND SAVE RESULTS
        alpha_1, alpha_0, gamma = self._first_moment_coefs(obs_moments)
        causal_moment = np.sum(gamma * obs_moments.E_X)
        gammas.append(gamma)
        causal_moments.append(causal_moment)

        for l in range(2, 2 * self.ngroups + 1):
            if l % 2 == 0:
                alpha_1, alpha_0, gamma = self._next_moment_coefs(
                    obs_moments, 
                    gammas[l-1]
                )
                causal_moment = np.sum(gamma * obs_moments.E_X)
            else:
                alpha_1, alpha_0, gamma = self._next_moment_coefs(
                    obs_moments,
                    gammas[l-1]
                )
                causal_moment = np.sum(gamma * obs_moments.E_X)
            gammas.append(gamma)
            causal_moments.append(causal_moment)

        if self.decomposition_method == "prony":
            source_probs, means = prony(causal_moments, self.ngroups)
        elif self.decomposition_method == "matrix_pencil":
            source_probs, means = matrix_pencil(causal_moments, self.ngroups)
        else:
            raise ValueError(f"Decomposition method '{self.decomposition_method}' not recognized")

        return dict(
            causal_moments=causal_moments,
            source_probs=source_probs,
            means=means
        )
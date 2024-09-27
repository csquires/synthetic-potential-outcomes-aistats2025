# === IMPORTS: BUILT-IN ===
from dataclasses import dataclass
from typing import List, Dict

# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from problem_dims import ProblemDimensions
from src.mixture_utils import prony, matrix_pencil


@dataclass
class HistoryItem:
    xsyn_moments: Dict[int, np.ndarray]
    cross_moments: Dict[int, np.ndarray]
    alpha: np.ndarray
    beta: np.ndarray
    cond0: float
    cond1: float



class SyntheticPotentialOutcomes:
    def __init__(self, problem_dims: ProblemDimensions, decomposition_method: str = "matrix_pencil"):
        self.problem_dims = problem_dims
        self.x_ixs = self.problem_dims.x_ixs
        self.z_ixs = self.problem_dims.z_ixs
        self.y_ix = self.problem_dims.y_ix
        self.ngroups = self.problem_dims.ngroups
        self.decomposition_method = decomposition_method

    def _first_moment_coefs(
        self, 
        conditional_second_moments: Dict[int, np.ndarray],
    ):
        # === CALCULATE ALPHA
        m_xx_1 = conditional_second_moments[1][np.ix_(self.z_ixs, self.x_ixs)]
        m_xy_1 = conditional_second_moments[1][self.z_ixs, self.y_ix]
        # alpha = inv(m_xx_1) @ m_xy_1
        alpha = np.linalg.lstsq(m_xx_1, m_xy_1, rcond=None)[0]  # needed when |self.x_ixs| > k

        # === CALCULATE BETA
        m_xx_0 = conditional_second_moments[0][np.ix_(self.z_ixs, self.x_ixs)]
        m_xy_0 = conditional_second_moments[0][self.z_ixs, self.y_ix]
        # beta = inv(m_xx_0) @ m_xy_0
        beta = np.linalg.lstsq(m_xx_0, m_xy_0, rcond=None)[0]  # needed when |self.x_ixs| > k

        # === FOR DEBUGGING: SAVE WHAT WAS CALCULATED IN THIS STEP 
        history_item = HistoryItem(
            xsyn_moments={0: m_xx_0, 1: m_xx_1},
            cross_moments={0: m_xy_0, 1: m_xy_1},
            alpha=alpha,
            beta=beta,
            cond0=np.linalg.cond(m_xx_0),
            cond1=np.linalg.cond(m_xx_1)
        )
        
        # === RETURN HISTORY AND GAMMA
        gamma = alpha - beta
        return history_item, gamma

    def _next_moment_coefs(
        self, 
        conditional_second_moments: Dict[int, np.ndarray],
        conditional_third_moments: Dict[int, np.ndarray], 
        gamma_previous: np.ndarray
    ):
        # === CALCULATE ALPHA
        m_xx_1 = conditional_second_moments[1][np.ix_(self.z_ixs, self.x_ixs)]
        m_xxy_1 = conditional_third_moments[1][np.ix_(self.z_ixs, self.x_ixs)]
        # alpha = inv(m_xx_1) @ m_xxy_1 @ gamma_previous
        alpha_pre = np.linalg.lstsq(m_xx_1, m_xxy_1, rcond=None)[0]  # needed when |xsyn| > k
        alpha = alpha_pre @ gamma_previous

        # === CALCULATE BETA
        m_xx_0 = conditional_second_moments[0][np.ix_(self.z_ixs, self.x_ixs)]
        m_xxy_0 = conditional_third_moments[0][np.ix_(self.z_ixs, self.x_ixs)]
        # beta = inv(m_xx_0) @ m_xxy_0 @ gamma_previous
        beta_pre = np.linalg.lstsq(m_xx_0, m_xxy_0, rcond=None)[0]  # needed when |xsyn| > k
        beta = beta_pre @ gamma_previous

        # === FOR DEBUGGING: SAVE WHAT WAS CALCULATED IN THIS STEP 
        history_item = HistoryItem(
            xsyn_moments={0: m_xx_0, 1: m_xx_1},
            cross_moments={0: m_xxy_0, 1: m_xxy_1},
            alpha=alpha,
            beta=beta,
            cond0=np.linalg.cond(m_xx_0),
            cond1=np.linalg.cond(m_xx_1)
        )
        
        # === RETURN HISTORY AND GAMMA
        gamma = alpha - beta
        return history_item, gamma
    
    def only_first_step(
        self, 
        expectations: np.ndarray,
        conditional_second_moments: Dict[int, np.ndarray], 
        xref: List[int], 
        xsyn: List[int]
    ):
        history_item, gamma = self._first_moment_coefs(conditional_second_moments, xref, xsyn)
        xsyn_mean = expectations[xsyn]

        y0_mean = np.sum(history_item.beta * xsyn_mean)
        y1_mean = np.sum(history_item.alpha * xsyn_mean)

        return y0_mean, y1_mean, history_item
    
    def fit_fixed_partition(
        self,
        expectations: np.ndarray,
        conditional_second_moments: Dict[int, np.ndarray],
        conditional_third_moments: Dict[int, np.ndarray],
    ):
        x_mean = expectations[self.x_ixs]

        # === LIST OF GAMMAS AND MOMENTS
        gammas = [None]
        moments = [1]
        history = []

        # === RUN FIRST STEP AND SAVE RESULTS
        history_item, gamma = self._first_moment_coefs(conditional_second_moments)
        moment = np.sum(gamma * x_mean)
        gammas.append(gamma)
        moments.append(moment)
        history.append(history_item)

        for l in range(2, 2 * self.ngroups + 1):
            if l % 2 == 0:
                history_item, gamma = self._next_moment_coefs(
                    conditional_second_moments, 
                    conditional_third_moments, 
                    gammas[l-1]
                )
                moment = np.sum(gamma * x_mean)
            else:
                history_item, gamma = self._next_moment_coefs(
                    conditional_second_moments, 
                    conditional_third_moments, 
                    gammas[l-1]
                )
                moment = np.sum(gamma * x_mean)
            gammas.append(gamma)
            moments.append(moment)
            history.append(history_item)

        if self.decomposition_method == "prony":
            source_probs, means = prony(moments, self.ngroups)
        elif self.decomposition_method == "matrix_pencil":
            source_probs, means = matrix_pencil(moments, self.ngroups)
        else:
            raise ValueError(f"Decomposition method '{self.decomposition_method}' not recognized")

        return dict(
            history=history,
            moments=moments,
            source_probs=source_probs,
            means=means
        )
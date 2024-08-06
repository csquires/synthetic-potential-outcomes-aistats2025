# === IMPORTS: BUILT-IN ===
from dataclasses import dataclass
from typing import List, Dict

# === IMPORTS: THIRD-PARTY ===
import numpy as np
from src.empirical_moments import EmpiricalMoments

# === IMPORTS: LOCAL ===
from src.problem_config import ProblemConfig
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
    def __init__(self, config: ProblemConfig, decomposition_method: str = "matrix_pencil"):
        self.config = config
        self.decomposition_method = decomposition_method

    def _first_moment_coefs(
        self, 
        conditional_second_moments: Dict[int, np.ndarray],
        xref: List[int], 
        xsyn: List[int],
    ):
        # === CALCULATE ALPHA
        m_xx_1 = conditional_second_moments[1][np.ix_(xref, xsyn)]
        m_xy_1 = conditional_second_moments[1][xref, self.config.y_ix]
        # alpha = inv(m_xx_1) @ m_xy_1
        alpha = np.linalg.lstsq(m_xx_1, m_xy_1, rcond=None)[0]  # needed when |xsyn| > k

        # === CALCULATE BETA
        m_xx_0 = conditional_second_moments[0][np.ix_(xref, xsyn)]
        m_xy_0 = conditional_second_moments[0][xref, self.config.y_ix]
        # beta = inv(m_xx_0) @ m_xy_0
        beta = np.linalg.lstsq(m_xx_0, m_xy_0, rcond=None)[0]  # needed when |xsyn| > k

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
        xref: List[int], 
        xsyn_prev: List[int], 
        xsyn_new: List[int], 
        gamma_previous: np.ndarray
    ):
        # === CALCULATE ALPHA
        m_xx_1 = conditional_second_moments[1][np.ix_(xref, xsyn_new)]
        m_xxy_1 = conditional_third_moments[1][np.ix_(xref, xsyn_prev)]
        # alpha = inv(m_xx_1) @ m_xxy_1 @ gamma_previous
        alpha_pre = np.linalg.lstsq(m_xx_1, m_xxy_1, rcond=None)[0]  # needed when |xsyn| > k
        alpha = alpha_pre @ gamma_previous

        # === CALCULATE BETA
        m_xx_0 = conditional_second_moments[0][np.ix_(xref, xsyn_new)]
        m_xxy_0 = conditional_third_moments[0][np.ix_(xref, xsyn_prev)]
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
        xref: List[int], 
        xsyn1: List[int], 
        xsyn2: List[int],
    ):
        xsyn1_mean = expectations[xsyn1]
        xsyn2_mean = expectations[xsyn2]

        # === LIST OF GAMMAS AND MOMENTS
        gammas = [None]
        moments = [1]
        history = []

        # === RUN FIRST STEP AND SAVE RESULTS
        history_item, gamma = self._first_moment_coefs(
            conditional_second_moments,
            xref, 
            xsyn1
        )
        moment = np.sum(gamma * xsyn1_mean)
        gammas.append(gamma)
        moments.append(moment)
        history.append(history_item)

        for l in range(2, 2 * self.config.ngroups + 1):
            if l % 2 == 0:
                history_item, gamma = self._next_moment_coefs(
                    conditional_second_moments, 
                    conditional_third_moments, 
                    xref, 
                    xsyn1,  # previous
                    xsyn2,  # new
                    gammas[l-1]
                )
                moment = np.sum(gamma * xsyn2_mean)
            else:
                history_item, gamma = self._next_moment_coefs(
                    conditional_second_moments, 
                    conditional_third_moments, 
                    xref, 
                    xsyn2,  # previous
                    xsyn1,  # new
                    gammas[l-1]
                )
                moment = np.sum(gamma * xsyn1_mean)
            gammas.append(gamma)
            moments.append(moment)
            history.append(history_item)

        if self.decomposition_method == "prony":
            source_probs, means = prony(moments, self.config.ngroups)
        elif self.decomposition_method == "matrix_pencil":
            source_probs, means = matrix_pencil(moments, self.config.ngroups)
        else:
            raise ValueError(f"Decomposition method '{self.decomposition_method}' not recognized")

        return dict(
            history=history,
            moments=moments,
            source_probs=source_probs,
            means=means
        )

    def fit(self, obs_samples: np.ndarray):
        moment_estimator = EmpiricalMoments(self.config, obs_samples)
        expectations = moment_estimator.expectations
        conditional_second_moments = moment_estimator.conditional_second_moments
        conditional_third_moments = moment_estimator.conditional_third_moments
        breakpoint()

        partitions = []
        for xref, xsyn1, xsyn2 in partitions:
            partition_results = self.fit_fixed_partition(
                conditional_second_moments, 
                conditional_third_moments, 
                xref, 
                xsyn1, 
                xsyn2
            )

            # TODO: select partition with best condition number

        # TODO: given synthetic moments, recover distribution using moment matching

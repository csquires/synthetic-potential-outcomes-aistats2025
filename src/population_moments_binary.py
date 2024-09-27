# === IMPORTS: BUILT-IN ===
from collections import defaultdict
from typing import Dict
import itertools as itr

# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.problem_dims import ProblemDimensions


def complement(nvariables: int, subset: int):
    return tuple(set(range(nvariables)) - subset)


def compute_source_probs_and_means(
    p_ytu: np.ndarray,  # tensor representing marginal P(Y, T, U)
):
    p_tu = p_ytu.sum(axis=0)                            # shape = (|T|, |U|)
    p_u = p_tu.sum(axis=0)                              # shape = (|U|,)

    # COMPUTE E[R | U=u]
    p_y_given_ut = p_ytu / p_tu                         # shape = (|Y|, |T|, |U|)
    mean_y1_given_u = p_y_given_ut[1, 1]                # shape = (|U|,)
    mean_y0_given_u = p_y_given_ut[1, 0]                # shape = (|U|,)
    mean_r_given_u = mean_y1_given_u - mean_y0_given_u  # shape = (|U|,)

    return p_u, mean_r_given_u
    

def compute_R_moments(
    p_ytu: np.ndarray,  # tensor representing marginal P(Y, T, U)
    max_order: int
):
    p_u, mean_r_given_u = compute_source_probs_and_means(p_ytu)

    moments = [1]
    for order in range(1, max_order+1):
        moments_r_given_u = mean_r_given_u ** order
        moment = (p_u * moments_r_given_u).sum()
        moments.append(moment)
    return moments


class PopulationMomentsBinary:
    def __init__(self, config: ProblemDimensions, full_marginal: np.ndarray):
        self.config = config
        self.full_marginal = full_marginal
        self.obs_marginal = full_marginal.sum(axis=config.u_ix)

        nvariables = len(self.obs_marginal.shape)
        self.p_ytu = self.full_marginal.sum(complement(nvariables, {config.y_ix, config.t_ix, config.u_ix}))
        self.p_tu = self.p_ytu.sum(axis=0)
        self.p_u = self.p_tu.sum(axis=0)

        # PRE-COMPUTE CONDITIONALS
        treatment_marginal = self.obs_marginal.sum(complement(nvariables, {config.t_ix}))
        self.t2conditionals = dict()
        for t in range(config.ntreatments):
            conditional = np.take(self.obs_marginal, t, axis=config.t_ix) / treatment_marginal[t]
            self.t2conditionals[t] = conditional

        self._stored_expectations = None
        self._stored_conditional_expectations = None
        self._stored_conditional_second_moments = None
        self._stored_conditional_third_moments = None
        self._stored_conditional_higher_moments = defaultdict(lambda: None)

    def moments_Y1(self, max_order: int):
        moments = [1]
        for order in range(1, max_order+1):
            mean_y1_given_u = self.p_ytu[1, 1] / self.p_tu[1]
            moments_r_given_u = mean_y1_given_u ** order
            moment = (self.p_u * moments_r_given_u).sum()
            moments.append(moment)
        return moments

    def moments_Y0(self, max_order: int):
        moments = [1]
        for order in range(1, max_order+1):
            mean_y0_given_u = self.p_ytu[1, 0] / self.p_tu[0]
            moments_r_given_u = mean_y0_given_u ** order
            moment = (self.p_u * moments_r_given_u).sum()
            moments.append(moment)
        return moments
    
    def moments_R(self, max_order: int):
        moments = [1]
        for order in range(1, max_order+1):
            mean_y1_given_u = self.p_ytu[1, 1] / self.p_tu[1]
            mean_y0_given_u = self.p_ytu[1, 0] / self.p_tu[0]
            mean_R_given_u = mean_y1_given_u - mean_y0_given_u
            moments_r_given_u = mean_R_given_u ** order
            moment = (self.p_u * moments_r_given_u).sum()
            moments.append(moment)
        return moments
    
    def prob_y0_given_U(self):
        return self.p_ytu[1, 1] / self.p_tu[1]
    
    def prob_y1_given_U(self):
        return self.p_ytu[1, 0] / self.p_tu[0]

    @property
    def expectations(self) -> Dict[int, np.ndarray]:
        if self._stored_expectations is None:
            self._compute_expectations()
        return self._stored_expectations

    @property
    def conditional_expectations(self) -> Dict[int, np.ndarray]:
        if self._stored_conditional_expectations is None:
            self._compute_conditional_expectations()
        return self._stored_conditional_expectations
    
    @property
    def conditional_second_moments(self) -> Dict[int, np.ndarray]:
        if self._stored_conditional_second_moments is None:
            self._compute_conditional_second_moments()
        return self._stored_conditional_second_moments
    
    @property
    def conditional_third_moments(self) -> Dict[int, np.ndarray]:
        if self._stored_conditional_third_moments is None:
            self._compute_conditional_third_moments()
        return self._stored_conditional_third_moments
    
    def conditional_higher_moments(self, order) -> Dict[int, np.ndarray]:
        if self._stored_conditional_higher_moments[order] is None:
            self._compute_conditional_higher_moments(order)
        return self._stored_conditional_higher_moments[order]
    
    def _compute_expectations(self):
        nvariables = len(self.obs_marginal.shape)
        self._stored_expectations = np.zeros(nvariables)
        for i in range(nvariables):
            p_i = np.sum(self.obs_marginal, axis=complement(nvariables, {i}))
            self._stored_expectations[i] = p_i[1]
    
    def _compute_conditional_expectations(self):
        self._stored_conditional_expectations = dict()

        for t, conditional in self.t2conditionals.items():
            nvariables = len(conditional.shape)
            result = np.zeros((nvariables))
            for i in range(nvariables):
                p_i = np.sum(conditional, axis=complement(nvariables, {i}))
                result[i] = p_i[1]
            
            self._stored_conditional_expectations[t] = result
    
    def _compute_conditional_second_moments(self):
        self._stored_conditional_second_moments = dict()
        for t, conditional in self.t2conditionals.items():
            nvariables = len(conditional.shape)
            result = np.zeros((nvariables, nvariables))

            for i in range(nvariables):
                p_i = np.sum(conditional, axis=complement(nvariables, {i}))
                result[i, i] = p_i[1]

            for i, j in itr.combinations(range(nvariables), 2):
                p_ij = np.sum(conditional, axis=complement(nvariables, {i, j}))
                result[i, j] = p_ij[1, 1]
                result[j, i] = p_ij[1, 1]
            
            self._stored_conditional_second_moments[t] = result

    def _compute_conditional_third_moments(self):
        self._stored_conditional_third_moments = dict()
        y_ix = self.config.y_ix

        for t, conditional in self.t2conditionals.items():
            nvariables = len(conditional.shape)
            result = np.zeros((nvariables-1, nvariables-1))

            for i in range(nvariables-1):
                p_i = np.sum(conditional, axis=complement(nvariables, {i, y_ix}))
                result[i, i] = p_i[1, 1]

            for i, j in itr.combinations(range(nvariables-1), 2):
                p_ij = np.sum(conditional, axis=complement(nvariables, {i, j, y_ix}))
                result[i, j] = p_ij[1, 1, 1]
                result[j, i] = p_ij[1, 1, 1]
            
            self._stored_conditional_third_moments[t] = result

    def _compute_conditional_higher_moments(self, order):
        self._stored_conditional_higher_moments[order] = dict()

        for t, conditional in self.t2conditionals.items():
            nvariables = len(conditional.shape)
            result = np.zeros((nvariables,) * order)

            # TODO: LOTS OF REPETITION WITH THIS APPROACH, CAN WE USE A DIFFERENT ONE?
            for indices in itr.combinations_with_replacement(range(nvariables), order):
                marginal = np.sum(conditional, axis=complement(nvariables, set(indices)))
                val = marginal[(1,) * len(marginal.shape)]
                for perm_indices in itr.permutations(indices):
                    print(perm_indices)
                    result[perm_indices] = val
            
            self._stored_conditional_higher_moments[order][t] = result


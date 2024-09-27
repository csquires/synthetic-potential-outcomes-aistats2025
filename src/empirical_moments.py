# === IMPORTS: BUILT-IN ===
from collections import defaultdict
from typing import Dict

# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.problem_dims import ProblemDimensions


class EmpiricalMoments:
    def __init__(self, problem_dims: ProblemDimensions, obs_samples: np.ndarray):
        self.problem_dims = problem_dims
        self.obs_samples = obs_samples

        # split samples by treatment
        self.t2samples: Dict[int, np.ndarray] = dict()
        for t in range(problem_dims.ntreatments):
            self.t2samples[t] = obs_samples[obs_samples[:, problem_dims.t_ix] == t][:, problem_dims.zxy_ixs]

        # split samples by treatment
        self._stored_expectations = None
        self._stored_conditional_expectations = None
        self._stored_conditional_second_moments = None
        self._stored_conditional_third_moments = None
        self._stored_third_moments = None
        self._stored_conditional_higher_moments = defaultdict(lambda: None)

    @property
    def expectations(self) -> Dict[int, np.ndarray]:
        if self._stored_expectations is None:
            self._stored_expectations = np.mean(self.obs_samples, axis=0)

        return self._stored_expectations

    @property
    def conditional_expectations(self) -> Dict[int, np.ndarray]:
        if self._stored_conditional_expectations is None:
            self._stored_conditional_expectations = dict()
            for t, samples_t in self.t2samples.items():
                conditional_expectation = samples_t.mean(axis=0)
                self._stored_conditional_expectations[t] = conditional_expectation

        return self._stored_conditional_expectations

    @property
    def conditional_second_moments(self) -> Dict[int, np.ndarray]:
        if self._stored_conditional_second_moments is None:
            self._stored_conditional_second_moments = dict()
            for t, samples_t in self.t2samples.items():
                moment = np.einsum('ij,ik->jk', samples_t, samples_t) / samples_t.shape[0]
                self._stored_conditional_second_moments[t] = moment

        return self._stored_conditional_second_moments
    
    @property
    def conditional_third_moments(self):
        if self._stored_conditional_third_moments is None:
            self._stored_conditional_third_moments = dict()
            for t, samples_t in self.t2samples.items():
                moment = np.einsum("ij,ik,i->jk", samples_t, samples_t, samples_t[:, self.config.y_ix]) / samples_t.shape[0]
                self._stored_conditional_third_moments[t] = moment

        return self._stored_conditional_third_moments
    
    @property
    def third_moments(self):
        if self._stored_third_moments is None:
            samples = self.obs_samples
            self._stored_third_moments = np.einsum("ij,ik,im->jkm", samples, samples, samples) / samples.shape[0]

        return self._stored_third_moments
    
    def conditional_higher_moments(self, order) -> Dict[int, np.ndarray]:
        if self._stored_conditional_higher_moments[order] is None:
            self._stored_conditional_higher_moments[order] = dict()
            dims = "jklmnopqrstuvwxyz"[:order]
            for t, samples_t in self.t2samples.items():
                pattern = ",".join([f"i{dim}" for dim in dims]) + "->" + dims
                print(pattern)
                moment = np.einsum(pattern, *(samples_t for _ in range(order))) / samples_t.shape[0]
                self._stored_conditional_higher_moments[order][t] = moment
        
        return self._stored_conditional_higher_moments[order]
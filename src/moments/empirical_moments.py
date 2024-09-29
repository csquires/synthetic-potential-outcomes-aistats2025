# === IMPORTS: BUILT-IN ===
from collections import defaultdict
from typing import Dict

# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.problem_dims import ProblemDimensions
from src.moments.moments import Moments


class EmpiricalMoments(Moments):
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

    @property
    def E_X(self) -> np.ndarray:
        if self._stored_expectations is None:
            self._stored_expectations = np.mean(self.obs_samples, axis=0)

        return self._stored_expectations
    
    @property
    def E_Z(self) -> np.ndarray:
        if self._stored_expectations is None:
            self._stored_expectations = np.mean(self.obs_samples, axis=0)

        return self._stored_expectations

    @property
    def E_X_T(self) -> Dict[int, np.ndarray]:
        if self._stored_conditional_expectations is None:
            self._stored_conditional_expectations = dict()
            for t, samples_t in self.t2samples.items():
                conditional_expectation = samples_t.mean(axis=0)
                self._stored_conditional_expectations[t] = conditional_expectation

        return self._stored_conditional_expectations
    
    @property
    def E_Z_T(self) -> Dict[int, np.ndarray]:
        if self._stored_conditional_expectations is None:
            self._stored_conditional_expectations = dict()
            for t, samples_t in self.t2samples.items():
                conditional_expectation = samples_t.mean(axis=0)
                self._stored_conditional_expectations[t] = conditional_expectation

        return self._stored_conditional_expectations

    @property
    def M_ZX_T(self) -> Dict[int, np.ndarray]:
        if self._stored_conditional_second_moments is None:
            self._stored_conditional_second_moments = dict()
            for t, samples_t in self.t2samples.items():
                moment = np.einsum('ij,ik->jk', samples_t, samples_t) / samples_t.shape[0]
                self._stored_conditional_second_moments[t] = moment

        return self._stored_conditional_second_moments
    
    @property
    def M_ZXY_T(self) -> Dict[int, np.ndarray]:
        if self._stored_conditional_third_moments is None:
            self._stored_conditional_third_moments = dict()
            for t, samples_t in self.t2samples.items():
                moment = np.einsum("ij,ik,i->jk", samples_t, samples_t, samples_t[:, self.problem_dims.y_ix]) / samples_t.shape[0]
                self._stored_conditional_third_moments[t] = moment

        return self._stored_conditional_third_moments
    
    @property
    def M_ZXtY(self):
        if self._stored_third_moments is None:
            samples = self.obs_samples
            t_onehot = (samples[:, self.problem_dims.t_ix] == np.array([[0], [1]])).T
            y_samples = samples[:, self.problem_dims.y_ix]
            y_tilde_samples = t_onehot * y_samples[:, None]
            self._stored_third_moments = np.einsum("ij,ik,im->jkm", samples, samples, y_tilde_samples) / samples.shape[0]

        return self._stored_third_moments
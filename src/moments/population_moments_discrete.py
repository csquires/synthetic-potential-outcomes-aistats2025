# === IMPORTS: BUILT-IN ===
from collections import defaultdict
from typing import Dict
import itertools as itr

# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.problem_dims import ProblemDimensions
from src.moments.moments import Moments



class PopulationMomentsDiscrete(Moments):
    def __init__(self, problem_dims: ProblemDimensions, full_marginal: np.ndarray):
        self.problem_dims = problem_dims
        self.dz = self.problem_dims.nz
        self.dx = self.problem_dims.nx

        self.full_marginal = full_marginal  # (z, x, y, t, u)
        self.Pzxyt = np.einsum("zxytu->zxyt", full_marginal)
        
        # === OBSERVED ===
        # 3-way marginals
        self.Pzxy = np.einsum("zxyt->zxy", self.Pzxyt)
        self.Pzxt = np.einsum("zxyt->zxt", self.Pzxyt)
        self.Pzyt = np.einsum("zxyt->zyt", self.Pzxyt)
        self.Pxyt = np.einsum("zxyt->xyt", self.Pzxyt)

        # 2-way marginals
        self.Pzx = np.einsum("zxy->zx", self.Pzxy)
        self.Pzy = np.einsum("zxy->zy", self.Pzxy)
        self.Pzt = np.einsum("zxt->zt", self.Pzxt)
        self.Pxt = np.einsum("zxt->xt", self.Pzxt)
        self.Pyt = np.einsum("xyt->yt", self.Pxyt)

        # univariate marginals
        self.Pz = np.einsum("zx->z", self.Pzx)
        self.Px = np.einsum("zx->x", self.Pzx)
        self.Py = np.einsum("zy->y", self.Pzy)
        self.Pt = np.einsum("tu->t", self.Ptu)
        
        # === UNOBSERVED ===
        # 3-way marginals
        self.Pytu = np.einsum("zxytu->ytu", full_marginal)
        # 2-way marginals
        self.Ptu = np.einsum("ytu->tu", self.Pytu)
        # univariate marginals
        self.Pu = np.einsum("tu->u", self.Ptu)

        # PRE-COMPUTE CONDITIONALS
        self.t2conditionals = dict()
        for t in range(problem_dims.ntreatments):
            conditional = self.Pzxyt[:, :, :, t] * (self.Pt[t] ** -1)
            self.t2conditionals[t] = conditional

        self._stored_expectations = None
        self._stored_conditional_expectations = None
        self._stored_conditional_second_moments = None
        self._stored_conditional_third_moments = None
        self._stored_third_moments = None
        self._stored_conditional_higher_moments = defaultdict(lambda: None)

    def moments_Y1(self, max_order: int):
        moments = [1]
        for order in range(1, max_order+1):
            mean_y1_given_u = np.einsum("u,u->u", self.Pytu[1, 1, :], self.Ptu[1, :]**-1)
            moments_y1_given_u = mean_y1_given_u ** order
            moment = np.einsum("u,u", self.Pu, moments_y1_given_u)
            moments.append(moment)
        return moments

    def moments_Y0(self, max_order: int):
        moments = [1]
        for order in range(1, max_order+1):
            mean_y0_given_u = np.einsum("u,u->u", self.Pytu[1, 0, :], self.Ptu[0, :]**-1)
            moments_y0_given_u = mean_y0_given_u ** order
            moment = np.einsum("u,u", self.Pu, moments_y0_given_u)
            moments.append(moment)
        return moments
    
    def moments_R(self, max_order: int):
        moments = [1]
        for order in range(1, max_order+1):
            mean_r_given_u = np.einsum("u,u->u", self.Pytu[1, 1, :] - self.Pytu[1, 0, :], self.Ptu[1, :]**-1)
            moments_r_given_u = mean_r_given_u ** order
            moment = np.einsum("u,u", self.Pu, moments_r_given_u)
            moments.append(moment)
        return moments
    
    @property
    def expectations(self) -> np.ndarray:
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
    
    def third_moments(self) -> np.ndarray:
        if self._stored_third_moments is None:
            self._compute_third_moments()
        return self._stored_third_moments
    
    def _compute_expectations(self):
        self.stored_expectations = np.concatenate((self.Pz, self.Px))
    
    def _compute_conditional_expectations(self):
        self._stored_conditional_expectations = dict()

        for t in range(self.problem_dims.ntreatments):
            Pz_t = self.Pzt[:, t] * (self.Pt[t] ** -1)
            Px_t = self.Pxt[:, t] * (self.Pt[t] ** -1)
            result = np.concatenate((Pz_t, Px_t))
            self._stored_conditional_expectations[t] = result
    
    def _compute_conditional_second_moments(self):
        self._stored_conditional_second_moments = dict()
        dz, dx = self.dz, self.dx

        for t in range(self.problem_dims.ntreatments):
            result = np.zeros([self.dz + self.dx + 1])

            Pz_t = self.Pzt[:, t] * (self.Pt[t] ** -1)
            Px_t = self.Pxt[:, t] * (self.Pt[t] ** -1)
            result[:dz, :dz] = np.diag(Pz_t)
            result[dz:(dz + dz), dz:(dz + dx)] = np.diag(Px_t)

            Pzx_t = self.Pzxt[:, :, t] * (self.Pt[t] ** -1)
            result[:dz, dz:(dz + dx)] = Pzx_t
            result[dz:(dz + dx), :dz] = Pzx_t.T

            Mzy_t = self.Pzyt[:, 1, t] * (self.Pt[t] ** -1)
            result[:dz, -1] = Mzy_t
            result[-1, :dz] = Mzy_t.T

            Mxy_t = self.Pxyt[:, 1, t] * (self.Pt[t] ** -1)
            result[dz:(dz+dx), -1] = Mxy_t
            result[-1, dz:(dz + dx)] = Mxy_t.T
            
            result[-1, -1] = self.Pyt[1, t] * (self.Pt[t] ** -1)
            self._stored_conditional_second_moments[t] = result

    def _compute_conditional_third_moments(self):
        self._stored_conditional_third_moments = dict()
        dz, dx = self.dz, self.dx

        for t in range(self.problem_dims.ntreatments):
            result = np.zeros([self.dz + self.dx])

            Mzy_t = self.Pzyt[:, 1, t] * (self.Pt[t] ** -1)
            Mxy_t = self.Pxyt[:, 1, t] * (self.Pt[t] ** -1)
            result[:dz, :dz] = np.diag(Mzy_t)
            result[dz:(dz + dz), dz:(dz + dx)] = np.diag(Mxy_t)

            Mzxy_t = self.Pzxyt[:, :, 1, t] * (self.Pt[t] ** -1)
            result[:dz, dz:(dz + dx)] = Mzxy_t
            result[dz:(dz + dx), :dz] = Mzxy_t.T



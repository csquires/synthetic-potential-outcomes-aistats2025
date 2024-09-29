# === IMPORTS: BUILT-IN ===
from abc import ABC
from typing import Dict

# === IMPORTS: THIRD-PARTY ===
import numpy as np


class Moments(ABC):
    @property
    def E_X(self) -> np.ndarray:
        raise NotImplementedError
    
    @property
    def E_Z(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def E_X_T(self) -> Dict[int, np.ndarray]:
        raise NotImplementedError
    
    @property
    def E_Z_t(self) -> Dict[int, np.ndarray]:
        raise NotImplementedError

    @property
    def M_ZX_T(self) -> Dict[int, np.ndarray]:
        raise NotImplementedError
    
    @property
    def M_ZXY_T(self):
        raise NotImplementedError
    
    @property
    def M_ZXtY(self):
        raise NotImplementedError
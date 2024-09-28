# === IMPORTS: BUILT-IN ===
from abc import ABC
from typing import Dict

# === IMPORTS: THIRD-PARTY ===
import numpy as np


class Moments(ABC):
    @property
    def expectations(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def conditional_expectations(self) -> Dict[int, np.ndarray]:
        raise NotImplementedError

    @property
    def conditional_second_moments(self) -> Dict[int, np.ndarray]:
        raise NotImplementedError
    
    @property
    def conditional_third_moments(self):
        raise NotImplementedError
    
    @property
    def third_moments(self):
        raise NotImplementedError
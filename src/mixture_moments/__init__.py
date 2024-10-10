# === IMPORTS: BUILT-IN ===
from dataclasses import dataclass
from typing import Dict

# === IMPORTS: THIRD-PARTY ===
import numpy as np


@dataclass
class MixtureMoments:
    # distributions
    Pu: np.ndarray

    # conditional first moments
    EZ_U: np.ndarray
    EX_U: np.ndarray
    ES_U: np.ndarray
    E_Z_U: Dict[int, float]
    E_X_U: Dict[int, float]
    E_S_U: Dict[int, float]
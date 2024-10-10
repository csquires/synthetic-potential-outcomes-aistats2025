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
    E_Z_U: Dict[int, float]
    E_X_U: Dict[int, float]
    E_tY_U: Dict[int, float]
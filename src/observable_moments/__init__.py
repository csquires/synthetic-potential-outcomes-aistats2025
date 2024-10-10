# === IMPORTS: BUILT-IN ===
from dataclasses import dataclass
from typing import Dict

# === IMPORTS: THIRD-PARTY ===
import numpy as np


@dataclass
class ObservableMoments:
    # first moments
    E_Z: np.ndarray
    E_X: np.ndarray
    E_S: np.ndarray
    # second moments
    M_ZX: np.ndarray
    M_ZS: np.ndarray
    M_XS: np.ndarray
    # third moments
    M_ZXS: np.ndarray

    # conditional first moments
    E_Z_T: Dict[int, np.ndarray]
    E_X_T: Dict[int, np.ndarray]
    # conditional second moments
    M_ZX_T: Dict[int, np.ndarray]
    M_ZY_T: Dict[int, np.ndarray]
    # conditional third moments
    M_ZXY_T: Dict[int, np.ndarray]

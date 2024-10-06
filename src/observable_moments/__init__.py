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
    E_tY: np.ndarray
    # second moments
    M_ZX: np.ndarray
    M_ZtY: np.ndarray
    M_XtY: np.ndarray
    # third moments
    M_ZXtY: np.ndarray

    # conditional first moments
    E_Z_T: Dict[int, np.ndarray]
    E_X_T: Dict[int, np.ndarray]
    # conditional second moments
    M_ZX_T: Dict[int, np.ndarray]
    M_ZY_T: Dict[int, np.ndarray]
    # conditional third moments
    M_ZXY_T: Dict[int, np.ndarray]



# first moments
# E_Z -> np.ndarray
# E_X -> np.ndarray
# E_tY -> np.ndarray
# second moments
# M_ZX -> np.ndarray
# M_ZtY -> np.ndarray
# M_XtY -> np.ndarray
# third moments
# M_ZXtY -> np.ndarray

# conditional first moments
# E_Z_T -> Dict[int, np.ndarray]
# E_X_T -> Dict[int, np.ndarray]
# conditional second moments
# M_ZX_T -> Dict[int, np.ndarray]
# conditional third moments
# M_ZXY_T  -> Dict[int, np.ndarray]





# E_Z = [E(Z_1), E(Z_2), ..., E(Z_dz)]
# E_X = [E(X_1), E(X_2), ..., E(X_dx)]
# E_tY = [E(Y * 1_{T=1}), E(Y * 1_{T=2}), ...]

# M_ZX = [
#  [E(Z_1, X_1), E(Z_1, X_2), ..., E(Z_1, X_dx)],
#  [E(Z_2, X_1), E(Z_2, X_2), ..., E(Z_2, X_dx)],
#  ...
#  [E(Z_dz, X_1), E(Z_dz, X_2), ..., E(Z_dz, X_dx)]
# ]

# M_ZtY(self) = [
#   [E(Z_1 * Y * 1_T=1}), E(Z_1 * Y * 1_T=2}), ..., E(Z_1 * Y * 1_T=dt})],
#   [E(Z_2 * Y * 1_T=2}), E(Z_2 * Y * 1_T=2}), ..., E(Z_2 * Y * 1_T=dt})],
#   ...
#   [E(Z_dz * Y * 1_T=2}), E(Z_dz * Y * 1_T=2}), ..., E(Z_dz * Y * 1_T=dt})]
# ]
    
# M_XtY =
# [[E(X_1 * Y * 1_T=1}), E(X_1 * Y * 1_T=2}), ..., E(X_1 * Y * 1_T=dt})],
#     [E(X_2 * Y * 1_T=2}), E(X_2 * Y * 1_T=2}), ..., E(X_2 * Y * 1_T=dt})],
#     ...
#     [E(X_dx * Y * 1_T=2}), E(X_dx * Y * 1_T=2}), ..., E(X_dx * Y * 1_T=dt})]]
    
# M_ZXtY =
# [
#     [[E(Z_1 * X_1 * Y * 1_{T=1}), E(Z_1 * X_2 * Y * 1_{T=1}), ..., E(Z_1 * X_dx * Y * 1_{T=1})],
#         [E(Z_2 * X_1 * Y * 1_{T=1}), E(Z_2 * X_2 * Y * 1_{T=1}), ..., E(Z_2 * X_dx * Y * 1_{T=1})],
#         ...
#         [E(Z_dz * X_1 * Y * 1_{T=1}), E(Z_dz * X_2 * Y * 1_{T=1}), ..., E(Z_dz * X_dx * Y * 1_{T=1})]],
#     [[E(Z_1 * X_1 * Y * 1_{T=2}), E(Z_1 * X_2 * Y * 1_{T=2}), ..., E(Z_1 * X_dx * Y * 1_{T=2})],
#         [E(Z_2 * X_1 * Y * 1_{T=2}), E(Z_2 * X_2 * Y * 1_{T=2}), ..., E(Z_2 * X_dx * Y * 1_{T=2})],
#         ...
#         [E(Z_dz * X_1 * Y * 1_{T=2}), E(Z_dz * X_2 * Y * 1_{T=2}), ..., E(Z_dz * X_dx * Y * 1_{T=2})]],
#     ...
# ]

# E_Z_T = {t: [E(Z_1 | T=t), E(Z_2 | T=t), ..., E(Z_dz | T=t)] }
# E_X_T = {t: [E(X_1 | T=t), E(X_2 | T=t), ..., E(X_dx | T=t)] }
    
# M_ZX_T = {t: 
#     [[E(Z_1 * X_1 | T=t), E(Z_1 * X_2 | T=t), ..., E(Z_1 * X_dx | T=t)],
#         [E(Z_2 * X_1 | T=t), E(Z_2 * X_2 | T=t), ..., E(Z_2 * X_dx | T=t)],
#         ...
#         [E(Z_dz * X_1 | T=t), E(Z_dz * X_2 | T=t), ..., E(Z_dz * X_dx | T=t)]]
# }

# M_ZXY_T = {t: 
#     [[E(Z_1 * X_1 * Y | T=t), E(Z_1 * X_2 * Y | T=t), ..., E(Z_1 * X_dx * Y | T=t)],
#         [E(Z_2 * X_1 * Y | T=t), E(Z_2 * X_2 * Y | T=t), ..., E(Z_2 * X_dx * Y | T=t)],
#         ...
#         [E(Z_dz * X_1 * Y | T=t), E(Z_dz * X_2 * Y | T=t), ..., E(Z_dz * X_dx * Y | T=t)]]
# }
    
    


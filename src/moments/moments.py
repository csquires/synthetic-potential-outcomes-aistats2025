# === IMPORTS: BUILT-IN ===
from abc import ABC
from typing import Dict

# === IMPORTS: THIRD-PARTY ===
import numpy as np


class Moments(ABC):
    @property
    def E_Z(self) -> np.ndarray:
        """
        Return [E(Z_1), E(Z_2), ..., E(Z_dz)]
        """
        raise NotImplementedError
    
    @property
    def E_X(self) -> np.ndarray:
        """
        Return [E(X_1), E(X_2), ..., E(Z_dx)]
        """
        raise NotImplementedError
    
    @property
    def M_ZX(self) -> np.ndarray:
        """
        Return 
        [[E(Z_1, X_1), E(Z_1, X_2), ..., E(Z_1, X_dx)],
         [E(Z_2, X_1), E(Z_2, X_2), ..., E(Z_2, X_dx)],
         ...
         [E(Z_dz, X_1), E(Z_dz, X_2), ..., E(Z_dz, X_dx)]]
        """
        raise NotImplementedError
    
    @property
    def M_ZtY(self) -> np.ndarray:
        """
        Return 
        [[E(Z_1 * Y * 1_T=1}), E(Z_1 * Y * 1_T=2}), ..., E(Z_1 * Y * 1_T=dt})],
         [E(Z_2 * Y * 1_T=2}), E(Z_2 * Y * 1_T=2}), ..., E(Z_2 * Y * 1_T=dt})],
         ...
         [E(Z_dz * Y * 1_T=2}), E(Z_dz * Y * 1_T=2}), ..., E(Z_dz * Y * 1_T=dt})]]
        """
        raise NotImplementedError
    
    @property
    def M_XtY(self) -> np.ndarray:
        """
        Return 
        [[E(X_1 * Y * 1_T=1}), E(X_1 * Y * 1_T=2}), ..., E(X_1 * Y * 1_T=dt})],
         [E(X_2 * Y * 1_T=2}), E(X_2 * Y * 1_T=2}), ..., E(X_2 * Y * 1_T=dt})],
         ...
         [E(X_dx * Y * 1_T=2}), E(X_dx * Y * 1_T=2}), ..., E(X_dx * Y * 1_T=dt})]]
        """
        raise NotImplementedError
    
    @property
    def M_ZXtY(self) -> np.ndarray:
        """
        Return
        [
            [[E(Z_1 * X_1 * Y * 1_{T=1}), E(Z_1 * X_2 * Y * 1_{T=1}), ..., E(Z_1 * X_dx * Y * 1_{T=1})],
             [E(Z_2 * X_1 * Y * 1_{T=1}), E(Z_2 * X_2 * Y * 1_{T=1}), ..., E(Z_2 * X_dx * Y * 1_{T=1})],
             ...
             [E(Z_dz * X_1 * Y * 1_{T=1}), E(Z_dz * X_2 * Y * 1_{T=1}), ..., E(Z_dz * X_dx * Y * 1_{T=1})]],
            [[E(Z_1 * X_1 * Y * 1_{T=2}), E(Z_1 * X_2 * Y * 1_{T=2}), ..., E(Z_1 * X_dx * Y * 1_{T=2})],
             [E(Z_2 * X_1 * Y * 1_{T=2}), E(Z_2 * X_2 * Y * 1_{T=2}), ..., E(Z_2 * X_dx * Y * 1_{T=2})],
             ...
             [E(Z_dz * X_1 * Y * 1_{T=2}), E(Z_dz * X_2 * Y * 1_{T=2}), ..., E(Z_dz * X_dx * Y * 1_{T=2})]],
            ...
        ]
        Size: dz * dx * dt
        """
        raise NotImplementedError
    
    # === CONDITIONAL ===
    @property
    def E_tY(self) -> np.ndarray:
        """
        Return [E(Y * 1_{T=1}), E(Y * 1_{T=2}), ...]
        """
        raise NotImplementedError
    
    @property
    def E_Z_t(self) -> Dict[int, np.ndarray]:
        """
        Return 
        {t: 
            [E(Z_1 | T=t), E(Z_2 | T=t), ..., E(Z_dz | T=t)]
        }
        """
        raise NotImplementedError

    @property
    def E_X_T(self) -> Dict[int, np.ndarray]:
        """
        Return 
        {t: 
            [E(X_1 | T=t), E(X_2 | T=t), ..., E(X_dx | T=t)]
        }
        """
        raise NotImplementedError
    
    @property
    def M_ZX_T(self) -> Dict[int, np.ndarray]:
        """
        Return 
        {t: 
            [[E(Z_1 * X_1 | T=t), E(Z_1 * X_2 | T=t), ..., E(Z_1 * X_dx | T=t)],
             [E(Z_2 * X_1 | T=t), E(Z_2 * X_2 | T=t), ..., E(Z_2 * X_dx | T=t)],
             ...
             [E(Z_dz * X_1 | T=t), E(Z_dz * X_2 | T=t), ..., E(Z_dz * X_dx | T=t)]]
        }
        """
        raise NotImplementedError
    
    @property
    def M_ZXY_T(self)  -> Dict[int, np.ndarray]:
        """
        Return 
        {t: 
            [[E(Z_1 * X_1 * Y | T=t), E(Z_1 * X_2 * Y | T=t), ..., E(Z_1 * X_dx * Y | T=t)],
             [E(Z_2 * X_1 * Y | T=t), E(Z_2 * X_2 * Y | T=t), ..., E(Z_2 * X_dx * Y | T=t)],
             ...
             [E(Z_dz * X_1 * Y | T=t), E(Z_dz * X_2 * Y | T=t), ..., E(Z_dz * X_dx * Y | T=t)]]
        }
        """
        raise NotImplementedError
    
    
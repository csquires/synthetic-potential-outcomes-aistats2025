# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.problem_dims import BinaryProblemDimensions


def lookup_binary(Vab: np.ndarray, avals, bvals):
    nsamples = len(avals)

    onehot = np.zeros((nsamples, 4))
    onehot[:, 0] = (1 - avals) * (1 - bvals)    # A = 0, B = 0
    onehot[:, 1] = (1 - avals) * bvals          # A = 0, B = 1
    onehot[:, 2] = avals * (1 - bvals)          # A = 1, B = 0
    onehot[:, 3] = avals * bvals                # A = 1, B = 1

    flatV = np.zeros(4)
    flatV[0] = Vab[0, 0]
    flatV[1] = Vab[0, 1]
    flatV[2] = Vab[1, 0]
    flatV[3] = Vab[1, 1]

    vals = np.einsum("id,d->i", onehot, flatV)
    return vals


class BinaryGeneratorMain:
    def __init__(
            self, 
            zt_strength: float = 0,
            xy_strength: float = 0
        ):
        self.zt_strength = zt_strength
        self.xy_strength = xy_strength
        self.problem_dims = BinaryProblemDimensions(nz=1, nx=1, ngroups=2, ntreatments=2)

        # P(U)
        self.Pu = np.array([0.5, 0.5])

        # P(Z | U)
        self.Pz_u = np.zeros((2, 2))
        self.Pz_u[:, 0] = np.array([3/4, 1/4])  # given U = 0
        self.Pz_u[:, 1] = np.array([1/4, 3/4])  # given U = 1

        # P(X | U)
        self.Px_u = np.zeros((2, 2))
        self.Px_u[:, 0] = np.array([3/4, 1/4])  # given U = 0
        self.Px_u[:, 1] = np.array([1/4, 3/4])  # given U = 1

        # P(T | Z, U)
        a = zt_strength
        self.Pt_zu = np.zeros((2, 2, 2))
        self.Pt_zu[:, 0, 0] = np.array([1/4, 3/4])              # given Z = 0, U = 0
        self.Pt_zu[:, 0, 1] = np.array([3/4 - a/2, 1/4 + a/2])  # given Z = 0, U = 1
        self.Pt_zu[:, 1, 0] = np.array([1/4 + a/2, 3/4 - a/2])  # given Z = 1, U = 0
        self.Pt_zu[:, 1, 1] = np.array([3/4, 1/4])              # given Z = 1, U = 1

        # E(Y0 | X, U)
        self.Py0_xu = np.zeros((2, 2, 2))
        b = self.xy_strength
        self.Py0_xu[:, 0, 0] = 1/8 * np.array([1, 7])                   # given X = 0, U = 0
        self.Py0_xu[:, 0, 1] = 1/8 * np.array([7 - 6 * b, 1 + 6 * b])   # given X = 0, U = 1
        self.Py0_xu[:, 1, 0] = 1/8 * np.array([1 + 6 * b, 7 - 6 * b])   # given X = 1, U = 0
        self.Py0_xu[:, 1, 1] = 1/8 * np.array([7, 1])                   # given X = 1, U = 1

        # E(Y1 | X, U)
        self.Py1_xu = np.zeros((2, 2, 2))
        b = self.xy_strength
        self.Py1_xu[:, 0, 0] = 1/8 * np.array([7, 1])                   # given X = 0, U = 0
        self.Py1_xu[:, 0, 1] = 1/8 * np.array([1 + 6 * b, 7 - 6 * b])   # given X = 0, U = 1
        self.Py1_xu[:, 1, 0] = 1/8 * np.array([7 - 6 * b, 1 + 6 * b])   # given X = 1, U = 0
        self.Py1_xu[:, 1, 1] = 1/8 * np.array([1, 7])                   # given X = 1, U = 1

        # E(Y | X, T, U) --- repetitive, but convenient
        self.Py_xtu = np.zeros((2, 2, 2, 2))
        self.Py_xtu[:, :, 0, :] = self.Py0_xu
        self.Py_xtu[:, :, 1, :] = self.Py1_xu

    def generate(self, nsamples: int):
        full_samples = np.ndarray((nsamples, 5))

        # U
        u_vals = np.random.binomial(n=1, p=0.5, size=nsamples)

        # X | U
        x_cutoffs = (1 - u_vals) * 1/4 + u_vals * 3/4
        x_vals = np.random.uniform(size=nsamples) < x_cutoffs

        # Z | U
        z_cutoffs = (1 - u_vals) * 1/4 + u_vals * 3/4
        z_vals = np.random.uniform(size=nsamples) < z_cutoffs
        
        # T | Z, U
        t_cutoffs = lookup_binary(self.Pt_zu[1, :, :], z_vals, u_vals)
        t_vals = np.random.uniform(size=nsamples) < t_cutoffs
        
        # Y | X, T, U
        y0_vals = lookup_binary(self.Py0_xu[1, :, :], x_vals, u_vals)
        y1_vals = lookup_binary(self.Py1_xu[1, :, :], x_vals, u_vals)
        y_cutoffs = (1 - t_vals) * y0_vals + t_vals * y1_vals
        y_vals = np.random.uniform(size=nsamples) < y_cutoffs

        z_ix, x_ix = 0, 1
        y_ix, t_ix, u_ix = 2, 3, 4
        full_samples[:, z_ix] = z_vals
        full_samples[:, x_ix] = x_vals
        full_samples[:, y_ix] = y_vals
        full_samples[:, t_ix] = t_vals
        full_samples[:, u_ix] = u_vals
        
        obs_samples = full_samples[:, :-1]
        return full_samples, obs_samples
    
    def true_marginal(self):
        marginal = np.einsum(
            "zu,xu,yxtu,tzu,u->zxytu",
            self.Pz_u,
            self.Px_u,
            self.Py_xtu, 
            self.Pt_zu, 
            self.Pu
        )

        return marginal
    







    # def get_observable_moments(self):
    #     EtY_zxu = np.einsum("xtu,tzu->tzxu", self.Ey_xtu, self.Pt_zu)
    #     EtY_u = np.einsum("tzxu,zu,xu->tu", EtY_zxu, self.Pz_u, self.Px_u)
    #     Pzxt_u = np.einsum("zu,xu,tzu->zxtu", self.Pz_u, self.Px_u, self.Pt_zu)

    #     Pzxt = np.einsum("zxtu,u->zxt", Pzxt_u, self.Pu)
    #     Pzt = np.einsum("zxt->zt", Pzxt)
    #     Pxt = np.einsum("zxt->xt", Pzxt)
    #     Pt = np.einsum("zt->t", Pzt)
    #     Pz_t = np.einsum("zt,t->zt", Pzt, Pt ** -1)
    #     Px_t = np.einsum("xt,t->xt", Pxt, Pt ** -1)
    #     Pzx_t = np.einsum("zxt->zxt", Pzxt, Pt ** -1)
    #     Mzy_t = np.einsum("zu,xtu,xu,u->zt", self.Pz_u, self.Ey_xtu, self.Px_u, self.Pu)
    #     Mzxy_t = np.einsum("zu,xtu,xu,u->zxt", self.Pz_u, self.Ey_xtu, self.Pu)

    #     # === UNCONDITIONAL ===
    #     # first moments
    #     E_Z = np.einsum("zu,u->z", self.Pz_u, self.Pu)
    #     E_X = np.einsum("xu,u->x", self.Px_u, self.Pu)
    #     E_tY = np.einsum("xtu,tzu,u->t", EtY_u, self.Pu)
    #     # second moments
    #     M_ZX = np.einsum("zu,xu,u->zx", self.Pz_u, self.Px_u, self.Pu)  # (Z, X) + marginalize over U
    #     M_ZtY = np.einsum("zu,tzxu,xu,u->zt", self.Pz_u, EtY_zxu, self.Px_u, self.Pu)  # (Z, tY) + marginalize over X, U
    #     M_XtY = np.einsum("xu,xtu,tzu,zu,u->xt", self.Px_u,  self.Pz_u, self.Pu)  # (X, tY) + marginalize over Z, U
    #     # third moments
    #     M_ZXtY = np.einsum("zu,xu,xtu,tzu,u->zxt", self.Pz_u, self.Px_u, self.Ey_xtu, self.Pt_zu, self.Pu)  # (Z, X, tY) + marginalize over U

    #     # === CONDITIONAL ===
    #     # first moments
    #     E_Z_T = {t: Pz_t[:, t] for t in range(2)}
    #     E_X_T = {t: Px_t[:, t] for t in range(2)}
    #     # second moments
    #     M_ZX_T = {t: Pzx_t[:, :, t] for t in range(2)}
    #     M_ZY_T = {t: Mzy_t[:, :, t] for t in range(2)}
    #     # third moments
    #     M_ZXY_T = {t: Mzxy_t[:, :, t] for t in range(2)}

    #     obs_moments = ObservableMoments(
    #         E_Z, E_X, E_tY,
    #         M_ZX, M_ZtY, M_XtY,
    #         M_ZXtY,
    #         E_Z_T, E_X_T,
    #         M_ZX_T, M_ZY_T,
    #         M_ZXY_T
    #     )
    #     return obs_moments
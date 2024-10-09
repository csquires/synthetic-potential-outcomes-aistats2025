# === IMPORTS: THIRD-PARTY ===
import numpy as np

# === IMPORTS: LOCAL ===
from src.problem_dims import BinaryProblemDimensions
from src.observable_moments import ObservableMoments


def lookup_binary(Vab: np.ndarray, avals, bvals):
    nsamples = len(avals)

    onehot = np.zeros((nsamples, 4))
    onehot[:, 0] = (1 - avals) * (1 - bvals)    # A = 0, B = 0
    onehot[:, 1] = (1 - avals) * bvals          # A = 0, B = 1
    onehot[:, 2] = avals * (1 - bvals)          # A = 1, B = 0
    onehot[:, 3] = avals * bvals                # A = 1, B = 1

    flatV = Vab.flatten()
    vals = np.einsum("id,d->i", onehot, flatV)
    return vals


class BinaryGenerator:
    def __init__(
            self, 
            problem_dims: BinaryProblemDimensions,
            lambda_treatment: float = 0,  # (0 = U->T only), (1 = W->T only)
            lambda_outcome: float = 0  # (0 = T,U->Y only), (1 = T,X->Y only)
        ):
        self.problem_dims = problem_dims
        self.lambda_treatment = lambda_treatment
        self.lambda_outcome = lambda_outcome

        # P(U)
        self.Pu = np.ndarray([0.5, 0.5])

        # P(Z | U)
        self.Pz_u = np.zeros((2, 2))
        self.Pz_u[:, 0] = np.array([1/4, 3/4])  # given U = 0
        self.Pz_u[:, 1] = np.array([3/4, 1/4])  # given U = 1

        # P(X | U)
        self.Px_u = np.zeros((2, 2))
        self.Px_u[:, 0] = np.array([1/4, 3/4])  # given U = 0
        self.Px_u[:, 1] = np.array([3/4, 1/4])  # given U = 1

        # P(T | Z, U)
        a = lambda_treatment
        self.Pt_zu = np.zeros((2, 2, 2))
        self.Pt_zu[:, 0, 0] = np.array([1/4, 3/4])              # given Z = 0, U = 0
        self.Pt_zu[:, 0, 1] = np.array([3/4 - a/2, 1/4 + a/2])  # given Z = 0, U = 1
        self.Pt_zu[:, 1, 0] = np.array([1/4 + a/2, 3/4 - a/2])  # given Z = 1, U = 0
        self.Pt_zu[:, 1, 1] = np.array([3/4, 1/4])              # given Z = 1, U = 1

        # E(Y0 | X, U)
        self.Ey0_xu = np.zeros((2, 2))
        b = self.lambda_outcome
        self.Ey0_xu[0, 0] = 1 + 3 * b      # given X = 0, U = 0
        self.Ey0_xu[0, 1] = 4 * b          # given X = 0, U = 1
        self.Ey0_xu[1, 0] = 1 - b          # given X = 1, U = 0
        self.Ey0_xu[1, 1] = 0              # given X = 1, U = 1

        # E(Y1 | X, U)
        self.Ey1_xu = np.zeros((2, 2))
        b = self.lambda_outcome
        self.Ey1_xu[0, 0] = 6 * b          # given X = 0, U = 0
        self.Ey1_xu[0, 1] = 5 + b          # given X = 0, U = 1
        self.Ey1_xu[1, 0] = 2 * b          # given X = 1, U = 0
        self.Ey1_xu[1, 1] = 5 - 3 * b      # given X = 1, U = 1

        # E(Y | X, T, U) --- repetitive, but convenient
        self.Ey_xtu = np.zeros((2, 2, 2))
        self.Ey_xtu[:, 0, :] = self.Ey0_xu
        self.Ey_xtu[:, 1, :] = self.Ey1_xu

    def generate(self, nsamples: int):
        nproxies = self.problem_dims.nz + self.problem_dims.nx
        y_ix = self.problem_dims.y_ix
        t_ix = self.problem_dims.t_ix
        u_ix = self.problem_dims.u_ix
        full_samples = np.ndarray((nsamples, nproxies + 3))

        # U
        u_vals = np.random.binomial(n=1, p=0.5, size=nsamples)

        # X | U
        x_cutoffs = (1 - u_vals) * 3/4 + u_vals * 1/4
        x_vals = np.random.uniform(size=nsamples) < x_cutoffs

        # Z | U
        z_cutoffs = (1 - u_vals) * 3/4 + u_vals * 1/4
        z_vals = np.random.uniform(size=nsamples) < z_cutoffs
        
        # T | Z, U
        t_cutoffs = lookup_binary(self.Pt_zu[1, :, :], z_vals, u_vals)
        t_vals = np.random.uniform(size=nsamples) < t_cutoffs
        
        # Y | X, T, U
        y0_vals = lookup_binary(self.Ey0_xu, x_vals, u_vals)
        y1_vals = lookup_binary(self.Ey1_xu, x_vals, u_vals)
        y_noise = np.random.normal(size=nsamples)
        y_vals = t_vals * y1_vals + (1 - t_vals) * y0_vals + y_noise

        z_ix, x_ix = 0, 1
        full_samples[:, z_ix] = z_vals
        full_samples[:, x_ix] = x_vals
        full_samples[:, y_ix] = y_vals
        full_samples[:, t_ix] = t_vals
        full_samples[:, u_ix] = u_vals
        
        obs_samples = full_samples[:, :-1]
        return full_samples, obs_samples
    
    def get_observable_moments(self):
        EtY_zxu = np.einsum("xtu,tzu->tzxu", self.Ey_xtu, self.Pt_zu)
        EtY_u = np.einsum("tzxu,zu,xu->tu", EtY_zxu, self.Pz_u, self.Px_u)
        Pzxt_u = np.einsum("zu,xu,tzu->zxtu", self.Pz_u, self.Px_u, self.Pt_zu)

        Pzxt = np.einsum("zxtu,u->zxt", Pzxt_u, self.Pu)
        Pzt = np.einsum("zxt->zt", Pzxt)
        Pxt = np.einsum("zxt->xt", Pzxt)
        Pt = np.einsum("zt->t", Pzt)
        Pz_t = np.einsum("zt,t->zt", Pzt, Pt ** -1)
        Px_t = np.einsum("xt,t->xt", Pxt, Pt ** -1)
        Pzx_t = np.einsum("zxt->zxt", Pzxt, Pt ** -1)
        Mzy_t = np.einsum("zu,xtu,xu,u->zt", self.Pz_u, self.Ey_xtu, self.Px_u, self.Pu)
        Mzxy_t = np.einsum("zu,xtu,xu,u->zxt", self.Pz_u, self.Ey_xtu, self.Pu)

        # === UNCONDITIONAL ===
        # first moments
        E_Z = np.einsum("zu,u->z", self.Pz_u, self.Pu)
        E_X = np.einsum("xu,u->x", self.Px_u, self.Pu)
        E_tY = np.einsum("xtu,tzu,u->t", EtY_u, self.Pu)
        # second moments
        M_ZX = np.einsum("zu,xu,u->zx", self.Pz_u, self.Px_u, self.Pu)  # (Z, X) + marginalize over U
        M_ZtY = np.einsum("zu,tzxu,xu,u->zt", self.Pz_u, EtY_zxu, self.Px_u, self.Pu)  # (Z, tY) + marginalize over X, U
        M_XtY = np.einsum("xu,xtu,tzu,zu,u->xt", self.Px_u,  self.Pz_u, self.Pu)  # (X, tY) + marginalize over Z, U
        # third moments
        M_ZXtY = np.einsum("zu,xu,xtu,tzu,u->zxt", self.Pz_u, self.Px_u, self.Ey_xtu, self.Pt_zu, self.Pu)  # (Z, X, tY) + marginalize over U

        # === CONDITIONAL ===
        # first moments
        E_Z_T = {t: Pz_t[:, t] for t in range(2)}
        E_X_T = {t: Px_t[:, t] for t in range(2)}
        # second moments
        M_ZX_T = {t: Pzx_t[:, :, t] for t in range(2)}
        M_ZY_T = {t: Mzy_t[:, :, t] for t in range(2)}
        # third moments
        M_ZXY_T = {t: Mzxy_t[:, :, t] for t in range(2)}

        obs_moments = ObservableMoments(
            E_Z, E_X, E_tY,
            M_ZX, M_ZtY, M_XtY,
            M_ZXtY,
            E_Z_T, E_X_T,
            M_ZX_T, M_ZY_T,
            M_ZXY_T
        )
        return obs_moments
    
    # def true_marginal(self):
    #     proxy_conditionals = [self.proxy_conditional(i) for i in range(self.problem_dims.nx + self.problem_dims.nz)]
        
    #     current_marginal = np.einsum("ytu,tu,u->ytu", self.Py_tu, self.Pt_u, self.Pu)
    #     for proxy_conditional in reversed(proxy_conditionals):
    #         current_marginal = np.einsum("vu,...u->v...u", proxy_conditional, current_marginal)

    #     dz, dx = self.problem_dims.dz, self.problem_dims.dx
    #     dy, dt, du = 2, self.problem_dims.ntreatments, self.problem_dims.ngroups
    #     final_marginal = current_marginal.reshape(dz, dx, dy, dt, du)
    #     return final_marginal
    


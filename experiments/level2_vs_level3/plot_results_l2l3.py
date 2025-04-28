# === IMPORTS: BUILT-IN ===
import os
import pickle

# === IMPORTS: THIRD-PARTY ===
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# === IMPORTS: LOCAL ===
from src.causal_moments.causal_moments_discrete import compute_potential_outcome_moments_discrete
from src.mixture_moments.mixture_moments_discrete import compute_mixture_moments_discrete
from src.mixture_moments import MixtureMoments

plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'

FIGURE_FOLDER = "experiments/level2_vs_level3/figures"
POSTER = True
PLOT_PARAFAC = False
if POSTER:
    FIGSIZE = (16,8)
    FILENAME = "l2_vs_l3_errors_poster.pdf"
else:
    FIGSIZE = (4,8)
    FILENAME = "l2_vs_l3_errors.pdf"

def mixture2mte(estimated_mixture: MixtureMoments):
    est_Pytu = estimated_mixture.ES_U.reshape((2, 2, 2))
    est_Ptu = np.einsum("ytu->tu", est_Pytu)
    est_Py_tu = np.einsum("ytu,tu->ytu", est_Pytu, est_Ptu ** -1)
    EY1_U = est_Py_tu[1, 1]
    EY0_U = est_Py_tu[1, 0]
    return EY1_U - EY0_U


def mte_error(
    true_source_probs,
    true_mtes,
    estimated_source_probs,
    estimated_mtes
):
    # TODO: because of permutations, this should do best matching
    true_mtes_sorted = np.sort(true_mtes)
    estimated_mtes_sorted = np.sort(estimated_mtes)
    term1 = np.sum((true_mtes_sorted - estimated_mtes_sorted) ** 2)
    term2 = np.sum((true_source_probs - estimated_source_probs) ** 2)
    return term1 + term2


def mixture_error(
    true_dist: np.ndarray,
    estimated_mixture: MixtureMoments
):
    true_dist = true_dist.reshape((2, 2, 4, 2))
    estimated_dist = np.einsum(
        "zu,xu,su,u->zxsu",
        estimated_mixture.EZ_U,
        estimated_mixture.EX_U,
        estimated_mixture.ES_U,
        estimated_mixture.Pu
    )
    estimated_dist_perm = np.zeros(estimated_dist.shape)
    estimated_dist_perm[:, :, :, 0] = estimated_dist[:, :, :, 1]
    estimated_dist_perm[:, :, :, 1] = estimated_dist[:, :, :, 0]

    diff = np.sum(np.abs(true_dist - estimated_dist))
    diff_perm = np.sum(np.abs(true_dist - estimated_dist_perm))
    return min(diff, diff_perm)



# === LOAD RESULTS ===
results = pickle.load(open("experiments/level2_vs_level3/results.pkl", "rb"))
zt_strengths = results["zt_strengths"]
true_dists = results["true_dists"]
estimated_source_probs = results["estimated_source_probs"]
estimated_mtes = results["estimated_mtes"]
estimated_mixtures = results["estimated_mixtures"]
nruns = estimated_source_probs[0].shape[0]

estimated_mtes_mixtures = {
    zt_strength: np.zeros((nruns, 2))
    for zt_strength in zt_strengths
}

mte_errors = np.zeros((len(zt_strengths), nruns))
mte_errors_parafac = np.zeros((len(zt_strengths), nruns))
mixture_errors = np.zeros((len(zt_strengths), nruns))
for s_ix, zt_strength in enumerate(zt_strengths):
    # get true moments
    true_causal_moments = compute_potential_outcome_moments_discrete(true_dists[zt_strength], 1)
    true_source_probs_mu = true_causal_moments.Pu
    true_mtes_mu = np.array([true_causal_moments.E_R_U[0], true_causal_moments.E_R_U[1]])

    for r_ix in range(nruns):
        estimated_source_probs_mu = estimated_source_probs[zt_strength][r_ix]
        estimated_mtes_mu = estimated_mtes[zt_strength][r_ix]
        estimated_mixture_moments_mu: MixtureMoments = estimated_mixtures[zt_strength][r_ix]
        estimated_mtes_mixture_mu = mixture2mte(estimated_mixture_moments_mu)
        estimated_mtes_mixtures[zt_strength][r_ix] = estimated_mtes_mixture_mu

        mte_errors[s_ix, r_ix] = mte_error(
            true_source_probs_mu,
            true_mtes_mu,
            estimated_source_probs_mu,
            estimated_mtes_mu
        )
        mixture_errors[s_ix, r_ix] = mixture_error(
            true_dists[zt_strength], 
            estimated_mixture_moments_mu
        )
        mte_errors_parafac[s_ix, r_ix] = mte_error(
            true_source_probs_mu,
            true_mtes_mu,
            estimated_mixture_moments_mu.Pu,
            estimated_mtes_mixture_mu
        )


# ngroups = 2
# fig, axes = plt.subplots(2, ngroups, figsize=(8, 6))
# for k in range(ngroups):
#     axes[0, k].hist(all_estimated_source_probs[:, k])
#     axes[0, k].axvline(true_source_probs[k], color='k')
#     label = fr"Estimated $\mathbb{{P}}(U={k})$"
#     axes[0, k].set_xlabel(label, fontsize=16)

#     axes[1, k].hist(all_estimated_means[:, k])
#     axes[1, k].axvline(true_means[k], color='k')
#     label = fr"Estimated $\mathbb{{E}}[Y^{{(1)}} - Y^{{(0)}} \mid U={k}]$"
#     axes[1, k].set_xlabel(label, fontsize=16)





os.makedirs(FIGURE_FOLDER, exist_ok=True)
sns.set_theme()
plt.clf()
pylab.rcParams.update({"xtick.labelsize": "large", "ytick.labelsize": "large"})

ax0_line = np.mean(mixture_errors)
ax0_ylim = (ax0_line - 2 * np.std(mixture_errors), ax0_line + 2 * np.std(mixture_errors))

ax1_line = np.mean(mte_errors)
ax1_ylim = (ax1_line - 2 * np.std(mixture_errors), ax1_line + 2 * np.std(mixture_errors))

ax0_middle = np.mean(mixture_errors, axis=1)
ax0_stds = np.std(mixture_errors, axis=1)
ax0_lower = ax0_middle - ax0_stds
ax0_upper = ax0_middle + ax0_stds
# ax0_lower = np.quantile(mixture_errors, 0.25, axis=1)
# ax0_upper = np.quantile(mixture_errors, 0.75, axis=1)

ax1_middle = np.mean(mte_errors, axis=1)
ax1_stds = np.std(mte_errors, axis=1)
ax1_lower = ax1_middle - ax1_stds
ax1_upper = ax1_middle + ax1_stds
ax1_middle_parafac = np.mean(mte_errors_parafac, axis=1)
ax1_stds_parafac = np.std(mte_errors_parafac, axis=1)
ax1_lower_parafac = ax1_middle_parafac - ax1_stds
ax1_upper_parafac = ax1_middle_parafac + ax1_stds
# ax1_lower = np.quantile(mte_errors, 0.25, axis=1)
# ax1_upper = np.quantile(mte_errors, 0.75, axis=1)

fig, axes = plt.subplots(2, 1, figsize=FIGSIZE)
axes[0].axhline(ax0_line, linestyle="--", color="gray")
axes[0].plot(zt_strengths, ax0_middle)
axes[0].fill_between(zt_strengths, ax0_lower, ax0_upper, alpha=0.5)
axes[0].set_ylim(*ax0_ylim)
axes[0].set_ylabel(fr"Mixture estimation error", fontsize=24)
axes[0].set_xticklabels([])

axes[1].axhline(ax1_line, linestyle="--", color="gray")
axes[1].plot(zt_strengths, ax1_middle, label="SPO")
axes[1].fill_between(zt_strengths, ax1_lower, ax1_upper, alpha=0.5)
if PLOT_PARAFAC:
    axes[1].plot(zt_strengths, ax1_middle_parafac, label="NN-CP")
    axes[1].fill_between(zt_strengths, ax1_lower_parafac, ax1_upper_parafac, alpha=0.5)
axes[1].set_ylim(*ax1_ylim)
axes[1].set_xlabel(fr"$\mu_{{zt}}$", fontsize=24)
if PLOT_PARAFAC:
    axes[1].legend()
axes[1].set_ylabel(fr"MTE estimation error", fontsize=24)


plt.tight_layout()
plt.savefig(f"{FIGURE_FOLDER}/{FILENAME}")
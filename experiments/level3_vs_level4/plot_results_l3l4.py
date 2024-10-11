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
from src.mixture_moments import MixtureMoments

plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'

FIGURE_FOLDER = "experiments/level3_vs_level4/figures"


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


def mixture2mte(estimated_mixture: MixtureMoments):
    est_Pytu = estimated_mixture.ES_U.reshape((2, 2, 2))
    est_Ptu = np.einsum("ytu->tu", est_Pytu)
    est_Py_tu = np.einsum("ytu,tu->ytu", est_Pytu, est_Ptu ** -1)
    EY1_U = est_Py_tu[1, 1]
    EY0_U = est_Py_tu[1, 0]
    return EY1_U - EY0_U


# === LOAD RESULTS ===
results = pickle.load(open("experiments/level3_vs_level4/results.pkl", "rb"))
xy_strengths = results["xy_strengths"]
true_dists = results["true_dists"]
estimated_source_probs = results["estimated_source_probs"]
estimated_mtes = results["estimated_mtes"]
estimated_ates = results["estimated_ates"]
estimated_mixtures = results["estimated_mixtures"]


nruns = estimated_source_probs[0].shape[0]
mte_errors = np.zeros((len(xy_strengths), nruns))
mte_errors_parafac = np.zeros((len(xy_strengths), nruns))
ate_errors = np.zeros((len(xy_strengths), nruns))
for s_ix, xy_strength in enumerate(xy_strengths):
    # get true moments
    causal_moments = compute_potential_outcome_moments_discrete(true_dists[xy_strength], 1)
    true_source_probs_mu = causal_moments.Pu
    true_mtes_mu = np.array([causal_moments.E_R_U[0], causal_moments.E_R_U[1]])
    true_ate_mu = causal_moments.moments_R[1]

    for r_ix in range(nruns):
        estimated_source_probs_r = estimated_source_probs[xy_strength][r_ix]
        estimated_mtes_r = estimated_mtes[xy_strength][r_ix]
        estimated_mixture_moments_mu: MixtureMoments = estimated_mixtures[xy_strength][r_ix]
        estimated_mtes_mixture_mu = mixture2mte(estimated_mixture_moments_mu)

        estimated_ate_r = estimated_ates[xy_strength][r_ix]
        mte_errors[s_ix, r_ix] = mte_error(
            true_source_probs_mu,
            true_mtes_mu,
            estimated_source_probs_r,
            estimated_mtes_r
        )
        ate_errors[s_ix, r_ix] = (estimated_ate_r - true_ate_mu)**2
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

ax0_line = np.mean(mte_errors)
ax0_ylim = (ax0_line - 2 * np.std(mte_errors), ax0_line + 2 * np.std(mte_errors))

ax1_line = np.mean(ate_errors)
ax1_ylim = (ax1_line - 2 * np.std(ate_errors), ax1_line + 2 * np.std(ate_errors))

ax0_middle = np.mean(mte_errors, axis=1)
ax0_stds = np.std(mte_errors, axis=1)
ax0_lower = ax0_middle - ax0_stds
ax0_upper = ax0_middle + ax0_stds
ax0_middle_parafac = np.mean(mte_errors_parafac, axis=1)
ax0_stds_parafac = np.std(mte_errors_parafac, axis=1)
ax0_lower_parafac = ax0_middle_parafac - ax0_stds_parafac
ax0_upper_parafac = ax0_middle_parafac + ax0_stds_parafac
# ax0_lower = np.quantile(mte_errors, 0.25, axis=1)
# ax0_upper = np.quantile(mte_errors, 0.75, axis=1)

ax1_middle = np.mean(ate_errors, axis=1)
ax1_stds = np.std(ate_errors, axis=1)
ax1_lower = ax1_middle - ax1_stds
ax1_upper = ax1_middle + ax1_stds
# ax1_lower = np.quantile(ate_errors, 0.25, axis=1)
# ax1_upper = np.quantile(ate_errors, 0.75, axis=1)

fig, axes = plt.subplots(2, 1, figsize=(4, 8))
axes[0].axhline(ax0_line, linestyle="--", color="gray")
axes[0].plot(xy_strengths, ax0_middle)
axes[0].fill_between(xy_strengths, ax0_lower, ax0_upper, alpha=0.5)
axes[0].plot(xy_strengths, ax0_middle_parafac)
axes[0].fill_between(xy_strengths, ax0_lower_parafac, ax0_upper_parafac, alpha=0.5)
axes[0].set_ylim(*ax0_ylim)
axes[0].set_ylabel(fr"MTE estimation error", fontsize=24)
axes[0].set_xticklabels([])

axes[1].axhline(ax1_line, linestyle="--", color="gray")
axes[1].plot(xy_strengths, ax1_middle)
axes[1].fill_between(xy_strengths, ax1_lower, ax1_upper, alpha=0.5)
axes[1].set_ylim(*ax1_ylim)
axes[1].set_xlabel(fr"$\mu_{{xy}}$", fontsize=24)
axes[1].set_ylabel(fr"ATE estimation error", fontsize=24)


plt.tight_layout()
plt.savefig(f"{FIGURE_FOLDER}/l3_vs_l4_errors.pdf")
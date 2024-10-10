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



# === LOAD RESULTS ===
results = pickle.load(open("experiments/level3_vs_level4/results.pkl", "rb"))
xy_strengths = results["xy_strengths"]
true_dists = results["true_dists"]
estimated_source_probs = results["estimated_source_probs"]
estimated_mtes = results["estimated_mtes"]
estimated_ates = results["estimated_ates"]


nruns = estimated_source_probs[0].shape[0]
mte_errors = np.zeros((len(xy_strengths), nruns))
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
        estimated_ate_r = estimated_ates[xy_strength][r_ix]
        mte_errors[s_ix, r_ix] = mte_error(
            true_source_probs_mu,
            true_mtes_mu,
            estimated_source_probs_r,
            estimated_mtes_r
        )
        ate_errors[s_ix, r_ix] = (estimated_ate_r - true_ate_mu)**2


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

ax0_mean = mte_errors.mean()
ax1_mean = ate_errors.mean()

fig, axes = plt.subplots(2, 1, figsize=(8, 8))
axes[0].plot(xy_strengths, mte_errors.mean(axis=1))
axes[0].fill_between(xy_strengths, np.quantile(mte_errors, 0.25, axis=1), np.quantile(mte_errors, 0.75, axis=1), alpha=0.5)
axes[0].axhline(ax0_mean, linestyle="--", color="gray")
axes[0].set_ylim(0, 2 * ax0_mean)
axes[0].set_ylabel(fr"MTE estimation error", fontsize=24)
axes[0].set_xticklabels([])

axes[1].plot(xy_strengths, ate_errors.mean(axis=1))
axes[1].fill_between(xy_strengths, np.quantile(ate_errors, 0.25, axis=1), np.quantile(ate_errors, 0.75, axis=1), alpha=0.5)
axes[1].axhline(ax1_mean, linestyle="--", color="gray")
axes[1].set_ylim(0, 2 * ax1_mean)
axes[1].set_xlabel(fr"$\mu_{{xy}}$", fontsize=24)
axes[1].set_ylabel(fr"ATE estimation error", fontsize=24)


plt.tight_layout()
plt.savefig(f"{FIGURE_FOLDER}/l3_vs_l4_errors.pdf")
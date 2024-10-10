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

FIGURE_FOLDER = "experiments/level2_vs_level3/figures"


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
results = pickle.load(open("experiments/level2_vs_level3/results.pkl", "rb"))
zt_strengths = results["zt_strengths"]
true_dists = results["true_dists"]
estimated_source_probs = results["estimated_source_probs"]
estimated_mtes = results["estimated_mtes"]
estimate_dists = results["estimated_dists"]


nruns = estimated_source_probs[0].shape[0]
mte_errors = np.zeros((len(zt_strengths), nruns))
for s_ix, zt_strength in enumerate(zt_strengths):
    # get true moments
    causal_moments = compute_potential_outcome_moments_discrete(true_dists[zt_strength], 1)
    true_source_probs_mu = causal_moments.Pu
    true_mtes_mu = np.array([causal_moments.E_R_U[0], causal_moments.E_R_U[1]])

    for r_ix in range(nruns):
        estimated_source_probs_mu = estimated_source_probs[zt_strength][r_ix]
        estimated_mtes_mu = estimated_mtes[zt_strength][r_ix]
        mte_errors[s_ix, r_ix] = mte_error(
            true_source_probs_mu,
            true_mtes_mu,
            estimated_source_probs_mu,
            estimated_mtes_mu
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

ax1_mean = mte_errors.mean()

fig, axes = plt.subplots(2, 1, figsize=(8, 8))
axes[0].set_ylabel(fr"Mixture estimation error", fontsize=24)

axes[1].axhline(ax1_mean, linestyle="--", color="gray")
axes[1].plot(zt_strengths, mte_errors.mean(axis=1))
axes[1].fill_between(zt_strengths, np.quantile(mte_errors, 0.25, axis=1), np.quantile(mte_errors, 0.75, axis=1), alpha=0.5)
axes[1].set_ylim(0, 2 * ax1_mean)
axes[1].set_xlabel(fr"$\mu_{{zt}}$", fontsize=24)
axes[1].set_ylabel(fr"MTE estimation error", fontsize=24)


plt.tight_layout()
plt.savefig(f"{FIGURE_FOLDER}/l2_vs_l3_mte_errors.pdf")
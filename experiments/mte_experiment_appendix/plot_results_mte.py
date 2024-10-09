# === IMPORTS: BUILT-IN ===
import os
import pickle

# === IMPORTS: THIRD-PARTY ===
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'

FIGURE_FOLDER = "experiments/mte_experiment_appendix/figures"


# === LOAD RESULTS ===
results = pickle.load(open("experiments/mte_experiment_appendix/results.pkl", "rb"))
true_source_probs = results["true_source_probs"]
true_means = results["true_means"]
all_estimated_source_probs = results["all_estimated_source_probs"]
all_estimated_means = results["all_estimated_means"]


os.makedirs(FIGURE_FOLDER, exist_ok=True)
sns.set_theme()
plt.clf()
ngroups = 2
fig, axes = plt.subplots(2, ngroups, figsize=(8, 6))
for k in range(ngroups):
    axes[0, k].hist(all_estimated_source_probs[:, k])
    axes[0, k].axvline(true_source_probs[k], color='k')
    label = fr"Estimated $\mathbb{{P}}(U={k})$"
    axes[0, k].set_xlabel(label, fontsize=16)

    axes[1, k].hist(all_estimated_means[:, k])
    axes[1, k].axvline(true_means[k], color='k')
    label = fr"Estimated $\mathbb{{E}}[Y^{{(1)}} - Y^{{(0)}} \mid U={k}]$"
    axes[1, k].set_xlabel(label, fontsize=16)


# === SHOW PLOTS ===
plt.tight_layout()
plt.savefig(f"{FIGURE_FOLDER}/source_probs_and_means.pdf")
# === IMPORTS: BUILT-IN ===
import os
import pickle

# === IMPORTS: THIRD-PARTY ===
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsfonts}'


FIGURE_FOLDER = "experiments/ate_experiment_appendix/figures"


# === LOAD RESULTS ===
results = pickle.load(open("experiments/ate_experiment_appendix/results.pkl", "rb"))
y0_ests = results["y0_ests"]
y1_ests = results["y1_ests"]
true_mean_y0 = results["true_mean_y0"]
true_mean_y1 = results["true_mean_y1"]
ate_ests = y1_ests - y0_ests

plt.clf()
plt.figure(figsize=(4, 6))
plt.hist(ate_ests)
plt.axvline(true_mean_y1 - true_mean_y0, color='k')
plt.xlabel(r'Estimated $\mathbb{E}[Y^{(1)} - Y^{(0)}]$', fontsize=18)
plt.tight_layout()
os.makedirs(FIGURE_FOLDER, exist_ok=True)
plt.savefig(f"{FIGURE_FOLDER}/ate_ests.pdf")
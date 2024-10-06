
# Code for *Synthetic Potential Outcomes and Causal Mixture Identifiability*

## Setup instructions
Our code was tested with Python 3.9 and should be compatible with Python>=3.9. To install the necessary packages:
```
bash setup.sh
```

## Instructions for replication

Replicating the results should take about 1-2 minutes.

All commands should be run from the top-level directory. First, activate the virtual environment and enter into `ipython`:
```
source venv/bin/activate
ipython
```

Then, run the experiments with these commands:

**Experiment 1**:
```
run experiments/ate_experiment/collect_results_ate.py
run experiments/ate_experiment/plot_results_ate.py
```

The resulting plot will be saved at `experiments/ate_experiment/figures/ate_ests.pdf`.

**Experiment 2**:
```
run experiments/mte_experiment/collect_results_mte.py
run experiments/mte_experiment/plot_results_mte.py
```

The resulting plot will be saved at `experiments/mte_experiment/figures/source_probs_and_means.pdf`.
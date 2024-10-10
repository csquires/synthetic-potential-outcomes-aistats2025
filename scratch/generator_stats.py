# === IMPORTS: BUILT-IN ===
import pickle

# === IMPORTS: THIRD-PARTY ===
import numpy as np
from tqdm import trange

# === IMPORTS: LOCAL ===
from src.data_generation.generator_main import BinaryGeneratorMain

from src.observable_moments.population_moments_discrete import compute_observable_moments_discrete


np.random.seed(123)
# ==== RUN METHOD ====
nsamples = int(1e3)
zt_strengths = np.linspace(0, 1, 11)
nruns = 100

true_dists = dict()
estimated_source_probs = {zt_strength: np.zeros((nruns, 2)) for zt_strength in zt_strengths}
estimated_mtes = {zt_strength: np.zeros((nruns, 2)) for zt_strength in zt_strengths}
estimated_mixtures = {zt_strength: dict() for zt_strength in zt_strengths}

generator = BinaryGeneratorMain(zt_strength=0, xy_strength=0)
# compute_observable_moments_discrete()
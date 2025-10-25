# src/bayesian_estimator/estimator.py
from typing import List, Dict, Tuple
import numpy as np
from .hierarchical_model import aggregate_over_K

class BayesianEntropyEstimator:
    """
    Main wrapper for Bayesian semantic entropy estimation per prompt.
    """

    def __init__(self,
                 prior_K_path: str,
                 alpha_prior: float = 0.5,
                 num_mc_samples: int = 500,
                 use_truncation: bool = True,
                 random_state: int = 0):
        self.prior_K_path = prior_K_path
        self.alpha_prior = alpha_prior
        self.num_mc_samples = num_mc_samples
        self.use_truncation = use_truncation
        self.random_state = random_state

    def _aggregate_counts_and_masses(self, samples: List[Dict]) -> Tuple[List[int], List[float]]:
        """
        samples: list of dicts like {'text': ..., 'meaning_id': int, 'prob': float}
        returns:
          counts: [c0, c1, ...] for observed unique meanings (ordered by meaning_id)
          observed_masses: [sum_prob_for_meaning0, ...] (same order)
        """
        # group by meaning_id
        from collections import defaultdict
        counts_map = defaultdict(int)
        mass_map = defaultdict(float)
        for s in samples:
            mid = s["meaning_id"]
            counts_map[mid] += 1
            mass_map[mid] += float(s.get("prob", 0.0))

        # order by meaning id to make deterministic
        keys = sorted(counts_map.keys())
        counts = [counts_map[k] for k in keys]
        masses = [mass_map[k] for k in keys]
        return counts, masses

    def estimate(self, samples: List[Dict]) -> Dict:
        """
        samples: list of {'text': ..., 'meaning_id': int, 'prob': float}
        returns: dict with E_h, Var_h, Kmin, N_samples
        """
        counts, masses = self._aggregate_counts_and_masses(samples)
        Kmin = len(counts)
        N = sum(counts)
        E_h, Var_h = aggregate_over_K(counts=counts,
                                      observed_mass_per_meaning=masses,
                                      prior_K_path=self.prior_K_path,
                                      alpha_scalar=self.alpha_prior,
                                      use_truncation=self.use_truncation,
                                      num_mc_samples=self.num_mc_samples,
                                      random_state=self.random_state)
        return {
            "E_h": E_h,
            "Var_h": Var_h,
            "Kmin": Kmin,
            "N": N
        }

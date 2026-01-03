import numpy as np
from scipy.stats import dirichlet

class BayesianSemanticEntropy:
    def __init__(self, alpha=1.0, max_mc_samples=5000):
        self.alpha = alpha
        self.max_mc_samples = max_mc_samples

    def _shannon_entropy(self, probability_vector):
        p = np.array(probability_vector) + 1e-12  # Small value to avoid log(0)
        return -np.sum(p * np.log(p))

    def estimate_entropy(self, samples):
        """
        This function estimates both entropy and variance from the sample set.
        Args:
        - samples: List of samples where each sample is a dictionary
                   containing {'meaning_id': ..., 'probability': ..., 'text': ...}
        Returns:
        - entropy: Estimated entropy of the samples.
        - variance: Estimated variance of the entropy.
        """
        if not samples:
            return 0.0, 0.0

        meaning_counts = {}
        for s in samples:
            m_id = s['meaning_id']
            prob = s['probability']
            meaning_counts[m_id] = meaning_counts.get(m_id, 0.0) + prob

        unique_meanings = sorted(meaning_counts.keys())
        lower_bounds = np.array([meaning_counts[m] for m in unique_meanings])

        if np.sum(lower_bounds) >= 1.0 - 1e-6:
            return 0.0, 0.0

        K_observed = len(unique_meanings)

        entropy_estimates = []
        for num_unseen in range(0, 5):  # Assuming 0 to 4 unseen categories
            K_total = K_observed + num_unseen
            alpha_vec = np.full(K_total, self.alpha)

            try:
                candidate_vectors = dirichlet.rvs(alpha_vec, size=self.max_mc_samples)
            except ValueError:
                continue

            observed_candidates = candidate_vectors[:, :K_observed]
            valid_mask = np.all(observed_candidates >= lower_bounds, axis=1)
            valid_vectors = candidate_vectors[valid_mask]

            if len(valid_vectors) == 0:
                continue

            entropies = [self._shannon_entropy(vec) for vec in valid_vectors]
            avg_entropy_for_K = np.mean(entropies)
            entropy_estimates.append(avg_entropy_for_K)

        if not entropy_estimates:
            return 0.0, 0.0

        final_entropy = np.mean(entropy_estimates)

        # Variance of entropy estimates
        entropy_variance = np.var(entropy_estimates)
        
        return final_entropy, entropy_variance

    def adaptive_estimator(self, samples):
        """
        This function estimates entropy and variance based on the adaptive estimation method.
        Args:
        - samples: List of samples where each sample is a dictionary
                   containing {'meaning_id': ..., 'probability': ..., 'text': ...}
        Returns:
        - entropy: Final estimated entropy of the samples.
        - variance: Final estimated variance of the entropy.
        """
        if not samples:
            return 0.0, 0.0

        meaning_counts = {}
        for s in samples:
            m_id = s['meaning_id']
            prob = s['probability']
            meaning_counts[m_id] = meaning_counts.get(m_id, 0.0) + prob

        unique_meanings = sorted(meaning_counts.keys())
        lower_bounds = np.array([meaning_counts[m] for m in unique_meanings])

        if np.sum(lower_bounds) >= 1.0 - 1e-6:
            return 0.0, 0.0

        K_observed = len(unique_meanings)

        means_per_K = []
        vars_per_K = []

        for num_unseen in range(0, 5):  # Considering unseen categories
            K_total = K_observed + num_unseen
            alpha_vec = np.full(K_total, self.alpha)

            try:
                candidate_vectors = dirichlet.rvs(alpha_vec, size=self.max_mc_samples)
            except ValueError:
                continue

            observed_candidates = candidate_vectors[:, :K_observed]
            valid_mask = np.all(observed_candidates >= lower_bounds, axis=1)
            valid_vectors = candidate_vectors[valid_mask]

            if len(valid_vectors) < 2:  # Need at least 2 for variance
                continue

            entropies = [self._shannon_entropy(vec) for vec in valid_vectors]
            means_per_K.append(np.mean(entropies))
            vars_per_K.append(np.var(entropies))

        if not means_per_K:
            return 0.0, 0.0
        
        final_entropy = np.mean(means_per_K)

        avg_of_variances = np.mean(vars_per_K)
        var_of_means = np.var(means_per_K)
        
        final_variance = avg_of_variances + var_of_means
        
        return final_entropy, final_variance
